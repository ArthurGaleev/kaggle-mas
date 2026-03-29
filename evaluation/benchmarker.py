"""
Benchmarker — comprehensive model benchmarking and architecture comparison.

Install dependencies:
  pip install scikit-learn pandas numpy scipy
"""
from __future__ import annotations

import json
import logging
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# sklearn imports — guarded so the module can be imported without sklearn
try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class Benchmarker:
    """
    Runs comprehensive K-fold cross-validation benchmarks and compares ML architectures.

    Parameters
    ----------
    cv_folds:
        Number of cross-validation folds (default 5).
    seed:
        Random seed for reproducibility (default 42).
    results_dir:
        Directory where JSON benchmark results are saved.

    Examples
    --------
    ::

        from evaluation.benchmarker import Benchmarker

        bench = Benchmarker(cv_folds=5, results_dir="./outputs/benchmarks")
        results = bench.run_benchmark(state, cfg)
        leaderboard = bench.generate_leaderboard([results])
        print(leaderboard)
    """

    def __init__(
        self,
        cv_folds: int = 5,
        seed: int = 42,
        results_dir: str = "./outputs/benchmarks",
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for Benchmarker. "
                "Install with: pip install scikit-learn"
            )
        self.cv_folds   = cv_folds
        self.seed       = seed
        self.results_dir = results_dir
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        state: Dict[str, Any],
        cfg: Any,
    ) -> Dict[str, Any]:
        """
        Run a comprehensive K-fold cross-validation benchmark for each model.

        Metrics recorded per model per fold:
          * MSE, RMSE, MAE, R²
          * Training time (s)
          * Inference time (s)
          * Peak memory delta (MB) via tracemalloc
          * Feature count

        Parameters
        ----------
        state:
            Pipeline state dict containing at minimum:
            ``train_feat`` (pd.DataFrame), ``target_series`` (pd.Series),
            ``models`` (dict of trained model objects, used for architecture only).
        cfg:
            Hydra config with ``pipeline.cv_folds`` and ``models.*`` sections.

        Returns
        -------
        dict
            Benchmark results keyed by model name.  Also saved as JSON.
        """
        train_feat    = state.get("train_feat")
        target_series = state.get("target_series")

        if train_feat is None or target_series is None:
            raise ValueError("state must contain 'train_feat' and 'target_series'.")

        X = np.asarray(train_feat, dtype=np.float32)
        y = np.asarray(target_series, dtype=np.float32)
        n_features = X.shape[1]

        # Resolve CV folds from config
        try:
            n_splits = int(cfg.pipeline.cv_folds)
        except Exception:
            n_splits = self.cv_folds

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        results: Dict[str, Any] = {
            "cv_folds":   n_splits,
            "n_samples":  len(y),
            "n_features": n_features,
            "models":     {},
        }

        # Determine which model types to benchmark from config
        model_types = self._get_model_types(cfg, state)

        for model_name, model_fn in model_types.items():
            logger.info("[Benchmarker] Running %d-fold CV for %s …", n_splits, model_name)
            fold_metrics = self._run_model_cv(
                model_fn, X, y, kf, n_features, model_name
            )
            results["models"][model_name] = fold_metrics

        # Summary stats across folds
        for model_name, fold_data in results["models"].items():
            fold_data["summary"] = self._summarise_folds(fold_data["folds"])

        # Save to disk
        save_path = self._save_results(results)
        results["results_path"] = save_path
        logger.info("[Benchmarker] Benchmark complete. Saved to %s", save_path)
        return results

    # ------------------------------------------------------------------
    # Architecture comparison
    # ------------------------------------------------------------------

    def compare_architectures(
        self,
        results_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare benchmark results from multiple pipeline runs.

        Performs paired t-tests between all model pairs to identify
        statistically significant performance differences.

        Parameters
        ----------
        results_list:
            List of result dicts from :meth:`run_benchmark`.

        Returns
        -------
        dict
            Comparison table and statistical significance results.
        """
        comparison: Dict[str, Any] = {
            "n_architectures": len(results_list),
            "per_run":         [],
            "pairwise_tests":  {},
            "best_architecture_index": None,
        }

        all_mse_per_run: List[Tuple[int, str, List[float]]] = []

        for run_idx, run_results in enumerate(results_list):
            run_entry: Dict[str, Any] = {"run": run_idx, "models": {}}
            for model_name, model_data in run_results.get("models", {}).items():
                summary = model_data.get("summary", {})
                run_entry["models"][model_name] = {
                    "mean_mse":  summary.get("mse", {}).get("mean"),
                    "std_mse":   summary.get("mse", {}).get("std"),
                    "mean_rmse": summary.get("rmse", {}).get("mean"),
                    "mean_r2":   summary.get("r2", {}).get("mean"),
                }
                fold_mse = [f.get("mse", 0) for f in model_data.get("folds", [])]
                all_mse_per_run.append((run_idx, model_name, fold_mse))
            comparison["per_run"].append(run_entry)

        # Pairwise t-tests
        for i, (run_i, name_i, mse_i) in enumerate(all_mse_per_run):
            for j, (run_j, name_j, mse_j) in enumerate(all_mse_per_run):
                if j <= i:
                    continue
                key_a = f"run{run_i}_{name_i}"
                key_b = f"run{run_j}_{name_j}"
                pair_key = f"{key_a} vs {key_b}"

                if len(mse_i) == len(mse_j) and len(mse_i) >= 2:
                    t_stat, p_value = stats.ttest_rel(mse_i, mse_j)
                    comparison["pairwise_tests"][pair_key] = {
                        "t_statistic": float(t_stat),
                        "p_value":     float(p_value),
                        "significant_at_05": bool(p_value < 0.05),
                        "better": key_a if np.mean(mse_i) < np.mean(mse_j) else key_b,
                    }

        # Identify best architecture by lowest mean MSE
        best_idx, best_mean = None, float("inf")
        for run_idx, run_entry in enumerate(comparison["per_run"]):
            for model_name, metrics in run_entry.get("models", {}).items():
                m = metrics.get("mean_mse")
                if m is not None and m < best_mean:
                    best_mean = m
                    best_idx  = run_idx

        comparison["best_architecture_index"] = best_idx
        comparison["best_mean_mse"] = best_mean if best_idx is not None else None
        return comparison

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def generate_leaderboard(
        self,
        results_list: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Generate a ranked leaderboard from a list of benchmark results.

        Ranks all model/run combinations by ascending mean MSE.

        Parameters
        ----------
        results_list:
            List of result dicts from :meth:`run_benchmark`.

        Returns
        -------
        pd.DataFrame
            Leaderboard with columns:
            rank, run, model, mean_mse, std_mse, mean_rmse, mean_mae,
            mean_r2, mean_train_time_s, mean_infer_time_s, mean_memory_mb,
            n_features.
        """
        rows = []
        for run_idx, run_results in enumerate(results_list):
            n_feats = run_results.get("n_features", 0)
            for model_name, model_data in run_results.get("models", {}).items():
                summary = model_data.get("summary", {})
                rows.append({
                    "run":               run_idx,
                    "model":             model_name,
                    "mean_mse":          summary.get("mse",        {}).get("mean"),
                    "std_mse":           summary.get("mse",        {}).get("std"),
                    "mean_rmse":         summary.get("rmse",       {}).get("mean"),
                    "mean_mae":          summary.get("mae",        {}).get("mean"),
                    "mean_r2":           summary.get("r2",         {}).get("mean"),
                    "mean_train_time_s": summary.get("train_time", {}).get("mean"),
                    "mean_infer_time_s": summary.get("infer_time", {}).get("mean"),
                    "mean_memory_mb":    summary.get("memory_mb",  {}).get("mean"),
                    "n_features":        n_feats,
                })

        if not rows:
            return pd.DataFrame(columns=[
                "rank", "run", "model", "mean_mse", "std_mse", "mean_rmse",
                "mean_mae", "mean_r2", "mean_train_time_s", "mean_infer_time_s",
                "mean_memory_mb", "n_features",
            ])

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_mse", ascending=True, na_position="last")
        df.insert(0, "rank", range(1, len(df) + 1))
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_model_cv(
        self,
        model_fn: Any,
        X: np.ndarray,
        y: np.ndarray,
        kf: "KFold",
        n_features: int,
        model_name: str,
    ) -> Dict[str, Any]:
        """Run K-fold CV for a single model factory function."""
        folds_data: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Memory tracking
            tracemalloc.start()
            t_train_start = time.perf_counter()

            try:
                model = model_fn(X_train, y_train, X_val, y_val)
            except Exception as exc:
                logger.warning(
                    "[Benchmarker] %s fold %d training failed: %s", model_name, fold_idx, exc
                )
                tracemalloc.stop()
                continue

            train_time = time.perf_counter() - t_train_start
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_mb = mem_peak / (1024 ** 2)

            # Inference time
            t_infer_start = time.perf_counter()
            try:
                preds = self._predict(model, X_val)
            except Exception as exc:
                logger.warning(
                    "[Benchmarker] %s fold %d inference failed: %s", model_name, fold_idx, exc
                )
                continue
            infer_time = time.perf_counter() - t_infer_start

            mse  = float(mean_squared_error(y_val, preds))
            rmse = float(np.sqrt(mse))
            mae  = float(mean_absolute_error(y_val, preds))
            r2   = float(r2_score(y_val, preds))

            folds_data.append({
                "fold":       fold_idx,
                "mse":        mse,
                "rmse":       rmse,
                "mae":        mae,
                "r2":         r2,
                "train_time": train_time,
                "infer_time": infer_time,
                "memory_mb":  mem_mb,
                "n_features": n_features,
            })
            logger.debug(
                "[Benchmarker] %s fold %d — MSE=%.4f RMSE=%.4f R²=%.4f",
                model_name, fold_idx, mse, rmse, r2,
            )

        return {"folds": folds_data}

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        """Unified predict call handling LightGBM, XGBoost, CatBoost, sklearn."""
        import pandas as pd  # noqa: PLC0415
        try:
            # LightGBM Booster
            import lightgbm as lgb  # noqa: PLC0415
            if isinstance(model, lgb.Booster):
                return model.predict(X)
        except ImportError:
            pass
        try:
            # XGBoost Booster
            import xgboost as xgb  # noqa: PLC0415
            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(X)
                return model.predict(dmat)
        except ImportError:
            pass
        # CatBoost / sklearn interface
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X), dtype=float)
        raise ValueError(f"Cannot predict with model of type {type(model)}")

    @staticmethod
    def _summarise_folds(folds: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute mean/std/min/max across fold metrics."""
        if not folds:
            return {}
        metric_keys = [k for k in folds[0] if k not in ("fold",)]
        summary: Dict[str, Dict[str, float]] = {}
        for key in metric_keys:
            vals = [f[key] for f in folds if key in f]
            arr  = np.asarray(vals, dtype=float)
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std":  float(np.std(arr)),
                "min":  float(np.min(arr)),
                "max":  float(np.max(arr)),
            }
        return summary

    def _get_model_types(
        self,
        cfg: Any,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a dict of model_name → training factory callables from config."""
        from tools.model_tools import ModelTools

        factories: Dict[str, Any] = {}

        def _lgbm_factory(X_train, y_train, X_val, y_val):
            try:
                params = dict(cfg.models.lightgbm.params)
            except Exception:
                params = {}
            return ModelTools.train_lightgbm(X_train, y_train, X_val, y_val, params)

        def _xgb_factory(X_train, y_train, X_val, y_val):
            try:
                params = dict(cfg.models.xgboost.params)
            except Exception:
                params = {}
            return ModelTools.train_xgboost(X_train, y_train, X_val, y_val, params)

        def _cat_factory(X_train, y_train, X_val, y_val):
            try:
                params = dict(cfg.models.catboost.params)
            except Exception:
                params = {}
            return ModelTools.train_catboost(X_train, y_train, X_val, y_val, params)

        # Only include models that are enabled in cfg
        try:
            if cfg.models.lightgbm.enabled:
                factories["lightgbm"] = _lgbm_factory
        except Exception:
            factories["lightgbm"] = _lgbm_factory

        try:
            if cfg.models.xgboost.enabled:
                factories["xgboost"] = _xgb_factory
        except Exception:
            factories["xgboost"] = _xgb_factory

        try:
            if cfg.models.catboost.enabled:
                factories["catboost"] = _cat_factory
        except Exception:
            factories["catboost"] = _cat_factory

        return factories

    def _save_results(self, results: Dict[str, Any]) -> str:
        """Save benchmark results dict to a timestamped JSON file."""
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.results_dir) / f"benchmark_{ts}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        # JSON-serialise numpy scalars
        def _default(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=_default)
        return str(path)


__all__ = ["Benchmarker"]
