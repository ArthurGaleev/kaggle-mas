"""
ModelComparator — statistical model and feature-set comparison.

Install dependencies:
  pip install scikit-learn pandas numpy scipy matplotlib
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score, KFold
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class ModelComparator:
    """
    Statistical comparison of multiple trained models and feature subsets.

    Parameters
    ----------
    seed:
        Random state for any internal CV splits (default 42).
    output_dir:
        Directory for saving comparison plots.

    Examples
    --------
    ::

        from evaluation.model_comparator import ModelComparator
        comp = ModelComparator(output_dir="./outputs/comparisons")
        report = comp.compare_models(models_dict, X_val, y_val)
        feat_report = comp.compare_feature_sets(best_model, feature_sets, X, y)
    """

    def __init__(self, seed: int = 42, output_dir: str = "./outputs/comparisons") -> None:
        self.seed       = seed
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_val: Any,
        y_val: Any,
    ) -> Dict[str, Any]:
        """
        Compute validation metrics for each model and run pairwise significance tests.

        Parameters
        ----------
        models_dict:
            Mapping of model_name → trained model object.
        X_val:
            Validation feature matrix (array-like).
        y_val:
            Validation target (array-like).

        Returns
        -------
        dict with keys:
          * ``metrics``       — per-model metrics dict
          * ``pairwise``      — pairwise significance test results
          * ``ranked``        — model names ranked by ascending MSE
          * ``best_model``    — name of best model by MSE
        """
        X = np.asarray(X_val, dtype=np.float32)
        y = np.asarray(y_val, dtype=np.float32)

        # Compute metrics per model
        metrics: Dict[str, Dict[str, float]] = {}
        preds_map: Dict[str, np.ndarray] = {}

        for name, model in models_dict.items():
            try:
                preds = self._predict(model, X)
                preds_map[name] = preds
                metrics[name] = self._compute_metrics(y, preds)
                logger.debug(
                    "[ModelComparator] %s — MSE=%.4f R²=%.4f",
                    name, metrics[name]["mse"], metrics[name]["r2"],
                )
            except Exception as exc:
                logger.warning("[ModelComparator] Could not evaluate %s: %s", name, exc)

        # Pairwise statistical tests (comparing residuals)
        pairwise: Dict[str, Any] = {}
        model_names = list(preds_map.keys())

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]
                pair_key = f"{name_a} vs {name_b}"

                resid_a = y - preds_map[name_a]
                resid_b = y - preds_map[name_b]
                sq_err_a = resid_a ** 2
                sq_err_b = resid_b ** 2

                # Paired t-test on squared errors
                if len(sq_err_a) >= 2:
                    t_stat, p_value = stats.ttest_rel(sq_err_a, sq_err_b)
                    better = name_a if np.mean(sq_err_a) < np.mean(sq_err_b) else name_b
                    pairwise[pair_key] = {
                        "t_statistic":      float(t_stat),
                        "p_value":          float(p_value),
                        "significant_at_05": bool(p_value < 0.05),
                        "better":           better,
                        "delta_mse":        float(np.mean(sq_err_a) - np.mean(sq_err_b)),
                    }

        # Rank models
        ranked = sorted(metrics.keys(), key=lambda n: metrics[n].get("mse", float("inf")))
        best_model = ranked[0] if ranked else None

        # Generate plots
        try:
            self._plot_comparison(metrics, preds_map, y)
        except Exception as exc:
            logger.warning("[ModelComparator] Could not generate plots: %s", exc)

        return {
            "metrics":    metrics,
            "pairwise":   pairwise,
            "ranked":     ranked,
            "best_model": best_model,
        }

    # ------------------------------------------------------------------
    # Feature set comparison
    # ------------------------------------------------------------------

    def compare_feature_sets(
        self,
        model: Any,
        feature_sets: Dict[str, Any],
        X: Any,
        y: Any,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare different feature subsets using cross-validation.

        Parameters
        ----------
        model:
            A sklearn-compatible model with ``fit`` / ``predict`` interface.
        feature_sets:
            Mapping of set_name → feature matrix (array-like).
            E.g. ``{"all": X_all, "no_text": X_no_text, "geo_only": X_geo}``.
        X:
            Full feature matrix (used as reference; not used directly if
            feature_sets values are provided).
        y:
            Target array (array-like).
        cv_folds:
            Number of CV folds (default 5).

        Returns
        -------
        dict with keys:
          * ``results``           — per-set CV summary
          * ``best_set``          — name of set with lowest mean CV MSE
          * ``feature_importance`` — importance scores (if model supports it)
          * ``ablation_deltas``   — MSE delta of each set vs full feature set
        """
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise ValueError(
                "model must be sklearn-compatible (has .fit and .predict methods)."
            )

        y_arr = np.asarray(y, dtype=np.float32)
        kf    = KFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        set_results: Dict[str, Dict[str, Any]] = {}

        for set_name, X_set in feature_sets.items():
            X_arr = np.asarray(X_set, dtype=np.float32)
            fold_mse: List[float] = []

            for train_idx, val_idx in kf.split(X_arr):
                X_tr, X_va = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
                try:
                    import copy
                    m = copy.deepcopy(model)
                    m.fit(X_tr, y_tr)
                    preds = np.asarray(m.predict(X_va), dtype=float)
                    fold_mse.append(float(mean_squared_error(y_va, preds)))
                except Exception as exc:
                    logger.warning(
                        "[ModelComparator] Feature set '%s' fold failed: %s", set_name, exc
                    )

            if fold_mse:
                set_results[set_name] = {
                    "mean_mse":  float(np.mean(fold_mse)),
                    "std_mse":   float(np.std(fold_mse)),
                    "n_features": X_arr.shape[1] if X_arr.ndim > 1 else 1,
                    "fold_mse":  fold_mse,
                }
            logger.debug(
                "[ModelComparator] Feature set '%s': mean CV MSE=%.4f ± %.4f",
                set_name,
                set_results.get(set_name, {}).get("mean_mse", float("nan")),
                set_results.get(set_name, {}).get("std_mse", float("nan")),
            )

        # Best set
        best_set = min(
            set_results.keys(),
            key=lambda k: set_results[k].get("mean_mse", float("inf")),
            default=None,
        )

        # Feature importance from trained model (on full data)
        importance: Dict[str, float] = {}
        try:
            full_X = np.asarray(
                next(iter(feature_sets.values())), dtype=np.float32
            )
            import copy
            m_full = copy.deepcopy(model)
            m_full.fit(full_X, y_arr)
            if hasattr(m_full, "feature_importances_"):
                imp_arr = m_full.feature_importances_
                for i, v in enumerate(imp_arr):
                    importance[f"feature_{i}"] = float(v)
        except Exception as exc:
            logger.debug("[ModelComparator] Could not extract feature importance: %s", exc)

        # Ablation deltas (vs full set)
        ablation_deltas: Dict[str, Optional[float]] = {}
        if "all" in set_results:
            ref_mse = set_results["all"]["mean_mse"]
            for set_name, res in set_results.items():
                ablation_deltas[set_name] = res["mean_mse"] - ref_mse

        return {
            "results":            set_results,
            "best_set":           best_set,
            "feature_importance": importance,
            "ablation_deltas":    ablation_deltas,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        """Dispatch predict call for LightGBM, XGBoost, CatBoost, or sklearn."""
        try:
            import lightgbm as lgb  # noqa: PLC0415
            if isinstance(model, lgb.Booster):
                return np.asarray(model.predict(X), dtype=float)
        except ImportError:
            pass
        try:
            import xgboost as xgb  # noqa: PLC0415
            if isinstance(model, xgb.Booster):
                dmat = xgb.DMatrix(X)
                return np.asarray(model.predict(dmat), dtype=float)
        except ImportError:
            pass
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X), dtype=float)
        raise ValueError(f"Cannot predict with model type: {type(model)}")

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard regression metrics."""
        mse  = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def _plot_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        preds_map: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Generate and save model comparison bar charts and residual plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        out_dir = Path(self.output_dir)

        # ---- Bar chart: MSE per model ----
        model_names = list(metrics.keys())
        mse_vals    = [metrics[n]["mse"] for n in model_names]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))
        ax.barh(model_names, mse_vals, color=colors, edgecolor="white")
        ax.set_xlabel("Validation MSE")
        ax.set_title("Model Comparison — Validation MSE", fontweight="bold")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_dir / "model_comparison_mse.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # ---- Scatter: predicted vs actual (best model only) ----
        if model_names:
            best = sorted(model_names, key=lambda n: metrics[n]["mse"])[0]
            y_pred = preds_map[best]
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
            mn = min(float(y_true.min()), float(y_pred.min()))
            mx = max(float(y_true.max()), float(y_pred.max()))
            ax2.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal")
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.set_title(f"Predicted vs Actual — {best}", fontweight="bold")
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig(out_dir / f"pred_vs_actual_{best}.png", dpi=120, bbox_inches="tight")
            plt.close(fig2)

        logger.info("[ModelComparator] Plots saved to %s", out_dir)


__all__ = ["ModelComparator"]
