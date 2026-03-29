"""
ModelAgent — trains gradient-boosting models guided by an LLM hyperparameter
plan, optimises with Optuna, and creates a weighted ensemble.

The LLM is consulted only to reason about which models to enable and which
parameter search-space adjustments are appropriate for the dataset size and
type. All training and tuning code is deterministic Python.

GPU handling is centralised in ``tools.model_tools.ModelTools``:
  - ``gpu_available()``  — cached CUDA probe + OpenCL ICD setup for LightGBM
  - ``ModelTools.train_lightgbm/xgboost/catboost`` — inject the correct
    device/tree_method/task_type param and fall back to CPU transparently.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

from agents.base import BaseAgent
from tools.model_tools import ModelTools, gpu_available
from utils.helpers import safe_json_parse

_logger = logging.getLogger("kaggle-mas.model_agent")


def _xgb_predict(booster: Any, X: np.ndarray) -> np.ndarray:
    """
    Predict with an ``xgb.Booster``, wrapping *X* in a ``DMatrix``.

    ``xgb.Booster.predict`` only accepts a ``DMatrix``; passing a raw
    ``np.ndarray`` raises ``TypeError: Expecting data to be a DMatrix object``.
    This helper centralises the wrap so callers stay clean.
    """
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(X))


class ModelAgent(BaseAgent):
    """
    Trains LightGBM / XGBoost / CatBoost models with Optuna tuning and
    K-fold cross-validation.

    Expected state keys consumed:
        train_feat    (pd.DataFrame): engineered feature matrix (train).
        test_feat     (pd.DataFrame): engineered feature matrix (test).
        target_series (pd.Series):   target values.
        test_ids      (pd.Series):   _id column for submission.
        feature_names (list[str]):   feature column names.
        improvement_plan (dict, opt): hints from OrchestratorAgent.

    State keys produced:
        models          (dict):          trained model objects per fold per algorithm.
        oof_predictions (dict):          out-of-fold predictions per algorithm.
        cv_scores       (dict):          per-fold metrics by model name.
        feature_importances (dict):      mean feature importance per algorithm.
        ensemble_weights (dict):         weights used for ensemble.
        ensemble_oof    (np.ndarray):    ensemble OOF predictions.
        test_predictions (dict):         test predictions per algorithm.
        ensemble_test   (np.ndarray):    weighted ensemble test predictions.
        submission_df   (pd.DataFrame):  submission-ready DataFrame.
        model_plan      (dict):          LLM-generated model plan.
    """

    SYSTEM_PROMPT = (
        "You are an expert ML engineer selecting and configuring gradient-boosting "
        "models for a rental-property regression competition (MSE metric). "
        "Return compact, resource-efficient hyperparameter search spaces in strict JSON."
    )

    # ------------------------------------------------------------------
    # LLM plan request
    # ------------------------------------------------------------------

    def _request_model_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ask LLM which models to train and what search-space adjustments to make."""
        n_rows, n_cols = state["train_feat"].shape
        improvement_hints = state.get("improvement_plan", {}).get("model_hints", "")

        prompt = f"""
You are selecting models and hyperparameter search spaces for a rental-property
price regression task (Kaggle, MSE metric).

## Dataset characteristics
- Train rows:    {n_rows}
- Train columns: {n_cols}
- Hardware:      Google Colab / Kaggle Notebook — T4 or P100 GPU, 16 GB RAM

## Improvement hints from previous iteration (if any)
{improvement_hints or "None — this is the first training run."}

## Available models
- lightgbm:  fast, GPU-accelerated (device='gpu'), memory-efficient
- xgboost:   GPU-accelerated via device='cuda' (XGBoost >=2.0) or tree_method='gpu_hist' (older)
- catboost:  strong with categoricals; GPU via task_type='GPU'

## Instructions
Decide which models to enable, how many Optuna trials to run (≤ 30 for notebook
constraints), and what param ranges to search.  Keep it practical.
Always include the GPU acceleration parameter for each enabled model.

Return strict JSON only:
{{
  "models": {{
    "lightgbm": {{
      "enabled": true,
      "n_trials": 20,
      "fixed_params": {{
        "verbosity": -1,
        "device": "gpu",
        "n_estimators": 1000,
        "early_stopping_rounds": 50
      }},
      "search_space": {{
        "num_leaves":        {{"type": "int",   "low": 31,  "high": 255}},
        "learning_rate":     {{"type": "float", "low": 0.01,"high": 0.3, "log": true}},
        "feature_fraction":  {{"type": "float", "low": 0.5, "high": 1.0}},
        "bagging_fraction":  {{"type": "float", "low": 0.5, "high": 1.0}},
        "min_child_samples": {{"type": "int",   "low": 20,  "high": 200}},
        "reg_alpha":         {{"type": "float", "low": 1e-8,"high": 10.0,"log": true}},
        "reg_lambda":        {{"type": "float", "low": 1e-8,"high": 10.0,"log": true}}
      }}
    }},
    "xgboost": {{
      "enabled": true,
      "n_trials": 15,
      "fixed_params": {{
        "device": "cuda",
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "verbosity": 0
      }},
      "search_space": {{
        "max_depth":        {{"type": "int",   "low": 3,   "high": 12}},
        "learning_rate":    {{"type": "float", "low": 0.01,"high": 0.3,  "log": true}},
        "subsample":        {{"type": "float", "low": 0.5, "high": 1.0}},
        "colsample_bytree": {{"type": "float", "low": 0.5, "high": 1.0}},
        "reg_alpha":        {{"type": "float", "low": 1e-8,"high": 10.0, "log": true}},
        "reg_lambda":       {{"type": "float", "low": 1e-8,"high": 10.0, "log": true}}
      }}
    }},
    "catboost": {{
      "enabled": false,
      "n_trials": 10,
      "fixed_params": {{
        "iterations": 1000,
        "early_stopping_rounds": 50,
        "task_type": "GPU",
        "verbose": 0
      }},
      "search_space": {{
        "depth":          {{"type": "int",   "low": 4,   "high": 10}},
        "learning_rate":  {{"type": "float", "low": 0.01,"high": 0.3, "log": true}},
        "l2_leaf_reg":    {{"type": "float", "low": 1e-3,"high": 10.0,"log": true}}
      }}
    }}
  }},
  "cv_folds": 5,
  "ensemble_method": "inverse_mse",
  "random_seed": 42
}}
Respond with JSON only.
"""
        default: Dict[str, Any] = {
            "models": {
                "lightgbm": {
                    "enabled": True,
                    "n_trials": 20,
                    "fixed_params": {
                        "verbosity": -1,
                        "device": "gpu",
                        "n_estimators": 1000,
                        "early_stopping_rounds": 50,
                    },
                    "search_space": {
                        "num_leaves":        {"type": "int",   "low": 31,   "high": 255},
                        "learning_rate":     {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
                        "feature_fraction":  {"type": "float", "low": 0.5,  "high": 1.0},
                        "min_child_samples": {"type": "int",   "low": 20,   "high": 200},
                        "reg_alpha":         {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                        "reg_lambda":        {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                    },
                },
                "xgboost": {
                    "enabled": True,
                    "n_trials": 15,
                    "fixed_params": {
                        "device": "cuda",
                        "n_estimators": 1000,
                        "early_stopping_rounds": 50,
                        "verbosity": 0,
                    },
                    "search_space": {
                        "max_depth":        {"type": "int",   "low": 3,    "high": 12},
                        "learning_rate":    {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
                        "subsample":        {"type": "float", "low": 0.5,  "high": 1.0},
                        "colsample_bytree": {"type": "float", "low": 0.5,  "high": 1.0},
                        "reg_alpha":        {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                        "reg_lambda":       {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                    },
                },
                "catboost": {"enabled": False},
            },
            "cv_folds": 5,
            "ensemble_method": "inverse_mse",
            "random_seed": 42,
        }
        return self._ask_llm_json(prompt, default=default)

    # ------------------------------------------------------------------
    # Optuna search-space builder
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a search-space spec dict into Optuna trial parameter samples."""
        params: Dict[str, Any] = {}
        for name, spec in search_space.items():
            ptype = spec.get("type", "float")
            if ptype == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif ptype == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    # ------------------------------------------------------------------
    # Model training helpers — delegate to ModelTools for GPU + training
    # ------------------------------------------------------------------

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        fixed: Dict[str, Any],
    ) -> Tuple[Any, float]:
        merged = {**fixed, **params}
        model = ModelTools.train_lightgbm(X_train, y_train, X_val, y_val, merged)
        val_pred = model.predict(X_val)
        return model, float(mean_squared_error(y_val, val_pred))

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        fixed: Dict[str, Any],
    ) -> Tuple[Any, float]:
        merged = {**fixed, **params}
        booster = ModelTools.train_xgboost(X_train, y_train, X_val, y_val, merged)
        val_pred = _xgb_predict(booster, X_val)
        return booster, float(mean_squared_error(y_val, val_pred))

    def _train_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        fixed: Dict[str, Any],
    ) -> Tuple[Any, float]:
        merged = {**fixed, **params}
        model = ModelTools.train_catboost(X_train, y_train, X_val, y_val, merged)
        val_pred = np.asarray(model.predict(X_val))
        return model, float(mean_squared_error(y_val, val_pred))

    # ------------------------------------------------------------------
    # Optuna tuning per algorithm
    # ------------------------------------------------------------------

    def _tune_algorithm(
        self,
        algo: str,
        algo_cfg: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run Optuna to find best hyperparameters for one algorithm using
        a single validation fold (not full CV, to stay within Colab time budget).
        """
        search_space = algo_cfg.get("search_space", {})
        fixed = algo_cfg.get("fixed_params", {})
        n_trials = algo_cfg.get("n_trials", 20)

        train_fn = {
            "lightgbm": self._train_lightgbm,
            "xgboost": self._train_xgboost,
            "catboost": self._train_catboost,
        }[algo]

        # Use 80/20 split for tuning (fast single-split)
        split_idx = int(len(X) * 0.8)
        idx = np.random.default_rng(seed).permutation(len(X))
        tune_train_idx, tune_val_idx = idx[:split_idx], idx[split_idx:]
        X_t, y_t = X[tune_train_idx], y[tune_train_idx]
        X_v, y_v = X[tune_val_idx], y[tune_val_idx]

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, search_space)
            try:
                _, mse = train_fn(X_t, y_t, X_v, y_v, params, dict(fixed))
                return mse
            except Exception as exc:
                self._log(f"Trial failed: {exc}", level="warning")
                return float("inf")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        self._log(f"[{algo}] Optuna best MSE={study.best_value:.4f} params={best_params}")
        return best_params

    # ------------------------------------------------------------------
    # K-fold CV training
    # ------------------------------------------------------------------

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        """
        Dispatch prediction to the correct API for each model type.

        - ``xgb.Booster`` requires a ``DMatrix``; passing a raw array raises
          ``TypeError: Expecting data to be a DMatrix object``.  We detect the
          Booster type and wrap via :func:`_xgb_predict`.
        - ``lgb.Booster`` (and sklearn-style models) expose a plain
          ``predict(array)`` API.
        """
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                return _xgb_predict(model, X)
        except ImportError:
            pass
        return np.asarray(model.predict(X))

    def _cross_validate_algorithm(
        self,
        algo: str,
        algo_cfg: Dict[str, Any],
        best_params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray,
        n_splits: int,
        seed: int,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Train the algorithm with best_params across n_splits folds.
        Returns OOF preds, fold MSEs, test preds, and feature importances.
        """
        fixed = dict(algo_cfg.get("fixed_params", {}))
        train_fn = {
            "lightgbm": self._train_lightgbm,
            "xgboost": self._train_xgboost,
            "catboost": self._train_catboost,
        }[algo]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(y))
        test_preds_folds = np.zeros((len(X_test), n_splits))
        fold_mses: List[float] = []
        models: List[Any] = []
        importances: List[np.ndarray] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]

            model, fold_mse = train_fn(X_tr, y_tr, X_vl, y_vl, dict(best_params), dict(fixed))
            # Use _predict() so xgb.Booster gets a DMatrix, not a raw ndarray
            oof_preds[val_idx] = self._predict(model, X_vl)
            test_preds_folds[:, fold_idx] = self._predict(model, X_test)
            fold_mses.append(fold_mse)
            models.append(model)

            # Feature importances
            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)
            elif hasattr(model, "feature_importance"):
                importances.append(model.feature_importance())

            self._log(f"[{algo}] Fold {fold_idx+1}/{n_splits} MSE={fold_mse:.4f}")
            gc.collect()

        mean_imp = np.mean(importances, axis=0) if importances else np.zeros(len(feature_names))
        imp_dict = dict(zip(feature_names, mean_imp.tolist()))

        return {
            "models": models,
            "oof_predictions": oof_preds,
            "test_predictions": test_preds_folds.mean(axis=1),
            "fold_mses": fold_mses,
            "mean_cv_mse": float(np.mean(fold_mses)),
            "std_cv_mse": float(np.std(fold_mses)),
            "feature_importances": imp_dict,
        }

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def _build_ensemble(
        self,
        oof_results: Dict[str, Dict[str, Any]],
        test_results: Dict[str, np.ndarray],
        y: np.ndarray,
        method: str = "inverse_mse",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Combine per-algorithm OOF and test predictions into a weighted ensemble.
        """
        algos = list(oof_results.keys())
        mses = np.array([oof_results[a]["mean_cv_mse"] for a in algos])

        if method == "inverse_mse":
            raw_weights = 1.0 / np.maximum(mses, 1e-9)
        else:  # equal weights
            raw_weights = np.ones(len(algos))

        weights = raw_weights / raw_weights.sum()
        weight_dict = dict(zip(algos, weights.tolist()))
        self._log(f"Ensemble weights: {weight_dict}")

        oof_ensemble = sum(
            weights[i] * oof_results[algos[i]]["oof_predictions"]
            for i in range(len(algos))
        )
        test_ensemble = sum(
            weights[i] * test_results[algos[i]]
            for i in range(len(algos))
        )
        return oof_ensemble, test_ensemble, weight_dict

    # ------------------------------------------------------------------
    # Submission builder (extracted for clarity and testability)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_submission(
        test_ensemble: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a submission DataFrame with columns ``[index, prediction]``.

        The ``index`` column is a simple integer range from 0 to
        ``len(test_ensemble) - 1``, matching the row order of the test set.
        No deduplication or sorting is performed.

        Parameters
        ----------
        test_ensemble:
            Array of ensemble predictions for the test set.

        Returns
        -------
        pd.DataFrame
            Columns: ``index`` (0-based integer), ``prediction`` (float).
        """
        return pd.DataFrame(
            {
                "index": np.arange(len(test_ensemble)),
                "prediction": test_ensemble,
            }
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Ask LLM for model plan.
        2. Tune each enabled algorithm with Optuna.
        3. Train with K-fold CV using best params.
        4. Build ensemble.
        5. Generate submission.csv.
        """
        X: np.ndarray = state["train_feat"].values.astype(np.float32)
        y: np.ndarray = state["target_series"].values.astype(np.float32)
        X_test: np.ndarray = state["test_feat"].values.astype(np.float32)
        test_ids: pd.Series = state["test_ids"]
        feature_names: List[str] = state["feature_names"]

        # Log GPU status once at the start (uses cached check from ModelTools)
        _logger.info("[GPU] GPU available: %s", gpu_available())

        # --- LLM plan ---
        plan = self._request_model_plan(state)
        state["model_plan"] = plan
        models_cfg: Dict[str, Any] = plan.get("models", {})
        n_splits: int = int(plan.get("cv_folds", self.cfg.get("cv_folds", 5)))
        ensemble_method: str = plan.get("ensemble_method", "inverse_mse")
        seed: int = int(plan.get("random_seed", 42))

        enabled_algos = [a for a, c in models_cfg.items() if c.get("enabled", False)]
        self._log(f"Training algorithms: {enabled_algos}, cv_folds={n_splits}")

        oof_results: Dict[str, Dict[str, Any]] = {}
        test_predictions: Dict[str, np.ndarray] = {}

        for algo in enabled_algos:
            algo_cfg = models_cfg[algo]
            self._log(f"Tuning {algo} with Optuna ({algo_cfg.get('n_trials', 20)} trials)\u2026")

            try:
                best_params = self._tune_algorithm(algo, algo_cfg, X, y, n_splits, seed)
            except Exception as exc:
                self._log(f"Optuna tuning failed for {algo}: {exc}. Using fixed params.", level="warning")
                best_params = {}

            self._log(f"CV training {algo}\u2026")
            try:
                result = self._cross_validate_algorithm(
                    algo, algo_cfg, best_params, X, y, X_test, n_splits, seed, feature_names
                )
                oof_results[algo] = result
                test_predictions[algo] = result["test_predictions"]
                self._log(
                    f"[{algo}] CV MSE={result['mean_cv_mse']:.4f} \u00b1 {result['std_cv_mse']:.4f}"
                )
            except Exception as exc:
                self._log(f"CV training failed for {algo}: {exc}", level="error")
                continue
            finally:
                gc.collect()

        if not oof_results:
            raise RuntimeError("No models trained successfully.")

        # --- Ensemble ---
        oof_ensemble, test_ensemble, ensemble_weights = self._build_ensemble(
            oof_results, test_predictions, y, method=ensemble_method
        )
        ensemble_mse = float(mean_squared_error(y, oof_ensemble))
        self._log(f"Ensemble OOF MSE={ensemble_mse:.4f}")

        # --- Submission ---
        # Columns: index (0..N-1), prediction — row order matches test set
        submission = self._build_submission(test_ensemble)
        output_dir = Path(self.cfg.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        sub_path = output_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        self._log(f"Submission saved to {sub_path}")

        # --- Store in state ---
        state["models"] = {a: r["models"] for a, r in oof_results.items()}
        state["oof_predictions"] = {a: r["oof_predictions"] for a, r in oof_results.items()}
        state["cv_scores"] = {
            a: {"fold_mses": r["fold_mses"], "mean": r["mean_cv_mse"], "std": r["std_cv_mse"]}
            for a, r in oof_results.items()
        }
        state["feature_importances"] = {a: r["feature_importances"] for a, r in oof_results.items()}
        state["ensemble_weights"] = ensemble_weights
        state["ensemble_oof"] = oof_ensemble
        state["test_predictions"] = test_predictions
        state["ensemble_test"] = test_ensemble
        state["submission_df"] = submission
        state["ensemble_cv_mse"] = ensemble_mse

        gc.collect()
        return state
