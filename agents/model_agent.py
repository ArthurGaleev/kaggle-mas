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

OOF TARGET ENCODING
--------------------
If the feature plan requests target encoding, the raw categorical columns
(host_name, location_cluster, type_house) are present in train_feat / test_feat
as object-dtype columns.  ModelAgent computes a smoothed mean encoding map
from the *training fold only* at the start of each KFold iteration and applies
it to both the fold's training and validation splits, preventing leakage.

OPTUNA TUNING SPLIT
--------------------
Hyperparameter search uses a KFold with a *different* random_state than the CV
loop (seed+7919) so the tuning validation fold never coincides with any of the
CV validation folds, eliminating optimistic bias in the found parameters.

Algo-specific TPE sampler seeds: lgbm=seed, xgb=seed+1, catboost=seed+2.
This ensures each algorithm's Optuna search explores a different initial
hyperparameter region and avoids degenerate convergence to the same params
(observed previously when all used the same seed=42).

TARGET TRANSFORM
----------------
Log-transform on the target is intentionally disabled (use_log_target=False).
GBDT models handle moderately skewed rental prices natively, and log1p/expm1
roundtripping was observed to hurt leaderboard MSE in practice.

PREDICTION CLIPPING
-------------------
Predictions are clipped to [0, cap] where cap = 99.9th percentile of training
target x 3.0. The old cap (99.5th pct x 1.5 ≈ 547) was cutting off legitimate
high-value listings. The new cap allows for the full realistic price range
while still blocking absurd outlier predictions.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

from agents.base import BaseAgent
from agents.feature_agent import _compute_target_encoding_map
from tools.model_tools import ModelTools, gpu_available
from utils.helpers import safe_json_parse

_logger = logging.getLogger("kaggle-mas.model_agent")

# Prime offset added to the CV random_state to produce a disjoint tuning fold.
_TUNE_SEED_OFFSET: int = 7919

# Per-algorithm seed offsets for Optuna TPE sampler.
# Ensures each algorithm explores a different initial hyperparameter region.
_ALGO_SEED_OFFSETS: Dict[str, int] = {
    "lightgbm": 0,
    "xgboost": 1,
    "catboost": 2,
}

# Max number of CV folds used inside each Optuna trial (keep low to limit tuning time).
_MAX_TUNE_FOLDS: int = 2


def _xgb_predict(booster: Any, X: np.ndarray) -> np.ndarray:
    """
    Predict with an ``xgb.Booster``, wrapping *X* in a ``DMatrix``.
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
        feature_plan  (dict, opt):   plan from FeatureAgent (target enc config).
        improvement_plan (dict, opt): hints from OrchestratorAgent.

    State keys produced:
        models          (dict):          trained model objects per fold per algorithm.
        oof_predictions (dict):          out-of-fold predictions per algorithm.
        cv_scores       (dict):          per-fold metrics by model name.
        feature_importances (dict):      mean feature importance per algorithm.
        ensemble_weights (dict):         weights used for ensemble.
        ensemble_oof    (np.ndarray):    ensemble OOF predictions (raw price scale).
        test_predictions (dict):         test predictions per algorithm.
        ensemble_test   (np.ndarray):    weighted ensemble test predictions (raw price scale).
        submission_df   (pd.DataFrame):  submission-ready DataFrame with integer index column.
        model_plan      (dict):          LLM-generated model plan.
        use_log_target  (bool):          whether log-transform was applied (always False).
    """

    SYSTEM_PROMPT = (
        "You are an expert ML engineer selecting and configuring gradient-boosting "
        "models for a rental-property regression competition (MSE metric). "
        "Return compact, resource-efficient hyperparameter search spaces in strict JSON."
    )

    # Categorical columns subject to target encoding (mirrors FeatureAgent)
    _TE_COLS = ("host_name", "location_cluster", "type_house")

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
Decide which models to enable, how many Optuna trials to run, and what param
ranges to search. Keep it practical for notebook constraints.
Always include the GPU acceleration parameter for each enabled model.

Return strict JSON only:
{{
  "models": {{
    "lightgbm": {{
      "enabled": true,
      "n_trials": 12,
      "fixed_params": {{
        "verbosity": -1,
        "device": "gpu",
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "bagging_freq": 1
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
      "n_trials": 10,
      "fixed_params": {{
        "device": "cuda",
        "n_estimators": 2000,
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
      "enabled": true,
      "n_trials": 6,
      "fixed_params": {{
        "iterations": 2000,
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
                    "n_trials": 12,
                    "fixed_params": {
                        "verbosity": -1,
                        "device": "gpu",
                        "n_estimators": 2000,
                        "early_stopping_rounds": 50,
                        "bagging_freq": 1,
                    },
                    "search_space": {
                        "num_leaves":        {"type": "int",   "low": 31,   "high": 255},
                        "learning_rate":     {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
                        "feature_fraction":  {"type": "float", "low": 0.5,  "high": 1.0},
                        "bagging_fraction":  {"type": "float", "low": 0.5,  "high": 1.0},
                        "min_child_samples": {"type": "int",   "low": 20,   "high": 200},
                        "reg_alpha":         {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                        "reg_lambda":        {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
                    },
                },
                "xgboost": {
                    "enabled": True,
                    "n_trials": 10,
                    "fixed_params": {
                        "device": "cuda",
                        "n_estimators": 2000,
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
                "catboost": {
                    "enabled": True,
                    "n_trials": 6,
                    "fixed_params": {
                        "iterations": 2000,
                        "early_stopping_rounds": 50,
                        "task_type": "GPU",
                        "verbose": 0,
                    },
                    "search_space": {
                        "depth":         {"type": "int",   "low": 4,    "high": 10},
                        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                        "l2_leaf_reg":   {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
                    },
                },
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
    # Model training helpers
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
    # Per-fold target encoding
    # ------------------------------------------------------------------

    def _apply_fold_target_encoding(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_fold: np.ndarray,
        global_mean: float,
        smoothing: float,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Apply smoothed target-mean encoding using only the training fold's
        labels, then return updated numpy arrays and extended feature names.
        """
        te_cols_present = [c for c in self._TE_COLS if c in train_df.columns]
        if not te_cols_present:
            return (
                train_df.drop(columns=list(self._TE_COLS), errors="ignore").values.astype(np.float32),
                val_df.drop(columns=list(self._TE_COLS), errors="ignore").values.astype(np.float32),
                test_df.drop(columns=list(self._TE_COLS), errors="ignore").values.astype(np.float32),
                feature_names,
            )

        new_names = list(feature_names)
        extra_train, extra_val, extra_test = [], [], []

        for col in te_cols_present:
            mapping = _compute_target_encoding_map(
                col=train_df[col],
                target=pd.Series(target_fold, index=train_df.index),
                global_mean=global_mean,
                smoothing=smoothing,
            )
            feat = f"te_{col}"
            extra_train.append(train_df[col].map(mapping).fillna(global_mean).values)
            extra_val.append(val_df[col].map(mapping).fillna(global_mean).values)
            extra_test.append(test_df[col].map(mapping).fillna(global_mean).values)
            new_names.append(feat)

        num_cols = [c for c in train_df.columns if c not in self._TE_COLS]

        def _stack(df: pd.DataFrame, extras: List[np.ndarray]) -> np.ndarray:
            base = df[num_cols].values.astype(np.float32)
            if extras:
                return np.hstack([base, np.column_stack(extras).astype(np.float32)])
            return base

        return (
            _stack(train_df, extra_train),
            _stack(val_df, extra_val),
            _stack(test_df, extra_test),
            new_names,
        )

    # ------------------------------------------------------------------
    # Optuna tuning per algorithm
    # ------------------------------------------------------------------

    def _tune_algorithm(
        self,
        algo: str,
        algo_cfg: Dict[str, Any],
        X_df: pd.DataFrame,
        y: np.ndarray,
        seed: int,
        n_splits: int,
        feature_names: List[str],
        te_smoothing: float = 10.0,
        global_mean: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run Optuna to find best hyperparameters for one algorithm.
        Uses a disjoint KFold (seed + _TUNE_SEED_OFFSET) to avoid
        optimistic bias vs the CV folds.

        Each algorithm uses a unique TPE sampler seed (seed + algo offset)
        to ensure different algorithms explore different hyperparameter regions.
        """
        search_space = algo_cfg.get("search_space", {})
        fixed = algo_cfg.get("fixed_params", {})
        n_trials = algo_cfg.get("n_trials", 12)

        train_fn = {
            "lightgbm": self._train_lightgbm,
            "xgboost": self._train_xgboost,
            "catboost": self._train_catboost,
        }[algo]

        tune_kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed + _TUNE_SEED_OFFSET)

        # Use algo-specific seed offset so each algorithm's TPE sampler starts
        # from a different random state, preventing degenerate convergence to
        # identical hyperparameters across algorithms.
        algo_seed = seed + _ALGO_SEED_OFFSETS.get(algo, 0)

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, search_space)
            try:
                fold_mses = []
                for fold_idx, (tune_train_idx, tune_val_idx) in enumerate(tune_kf.split(X_df)):
                    if fold_idx >= _MAX_TUNE_FOLDS:
                        break
                    train_fold_df = X_df.iloc[tune_train_idx].reset_index(drop=True)
                    val_fold_df = X_df.iloc[tune_val_idx].reset_index(drop=True)
                    y_t = y[tune_train_idx]
                    y_v = y[tune_val_idx]

                    X_t, X_v, _, _ = self._apply_fold_target_encoding(
                        train_df=train_fold_df,
                        val_df=val_fold_df,
                        test_df=val_fold_df,
                        target_fold=y_t,
                        global_mean=global_mean,
                        smoothing=te_smoothing,
                        feature_names=feature_names,
                    )

                    _, mse = train_fn(X_t, y_t, X_v, y_v, params, dict(fixed))
                    fold_mses.append(mse)
                return float(np.mean(fold_mses))
            except Exception as exc:
                self._log(f"Trial failed: {exc}", level="warning")
                return float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=algo_seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        self._log(f"[{algo}] Optuna best MSE={study.best_value:.4f} params={best_params}")
        return best_params

    # ------------------------------------------------------------------
    # K-fold CV training
    # ------------------------------------------------------------------

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> np.ndarray:
        """Dispatch prediction to the correct API for each model type."""
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
        X_df: pd.DataFrame,
        y: np.ndarray,
        test_df: pd.DataFrame,
        n_splits: int,
        seed: int,
        feature_names: List[str],
        te_smoothing: float,
        global_mean: float,
    ) -> Dict[str, Any]:
        """
        Train the algorithm with best_params across n_splits folds.
        All MSE values are in raw price space (no log transform).
        """
        fixed = dict(algo_cfg.get("fixed_params", {}))
        train_fn = {
            "lightgbm": self._train_lightgbm,
            "xgboost": self._train_xgboost,
            "catboost": self._train_catboost,
        }[algo]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(y))
        test_preds_folds: List[np.ndarray] = []
        fold_mses: List[float] = []
        models: List[Any] = []
        importances: List[np.ndarray] = []

        indices = np.arange(len(y))
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            train_fold_df = X_df.iloc[train_idx].reset_index(drop=True)
            val_fold_df   = X_df.iloc[val_idx].reset_index(drop=True)
            y_tr = y[train_idx]
            y_vl = y[val_idx]

            X_tr, X_vl, X_te, fold_feat_names = self._apply_fold_target_encoding(
                train_df=train_fold_df,
                val_df=val_fold_df,
                test_df=test_df.reset_index(drop=True),
                target_fold=y_tr,
                global_mean=global_mean,
                smoothing=te_smoothing,
                feature_names=feature_names,
            )

            model, fold_mse = train_fn(X_tr, y_tr, X_vl, y_vl, dict(best_params), dict(fixed))
            oof_preds[val_idx] = self._predict(model, X_vl)
            test_preds_folds.append(self._predict(model, X_te))
            fold_mses.append(fold_mse)
            models.append(model)

            if hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)
            elif hasattr(model, "feature_importance"):
                importances.append(model.feature_importance())
            elif hasattr(model, "get_score"):
                score = model.get_score(importance_type="gain")
                imp_arr = np.zeros(len(fold_feat_names))
                for fname, val in score.items():
                    try:
                        idx = int(fname.replace("f", ""))
                        if idx < len(imp_arr):
                            imp_arr[idx] = val
                    except (ValueError, IndexError):
                        pass
                importances.append(imp_arr)

            self._log(f"[{algo}] Fold {fold_idx+1}/{n_splits} MSE={fold_mse:.4f}")

        gc.collect()

        test_preds_mean = np.mean(np.column_stack(test_preds_folds), axis=1)
        mean_imp = np.mean(importances, axis=0) if importances else np.zeros(len(fold_feat_names))
        imp_dict = dict(zip(fold_feat_names, mean_imp.tolist()))

        return {
            "models": models,
            "oof_predictions": oof_preds,
            "test_predictions": test_preds_mean,
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
        Attempts RidgeCV stacking first; falls back to inverse-MSE weighted average.
        All inputs and outputs are in raw price space.
        """
        algos = list(oof_results.keys())

        if len(algos) >= 2:
            try:
                from sklearn.linear_model import RidgeCV
                from sklearn.model_selection import cross_val_predict, KFold

                oof_matrix = np.column_stack(
                    [oof_results[a]["oof_predictions"] for a in algos]
                )
                test_matrix = np.column_stack(
                    [test_results[a] for a in algos]
                )

                meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0])
                stacking_cv = KFold(n_splits=5, shuffle=True, random_state=42)
                oof_ensemble = cross_val_predict(
                    meta, oof_matrix, y, cv=stacking_cv
                )

                meta.fit(oof_matrix, y)
                test_ensemble = meta.predict(test_matrix)

                weight_dict = dict(zip(algos, meta.coef_.tolist()))
                self._log(f"Stacking meta-learner (RidgeCV) weights: {weight_dict}, "
                          f"alpha={meta.alpha_}")

                # Sanity check: if any weight is strongly negative (< -0.2),
                # stacking is unstable — fall back to inverse-MSE average.
                if any(w < -0.2 for w in meta.coef_):
                    self._log(
                        "Stacking produced strongly negative weights — falling back to "
                        "inverse-MSE weighted average.",
                        level="warning",
                    )
                    raise ValueError("Negative stacking weights")

                return oof_ensemble, test_ensemble, weight_dict
            except Exception as exc:
                self._log(f"Stacking failed, falling back to weighted average: {exc}",
                          level="warning")

        mses = np.array([oof_results[a]["mean_cv_mse"] for a in algos])

        if method == "inverse_mse":
            raw_weights = 1.0 / np.maximum(mses, 1e-9)
        else:
            raw_weights = np.ones(len(algos))

        weights = raw_weights / raw_weights.sum()
        weight_dict = dict(zip(algos, weights.tolist()))
        self._log(f"Ensemble weights (weighted avg): {weight_dict}")

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
    # Submission builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_submission(test_ensemble: np.ndarray) -> pd.DataFrame:
        """
        Build the submission DataFrame with a plain integer index column.

        The competition expects:
            index  | prediction
            0      | 1234.5
            1      | 2345.6
            ...
        """
        n = len(test_ensemble)
        return pd.DataFrame(
            {
                "index": np.arange(n),
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
        3. Train with K-fold CV; apply target encoding inside each fold.
        4. Build ensemble.
        5. Populate state with submission_df.

        All predictions and MSE values are in raw price space.
        Log-transform is intentionally disabled (use_log_target=False).
        """
        X_df: pd.DataFrame = state["train_feat"]
        test_df: pd.DataFrame = state["test_feat"]
        y: np.ndarray = state["target_series"].values.astype(np.float32)
        test_ids: pd.Series = state["test_ids"]
        feature_names: List[str] = state["feature_names"]

        # Log-transform disabled: GBDT models handle rental price scale natively.
        # Empirically log1p/expm1 roundtripping increased leaderboard MSE.
        use_log_target: bool = False
        self._log("Training on raw target values (use_log_target=False).")
        self._log(f"Target stats: mean={float(y.mean()):.2f}, "
                  f"std={float(y.std()):.2f}, "
                  f"min={float(y.min()):.2f}, "
                  f"max={float(y.max()):.2f}, "
                  f"99.9th_pct={float(np.percentile(y, 99.9)):.2f}")

        feature_plan = state.get("feature_plan", {})
        te_cfg = feature_plan.get("groups", {}).get("target_encoding", {})
        te_enabled: bool = te_cfg.get("enabled", True)
        te_smoothing: float = float(te_cfg.get("smoothing", 10.0))
        global_mean: float = float(y.mean())

        if not te_enabled:
            drop_te = [c for c in self._TE_COLS if c in X_df.columns]
            if drop_te:
                X_df = X_df.drop(columns=drop_te)
                test_df = test_df.drop(columns=drop_te, errors="ignore")

        _logger.info("[GPU] GPU available: %s", gpu_available())

        plan = self._request_model_plan(state)
        state["model_plan"] = plan
        models_cfg: Dict[str, Any] = plan.get("models", {})
        n_splits: int = int(plan.get("cv_folds", OmegaConf.select(self.cfg, "cv_folds", default=5)))
        ensemble_method: str = plan.get("ensemble_method", "inverse_mse")
        seed: int = int(plan.get("random_seed", 42))

        enabled_algos = [a for a, c in models_cfg.items() if c.get("enabled", False)]
        self._log(f"Training algorithms: {enabled_algos}, cv_folds={n_splits}")

        oof_results: Dict[str, Dict[str, Any]] = {}
        test_predictions: Dict[str, np.ndarray] = {}

        for algo in enabled_algos:
            algo_cfg = models_cfg[algo]
            self._log(f"Tuning {algo} with Optuna ({algo_cfg.get('n_trials', 12)} trials)\u2026")

            try:
                best_params = self._tune_algorithm(
                    algo, algo_cfg, X_df, y, seed, n_splits,
                    feature_names=feature_names,
                    te_smoothing=te_smoothing,
                    global_mean=global_mean,
                )
            except Exception as exc:
                self._log(f"Optuna tuning failed for {algo}: {exc}. Using fixed params.", level="warning")
                best_params = {}

            self._log(f"CV training {algo}\u2026")
            try:
                result = self._cross_validate_algorithm(
                    algo=algo,
                    algo_cfg=algo_cfg,
                    best_params=best_params,
                    X_df=X_df,
                    y=y,
                    test_df=test_df,
                    n_splits=n_splits,
                    seed=seed,
                    feature_names=feature_names,
                    te_smoothing=te_smoothing,
                    global_mean=global_mean,
                )
                oof_results[algo] = result
                test_predictions[algo] = result["test_predictions"]
                self._log(
                    f"[{algo}] CV MSE={result['mean_cv_mse']:.4f} \u00b1 {result['std_cv_mse']:.4f} (raw price scale)"
                )
            except Exception as exc:
                self._log(f"CV training failed for {algo}: {exc}", level="error")
                continue

        gc.collect()

        if not oof_results:
            raise RuntimeError("No models trained successfully.")

        oof_ensemble, test_ensemble, ensemble_weights = self._build_ensemble(
            oof_results, test_predictions, y, method=ensemble_method
        )

        # Clip predictions: rental prices cannot be negative.
        # Use 99.9th percentile x 3.0 to allow for genuine high-value listings.
        # The old cap (99.5th pct x 1.5) was ~547 which cut off expensive apartments.
        upper_cap = float(np.percentile(y, 99.9)) * 3.0
        test_ensemble = np.clip(test_ensemble, 0, upper_cap)
        oof_ensemble = np.clip(oof_ensemble, 0, upper_cap)
        self._log(f"Predictions clipped to [0, {upper_cap:.1f}] (99.9th pct of target \u00d7 3.0)")

        ensemble_mse = float(mean_squared_error(y, oof_ensemble))
        self._log(f"Ensemble OOF MSE={ensemble_mse:.4f} (raw price scale)")

        submission = self._build_submission(test_ensemble)

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
        state["use_log_target"] = use_log_target

        return state
