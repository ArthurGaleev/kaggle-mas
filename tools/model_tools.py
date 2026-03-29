"""
ModelTools — static utility methods used by ModelAgent.

Install dependencies:
  pip install lightgbm xgboost catboost optuna scikit-learn joblib numpy
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    """Check if a CUDA GPU is available for tree-model training."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # Fallback: check nvidia-smi
    import shutil, subprocess
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return True
        except Exception:
            pass
    return False


def _ensure_opencl_icd() -> None:
    """Set up the NVIDIA OpenCL ICD required by LightGBM GPU on Colab/Kaggle."""
    import os, pathlib
    icd_path = pathlib.Path("/etc/OpenCL/vendors/nvidia.icd")
    if not icd_path.exists():
        try:
            icd_path.parent.mkdir(parents=True, exist_ok=True)
            icd_path.write_text("libnvidia-opencl.so.1\n")
            logger.info("Created OpenCL ICD at %s (needed for LightGBM GPU).", icd_path)
        except PermissionError:
            # Try with subprocess (Colab cells run as root)
            import subprocess
            try:
                subprocess.run(
                    ["bash", "-c",
                     "mkdir -p /etc/OpenCL/vendors && "
                     'echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd'],
                    check=True, capture_output=True,
                )
                logger.info("Created OpenCL ICD via subprocess.")
            except Exception as e:
                logger.warning("Could not create OpenCL ICD: %s. LightGBM GPU may fail.", e)


_GPU_READY: bool | None = None


def gpu_available() -> bool:
    """Cached GPU check (called once per process)."""
    global _GPU_READY
    if _GPU_READY is None:
        _GPU_READY = _gpu_available()
        if _GPU_READY:
            _ensure_opencl_icd()
        logger.info("GPU available: %s", _GPU_READY)
    return _GPU_READY


class ModelTools:
    """
    Static utility methods for model training, hyperparameter optimisation,
    ensemble creation, and model serialisation.

    All methods are ``@staticmethod`` — no instance is needed.

    Examples
    --------
    ::

        from tools.model_tools import ModelTools

        booster = ModelTools.train_lightgbm(X_tr, y_tr, X_val, y_val, params)
        best_params = ModelTools.optimize_hyperparams("lightgbm", X, y, n_trials=30)
        ensemble_preds = ModelTools.create_ensemble(models, weights, X_test)
        ModelTools.save_model(booster, "./outputs/lgbm.pkl")
    """

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    @staticmethod
    def train_lightgbm(
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Train a LightGBM regressor with early stopping.

        Parameters
        ----------
        X_train, y_train:
            Training features and target.
        X_val, y_val:
            Validation features and target (used for early stopping).
        params:
            LightGBM parameters dict.  ``num_boost_round`` is extracted and
            removed from params; defaults to 1000.

        Returns
        -------
        lgb.Booster
            Trained booster object.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        params = dict(params)
        n_estimators      = int(params.pop("n_estimators",      1000))
        early_stopping    = int(params.pop("early_stopping_rounds", 50))
        verbose_eval      = params.pop("verbose", -1)

        params.setdefault("objective",  "regression")
        params.setdefault("metric",     "mse")
        params.setdefault("verbosity",  -1)
        params.setdefault("seed",       42)

        # GPU acceleration
        if gpu_available() and "device" not in params:
            params["device"] = "gpu"
            params.setdefault("gpu_use_dp", False)
            logger.info("[ModelTools] LightGBM using GPU.")

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
        if verbose_eval != -1:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

        t0 = time.perf_counter()
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        elapsed = time.perf_counter() - t0

        best_iter = booster.best_iteration
        best_score = booster.best_score.get("valid_0", {}).get("mse", None)
        logger.info(
            "[ModelTools] LightGBM trained: best_iter=%d, val_mse=%.4f, time=%.1fs",
            best_iter, best_score or float("nan"), elapsed,
        )
        return booster

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    @staticmethod
    def train_xgboost(
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Train an XGBoost regressor with early stopping.

        Parameters
        ----------
        X_train, y_train:
            Training data.
        X_val, y_val:
            Validation data.
        params:
            XGBoost parameters dict.

        Returns
        -------
        xgb.Booster
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        params = dict(params)
        n_estimators   = int(params.pop("n_estimators",      1000))
        early_stopping = int(params.pop("early_stopping_rounds", 50))

        params.setdefault("objective",  "reg:squarederror")
        params.setdefault("eval_metric", "rmse")
        params.setdefault("seed",        42)
        params.setdefault("verbosity",   0)

        # GPU acceleration
        if gpu_available() and "device" not in params:
            params["device"] = "cuda"
            params.setdefault("tree_method", "hist")
            logger.info("[ModelTools] XGBoost using GPU (cuda).")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        evals_result: Dict[str, Any] = {}
        t0 = time.perf_counter()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping,
            evals_result=evals_result,
            verbose_eval=False,
        )
        elapsed = time.perf_counter() - t0

        best_iter  = booster.best_iteration
        val_scores = evals_result.get("val", {}).get("rmse", [])
        best_rmse  = min(val_scores) if val_scores else float("nan")
        logger.info(
            "[ModelTools] XGBoost trained: best_iter=%d, val_rmse=%.4f, time=%.1fs",
            best_iter, best_rmse, elapsed,
        )
        return booster

    # ------------------------------------------------------------------
    # CatBoost
    # ------------------------------------------------------------------

    @staticmethod
    def train_catboost(
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        params: Dict[str, Any],
    ) -> Any:
        """
        Train a CatBoost regressor with early stopping.

        Parameters
        ----------
        X_train, y_train:
            Training data.
        X_val, y_val:
            Validation data.
        params:
            CatBoost parameters dict.

        Returns
        -------
        CatBoostRegressor
        """
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            raise ImportError("CatBoost not installed. Run: pip install catboost")

        params = dict(params)
        early_stopping = int(params.pop("early_stopping_rounds", 50))
        params.setdefault("loss_function",  "RMSE")
        params.setdefault("eval_metric",    "RMSE")
        params.setdefault("random_seed",    42)
        params.setdefault("verbose",        0)

        # GPU acceleration
        if gpu_available() and "task_type" not in params:
            params["task_type"] = "GPU"
            params.setdefault("devices", "0")
            logger.info("[ModelTools] CatBoost using GPU.")

        train_pool = Pool(X_train, label=y_train)
        val_pool   = Pool(X_val,   label=y_val)

        model = CatBoostRegressor(**params)
        t0 = time.perf_counter()
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        best_score = model.get_best_score().get("validation", {}).get("RMSE", float("nan"))
        logger.info(
            "[ModelTools] CatBoost trained: val_rmse=%.4f, time=%.1fs", best_score, elapsed
        )
        return model

    # ------------------------------------------------------------------
    # Hyperparameter optimisation
    # ------------------------------------------------------------------

    @staticmethod
    def optimize_hyperparams(
        model_type: str,
        X: Any,
        y: Any,
        n_trials: int = 20,
        cv_folds: int = 5,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimise hyperparameters using Optuna (TPE sampler).

        Parameters
        ----------
        model_type:
            One of ``"lightgbm"``, ``"xgboost"``, ``"catboost"``.
        X:
            Full feature matrix (CV split handled internally).
        y:
            Target vector.
        n_trials:
            Number of Optuna trials (default 20).
        cv_folds:
            Number of CV folds for evaluation (default 5).
        timeout:
            Optional time limit in seconds for the study.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna not installed. Run: pip install optuna")

        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        kf    = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        def _cv_mse(params: Dict[str, Any]) -> float:
            oof_mse: List[float] = []
            for tr_idx, va_idx in kf.split(X_arr):
                X_tr, X_va = X_arr[tr_idx], X_arr[va_idx]
                y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
                try:
                    if model_type == "lightgbm":
                        m = ModelTools.train_lightgbm(X_tr, y_tr, X_va, y_va, params)
                        preds = m.predict(X_va)
                    elif model_type == "xgboost":
                        import xgboost as xgb
                        m = ModelTools.train_xgboost(X_tr, y_tr, X_va, y_va, params)
                        preds = m.predict(xgb.DMatrix(X_va))
                    elif model_type == "catboost":
                        m = ModelTools.train_catboost(X_tr, y_tr, X_va, y_va, params)
                        preds = np.asarray(m.predict(X_va))
                    else:
                        return float("inf")
                    oof_mse.append(float(mean_squared_error(y_va, preds)))
                except Exception as exc:
                    logger.warning("[ModelTools] Optuna fold failed: %s", exc)
                    oof_mse.append(float("inf"))
            return float(np.mean(oof_mse))

        def _objective_lgbm(trial: Any) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":         trial.suggest_int("max_depth", 4, 12),
                "num_leaves":        trial.suggest_int("num_leaves", 16, 256),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "early_stopping_rounds": 30,
            }
            return _cv_mse(params)

        def _objective_xgb(trial: Any) -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 2000, step=100),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "early_stopping_rounds": 30,
            }
            return _cv_mse(params)

        def _objective_cat(trial: Any) -> float:
            params = {
                "iterations":       trial.suggest_int("iterations", 200, 2000, step=100),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth":            trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "early_stopping_rounds": 30,
            }
            return _cv_mse(params)

        objective_map = {
            "lightgbm": _objective_lgbm,
            "xgboost":  _objective_xgb,
            "catboost": _objective_cat,
        }
        if model_type not in objective_map:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from: {list(objective_map)}"
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective_map[model_type],
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

        best = study.best_params
        best_value = study.best_value
        logger.info(
            "[ModelTools] Optuna %s: best MSE=%.4f after %d trials. Params: %s",
            model_type, best_value, len(study.trials), best,
        )
        return best

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    @staticmethod
    def create_ensemble(
        models: Dict[str, Any],
        weights: Dict[str, float],
        X: Any,
    ) -> np.ndarray:
        """
        Create a weighted ensemble prediction from multiple models.

        Parameters
        ----------
        models:
            Mapping of model_name → trained model object.
        weights:
            Mapping of model_name → weight (will be normalised to sum=1).
        X:
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray
            Weighted average prediction array.
        """
        try:
            import xgboost as xgb
            _xgb = xgb
        except ImportError:
            _xgb = None

        X_arr = np.asarray(X, dtype=np.float32)
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("All ensemble weights are zero.")

        ensemble: Optional[np.ndarray] = None

        for name, model in models.items():
            w = weights.get(name, 0.0) / total_weight
            if w == 0:
                continue

            try:
                import lightgbm as lgb
                if isinstance(model, lgb.Booster):
                    preds = np.asarray(model.predict(X_arr), dtype=float)
                    ensemble = preds * w if ensemble is None else ensemble + preds * w
                    continue
            except ImportError:
                pass

            if _xgb is not None and isinstance(model, _xgb.Booster):
                dmat  = _xgb.DMatrix(X_arr)
                preds = np.asarray(model.predict(dmat), dtype=float)
                ensemble = preds * w if ensemble is None else ensemble + preds * w
                continue

            if hasattr(model, "predict"):
                preds = np.asarray(model.predict(X_arr), dtype=float)
                ensemble = preds * w if ensemble is None else ensemble + preds * w
                continue

            logger.warning("[ModelTools] Skipping unknown model type for '%s'.", name)

        if ensemble is None:
            raise ValueError("No valid model predictions could be generated for the ensemble.")

        logger.debug(
            "[ModelTools] Ensemble created from %d models with weights %s.",
            len(models), weights,
        )
        return ensemble

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    @staticmethod
    def save_model(model: Any, path: str) -> None:
        """
        Save a trained model to disk.

        Supports native save formats for LightGBM / XGBoost, and falls back
        to ``joblib`` for all other sklearn-compatible models.

        Parameters
        ----------
        model:
            Trained model object.
        path:
            File path (will create parent directories as needed).
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                model.save_model(str(out_path))
                logger.info("[ModelTools] LightGBM model saved to %s", out_path)
                return
        except ImportError:
            pass

        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                model.save_model(str(out_path))
                logger.info("[ModelTools] XGBoost model saved to %s", out_path)
                return
        except ImportError:
            pass

        try:
            import joblib
            joblib.dump(model, str(out_path))
            logger.info("[ModelTools] Model saved via joblib to %s", out_path)
        except ImportError:
            import pickle
            with open(out_path, "wb") as f:
                pickle.dump(model, f)
            logger.info("[ModelTools] Model saved via pickle to %s", out_path)

    @staticmethod
    def load_model(path: str) -> Any:
        """
        Load a trained model from disk.

        Tries LightGBM / XGBoost native formats first, then joblib, then pickle.

        Parameters
        ----------
        path:
            File path to load from.

        Returns
        -------
        Any
            Loaded model object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Try LightGBM text format
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(in_path))
            logger.info("[ModelTools] Loaded LightGBM booster from %s", in_path)
            return booster
        except Exception:
            pass

        # Try XGBoost
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(in_path))
            logger.info("[ModelTools] Loaded XGBoost booster from %s", in_path)
            return booster
        except Exception:
            pass

        # joblib / pickle fallback
        try:
            import joblib
            model = joblib.load(str(in_path))
            logger.info("[ModelTools] Loaded model via joblib from %s", in_path)
            return model
        except Exception:
            pass

        import pickle
        with open(in_path, "rb") as f:
            model = pickle.load(f)
        logger.info("[ModelTools] Loaded model via pickle from %s", in_path)
        return model


__all__ = ["ModelTools"]
