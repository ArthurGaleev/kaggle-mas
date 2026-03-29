"""
EvaluatorAgent — computes comprehensive metrics for each model and the
ensemble, analyses feature importances and residuals, then asks the LLM
to interpret results and provide improvement recommendations.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from agents.base import BaseAgent
from agents.feature_agent import _compute_target_encoding_map
from utils.helpers import safe_json_parse


def _xgb_predict(booster: Any, X: np.ndarray) -> np.ndarray:
    """Predict with an xgb.Booster, wrapping X in a DMatrix."""
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(X))


class EvaluatorAgent(BaseAgent):
    """
    Computes evaluation metrics and asks LLM to interpret them.

    Expected state keys consumed:
        target_series    (pd.Series):     ground-truth labels.
        oof_predictions  (dict):          OOF preds per algorithm.
        cv_scores        (dict):          fold MSEs per algorithm.
        ensemble_oof     (np.ndarray):    ensemble OOF predictions.
        ensemble_cv_mse  (float):         ensemble OOF MSE.
        feature_importances (dict):       mean importance per algorithm.
        feature_names    (list[str]):      feature column names.
        models           (dict):          trained model objects (for train-score check).
        train_feat       (pd.DataFrame):  training feature matrix.

    State keys produced:
        evaluation_report (dict):  structured evaluation metrics.
        llm_interpretation (str):  LLM textual interpretation + recommendations.
    """

    SYSTEM_PROMPT = (
        "You are a senior ML evaluation specialist. You interpret cross-validation "
        "results, identify overfitting and instability, and provide actionable "
        "improvement recommendations for a rental-property price regression task (MSE metric)."
    )

    TOP_K_FEATURES = 20

    # TE placeholder columns (object dtype) that must be encoded before predict
    _TE_COLS = ("host_name", "location_cluster", "type_house")

    # ------------------------------------------------------------------
    # Metric computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return MSE, RMSE, MAE, R² for a predictions array."""
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mse":  round(mse, 6),
            "rmse": round(float(np.sqrt(mse)), 6),
            "mae":  round(float(mean_absolute_error(y_true, y_pred)), 6),
            "r2":   round(float(r2_score(y_true, y_pred)), 6),
        }

    @staticmethod
    def _cv_stability(fold_mses: List[float]) -> Dict[str, float]:
        """Return mean, std, and coefficient of variation for fold MSEs."""
        arr = np.array(fold_mses)
        mean = float(arr.mean())
        std = float(arr.std())
        return {
            "mean_mse": round(mean, 6),
            "std_mse":  round(std, 6),
            "cv_pct":   round(100 * std / max(mean, 1e-9), 2),
        }

    @staticmethod
    def _residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute residual statistics including normality test."""
        residuals = y_true - y_pred
        skewness = float(stats.skew(residuals))
        kurtosis = float(stats.kurtosis(residuals))
        sample = residuals if len(residuals) <= 5000 else np.random.default_rng(42).choice(residuals, 5000)
        try:
            _, shapiro_p = stats.shapiro(sample)
            is_normal = bool(shapiro_p > 0.05)
        except Exception:
            shapiro_p = float("nan")
            is_normal = False

        return {
            "mean":      round(float(residuals.mean()), 6),
            "std":       round(float(residuals.std()), 6),
            "min":       round(float(residuals.min()), 6),
            "max":       round(float(residuals.max()), 6),
            "p5":        round(float(np.percentile(residuals, 5)), 6),
            "p95":       round(float(np.percentile(residuals, 95)), 6),
            "skewness":  round(skewness, 4),
            "kurtosis":  round(kurtosis, 4),
            "shapiro_p": round(float(shapiro_p), 6) if not np.isnan(shapiro_p) else None,
            "is_normal": is_normal,
        }

    # ------------------------------------------------------------------
    # X_train reconstruction with TE features
    # ------------------------------------------------------------------

    def _reconstruct_X_train(
        self,
        train_feat: pd.DataFrame,
        y_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Rebuild the training feature matrix exactly as ModelAgent did:
        numeric columns + target-encoded columns appended at the end.

        This is required for the overfitting check — models were trained on
        54 features (51 numeric + 3 TE), but train_feat contains the TE cols
        as object-dtype placeholders that cannot be cast to float32 directly.

        We apply TE using the full training set (no fold split needed here
        since we only use this for a train-score sanity check, not for
        generating OOF predictions).
        """
        te_cols_present = [c for c in self._TE_COLS if c in train_feat.columns]
        num_cols = [c for c in train_feat.columns if c not in self._TE_COLS]
        base = train_feat[num_cols].values.astype(np.float32)

        if not te_cols_present:
            return base

        global_mean = float(y_raw.mean())
        extras = []
        for col in te_cols_present:
            mapping = _compute_target_encoding_map(
                col=train_feat[col],
                target=pd.Series(y_raw, index=train_feat.index),
                global_mean=global_mean,
                smoothing=10.0,
            )
            extras.append(train_feat[col].map(mapping).fillna(global_mean).values)

        return np.hstack([base, np.column_stack(extras).astype(np.float32)])

    # ------------------------------------------------------------------
    # Overfitting check
    # ------------------------------------------------------------------

    def _overfitting_check(
        self,
        algo: str,
        algo_models: List[Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        oof_mse: float,
    ) -> Dict[str, Any]:
        """
        Estimate train MSE by averaging predictions from all fold models on
        the full training set and comparing to OOF MSE.

        Uses _xgb_predict for XGBoost Boosters (requires DMatrix, not ndarray).
        X_train must have the same feature count as the models were trained on
        (i.e. include TE-encoded columns).
        """
        try:
            preds = np.zeros(len(y_train))
            n_models = len(algo_models)
            for model in algo_models:
                try:
                    import xgboost as xgb
                    if isinstance(model, xgb.Booster):
                        preds += _xgb_predict(model, X_train) / n_models
                        continue
                except ImportError:
                    pass
                preds += np.asarray(model.predict(X_train)) / n_models

            train_mse = float(mean_squared_error(y_train, preds))
            overfit_ratio = oof_mse / max(train_mse, 1e-9)
            return {
                "train_mse": round(train_mse, 6),
                "oof_mse":   round(oof_mse, 6),
                "ratio":     round(overfit_ratio, 4),
                "overfitting": overfit_ratio > 1.5,
            }
        except Exception as exc:
            self._log(f"Overfitting check failed for {algo}: {exc}", level="warning")
            return {"error": str(exc)}

    def _top_features(
        self, importances: Dict[str, float], top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Return top-K features sorted by mean importance."""
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": k, "importance": round(v, 4)} for k, v in sorted_feats[:top_k]]

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct the full evaluation report dict from state.
        """
        y = state["target_series"].values.astype(np.float64)
        oof_preds = state["oof_predictions"]
        cv_scores = state["cv_scores"]
        ensemble_oof = state["ensemble_oof"].astype(np.float64)
        feat_importances = state.get("feature_importances", {})
        models_dict = state.get("models", {})
        feature_names = state.get("feature_names", [])

        # Reconstruct X_train with TE columns appended — matches the feature
        # count the models were actually trained on (fixes 51 vs 54 mismatch).
        X_train = self._reconstruct_X_train(
            train_feat=state["train_feat"],
            y_raw=y.astype(np.float32),
        )

        report: Dict[str, Any] = {"per_algorithm": {}, "ensemble": {}, "feature_analysis": {}}

        for algo, oof in oof_preds.items():
            oof_arr = oof.astype(np.float64)
            metrics = self._compute_metrics(y, oof_arr)
            stability = self._cv_stability(cv_scores.get(algo, {}).get("fold_mses", [metrics["mse"]]))
            residuals = self._residual_analysis(y, oof_arr)

            algo_models = models_dict.get(algo, [])
            overfit = self._overfitting_check(algo, algo_models, X_train, y, metrics["mse"])

            top_feats = (
                self._top_features(feat_importances.get(algo, {}), self.TOP_K_FEATURES)
                if algo in feat_importances
                else []
            )

            report["per_algorithm"][algo] = {
                "oof_metrics": metrics,
                "cv_stability": stability,
                "residuals": residuals,
                "overfitting_check": overfit,
                "top_features": top_feats,
            }

        ens_metrics = self._compute_metrics(y, ensemble_oof)
        ens_residuals = self._residual_analysis(y, ensemble_oof)
        report["ensemble"] = {
            "oof_metrics": ens_metrics,
            "residuals": ens_residuals,
            "weights": state.get("ensemble_weights", {}),
        }

        if feat_importances:
            combined: Dict[str, float] = {}
            for algo, imp in feat_importances.items():
                weight = state.get("ensemble_weights", {}).get(algo, 1.0 / max(len(feat_importances), 1))
                for feat, val in imp.items():
                    combined[feat] = combined.get(feat, 0.0) + weight * val
            report["feature_analysis"]["weighted_top_features"] = self._top_features(combined, self.TOP_K_FEATURES)
            report["feature_analysis"]["total_features"] = len(feature_names)

        return report

    # ------------------------------------------------------------------
    # LLM interpretation
    # ------------------------------------------------------------------

    def _request_interpretation(self, report: Dict[str, Any]) -> str:
        """Ask the LLM to interpret the evaluation report and suggest improvements."""
        algo_summaries = {}
        for algo, data in report.get("per_algorithm", {}).items():
            m = data.get("oof_metrics", {})
            s = data.get("cv_stability", {})
            o = data.get("overfitting_check", {})
            algo_summaries[algo] = {
                "oof_mse": m.get("mse"),
                "oof_rmse": m.get("rmse"),
                "oof_r2": m.get("r2"),
                "cv_std_mse": s.get("std_mse"),
                "cv_pct": s.get("cv_pct"),
                "overfit_ratio": o.get("ratio"),
                "is_overfit": o.get("overfitting"),
            }

        ens = report.get("ensemble", {}).get("oof_metrics", {})
        top_feats = report.get("feature_analysis", {}).get("weighted_top_features", [])[:10]

        prompt = f"""
## Evaluation Report Summary — Rental Property Regression (MSE metric)

### Per-algorithm OOF performance
{json.dumps(algo_summaries, indent=2)}

### Ensemble OOF metrics
MSE={ens.get('mse')}, RMSE={ens.get('rmse')}, R\u00b2={ens.get('r2')}

### Top-10 features (weighted importance)
{json.dumps(top_feats, indent=2)}

## Instructions
Interpret these results concisely. Address:
1. Which model performs best and why it might be so.
2. Whether there are signs of overfitting or instability.
3. Are residuals acceptable? Any systematic bias?
4. What are the 2-3 most impactful improvements to try next?
5. Confidence level: should we accept results or iterate?

Be specific and actionable. Respond in plain text (no JSON).
"""
        return self._ask_llm(prompt)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build evaluation report deterministically, then get LLM interpretation."""
        self._log("Building evaluation report\u2026")
        report = self._build_report(state)

        self._log("Requesting LLM interpretation\u2026")
        interpretation = self._request_interpretation(report)

        state["evaluation_report"] = report
        state["llm_interpretation"] = interpretation

        best_algo = min(
            report["per_algorithm"],
            key=lambda a: report["per_algorithm"][a]["oof_metrics"]["mse"],
            default="N/A",
        )
        ens_mse = report["ensemble"]["oof_metrics"]["mse"]
        self._log(f"Best single model: {best_algo}  |  Ensemble OOF MSE: {ens_mse:.4f}")

        return state
