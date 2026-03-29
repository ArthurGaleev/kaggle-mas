"""
Output validation guardrails for the multi-agent ML pipeline.

Validates model predictions, competition submission files, and serialized
model artifacts before they are used or submitted.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class OutputValidator:
    """Validates pipeline outputs: predictions, submissions, and model artifacts.

    All public methods return ``(is_valid: bool, issues: List[str])``.
    """

    # ------------------------------------------------------------------
    # Prediction validation
    # ------------------------------------------------------------------

    def validate_predictions(
        self,
        predictions: Union[np.ndarray, list],
        cfg: Any,
    ) -> Tuple[bool, List[str]]:
        """Validate a raw prediction array.

        Checks performed
        ----------------
        * All values are finite (no NaN or Inf).
        * Values fall within configured bounds.
        * Array length matches expected length (if configured).
        * Predictions are not suspiciously constant (std > 0).

        Parameters
        ----------
        predictions:
            1-D array-like of predicted values.
        cfg:
            Config object / dict.  Optional keys:

            * ``pred_min`` – lower bound (default ``-1e9``).
            * ``pred_max`` – upper bound (default ``1e9``).
            * ``expected_length`` – expected number of predictions.
            * ``constant_std_threshold`` – minimum acceptable std
              (default ``1e-6``).

        Returns
        -------
        (is_valid, issues)
        """
        issues: List[str] = []

        preds = np.asarray(predictions, dtype=float)

        cfg_dict = _to_dict(cfg)
        pred_min: float = cfg_dict.get("pred_min", -1e9)
        pred_max: float = cfg_dict.get("pred_max", 1e9)
        expected_len: Optional[int] = cfg_dict.get("expected_length", None)
        const_thresh: float = cfg_dict.get("constant_std_threshold", 1e-6)

        # NaN check
        n_nan = int(np.isnan(preds).sum())
        if n_nan > 0:
            issues.append(f"Predictions contain {n_nan} NaN value(s).")

        # Inf check
        n_inf = int(np.isinf(preds).sum())
        if n_inf > 0:
            issues.append(f"Predictions contain {n_inf} Inf value(s).")

        # Bounds check (only on finite values)
        finite_mask = np.isfinite(preds)
        if finite_mask.any():
            below = int((preds[finite_mask] < pred_min).sum())
            above = int((preds[finite_mask] > pred_max).sum())
            if below > 0:
                issues.append(
                    f"{below} predictions below lower bound {pred_min}."
                )
            if above > 0:
                issues.append(
                    f"{above} predictions above upper bound {pred_max}."
                )

        # Length check
        if expected_len is not None and len(preds) != expected_len:
            issues.append(
                f"Prediction length {len(preds)} != expected {expected_len}."
            )

        # Constant predictions
        if finite_mask.any():
            std_val = float(np.std(preds[finite_mask]))
            if std_val < const_thresh:
                issues.append(
                    f"Predictions appear constant (std={std_val:.2e} < {const_thresh})."
                )

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning("Prediction validation failed: %s", issues)
        else:
            logger.info(
                "Prediction validation passed (n=%d, mean=%.4f, std=%.4f).",
                len(preds),
                float(np.nanmean(preds)),
                float(np.nanstd(preds)),
            )
        return is_valid, issues

    # ------------------------------------------------------------------
    # Submission file validation
    # ------------------------------------------------------------------

    def validate_submission(
        self,
        submission_df: Any,
        sample_submission_path: Union[str, Path],
    ) -> Tuple[bool, List[str]]:
        """Validate a submission DataFrame against the sample submission file.

        Parameters
        ----------
        submission_df:
            ``pandas.DataFrame`` to be submitted.
        sample_submission_path:
            Path to the competition's ``sample_submission.csv`` file.

        Returns
        -------
        (is_valid, issues)
        """
        import pandas as pd  # noqa: PLC0415

        issues: List[str] = []

        if not isinstance(submission_df, pd.DataFrame):
            return False, ["submission_df is not a pandas DataFrame."]

        sample_path = Path(sample_submission_path)
        if not sample_path.exists():
            issues.append(f"Sample submission file not found: {sample_path}.")
            return False, issues

        try:
            sample = pd.read_csv(sample_path)
        except Exception as exc:
            return False, [f"Could not read sample submission: {exc}"]

        # Column check
        missing_cols = [c for c in sample.columns if c not in submission_df.columns]
        extra_cols = [c for c in submission_df.columns if c not in sample.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}.")
        if extra_cols:
            issues.append(f"Unexpected extra columns: {extra_cols}.")

        # Row count
        if len(submission_df) != len(sample):
            issues.append(
                f"Row count mismatch: got {len(submission_df)}, expected {len(sample)}."
            )

        # ID column match (first column assumed to be the ID)
        id_col = sample.columns[0]
        if id_col in submission_df.columns and id_col in sample.columns:
            sub_ids = set(submission_df[id_col].tolist())
            smp_ids = set(sample[id_col].tolist())
            extra_ids = sub_ids - smp_ids
            missing_ids = smp_ids - sub_ids
            if extra_ids:
                issues.append(f"{len(extra_ids)} unexpected IDs in submission.")
            if missing_ids:
                issues.append(f"{len(missing_ids)} IDs missing from submission.")

        # Target column NaN / infinite
        target_col = sample.columns[-1]
        if target_col in submission_df.columns:
            col_vals = submission_df[target_col]
            n_nan = col_vals.isna().sum()
            if n_nan > 0:
                issues.append(f"Target column '{target_col}' has {n_nan} NaN value(s).")
            numeric = pd.to_numeric(col_vals, errors="coerce")
            n_inf = int(np.isinf(numeric.dropna().values).sum())
            if n_inf > 0:
                issues.append(f"Target column '{target_col}' has {n_inf} Inf value(s).")

            # Range check vs sample (if sample has non-trivial values)
            if pd.api.types.is_numeric_dtype(sample[target_col]):
                sample_min = sample[target_col].min()
                sample_max = sample[target_col].max()
                if sample_min != sample_max:  # sample has actual values
                    far_below = int((numeric < sample_min * 0.01).sum())
                    far_above = int((numeric > sample_max * 100).sum())
                    if far_below + far_above > 0:
                        issues.append(
                            f"{far_below + far_above} target values are wildly outside "
                            f"sample range [{sample_min}, {sample_max}]."
                        )

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning("Submission validation failed: %s", issues)
        else:
            logger.info("Submission validation passed (%d rows).", len(submission_df))
        return is_valid, issues

    # ------------------------------------------------------------------
    # Model artifacts validation
    # ------------------------------------------------------------------

    def validate_model_artifacts(
        self,
        state: Any,
    ) -> Tuple[bool, List[str]]:
        """Validate that the pipeline state contains serializable model artifacts.

        Parameters
        ----------
        state:
            Pipeline state dict (or object with ``__dict__``) that should
            contain ``models``, ``feature_names``, and ``preprocessing``
            entries.

        Returns
        -------
        (is_valid, issues)
        """
        issues: List[str] = []

        state_dict = state if isinstance(state, dict) else (
            state.__dict__ if hasattr(state, "__dict__") else {}
        )

        # Models present
        models = state_dict.get("models", None)
        if not models:
            issues.append("No models found in state.")
        else:
            if isinstance(models, dict):
                model_items = list(models.items())
            elif isinstance(models, (list, tuple)):
                model_items = list(enumerate(models))
            else:
                model_items = [("model", models)]

            for name, model in model_items:
                try:
                    pickle.dumps(model)
                except Exception as exc:
                    issues.append(f"Model '{name}' is not serializable: {exc}.")

        # Feature names
        feature_names = state_dict.get("feature_names", None)
        if feature_names is None:
            issues.append("'feature_names' not found in state.")
        elif not isinstance(feature_names, (list, tuple, np.ndarray)):
            issues.append(
                f"'feature_names' has unexpected type {type(feature_names).__name__}."
            )
        elif len(feature_names) == 0:
            issues.append("'feature_names' is empty.")

        # Preprocessing pipeline
        preprocessing = state_dict.get("preprocessing", None)
        if preprocessing is None:
            issues.append("'preprocessing' pipeline not found in state.")
        else:
            try:
                pickle.dumps(preprocessing)
            except Exception as exc:
                issues.append(f"Preprocessing pipeline is not serializable: {exc}.")

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning("Model artifact validation failed: %s", issues)
        else:
            logger.info("Model artifact validation passed.")
        return is_valid, issues


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return cfg.__dict__
    return {}
