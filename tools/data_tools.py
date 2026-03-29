"""
DataTools — static utility methods used by DataAgent.

Install dependencies:
  pip install pandas numpy scikit-learn
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataTools:
    """
    Static utility methods for data loading, profiling, and cleaning.

    All methods are ``@staticmethod`` — no instance is needed.

    Examples
    --------
    ::

        from tools.data_tools import DataTools

        df = DataTools.load_csv("./data/train.csv")
        profile = DataTools.compute_profile(df)
        df_clean = DataTools.apply_cleaning_plan(df, plan)
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Handles common encodings (utf-8, latin-1) and reports file info.

        Parameters
        ----------
        path:
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError:
            If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(p, encoding=encoding, low_memory=False)
                logger.info(
                    "Loaded %s — %d rows × %d cols (encoding=%s).",
                    p.name, len(df), len(df.columns), encoding,
                )
                return df
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                raise RuntimeError(f"Failed to load CSV {path}: {exc}") from exc

        raise RuntimeError(f"Could not decode CSV file: {path}")

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    @staticmethod
    def compute_profile(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute a comprehensive data profile of a DataFrame.

        Metrics included:
          * Shape (n_rows, n_cols)
          * Dtypes summary
          * Missing values (count and fraction per column)
          * Unique value counts
          * Numeric statistics (mean, std, min, max, skew, kurtosis)
          * Duplicate rows count
          * Memory usage (MB)

        Parameters
        ----------
        df:
            Input DataFrame to profile.

        Returns
        -------
        dict
            Structured profiling results.
        """
        n_rows, n_cols = df.shape

        # Dtype summary
        dtype_counts: Dict[str, int] = {}
        for dtype in df.dtypes:
            key = str(dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + 1

        # Missing values
        missing_counts = df.isnull().sum().to_dict()
        missing_fracs  = (df.isnull().mean() * 100).round(2).to_dict()
        total_missing  = int(df.isnull().sum().sum())
        overall_missing_pct = round(total_missing / max(n_rows * n_cols, 1) * 100, 2)

        # Cardinality
        unique_counts = df.nunique().to_dict()

        # Numeric stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            numeric_stats[col] = {
                "mean":     float(s.mean()),
                "std":      float(s.std()),
                "min":      float(s.min()),
                "max":      float(s.max()),
                "median":   float(s.median()),
                "skew":     float(s.skew()),
                "kurtosis": float(s.kurtosis()),
                "q25":      float(s.quantile(0.25)),
                "q75":      float(s.quantile(0.75)),
            }

        # Constant columns
        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

        # High-cardinality categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        high_card = {
            col: int(unique_counts.get(col, 0))
            for col in cat_cols
            if unique_counts.get(col, 0) > 50
        }

        profile = {
            "n_rows":               n_rows,
            "n_cols":               n_cols,
            "dtype_counts":         dtype_counts,
            "numeric_cols":         numeric_cols,
            "categorical_cols":     cat_cols,
            "constant_cols":        constant_cols,
            "missing_counts":       missing_counts,
            "missing_fracs":        missing_fracs,
            "total_missing":        total_missing,
            "overall_missing_pct":  overall_missing_pct,
            "unique_counts":        unique_counts,
            "high_cardinality_cols": high_card,
            "duplicate_rows":       int(df.duplicated().sum()),
            "memory_mb":            round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "numeric_stats":        numeric_stats,
        }

        logger.info(
            "Profile: %d rows, %d cols, %.1f%% missing, %d duplicates.",
            n_rows, n_cols, overall_missing_pct, profile["duplicate_rows"],
        )
        return profile

    # ------------------------------------------------------------------
    # Cleaning plan executor
    # ------------------------------------------------------------------

    @staticmethod
    def apply_cleaning_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply a structured cleaning plan to a DataFrame.

        Supported plan keys (all optional):
          * ``drop_cols``        — list of column names to drop
          * ``impute``           — dict of col → strategy ("mean"/"median"/"mode"/value)
          * ``rename_cols``      — dict of old_name → new_name
          * ``parse_dates``      — list of columns to parse as datetime
          * ``drop_duplicates``  — bool (default False)
          * ``cast``             — dict of col → dtype string
          * ``clip``             — dict of col → {"min": val, "max": val}

        Parameters
        ----------
        df:
            Input DataFrame.
        plan:
            Cleaning instructions produced by the LLM or user.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame (copy).
        """
        df = df.copy()

        # Drop columns
        for col in plan.get("drop_cols", []):
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                logger.debug("Dropped column: %s", col)

        # Impute
        impute_plan = plan.get("impute", {})
        for col, strategy in impute_plan.items():
            if col not in df.columns:
                continue
            df = DataTools.impute_column(df, col, strategy)

        # Rename
        rename_map = plan.get("rename_cols", {})
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.debug("Renamed columns: %s", rename_map)

        # Parse dates
        for col in plan.get("parse_dates", []):
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    logger.debug("Parsed dates: %s", col)
                except Exception as exc:
                    logger.warning("Could not parse dates for %s: %s", col, exc)

        # Drop duplicates
        if plan.get("drop_duplicates", False):
            before = len(df)
            df.drop_duplicates(inplace=True)
            logger.debug("Dropped %d duplicate rows.", before - len(df))

        # Cast dtypes
        cast_plan = plan.get("cast", {})
        for col, dtype_str in cast_plan.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype_str)
                    logger.debug("Cast %s → %s", col, dtype_str)
                except Exception as exc:
                    logger.warning("Could not cast %s to %s: %s", col, dtype_str, exc)

        # Clip values
        clip_plan = plan.get("clip", {})
        for col, bounds in clip_plan.items():
            if col in df.columns:
                try:
                    lo = bounds.get("min")
                    hi = bounds.get("max")
                    df[col] = df[col].clip(lower=lo, upper=hi)
                    logger.debug("Clipped %s to [%s, %s].", col, lo, hi)
                except Exception as exc:
                    logger.warning("Could not clip %s: %s", col, exc)

        logger.info("Cleaning plan applied. Shape: %s.", df.shape)
        return df

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_outliers_iqr(
        series: pd.Series,
        factor: float = 1.5,
    ) -> pd.Series:
        """
        Detect outliers using the IQR (interquartile range) method.

        An observation is flagged as an outlier if it falls below
        ``Q1 - factor * IQR`` or above ``Q3 + factor * IQR``.

        Parameters
        ----------
        series:
            Numeric pandas Series.
        factor:
            IQR multiplier (default 1.5; use 3.0 for extreme outliers).

        Returns
        -------
        pd.Series
            Boolean mask where ``True`` indicates an outlier.
        """
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning(
                "detect_outliers_iqr: column '%s' is not numeric. Returning all False.",
                series.name,
            )
            return pd.Series(False, index=series.index)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            return pd.Series(False, index=series.index)

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask  = (series < lower) | (series > upper)
        n_out = int(mask.sum())
        logger.debug(
            "IQR outliers in '%s': %d (%.2f%%) — bounds [%.4f, %.4f].",
            series.name, n_out, 100 * n_out / max(len(series), 1), lower, upper,
        )
        return mask

    # ------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------

    @staticmethod
    def impute_column(
        df: pd.DataFrame,
        col: str,
        strategy: Any,
    ) -> pd.DataFrame:
        """
        Impute missing values in a single column.

        Parameters
        ----------
        df:
            DataFrame containing *col*.
        col:
            Column name to impute.
        strategy:
            Imputation strategy:
            - ``"mean"``   → column mean (numeric only)
            - ``"median"`` → column median (numeric only)
            - ``"mode"``   → most frequent value
            - ``"zero"``   → fill with 0
            - ``"unknown"``→ fill with string ``"unknown"``
            - Any other value → fill with that value directly.

        Returns
        -------
        pd.DataFrame
            DataFrame with the imputed column (copy of the input).
        """
        if col not in df.columns:
            logger.warning("impute_column: column '%s' not found.", col)
            return df

        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            return df

        df = df.copy()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if strategy == "mean":
            if is_numeric:
                fill_value = df[col].mean()
            else:
                logger.warning("impute_column: 'mean' strategy used on non-numeric '%s'. Using mode.", col)
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
        elif strategy == "median":
            if is_numeric:
                fill_value = df[col].median()
            else:
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
        elif strategy == "mode":
            fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else (0 if is_numeric else "unknown")
        elif strategy == "zero":
            fill_value = 0
        elif strategy == "unknown":
            fill_value = "unknown"
        else:
            # Treat strategy as a literal fill value
            fill_value = strategy

        df[col] = df[col].fillna(fill_value)
        logger.debug(
            "Imputed %d missing in '%s' with %s=%r.",
            n_missing, col, strategy, fill_value,
        )
        return df


__all__ = ["DataTools"]
