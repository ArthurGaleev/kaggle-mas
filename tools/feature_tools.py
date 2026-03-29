"""
FeatureTools — static utility methods used by FeatureAgent.

Install dependencies:
  pip install pandas numpy scikit-learn scipy
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureTools:
    """
    Static utility methods for feature engineering.

    All methods are ``@staticmethod`` — no instance is needed.

    Examples
    --------
    ::

        from tools.feature_tools import FeatureTools

        df = FeatureTools.extract_datetime_features(df, "date_col")
        df = FeatureTools.create_geo_features(df, "lat", "lon", n_clusters=8)
        train, test = FeatureTools.target_encode(train, test, "city", "price")
    """

    # ------------------------------------------------------------------
    # Datetime features
    # ------------------------------------------------------------------

    @staticmethod
    def extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Extract calendar and cyclical features from a datetime column.

        New columns added (prefix = col name):
          year, month, day, dayofweek, hour (if time present),
          quarter, dayofyear, is_weekend,
          month_sin, month_cos (cyclical encoding).

        Parameters
        ----------
        df:
            Input DataFrame.
        col:
            Name of the datetime column (will be parsed if not already datetime).

        Returns
        -------
        pd.DataFrame
            DataFrame with new datetime feature columns appended.
        """
        df = df.copy()
        if col not in df.columns:
            logger.warning("extract_datetime_features: column '%s' not found.", col)
            return df

        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        dt   = df[col].dt
        pfx  = f"{col}_"

        df[pfx + "year"]      = dt.year
        df[pfx + "month"]     = dt.month
        df[pfx + "day"]       = dt.day
        df[pfx + "dayofweek"] = dt.dayofweek
        df[pfx + "quarter"]   = dt.quarter
        df[pfx + "dayofyear"] = dt.dayofyear
        df[pfx + "is_weekend"] = (dt.dayofweek >= 5).astype(np.int8)

        # Hour (only if the column has time information)
        if hasattr(dt, "hour") and df[col].dt.hour.nunique() > 1:
            df[pfx + "hour"] = dt.hour

        # Cyclical month encoding
        df[pfx + "month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        df[pfx + "month_cos"] = np.cos(2 * np.pi * dt.month / 12)

        logger.debug("extract_datetime_features: added features for '%s'.", col)
        return df

    # ------------------------------------------------------------------
    # Geographic features
    # ------------------------------------------------------------------

    @staticmethod
    def create_geo_features(
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        n_clusters: int = 10,
    ) -> pd.DataFrame:
        """
        Create geographic cluster features from latitude and longitude.

        Adds:
          * ``geo_cluster`` — KMeans cluster label (int)
          * ``geo_dist_to_center`` — Euclidean distance to cluster centroid
          * ``geo_lat_norm``, ``geo_lon_norm`` — normalised coordinates

        Parameters
        ----------
        df:
            Input DataFrame.
        lat_col:
            Latitude column name.
        lon_col:
            Longitude column name.
        n_clusters:
            Number of geographic clusters (default 10).

        Returns
        -------
        pd.DataFrame
        """
        from sklearn.cluster import KMeans  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        df = df.copy()

        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning(
                "create_geo_features: columns '%s' or '%s' not found.", lat_col, lon_col
            )
            return df

        coords = df[[lat_col, lon_col]].copy()
        valid_mask = coords.notna().all(axis=1)
        coords_valid = coords[valid_mask]

        if len(coords_valid) < n_clusters:
            logger.warning(
                "create_geo_features: fewer valid rows (%d) than clusters (%d). Reducing.",
                len(coords_valid), n_clusters,
            )
            n_clusters = max(1, len(coords_valid))

        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords_valid)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels    = km.fit_predict(coords_scaled)
        centroids = km.cluster_centers_

        df["geo_cluster"] = -1
        df.loc[valid_mask, "geo_cluster"] = labels

        # Distance to assigned centroid
        dist = np.linalg.norm(
            coords_scaled - centroids[labels], axis=1
        )
        df["geo_dist_to_center"] = 0.0
        df.loc[valid_mask, "geo_dist_to_center"] = dist

        # Normalised raw coordinates
        norm_coords = scaler.transform(coords_valid)
        df["geo_lat_norm"] = 0.0
        df["geo_lon_norm"] = 0.0
        df.loc[valid_mask, "geo_lat_norm"] = norm_coords[:, 0]
        df.loc[valid_mask, "geo_lon_norm"] = norm_coords[:, 1]

        logger.debug(
            "create_geo_features: %d clusters from (%s, %s).", n_clusters, lat_col, lon_col
        )
        return df

    # ------------------------------------------------------------------
    # Text features
    # ------------------------------------------------------------------

    @staticmethod
    def create_text_features(
        df: pd.DataFrame,
        col: str,
        n_components: int = 5,
    ) -> pd.DataFrame:
        """
        Create TF-IDF + SVD (LSA) features from a text column.

        Adds *n_components* new columns named ``{col}_svd_0``, ``{col}_svd_1``, …

        Parameters
        ----------
        df:
            Input DataFrame.
        col:
            Text column name.
        n_components:
            Number of SVD components to retain (default 5).

        Returns
        -------
        pd.DataFrame
        """
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.decomposition import TruncatedSVD                # noqa: PLC0415

        df = df.copy()
        if col not in df.columns:
            logger.warning("create_text_features: column '%s' not found.", col)
            return df

        texts  = df[col].fillna("").astype(str)
        valid  = texts.str.len() > 0

        if valid.sum() < 2:
            logger.warning(
                "create_text_features: too few non-empty texts in '%s'.", col
            )
            return df

        n_comp = min(n_components, valid.sum() - 1)

        tfidf  = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        X_tfidf = tfidf.fit_transform(texts[valid])

        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_svd = svd.fit_transform(X_tfidf)

        for i in range(n_comp):
            feat_col = f"{col}_svd_{i}"
            df[feat_col] = 0.0
            df.loc[valid, feat_col] = X_svd[:, i]

        explained = svd.explained_variance_ratio_.sum()
        logger.debug(
            "create_text_features: %d SVD components from '%s' (explained=%.2f%%).",
            n_comp, col, 100 * explained,
        )
        return df

    # ------------------------------------------------------------------
    # Target encoding
    # ------------------------------------------------------------------

    @staticmethod
    def target_encode(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        col: str,
        target_col: str,
        n_folds: int = 5,
        smoothing: float = 10.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply smoothed K-fold target encoding to a categorical column.

        Uses out-of-fold estimates on the training set to prevent leakage,
        and the global training mean for unseen test categories.

        Parameters
        ----------
        train_df:
            Training DataFrame (must contain *col* and *target_col*).
        test_df:
            Test DataFrame (must contain *col*).
        col:
            Categorical column to encode.
        target_col:
            Target column name.
        n_folds:
            Number of folds for OOF encoding (default 5).
        smoothing:
            Smoothing factor (higher → more shrinkage toward global mean).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Updated (train_df, test_df) with a new column ``{col}_target_enc``.
        """
        from sklearn.model_selection import KFold  # noqa: PLC0415

        train_df = train_df.copy()
        test_df  = test_df.copy()

        if col not in train_df.columns:
            logger.warning("target_encode: column '%s' not found.", col)
            return train_df, test_df

        out_col    = f"{col}_target_enc"
        global_mean = train_df[target_col].mean()
        oof_enc    = np.full(len(train_df), global_mean, dtype=float)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for tr_idx, va_idx in kf.split(train_df):
            tr_fold = train_df.iloc[tr_idx]
            va_fold = train_df.iloc[va_idx]

            agg = tr_fold.groupby(col)[target_col].agg(["mean", "count"])
            agg.columns = ["cat_mean", "cat_count"]
            # Smoothed mean: blend category mean with global mean
            agg["smoothed"] = (
                (agg["cat_mean"] * agg["cat_count"] + smoothing * global_mean)
                / (agg["cat_count"] + smoothing)
            )
            mapping = agg["smoothed"].to_dict()
            oof_enc[va_idx] = va_fold[col].map(mapping).fillna(global_mean).values

        train_df[out_col] = oof_enc

        # Test set: use full training stats
        agg_full = train_df.groupby(col)[target_col].agg(["mean", "count"])
        agg_full.columns = ["cat_mean", "cat_count"]
        agg_full["smoothed"] = (
            (agg_full["cat_mean"] * agg_full["cat_count"] + smoothing * global_mean)
            / (agg_full["cat_count"] + smoothing)
        )
        test_df[out_col] = test_df[col].map(agg_full["smoothed"]).fillna(global_mean)

        logger.debug("target_encode: encoded '%s' → '%s'.", col, out_col)
        return train_df, test_df

    # ------------------------------------------------------------------
    # Frequency encoding
    # ------------------------------------------------------------------

    @staticmethod
    def frequency_encode(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode a categorical column with its frequency (proportion) in training data.

        Parameters
        ----------
        train_df:
            Training DataFrame.
        test_df:
            Test DataFrame.
        col:
            Column to encode.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Updated DataFrames with a new column ``{col}_freq_enc``.
        """
        train_df = train_df.copy()
        test_df  = test_df.copy()

        if col not in train_df.columns:
            logger.warning("frequency_encode: column '%s' not found.", col)
            return train_df, test_df

        out_col  = f"{col}_freq_enc"
        freq_map = train_df[col].value_counts(normalize=True).to_dict()

        train_df[out_col] = train_df[col].map(freq_map).fillna(0.0)
        test_df[out_col]  = test_df[col].map(freq_map).fillna(0.0)

        logger.debug("frequency_encode: encoded '%s' → '%s'.", col, out_col)
        return train_df, test_df

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------

    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Create pairwise interaction features (products) for specified column pairs.

        For each pair ``(a, b)`` a new column ``{a}_x_{b}`` is added containing
        the element-wise product.

        Parameters
        ----------
        df:
            Input DataFrame.
        pairs:
            List of ``(col_a, col_b)`` tuples.

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        for col_a, col_b in pairs:
            if col_a not in df.columns or col_b not in df.columns:
                logger.warning(
                    "create_interaction_features: skipping pair ('%s', '%s') — not found.",
                    col_a, col_b,
                )
                continue
            out_col = f"{col_a}_x_{col_b}"
            try:
                df[out_col] = df[col_a] * df[col_b]
                logger.debug("Interaction: %s × %s → %s", col_a, col_b, out_col)
            except Exception as exc:
                logger.warning(
                    "create_interaction_features: could not compute %s × %s: %s",
                    col_a, col_b, exc,
                )
        return df

    # ------------------------------------------------------------------
    # Log transform
    # ------------------------------------------------------------------

    @staticmethod
    def log_transform(
        df: pd.DataFrame,
        cols: List[str],
        shift: float = 1.0,
    ) -> pd.DataFrame:
        """
        Apply log1p (log(x + shift)) transformation to specified columns.

        The shift ensures that zero values are handled correctly.
        Only numeric columns are transformed; others are silently skipped.

        Parameters
        ----------
        df:
            Input DataFrame.
        cols:
            List of column names to transform.
        shift:
            Value added before taking the log (default 1.0).

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                logger.warning("log_transform: column '%s' not found.", col)
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("log_transform: column '%s' is not numeric — skipping.", col)
                continue
            df[col] = np.log1p(df[col].clip(lower=-shift + 1e-9) + shift - 1)
            logger.debug("log_transform applied to '%s'.", col)
        return df


__all__ = ["FeatureTools"]
