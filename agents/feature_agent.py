"""
FeatureAgent — asks the LLM to plan feature engineering, then executes the
plan deterministically using pandas / sklearn / scipy.

All heavy computation (KMeans, TF-IDF, SVD, scaling …) is done in Python;
the LLM only decides which feature groups to activate and any non-trivial
parameter choices.

NOTE ON TARGET ENCODING
-----------------------
Target encoding is intentionally NOT applied inside this agent to prevent
data leakage.  If the feature plan enables `target_encoding`, this agent
stores the request in the feature plan but defers execution to ModelAgent,
which applies the encoding *inside* each KFold split using only the training
fold.  Use `_compute_target_encoding_map` (exported below) for that purpose.

NOTE ON FREQUENCY ENCODING
---------------------------
Frequency encoding is computed on the full training set here (before CV
splits), which technically leaks test-fold category frequencies into the
training folds.  The leakage is minor because frequency does not directly
encode the target, but it is non-zero when category frequency correlates
with price (e.g., popular hosts tend to charge more).  A fully leak-free
alternative is to recompute the frequency map inside each CV fold in
ModelAgent — the same pattern used for target encoding.  The current
implementation is kept for simplicity and computational speed; replace with
fold-level frequency encoding if you observe CV/LB gap on frequency features.

NOTE ON SCALING
---------------
StandardScaler is deliberately omitted from the pipeline.  Tree-based GBDT
models (LightGBM / XGBoost / CatBoost) are invariant to monotonic feature
transformations, so scaling neither hurts nor helps them — but fitting a
scaler on the full training set before CV splits leaks validation-fold
statistics into training data, inflating OOF performance estimates.

NOTE ON REFERENCE DATE
-----------------------
All datetime features that measure elapsed time (e.g. dt_days_since_review)
are computed relative to REFERENCE_DATE, a module-level constant set to
2026-01-01.  Using a fixed date guarantees that re-running the pipeline
(or looping through multiple feedback iterations) always produces identical
feature values, which is essential for reproducible CV scores and for
isolating the effect of other changes between iterations.
"""

import gc
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from agents.base import BaseAgent
from utils.helpers import safe_json_parse

# Fixed reference date for all elapsed-time features.
# Must never be pd.Timestamp.now() — that makes features non-deterministic
# across re-runs and across feedback loop iterations.
REFERENCE_DATE: pd.Timestamp = pd.Timestamp("2026-01-01")


def _compute_target_encoding_map(
    col: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float = 10.0,
) -> Dict[Any, float]:
    """
    Compute a smoothed target-mean encoding map from a *single fold's*
    training data.  Must be called inside each CV fold by ModelAgent.

    Parameters
    ----------
    col:
        Categorical column values (training fold only).
    target:
        Corresponding target values (training fold only).
    global_mean:
        Global mean of the target computed on the *full* training set
        (used as the smoothing prior — acceptable to compute once).
    smoothing:
        Bayesian smoothing weight; higher = stronger shrinkage towards mean.

    Returns
    -------
    dict mapping category value → smoothed mean.
    """
    stats = (
        pd.DataFrame({"col": col.values, "target": target.values})
        .groupby("col")["target"]
        .agg(["count", "mean"])
    )
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )
    return stats["smoothed"].to_dict()


class FeatureAgent(BaseAgent):
    """
    Performs feature engineering guided by an LLM plan.

    Expected state keys consumed:
        train_df     (pd.DataFrame): cleaned training data (with target).
        test_df      (pd.DataFrame): cleaned test data.
        data_profile (dict):         profile produced by DataAgent.

    State keys produced:
        train_feat   (pd.DataFrame): engineered feature matrix (train).
        test_feat    (pd.DataFrame): engineered feature matrix (test).
        feature_names (list[str]):   ordered list of feature column names.
        feature_plan  (dict):        LLM-generated plan for reproducibility.
        target_series (pd.Series):  target column extracted from train.
        test_ids      (pd.Series):  _id column from test.
    """

    SYSTEM_PROMPT = (
        "You are a feature-engineering expert for rental-property price prediction "
        "(MSE metric on Kaggle). Recommend practical, high-impact feature groups "
        "and return a strict JSON plan."
    )

    TARGET_COL = "target"
    ID_COL = "_id"

    # Hard guardrail: never produce more features than this
    MAX_FEATURES: int = 500

    # Column roles (used to build prompt context)
    COLUMN_ROLES = {
        "name": "text — listing title",
        "host_name": "categorical — host identifier",
        "location_cluster": "categorical — pre-clustered location group",
        "location": "text — freeform location description",
        "lat": "numeric — latitude",
        "lon": "numeric — longitude",
        "type_house": "categorical — property type",
        "sum": "numeric — price per night (could be raw price)",
        "min_days": "numeric — minimum stay in days",
        "amt_reviews": "numeric — number of reviews",
        "last_dt": "datetime — date of last review",
        "avg_reviews": "numeric — average review score",
        "total_host": "numeric — total listings by this host",
        "target": "numeric — rental price to predict",
    }

    # Categorical columns that target encoding operates on;
    # encoding itself is deferred to ModelAgent (inside each CV fold).
    TARGET_ENCODE_COLS = ("host_name", "location_cluster", "type_house")

    # ------------------------------------------------------------------
    # LLM plan request
    # ------------------------------------------------------------------

    def _request_feature_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask LLM which feature engineering groups to enable and with what params.
        """
        profile = state.get("data_profile", {})
        train_shape = profile.get("train", {}).get("shape", ["?", "?"])

        prompt = f"""
You are engineering features for a Kaggle rental-property regression task.
Target: 'target' (rental price, numeric). Metric: MSE.
Train shape after cleaning: {train_shape}.

## Column descriptions
{json.dumps(self.COLUMN_ROLES, indent=2)}

## Available feature groups
1. datetime_features   — extract year/month/day_of_week/is_weekend/days_since_last_review from 'last_dt'
2. geo_features        — KMeans geo-clusters from lat/lon; haversine distance to city centroid
3. text_features       — TF-IDF + SVD on 'name' and 'location' columns (low-dimensional)
4. target_encoding     — smoothed target-mean encoding (applied INSIDE each CV fold by ModelAgent
                         to prevent leakage; enabling this flag just reserves the column set)
5. frequency_encoding  — count/frequency encoding for high-cardinality categoricals
6. interaction_features — pairwise products of key numeric pairs
7. log_transforms      — log1p of right-skewed numeric columns
8. polynomial_features — degree-2 polynomial of top numeric features (careful with memory)
9. label_encoding      — ordinal label encoding for categoricals

## Instructions
Return a JSON plan (no markdown, strict JSON). Enable only the groups that are likely
to improve MSE on this dataset. Keep total feature count under {self.MAX_FEATURES}.

Schema:
{{
  "groups": {{
    "datetime_features":    {{"enabled": true}},
    "geo_features":         {{"enabled": true, "n_clusters": 8}},
    "text_features":        {{"enabled": true, "n_components": 5, "max_features": 300}},
    "target_encoding":      {{"enabled": true, "smoothing": 10}},
    "frequency_encoding":   {{"enabled": true}},
    "interaction_features": {{"enabled": true, "pairs": [["sum", "min_days"], ["amt_reviews", "avg_reviews"]]}},
    "log_transforms":       {{"enabled": true, "columns": ["sum", "min_days", "amt_reviews", "total_host"]}},
    "polynomial_features":  {{"enabled": false}},
    "label_encoding":       {{"enabled": true, "columns": ["type_house", "location_cluster"]}}
  }},
  "drop_low_importance": true
}}
Respond with JSON only.
"""
        default: Dict[str, Any] = {
            "groups": {
                "datetime_features":    {"enabled": True},
                "geo_features":         {"enabled": True, "n_clusters": 8},
                "text_features":        {"enabled": True, "n_components": 5, "max_features": 300},
                "target_encoding":      {"enabled": True, "smoothing": 10},
                "frequency_encoding":   {"enabled": True},
                "interaction_features": {
                    "enabled": True,
                    "pairs": [["sum", "min_days"], ["amt_reviews", "avg_reviews"]],
                },
                "log_transforms":       {"enabled": True, "columns": ["sum", "min_days", "amt_reviews", "total_host"]},
                "polynomial_features":  {"enabled": False},
                "label_encoding":       {"enabled": True, "columns": ["type_house", "location_cluster"]},
            },
            "drop_low_importance": True,
        }
        return self._ask_llm_json(prompt, default=default)

    # ------------------------------------------------------------------
    # Feature-group implementations (all deterministic)
    # ------------------------------------------------------------------

    def _add_datetime_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract temporal signals from 'last_dt'.

        All elapsed-time features are computed relative to the module-level
        REFERENCE_DATE constant (2026-01-01) rather than pd.Timestamp.now().
        This guarantees identical feature values across pipeline re-runs and
        across feedback-loop iterations, which is essential for reproducible
        CV scores.
        """
        col = "last_dt"

        for df in (train, test):
            if col not in df.columns:
                continue
            dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
            df["dt_year"] = dt.dt.year.fillna(0).astype(int)
            df["dt_month"] = dt.dt.month.fillna(0).astype(int)
            df["dt_day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
            df["dt_is_weekend"] = (df["dt_day_of_week"] >= 5).astype(int)
            df["dt_quarter"] = dt.dt.quarter.fillna(0).astype(int)
            df["dt_days_since_review"] = (REFERENCE_DATE - dt).dt.days.fillna(-1).astype(int)
            df.drop(columns=[col], inplace=True, errors="ignore")

        return train, test

    def _add_geo_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        n_clusters: int = 8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """KMeans geo-clusters + haversine distance to centroid."""
        lat_col, lon_col = "lat", "lon"
        if lat_col not in train.columns or lon_col not in train.columns:
            self._log("lat/lon columns missing; skipping geo features.", level="warning")
            return train, test

        train_coords = train[[lat_col, lon_col]].fillna(0).values
        test_coords = test[[lat_col, lon_col]].fillna(0).values

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(train_coords)

        train["geo_cluster"] = kmeans.predict(train_coords)
        test["geo_cluster"] = kmeans.predict(test_coords)

        centers = kmeans.cluster_centers_
        city_lat = float(np.mean(train[lat_col].dropna()))
        city_lon = float(np.mean(train[lon_col].dropna()))

        def haversine(lats: np.ndarray, lons: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
            R = 6371.0
            dlat = np.radians(lats - ref_lat)
            dlon = np.radians(lons - ref_lon)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lats)) * np.cos(np.radians(ref_lat)) * np.sin(dlon / 2) ** 2
            return R * 2 * np.arcsin(np.sqrt(a))

        train["geo_dist_center"] = haversine(
            train[lat_col].fillna(city_lat).values,
            train[lon_col].fillna(city_lon).values,
            city_lat, city_lon,
        )
        test["geo_dist_center"] = haversine(
            test[lat_col].fillna(city_lat).values,
            test[lon_col].fillna(city_lon).values,
            city_lat, city_lon,
        )

        return train, test

    def _add_text_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        n_components: int = 5,
        max_features: int = 300,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """TF-IDF + TruncatedSVD for text columns."""
        text_cols = [c for c in ("name", "location") if c in train.columns]

        for col in text_cols:
            train_text = train[col].fillna("").astype(str).tolist()
            test_text = test[col].fillna("").astype(str).tolist()

            tfidf = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )
            svd = TruncatedSVD(n_components=n_components, random_state=42)

            train_tfidf = tfidf.fit_transform(train_text)
            test_tfidf = tfidf.transform(test_text)

            train_svd = svd.fit_transform(train_tfidf)
            test_svd = svd.transform(test_tfidf)

            for i in range(n_components):
                feat_name = f"text_{col}_svd_{i}"
                train[feat_name] = train_svd[:, i]
                test[feat_name] = test_svd[:, i]

            # Drop original text column (not useful as raw text for trees)
            train.drop(columns=[col], inplace=True, errors="ignore")
            test.drop(columns=[col], inplace=True, errors="ignore")

        return train, test

    def _add_frequency_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Count/frequency encoding for high-cardinality categoricals.

        NOTE ON LEAKAGE: frequency maps are computed on the full training set
        (before CV splits).  This is a minor pre-CV leak — the test-fold rows
        contribute their category counts to the map used during training.
        The leakage is smaller than target encoding because frequencies do not
        directly encode the target value, but it is non-zero when popular
        categories correlate with price.  See module docstring for details on
        a fully leak-free alternative.
        """
        # Operate on a snapshot of column names so we don't miss cols that
        # other encoding steps may have already dropped.
        cat_cols = [c for c in self.TARGET_ENCODE_COLS if c in train.columns]

        for col in cat_cols:
            freq_map = train[col].value_counts(normalize=True).to_dict()
            feat = f"freq_{col}"
            train[feat] = train[col].map(freq_map).fillna(0)
            test[feat] = test[col].map(freq_map).fillna(0)

        return train, test

    def _add_interaction_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        pairs: Optional[List[List[str]]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pairwise product interaction features."""
        if pairs is None:
            pairs = [["sum", "min_days"], ["amt_reviews", "avg_reviews"]]

        for (col_a, col_b) in pairs:
            if col_a in train.columns and col_b in train.columns:
                feat = f"inter_{col_a}_{col_b}"
                train[feat] = train[col_a] * train[col_b]
                test[feat] = test[col_a] * test[col_b]

        return train, test

    def _add_log_transforms(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """log1p transform for right-skewed columns; original column is kept
        so models can choose which representation is more useful."""
        if columns is None:
            columns = ["sum", "min_days", "amt_reviews", "total_host"]

        for col in columns:
            if col in train.columns:
                log_feat = f"log_{col}"
                train[log_feat] = np.log1p(train[col].clip(lower=0))
                test[log_feat] = np.log1p(test[col].clip(lower=0))

        return train, test

    def _add_label_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ordinal label encoding (fit on combined to avoid unseen categories)."""
        if columns is None:
            columns = ["type_house", "location_cluster"]

        for col in columns:
            if col not in train.columns:
                continue
            le = LabelEncoder()
            combined = pd.concat(
                [train[col].astype(str), test[col].astype(str)],
                ignore_index=True,
            )
            le.fit(combined)
            train[f"le_{col}"] = le.transform(train[col].astype(str))
            test[f"le_{col}"] = le.transform(test[col].astype(str))

        return train, test
