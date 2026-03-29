"""
FeatureAgent — asks the LLM to plan feature engineering, then executes the
plan deterministically using pandas / sklearn / scipy.

All heavy computation (KMeans, TF-IDF, SVD, scaling …) is done in Python;
the LLM only decides which feature groups to activate and any non-trivial
parameter choices.
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

from agents.base import BaseAgent
from utils.helpers import safe_json_parse


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

IMPORTANT: The target variable is right-skewed. The ModelAgent will apply
log1p to the target before training and expm1 after prediction — this is
already handled. Your job is to maximize feature quality for the model.

## Column descriptions
{json.dumps(self.COLUMN_ROLES, indent=2)}

## Available feature groups
1. datetime_features   — extract year/month/day_of_week/is_weekend/days_since_last_review from 'last_dt'
2. geo_features        — KMeans geo-clusters from lat/lon (use 30 clusters); haversine distance to city centroid
3. text_features       — TF-IDF + SVD on 'name' and 'location' columns (use 20 SVD components, 1000 max vocab)
4. target_encoding     — target-mean encoding for 'host_name', 'location_cluster', 'type_house', 'geo_cluster'
5. frequency_encoding  — count/frequency encoding for high-cardinality categoricals
6. interaction_features — pairwise products of key numeric pairs
7. log_transforms      — log1p of right-skewed numeric columns (do NOT log the target — ModelAgent handles that)
8. polynomial_features — degree-2 polynomial of top numeric features (careful with memory)
9. label_encoding      — ordinal label encoding for categoricals

## Instructions
Return a JSON plan (no markdown, strict JSON). Enable only the groups that are likely
to improve MSE on this dataset. Keep total feature count under {self.MAX_FEATURES}.
Use 30 geo clusters and 20 SVD text components for richer representations.

Schema:
{{
  "groups": {{
    "datetime_features":    {{"enabled": true}},
    "geo_features":         {{"enabled": true, "n_clusters": 30}},
    "text_features":        {{"enabled": true, "n_components": 20, "max_features": 1000}},
    "target_encoding":      {{"enabled": true, "smoothing": 10}},
    "frequency_encoding":   {{"enabled": true}},
    "interaction_features": {{"enabled": true, "pairs": [["sum", "min_days"], ["amt_reviews", "avg_reviews"], ["total_host", "amt_reviews"]]}},
    "log_transforms":       {{"enabled": true, "columns": ["sum", "min_days", "amt_reviews", "total_host", "avg_reviews"]}},
    "polynomial_features":  {{"enabled": false}},
    "label_encoding":       {{"enabled": true, "columns": ["type_house", "location_cluster"]}}
  }},
  "drop_low_importance": true,
  "scale_features": true
}}
Respond with JSON only.
"""
        default: Dict[str, Any] = {
            "groups": {
                "datetime_features":    {"enabled": True},
                "geo_features":         {"enabled": True, "n_clusters": 30},
                "text_features":        {"enabled": True, "n_components": 20, "max_features": 1000},
                "target_encoding":      {"enabled": True, "smoothing": 10},
                "frequency_encoding":   {"enabled": True},
                "interaction_features": {
                    "enabled": True,
                    "pairs": [
                        ["sum", "min_days"],
                        ["amt_reviews", "avg_reviews"],
                        ["total_host", "amt_reviews"],
                    ],
                },
                "log_transforms":       {"enabled": True, "columns": ["sum", "min_days", "amt_reviews", "total_host", "avg_reviews"]},
                "polynomial_features":  {"enabled": False},
                "label_encoding":       {"enabled": True, "columns": ["type_house", "location_cluster"]},
            },
            "drop_low_importance": True,
            "scale_features": True,
        }
        return self._ask_llm_json(prompt, default=default)

    # ------------------------------------------------------------------
    # Feature-group implementations (all deterministic)
    # ------------------------------------------------------------------

    def _add_datetime_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract temporal signals from 'last_dt'."""
        col = "last_dt"
        reference_date = pd.Timestamp.now()

        for df in (train, test):
            if col not in df.columns:
                continue
            dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
            df["dt_year"] = dt.dt.year.fillna(0).astype(int)
            df["dt_month"] = dt.dt.month.fillna(0).astype(int)
            df["dt_day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
            df["dt_is_weekend"] = (df["dt_day_of_week"] >= 5).astype(int)
            df["dt_quarter"] = dt.dt.quarter.fillna(0).astype(int)
            df["dt_days_since_review"] = (reference_date - dt).dt.days.fillna(-1).astype(int)
            df.drop(columns=[col], inplace=True, errors="ignore")

        return train, test

    def _add_geo_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        n_clusters: int = 30,
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
        n_components: int = 20,
        max_features: int = 1000,
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

    def _add_target_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: pd.Series,
        smoothing: float = 10.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Smoothed target-mean encoding for categorical columns.

        Now includes 'geo_cluster' so neighbourhood-level price signal is
        directly available as a feature alongside the KMeans cluster label.
        """
        cat_cols = [
            c for c in ("host_name", "location_cluster", "type_house", "geo_cluster")
            if c in train.columns
        ]
        global_mean = float(target.mean())

        for col in cat_cols:
            stats = (
                pd.DataFrame({col: train[col], "target": target.values})
                .groupby(col)["target"]
                .agg(["count", "mean"])
            )
            stats["smoothed"] = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            mapping = stats["smoothed"].to_dict()

            feat = f"te_{col}"
            train[feat] = train[col].map(mapping).fillna(global_mean)
            test[feat] = test[col].map(mapping).fillna(global_mean)

        # Drop original categoricals that have been encoded
        # Note: geo_cluster is kept (integer cluster id is also useful for trees)
        drop_cats = [c for c in ("host_name", "location_cluster", "type_house") if c in train.columns]
        train.drop(columns=drop_cats, inplace=True)
        test.drop(columns=[c for c in drop_cats if c in test.columns], inplace=True)

        return train, test

    def _add_frequency_encoding(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Count/frequency encoding for high-cardinality categoricals."""
        cat_cols = [c for c in ("host_name", "location_cluster", "type_house") if c in train.columns]

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
            pairs = [
                ["sum", "min_days"],
                ["amt_reviews", "avg_reviews"],
                ["total_host", "amt_reviews"],
            ]

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
        """log1p transform for right-skewed columns.

        NOTE: Do NOT include the target here — ModelAgent applies log1p to the
        target internally before training and expm1 after prediction.
        """
        if columns is None:
            columns = ["sum", "min_days", "amt_reviews", "total_host", "avg_reviews"]

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
                [train[col].astype(str), test[col].astype(str)], ignore_index=True
            )
            le.fit(combined)
            train[f"le_{col}"] = le.transform(train[col].astype(str))
            test[f"le_{col}"] = le.transform(test[col].astype(str))

        return train, test

    def _apply_scaling(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        exclude: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Standard-scale all numeric features (fit on train only)."""
        if exclude is None:
            exclude = []
        num_cols = [
            c for c in train.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]
        scaler = StandardScaler()
        train[num_cols] = scaler.fit_transform(train[num_cols])
        test_num_cols = [c for c in num_cols if c in test.columns]
        test[test_num_cols] = scaler.transform(test[test_num_cols])
        return train, test, scaler

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the LLM for a feature plan, execute it, validate, and store results.
        """
        # --- Pull DataFrames from state ---
        train: pd.DataFrame = state["train_df"].copy()
        test: pd.DataFrame = state["test_df"].copy()

        # Separate target and IDs before feature engineering
        target = train.pop(self.TARGET_COL)
        test_ids = test[self.ID_COL].copy()

        # Drop ID from features (not predictive)
        train.drop(columns=[self.ID_COL], inplace=True, errors="ignore")
        test.drop(columns=[self.ID_COL], inplace=True, errors="ignore")

        # --- Get LLM plan ---
        plan = self._request_feature_plan(state)
        state["feature_plan"] = plan
        groups = plan.get("groups", {})
        self._log(f"Feature plan enabled groups: {[g for g, cfg in groups.items() if cfg.get('enabled')]}")

        # --- Execute each feature group ---
        if groups.get("datetime_features", {}).get("enabled", True):
            train, test = self._add_datetime_features(train, test)

        if groups.get("geo_features", {}).get("enabled", True):
            n_clusters = groups["geo_features"].get("n_clusters", 30)
            train, test = self._add_geo_features(train, test, n_clusters=n_clusters)

        # frequency_encoding must run BEFORE target_encoding so that original
        # categorical columns are still present when frequency maps are built.
        if groups.get("frequency_encoding", {}).get("enabled", True):
            train, test = self._add_frequency_encoding(train, test)

        if groups.get("text_features", {}).get("enabled", True):
            n_comp = groups.get("text_features", {}).get("n_components", 20)
            max_feat = groups.get("text_features", {}).get("max_features", 1000)
            train, test = self._add_text_features(train, test, n_components=n_comp, max_features=max_feat)

        if groups.get("target_encoding", {}).get("enabled", True):
            smoothing = groups.get("target_encoding", {}).get("smoothing", 10.0)
            train, test = self._add_target_encoding(train, test, target, smoothing=smoothing)

        if groups.get("interaction_features", {}).get("enabled", True):
            pairs = groups.get("interaction_features", {}).get("pairs")
            train, test = self._add_interaction_features(train, test, pairs=pairs)

        if groups.get("log_transforms", {}).get("enabled", True):
            log_cols = groups.get("log_transforms", {}).get("columns")
            train, test = self._add_log_transforms(train, test, columns=log_cols)

        if groups.get("label_encoding", {}).get("enabled", True):
            le_cols = groups.get("label_encoding", {}).get("columns")
            train, test = self._add_label_encoding(train, test, columns=le_cols)

        # --- Drop any remaining object columns (not encoded yet) ---
        obj_cols_train = train.select_dtypes(include="object").columns.tolist()
        obj_cols_test = test.select_dtypes(include="object").columns.tolist()
        drop_obj = list(set(obj_cols_train + obj_cols_test))
        if drop_obj:
            self._log(f"Dropping remaining object columns: {drop_obj}", level="warning")
            train.drop(columns=[c for c in drop_obj if c in train.columns], inplace=True)
            test.drop(columns=[c for c in drop_obj if c in test.columns], inplace=True)

        # Also drop datetime-type columns if any remain
        dt_cols = train.select_dtypes(include=["datetime64"]).columns.tolist()
        if dt_cols:
            train.drop(columns=dt_cols, inplace=True, errors="ignore")
            test.drop(columns=[c for c in dt_cols if c in test.columns], inplace=True, errors="ignore")

        # --- Align train/test columns ---
        common_cols = [c for c in train.columns if c in test.columns]
        train_only = [c for c in train.columns if c not in test.columns]
        if train_only:
            self._log(f"Columns in train but not test (dropping): {train_only}", level="warning")
        train = train[common_cols]
        test = test[common_cols]

        # --- Validate guardrail ---
        if train.shape[1] > self.MAX_FEATURES:
            self._log(
                f"Feature count {train.shape[1]} exceeds guardrail {self.MAX_FEATURES}. "
                "Truncating to top columns by variance.",
                level="warning",
            )
            variances = train.var().sort_values(ascending=False)
            keep_cols = variances.index[: self.MAX_FEATURES].tolist()
            train = train[keep_cols]
            test = test[keep_cols]

        # --- Optional scaling ---
        if plan.get("scale_features", True):
            train, test, _ = self._apply_scaling(train, test)

        feature_names: List[str] = train.columns.tolist()
        self._log(f"Final feature count: {len(feature_names)}")

        # --- Store in state ---
        state["train_feat"] = train
        state["test_feat"] = test
        state["target_series"] = target
        state["test_ids"] = test_ids
        state["feature_names"] = feature_names

        gc.collect()
        return state
