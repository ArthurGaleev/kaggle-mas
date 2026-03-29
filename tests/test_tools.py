"""
Unit tests for DataAgent and FeatureAgent tool methods.

These tests exercise the deterministic helper functions (no LLM calls required).
They use small synthetic DataFrames and call the agent methods directly,
bypassing the full pipeline.

Run with:
    pytest tests/test_tools.py -v
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers for constructing minimal agent instances
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg(**overrides) -> Dict[str, Any]:
    """Return a minimal config dict suitable for both agents."""
    base = {
        "target_col": "target",
        "id_col": "_id",
        "max_rows": 1_000_000,
        "max_cols": 500,
        "max_iterations": 3,
        "n_folds": 3,
    }
    base.update(overrides)
    return base


def _make_feature_agent():
    """Build a FeatureAgent with a mocked LLM client."""
    from agents.feature_agent import FeatureAgent
    cfg = _make_cfg()
    mock_llm = MagicMock()
    agent = FeatureAgent(cfg=cfg, llm_client=mock_llm)
    return agent


def _make_data_agent():
    """Build a DataAgent with a mocked LLM client."""
    from agents.data_agent import DataAgent
    cfg = _make_cfg()
    mock_llm = MagicMock()
    agent = DataAgent(cfg=cfg, llm_client=mock_llm)
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_rental_df(n: int = 80, include_target: bool = True) -> pd.DataFrame:
    """Return a small rental-property DataFrame that mimics the competition data."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "_id": range(n),
            "name": [f"Listing {i}" for i in range(n)],
            "host_name": rng.choice(["Alice", "Bob", "Carol", "Dave"], n),
            "location_cluster": rng.choice(["North", "South", "East", "West"], n),
            "location": [f"Street {i}, Moscow" for i in range(n)],
            "lat": rng.uniform(55.5, 56.0, n),
            "lon": rng.uniform(37.3, 37.9, n),
            "type_house": rng.choice(["Apartment", "House", "Room"], n),
            "sum": rng.uniform(1_000, 10_000, n),
            "min_days": rng.integers(1, 30, n).astype(float),
            "amt_reviews": rng.integers(0, 500, n).astype(float),
            "last_dt": pd.date_range("2022-01-01", periods=n, freq="3D"),
            "avg_reviews": rng.uniform(1.0, 5.0, n),
            "total_host": rng.integers(1, 50, n).astype(float),
        }
    )
    if include_target:
        df["target"] = rng.uniform(500, 5_000, n)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# test_compute_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeProfile:
    """Tests for DataAgent._compute_profile."""

    def test_compute_profile_shape(self):
        """Profile must report correct shape for train and test splits."""
        agent = _make_data_agent()
        train = _make_rental_df(60)
        test = _make_rental_df(30, include_target=False)
        profile = agent._compute_profile(train, test)

        assert profile["train"]["shape"] == [60, len(train.columns)]
        assert profile["test"]["shape"] == [30, len(test.columns)]

    def test_compute_profile_keys(self):
        """Profile dict must contain expected top-level keys."""
        agent = _make_data_agent()
        train = _make_rental_df(40)
        test = _make_rental_df(20, include_target=False)
        profile = agent._compute_profile(train, test)

        for key in ("train", "test", "train_duplicate_rows", "test_duplicate_rows"):
            assert key in profile, f"Missing profile key: {key!r}"
        # Column overlap info is stored as train_only_columns / test_only_columns
        assert ("train_only_columns" in profile or "test_only_columns" in profile or
                "column_overlap" in profile), (
            f"Expected column overlap key in profile. Got: {list(profile.keys())}"
        )

    def test_compute_profile_column_stats(self):
        """Numeric columns should have mean/std/min/max statistics."""
        agent = _make_data_agent()
        train = _make_rental_df(50)
        test = _make_rental_df(25, include_target=False)
        profile = agent._compute_profile(train, test)

        lat_info = profile["train"]["columns"].get("lat", {})
        for stat in ("mean", "std", "min", "max"):
            assert stat in lat_info, f"Numeric stat '{stat}' missing from lat column profile"

    def test_compute_profile_null_tracking(self):
        """Profile must report null counts accurately."""
        agent = _make_data_agent()
        train = _make_rental_df(40)
        train.loc[:9, "avg_reviews"] = np.nan  # introduce 10 nulls
        test = _make_rental_df(20, include_target=False)
        profile = agent._compute_profile(train, test)

        null_pct = profile["train"]["columns"]["avg_reviews"]["null_pct"]
        assert null_pct == pytest.approx(25.0, abs=1.0)  # 10/40 = 25 %


# ─────────────────────────────────────────────────────────────────────────────
# test_apply_cleaning_plan
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyCleaningPlan:
    """Tests for DataAgent._execute_cleaning_plan."""

    def test_apply_cleaning_drops_columns(self):
        """Columns listed in drop_columns must be removed."""
        agent = _make_data_agent()
        train = _make_rental_df(30)
        test = _make_rental_df(15, include_target=False)
        plan = {
            "drop_columns": ["host_name", "name"],
            "imputation": {},
            "outlier_handling": {},
            "type_conversions": {},
        }
        agent._execute_cleaning_plan(train, test, plan)
        assert "host_name" not in train.columns
        assert "name" not in train.columns
        assert "host_name" not in test.columns

    def test_apply_cleaning_never_drops_id_or_target(self):
        """_id and target columns must not be dropped even if listed."""
        agent = _make_data_agent()
        train = _make_rental_df(30)
        test = _make_rental_df(15, include_target=False)
        plan = {
            "drop_columns": ["_id", "target"],  # forbidden
            "imputation": {},
            "outlier_handling": {},
            "type_conversions": {},
        }
        agent._execute_cleaning_plan(train, test, plan)
        assert "_id" in train.columns
        assert "target" in train.columns

    def test_apply_cleaning_median_imputation(self):
        """Median imputation must fill all NaN values in the column."""
        agent = _make_data_agent()
        train = _make_rental_df(40)
        test = _make_rental_df(20, include_target=False)
        # Introduce nulls
        train.loc[:4, "sum"] = np.nan
        test.loc[:2, "sum"] = np.nan
        plan = {
            "drop_columns": [],
            "imputation": {"sum": "median"},
            "outlier_handling": {},
            "type_conversions": {},
        }
        agent._execute_cleaning_plan(train, test, plan)
        assert train["sum"].isna().sum() == 0
        assert test["sum"].isna().sum() == 0

    def test_apply_cleaning_clip_outliers(self):
        """Outlier clipping must keep values within [lower, upper]."""
        agent = _make_data_agent()
        train = _make_rental_df(50)
        test = _make_rental_df(25, include_target=False)
        train["sum"] = np.where(np.arange(50) < 5, 999_999, train["sum"])
        plan = {
            "drop_columns": [],
            "imputation": {},
            "outlier_handling": {"sum": {"method": "clip", "lower": 0, "upper": 15_000}},
            "type_conversions": {},
        }
        agent._execute_cleaning_plan(train, test, plan)
        assert train["sum"].max() <= 15_000


# ─────────────────────────────────────────────────────────────────────────────
# test_datetime_features
# ─────────────────────────────────────────────────────────────────────────────

class TestDatetimeFeatures:
    """Tests for FeatureAgent._add_datetime_features."""

    def test_datetime_features_adds_columns(self):
        """Datetime feature extraction must add the expected columns."""
        agent = _make_feature_agent()
        train = _make_rental_df(40)
        test = _make_rental_df(20, include_target=False)

        # last_dt is a datetime column; convert to string to simulate CSV load
        train["last_dt"] = train["last_dt"].astype(str)
        test["last_dt"] = test["last_dt"].astype(str)

        train_out, test_out = agent._add_datetime_features(train.copy(), test.copy())

        expected_cols = [
            "dt_year", "dt_month", "dt_day_of_week",
            "dt_is_weekend", "dt_quarter", "dt_days_since_review",
        ]
        for col in expected_cols:
            assert col in train_out.columns, f"Expected column '{col}' not found in train"
            assert col in test_out.columns, f"Expected column '{col}' not found in test"

    def test_datetime_features_drops_raw_column(self):
        """The raw 'last_dt' column must be removed after feature extraction."""
        agent = _make_feature_agent()
        train = _make_rental_df(30)
        test = _make_rental_df(15, include_target=False)
        train["last_dt"] = train["last_dt"].astype(str)
        test["last_dt"] = test["last_dt"].astype(str)

        train_out, test_out = agent._add_datetime_features(train.copy(), test.copy())
        assert "last_dt" not in train_out.columns
        assert "last_dt" not in test_out.columns

    def test_datetime_features_no_column_no_error(self):
        """If 'last_dt' is absent, the method must return DataFrames unchanged."""
        agent = _make_feature_agent()
        train = _make_rental_df(20).drop(columns=["last_dt"])
        test = _make_rental_df(10, include_target=False).drop(columns=["last_dt"])
        original_cols = set(train.columns)

        train_out, test_out = agent._add_datetime_features(train.copy(), test.copy())
        assert set(train_out.columns) == original_cols

    def test_datetime_features_is_weekend_binary(self):
        """dt_is_weekend must be strictly 0 or 1."""
        agent = _make_feature_agent()
        train = _make_rental_df(50)
        test = _make_rental_df(25, include_target=False)
        train["last_dt"] = train["last_dt"].astype(str)
        test["last_dt"] = test["last_dt"].astype(str)

        train_out, _ = agent._add_datetime_features(train.copy(), test.copy())
        assert set(train_out["dt_is_weekend"].unique()).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# test_frequency_encode
# ─────────────────────────────────────────────────────────────────────────────

class TestFrequencyEncode:
    """Tests for FeatureAgent._add_frequency_encoding."""

    def test_frequency_encode_creates_columns(self):
        """Frequency encoding must produce freq_* columns for categoricals."""
        agent = _make_feature_agent()
        train = _make_rental_df(60)
        test = _make_rental_df(30, include_target=False)

        train_out, test_out = agent._add_frequency_encoding(train.copy(), test.copy())

        for col in ("host_name", "location_cluster", "type_house"):
            assert f"freq_{col}" in train_out.columns, f"Missing freq_{col} in train"
            assert f"freq_{col}" in test_out.columns, f"Missing freq_{col} in test"

    def test_frequency_encode_values_between_0_and_1(self):
        """Frequency values must be probabilities in [0, 1]."""
        agent = _make_feature_agent()
        train = _make_rental_df(60)
        test = _make_rental_df(30, include_target=False)

        train_out, test_out = agent._add_frequency_encoding(train.copy(), test.copy())

        for col in ("host_name", "location_cluster", "type_house"):
            feat = f"freq_{col}"
            assert train_out[feat].min() >= 0.0
            assert train_out[feat].max() <= 1.0

    def test_frequency_encode_unseen_categories_zero(self):
        """Unseen categories in test must map to 0 (not NaN)."""
        agent = _make_feature_agent()
        train = _make_rental_df(40)
        test = _make_rental_df(20, include_target=False)
        # Introduce an unseen category in test
        test.loc[0, "type_house"] = "Mansion"

        _, test_out = agent._add_frequency_encoding(train.copy(), test.copy())
        assert test_out["freq_type_house"].isna().sum() == 0

    def test_frequency_encode_sums_to_one(self):
        """Frequency mapping for a column must sum to approximately 1.0."""
        agent = _make_feature_agent()
        train = _make_rental_df(80)
        test = _make_rental_df(40, include_target=False)

        train_out, _ = agent._add_frequency_encoding(train.copy(), test.copy())
        # For each unique category, its frequency sums to 1 across all rows
        freq_sum = train_out.groupby(train["type_house"])["freq_type_house"].first().sum()
        assert abs(freq_sum - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# test_target_encode
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetEncode:
    """Tests for FeatureAgent._add_target_encoding."""

    def test_target_encode_creates_columns(self):
        """Target encoding must produce te_* columns for categoricals."""
        agent = _make_feature_agent()
        train = _make_rental_df(60)
        test = _make_rental_df(30, include_target=False)
        target = train.pop("target")

        train_out, test_out = agent._add_target_encoding(
            train.copy(), test.copy(), target
        )

        for col in ("host_name", "location_cluster", "type_house"):
            assert f"te_{col}" in train_out.columns, f"Missing te_{col} in train"
            assert f"te_{col}" in test_out.columns, f"Missing te_{col} in test"

    def test_target_encode_no_nans(self):
        """Target encoded columns must not contain NaN values."""
        agent = _make_feature_agent()
        train = _make_rental_df(80)
        test = _make_rental_df(40, include_target=False)
        target = train.pop("target")
        # Add unseen test category
        test.loc[0, "type_house"] = "Treehouse"

        _, test_out = agent._add_target_encoding(train.copy(), test.copy(), target)
        for col in ("host_name", "location_cluster", "type_house"):
            assert test_out[f"te_{col}"].isna().sum() == 0, f"NaN in te_{col}"

    def test_target_encode_smoothing_effect(self):
        """Categories with very few samples should be pulled toward the global mean."""
        agent = _make_feature_agent()
        rng = np.random.default_rng(1)
        n = 100
        train = pd.DataFrame(
            {
                "type_house": ["Rare"] + ["Common"] * (n - 1),
                "host_name": ["A"] * n,
                "location_cluster": ["Z"] * n,
            }
        )
        target = pd.Series(rng.uniform(1_000, 5_000, n))
        test = train.copy()

        global_mean = float(target.mean())
        rare_true_mean = float(target.iloc[0])

        train_out, _ = agent._add_target_encoding(
            train.copy(), test.copy(), target, smoothing=50.0
        )

        rare_encoded = float(
            train_out.loc[train["type_house"] == "Rare", "te_type_house"].iloc[0]
        )
        # High smoothing pulls encoded value toward global mean
        assert abs(rare_encoded - global_mean) < abs(rare_true_mean - global_mean) + 1e-6

    def test_target_encode_numeric_output(self):
        """All te_* columns must be numeric (float)."""
        agent = _make_feature_agent()
        train = _make_rental_df(60)
        test = _make_rental_df(30, include_target=False)
        target = train.pop("target")

        train_out, test_out = agent._add_target_encoding(
            train.copy(), test.copy(), target
        )

        for col in ("host_name", "location_cluster", "type_house"):
            feat = f"te_{col}"
            assert pd.api.types.is_numeric_dtype(train_out[feat]), (
                f"{feat} is not numeric in train"
            )
            assert pd.api.types.is_numeric_dtype(test_out[feat]), (
                f"{feat} is not numeric in test"
            )
