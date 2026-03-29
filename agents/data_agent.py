"""
DataAgent — responsible for loading, profiling, and cleaning the raw datasets.

Phase 1 (profile_data): loads CSVs, computes a comprehensive data profile,
    and stores it in state without any LLM involvement.
Phase 2 (clean_data): asks the LLM to reason over the profile and produce a
    JSON cleaning plan, then executes that plan deterministically.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from agents.base import BaseAgent
from utils.helpers import safe_json_parse


class DataAgent(BaseAgent):
    """
    Loads, profiles, and cleans the rental-property regression datasets.

    Expected state keys consumed:
        data_dir (str): directory containing train.csv and test.csv.

    State keys produced:
        train_df       (pd.DataFrame): cleaned training data.
        test_df        (pd.DataFrame): cleaned test data.
        data_profile   (dict):         comprehensive profile of raw data.
        cleaning_plan  (dict):         LLM-generated cleaning plan.
    """

    SYSTEM_PROMPT = (
        "You are a senior data scientist specialising in rental-property price "
        "modelling. You analyse data profiles and produce concise, actionable "
        "data-cleaning plans in strict JSON format."
    )

    # Columns we know about from the competition spec
    KNOWN_COLUMNS = [
        "name", "_id", "host_name", "location_cluster", "location",
        "lat", "lon", "type_house", "sum", "min_days",
        "amt_reviews", "last_dt", "avg_reviews", "total_host", "target",
    ]
    TARGET_COL = "target"
    ID_COL = "_id"

    # ------------------------------------------------------------------
    # Phase 1: Profile
    # ------------------------------------------------------------------

    def profile_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load train.csv and test.csv, compute a comprehensive profile, and
        store raw DataFrames + profile in state.

        No LLM call is made here — profiling is fully deterministic.
        """
        data_dir = Path(state.get("data_dir", self.cfg.get("data_dir", "data")))
        self._log(f"Loading data from {data_dir}")

        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self._log(f"Loaded train shape={train_df.shape}, test shape={test_df.shape}")

        profile = self._compute_profile(train_df, test_df)
        state["train_df"] = train_df
        state["test_df"] = test_df
        state["data_profile"] = profile

        self._log("Data profile computed and stored in state.")
        return state

    def _compute_profile(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build a JSON-serialisable profile dict from raw DataFrames."""

        def _col_stats(df: pd.DataFrame, split: str) -> Dict[str, Any]:
            stats: Dict[str, Any] = {"split": split, "shape": list(df.shape), "columns": {}}
            for col in df.columns:
                s = df[col]
                col_info: Dict[str, Any] = {
                    "dtype": str(s.dtype),
                    "null_count": int(s.isna().sum()),
                    "null_pct": round(float(s.isna().mean()) * 100, 2),
                    "unique_count": int(s.nunique()),
                }
                if pd.api.types.is_numeric_dtype(s):
                    desc = s.describe()
                    col_info.update({
                        "mean": round(float(desc["mean"]), 4) if "mean" in desc else None,
                        "std": round(float(desc["std"]), 4) if "std" in desc else None,
                        "min": round(float(desc["min"]), 4) if "min" in desc else None,
                        "p25": round(float(desc["25%"]), 4) if "25%" in desc else None,
                        "p50": round(float(desc["50%"]), 4) if "50%" in desc else None,
                        "p75": round(float(desc["75%"]), 4) if "75%" in desc else None,
                        "max": round(float(desc["max"]), 4) if "max" in desc else None,
                        "skewness": round(float(s.skew()), 4),
                    })
                else:
                    top_vals = s.value_counts().head(5).to_dict()
                    col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
                stats["columns"][col] = col_info
            return stats

        train_stats = _col_stats(train, "train")
        test_stats = _col_stats(test, "test")

        # Duplicate check
        train_dup = int(train.duplicated().sum())
        test_dup = int(test.duplicated().sum())

        # Column overlap
        train_only = list(set(train.columns) - set(test.columns))
        test_only = list(set(test.columns) - set(train.columns))

        return {
            "train": train_stats,
            "test": test_stats,
            "train_duplicate_rows": train_dup,
            "test_duplicate_rows": test_dup,
            "train_only_columns": train_only,
            "test_only_columns": test_only,
        }

    # ------------------------------------------------------------------
    # Phase 2: Clean
    # ------------------------------------------------------------------

    def clean_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask LLM to reason over the data profile and produce a cleaning plan,
        then execute that plan deterministically.
        """
        profile = state["data_profile"]
        train_df: pd.DataFrame = state["train_df"].copy()
        test_df: pd.DataFrame = state["test_df"].copy()

        # --- Ask LLM for cleaning plan ---
        cleaning_plan = self._request_cleaning_plan(profile)
        state["cleaning_plan"] = cleaning_plan
        self._log(f"Cleaning plan received: {json.dumps(cleaning_plan, indent=2)}")

        # --- Execute plan deterministically ---
        train_df, test_df = self._execute_cleaning_plan(train_df, test_df, cleaning_plan)

        # --- Validate ---
        self._validate(train_df, test_df)

        state["train_df"] = train_df
        state["test_df"] = test_df
        self._log(
            f"Cleaning complete. train shape={train_df.shape}, test shape={test_df.shape}"
        )
        return state

    def _request_cleaning_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a prompt from the data profile and ask the LLM to return a
        JSON cleaning plan.
        """
        # Summarise missing values for prompt brevity
        missing_summary = {}
        for col, info in profile["train"]["columns"].items():
            if info["null_pct"] > 0:
                missing_summary[col] = {
                    "null_pct": info["null_pct"],
                    "dtype": info["dtype"],
                }

        prompt = f"""
You are reviewing a rental-property regression dataset for a Kaggle competition.
The target column is '{self.TARGET_COL}' (rental price, numeric).
The test set does not contain the target column.

## Data profile summary
- Train shape: {profile['train']['shape']}
- Test shape:  {profile['test']['shape']}
- Train duplicate rows: {profile['train_duplicate_rows']}

## Columns with missing values (train)
{json.dumps(missing_summary, indent=2)}

## Column dtypes (train)
{json.dumps({col: info['dtype'] for col, info in profile['train']['columns'].items()}, indent=2)}

## Known domain columns
{self.KNOWN_COLUMNS}

## Task
Produce a data-cleaning plan in **strict JSON** (no markdown fences, no extra keys).
Schema:
{{
  "drop_columns": ["col1", ...],           // columns to drop from both splits (never drop '{self.ID_COL}' or '{self.TARGET_COL}')
  "imputation": {{                           // per-column imputation for remaining nulls
    "col_name": "median" | "mean" | "mode" | "zero" | "unknown" | "ffill"
  }},
  "outlier_handling": {{                     // optional outlier clipping per column
    "col_name": {{"method": "clip", "lower": <number>, "upper": <number>}}
  }},
  "type_conversions": {{                     // pandas dtype coercions
    "col_name": "datetime" | "int" | "float" | "str" | "category"
  }}
}}
Only include keys that need action. For columns with no nulls and correct types, omit them.
Respond with JSON only.
"""
        default_plan: Dict[str, Any] = {
            "drop_columns": [],
            "imputation": {
                col: "median"
                for col, info in profile["train"]["columns"].items()
                if info["null_pct"] > 0 and info["dtype"] not in ("object",)
            },
            "outlier_handling": {
                self.TARGET_COL: {"method": "clip", "lower": 0, "upper": 100_000}
            },
            "type_conversions": {"last_dt": "datetime"},
        }
        return self._ask_llm_json(prompt, default=default_plan)

    def _execute_cleaning_plan(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        plan: Dict[str, Any],
    ):
        """Apply the cleaning plan to both DataFrames deterministically."""

        # 1. Type conversions first (some operations depend on correct dtype)
        for col, dtype in plan.get("type_conversions", {}).items():
            for df in (train, test):
                if col not in df.columns:
                    continue
                try:
                    if dtype == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    elif dtype == "int":
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    elif dtype == "float":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif dtype == "str":
                        df[col] = df[col].astype(str)
                    elif dtype == "category":
                        df[col] = df[col].astype("category")
                except Exception as exc:
                    self._log(f"Type conversion failed for {col} → {dtype}: {exc}", level="warning")

        # 2. Drop columns (never drop ID or target)
        safe_to_drop = [
            c for c in plan.get("drop_columns", [])
            if c not in (self.ID_COL, self.TARGET_COL)
        ]
        for col in safe_to_drop:
            train.drop(columns=[c for c in [col] if c in train.columns], inplace=True)
            test.drop(columns=[c for c in [col] if c in test.columns], inplace=True)
        if safe_to_drop:
            self._log(f"Dropped columns: {safe_to_drop}")

        # 3. Outlier handling (train only for target; both for features)
        for col, spec in plan.get("outlier_handling", {}).items():
            method = spec.get("method", "clip")
            lower = spec.get("lower")
            upper = spec.get("upper")
            if method == "clip":
                if col in train.columns:
                    train[col] = train[col].clip(lower=lower, upper=upper)
                    self._log(f"Clipped {col} to [{lower}, {upper}] in train.")
                # clip feature cols in test too (but not target)
                if col != self.TARGET_COL and col in test.columns:
                    test[col] = test[col].clip(lower=lower, upper=upper)

        # 4. Imputation
        imputation_map = plan.get("imputation", {})

        # Compute fill values from train (to avoid leakage)
        fill_values: Dict[str, Any] = {}
        for col, strategy in imputation_map.items():
            if col not in train.columns:
                continue
            if strategy == "median":
                fill_values[col] = train[col].median()
            elif strategy == "mean":
                fill_values[col] = train[col].mean()
            elif strategy == "mode":
                mode_vals = train[col].mode()
                fill_values[col] = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            elif strategy == "zero":
                fill_values[col] = 0
            elif strategy == "unknown":
                fill_values[col] = "unknown"
            elif strategy == "ffill":
                fill_values[col] = None  # handle separately

        for col, strategy in imputation_map.items():
            for df in (train, test):
                if col not in df.columns:
                    continue
                if strategy == "ffill":
                    df[col].fillna(method="ffill", inplace=True)
                    df[col].fillna(method="bfill", inplace=True)  # catch leading NaNs
                elif col in fill_values and fill_values[col] is not None:
                    df[col].fillna(fill_values[col], inplace=True)

        return train, test

    def _validate(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Run basic unit-test style checks on the cleaned DataFrames."""
        errors: List[str] = []

        # Check train has target column
        if self.TARGET_COL not in train.columns:
            errors.append(f"train missing '{self.TARGET_COL}' column.")

        # Check test has ID column
        if self.ID_COL not in test.columns:
            errors.append(f"test missing '{self.ID_COL}' column.")

        # Check no duplicated IDs in train
        if self.ID_COL in train.columns:
            n_dup_ids = int(train[self.ID_COL].duplicated().sum())
            if n_dup_ids > 0:
                errors.append(f"train has {n_dup_ids} duplicated '{self.ID_COL}' values.")

        # Check no NaN in target
        if self.TARGET_COL in train.columns:
            n_nan_target = int(train[self.TARGET_COL].isna().sum())
            if n_nan_target > 0:
                errors.append(f"target column has {n_nan_target} NaN values after cleaning.")

        # Check no NaN in test ID
        if self.ID_COL in test.columns:
            n_nan_id = int(test[self.ID_COL].isna().sum())
            if n_nan_id > 0:
                errors.append(f"test '{self.ID_COL}' has {n_nan_id} NaN values.")

        if errors:
            msg = "Data validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            self._log(msg, level="error")
            raise ValueError(msg)

        self._log("Data validation passed.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run Phase 1 (profile) then Phase 2 (clean)."""
        state = self.profile_data(state)
        state = self.clean_data(state)
        return state
