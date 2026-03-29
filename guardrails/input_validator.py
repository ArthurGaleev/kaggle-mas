"""
Input validation guardrails for the multi-agent ML pipeline.

Validates datasets, feature matrices, and raw LLM outputs before they
are passed downstream.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates pipeline inputs: datasets, feature matrices, and LLM outputs.

    All ``validate_*`` methods return ``(is_valid: bool, issues: List[str])``.
    An empty ``issues`` list means validation passed.
    """

    # ------------------------------------------------------------------
    # Dataset validation
    # ------------------------------------------------------------------

    def validate_dataset(
        self,
        df: Any,  # pd.DataFrame — avoid hard import at module level
        cfg: Any,
    ) -> Tuple[bool, List[str]]:
        """Validate a raw dataset DataFrame.

        Checks performed
        ----------------
        * Row count is within configured limits.
        * Column count is within configured limits.
        * No completely empty columns exist.
        * Column dtypes are from an allowed set.
        * Target column exists and is numeric (for training data).
        * No duplicate IDs (if an ID column is configured).
        * File/memory size is reasonable.

        Parameters
        ----------
        df:
            Input ``pandas.DataFrame`` to validate.
        cfg:
            Config object / dict with keys such as ``max_rows``,
            ``max_cols``, ``target_col``, ``id_col``, ``max_file_size_mb``.

        Returns
        -------
        (is_valid, issues)
        """
        import pandas as pd  # noqa: PLC0415

        issues: List[str] = []

        if not isinstance(df, pd.DataFrame):
            return False, ["Input is not a pandas DataFrame."]

        # Resolve config values with sensible defaults
        cfg_dict = cfg if isinstance(cfg, dict) else (cfg.__dict__ if hasattr(cfg, "__dict__") else {})
        max_rows: int = cfg_dict.get("max_rows", 2_000_000)
        max_cols: int = cfg_dict.get("max_cols", 500)
        target_col: Optional[str] = cfg_dict.get("target_col", None)
        id_col: Optional[str] = cfg_dict.get("id_col", None)
        max_mb: float = cfg_dict.get("max_file_size_mb", 500.0)
        allowed_dtypes: set = cfg_dict.get(
            "allowed_dtypes", {"int64", "int32", "float64", "float32", "object", "bool", "category"}
        )

        n_rows, n_cols = df.shape

        # Row / column limits
        if n_rows == 0:
            issues.append("Dataset has 0 rows.")
        elif n_rows > max_rows:
            issues.append(f"Row count {n_rows:,} exceeds limit {max_rows:,}.")

        if n_cols == 0:
            issues.append("Dataset has 0 columns.")
        elif n_cols > max_cols:
            issues.append(f"Column count {n_cols} exceeds limit {max_cols}.")

        # Completely empty columns
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            issues.append(f"Completely empty columns: {empty_cols}.")

        # Dtype check
        bad_dtypes = [
            f"{c} ({df[c].dtype})"
            for c in df.columns
            if str(df[c].dtype) not in allowed_dtypes
        ]
        if bad_dtypes:
            issues.append(f"Columns with disallowed dtypes: {bad_dtypes[:10]}.")

        # Target column
        if target_col:
            if target_col not in df.columns:
                issues.append(f"Target column '{target_col}' not found in dataset.")
            elif not pd.api.types.is_numeric_dtype(df[target_col]):
                issues.append(
                    f"Target column '{target_col}' is not numeric "
                    f"(dtype={df[target_col].dtype})."
                )

        # Duplicate IDs
        if id_col and id_col in df.columns:
            n_dups = df[id_col].duplicated().sum()
            if n_dups > 0:
                issues.append(f"ID column '{id_col}' has {n_dups} duplicate values.")

        # Memory size
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        if mem_mb > max_mb:
            issues.append(
                f"Dataset in-memory size {mem_mb:.1f} MB exceeds limit {max_mb:.0f} MB."
            )

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning("Dataset validation failed: %s", issues)
        else:
            logger.info("Dataset validation passed (%d rows, %d cols).", n_rows, n_cols)
        return is_valid, issues

    # ------------------------------------------------------------------
    # Feature matrix validation
    # ------------------------------------------------------------------

    def validate_features(
        self,
        df: Any,
        cfg: Any,
    ) -> Tuple[bool, List[str]]:
        """Validate an engineered feature DataFrame.

        Checks performed
        ----------------
        * Feature count within limits.
        * No constant features (zero variance).
        * No features with >95 % missing values.
        * No duplicate feature columns.
        * No infinite values.

        Parameters
        ----------
        df:
            Feature ``pandas.DataFrame`` (should contain only feature columns,
            not the target).
        cfg:
            Config object / dict with optional ``max_features`` key.

        Returns
        -------
        (is_valid, issues)
        """
        import pandas as pd  # noqa: PLC0415

        issues: List[str] = []

        if not isinstance(df, pd.DataFrame):
            return False, ["Input is not a pandas DataFrame."]

        cfg_dict = cfg if isinstance(cfg, dict) else (cfg.__dict__ if hasattr(cfg, "__dict__") else {})
        max_features: int = cfg_dict.get("max_features", 1000)
        missing_threshold: float = cfg_dict.get("missing_threshold", 0.95)

        n_rows, n_cols = df.shape

        # Feature count limit
        if n_cols > max_features:
            issues.append(f"Feature count {n_cols} exceeds limit {max_features}.")

        # Constant features (numeric only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            issues.append(f"Constant (zero-variance) features: {constant_cols}.")

        # High-missing features
        missing_frac = df.isnull().mean()
        high_missing = missing_frac[missing_frac > missing_threshold].index.tolist()
        if high_missing:
            issues.append(
                f"Features with >{missing_threshold:.0%} missing: {high_missing}."
            )

        # Duplicate columns
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            issues.append(f"Duplicate feature columns: {dup_cols}.")

        # Infinite values
        inf_cols = [
            c for c in numeric_cols if np.isinf(df[c].values).any()
        ]
        if inf_cols:
            issues.append(f"Infinite values found in columns: {inf_cols}.")

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning("Feature validation failed: %s", issues)
        else:
            logger.info("Feature validation passed (%d features).", n_cols)
        return is_valid, issues

    # ------------------------------------------------------------------
    # LLM output validation
    # ------------------------------------------------------------------

    def validate_llm_output(
        self,
        output: str,
        expected_format: str = "json",
    ) -> Tuple[bool, str]:
        """Validate and sanitize raw LLM output.

        Parameters
        ----------
        output:
            Raw string returned by the LLM.
        expected_format:
            Currently supports ``"json"`` (default) or ``"text"``.

        Returns
        -------
        (is_valid, sanitized_output)
            ``is_valid`` is ``False`` if parsing fails or the output is
            suspiciously short/long.  ``sanitized_output`` is the cleaned
            string (or an empty string on failure).
        """
        if not isinstance(output, str):
            logger.warning("LLM output is not a string: %s", type(output))
            return False, ""

        MAX_LEN = 50_000
        MIN_LEN = 2

        stripped = output.strip()

        if len(stripped) < MIN_LEN:
            logger.warning("LLM output is too short (%d chars).", len(stripped))
            return False, ""

        if len(stripped) > MAX_LEN:
            logger.warning(
                "LLM output is too long (%d chars > %d). Truncating.", len(stripped), MAX_LEN
            )
            stripped = stripped[:MAX_LEN]

        # Sanitize: remove potential injection / dangerous calls
        stripped = _sanitize_text(stripped)

        if expected_format == "json":
            # Try to extract JSON block if surrounded by markdown fences
            json_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", stripped)
            json_str = json_match.group(1).strip() if json_match else stripped

            try:
                json.loads(json_str)
                return True, json_str
            except json.JSONDecodeError as exc:
                logger.warning("LLM output is not valid JSON: %s", exc)
                return False, json_str

        # Plain text — return as-is after sanitisation
        return True, stripped


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    (re.compile(r"os\.system\s*\(", re.I), "os.system("),
    (re.compile(r"subprocess\.", re.I), "subprocess."),
    (re.compile(r"\beval\s*\(", re.I), "eval("),
    (re.compile(r"\bexec\s*\(", re.I), "exec("),
    (re.compile(r"__import__\s*\(", re.I), "__import__("),
    (re.compile(r"open\s*\(.*['\"]w['\"]", re.I), "open(..., 'w')"),
    (re.compile(r"shutil\.rmtree", re.I), "shutil.rmtree"),
    (re.compile(r"os\.remove\s*\(", re.I), "os.remove("),
    (re.compile(r"os\.unlink\s*\(", re.I), "os.unlink("),
    (re.compile(r"Ignore previous instructions", re.I), "prompt injection"),
    (re.compile(r"system prompt", re.I), "system prompt reference"),
]


def _sanitize_text(text: str) -> str:
    """Remove dangerous patterns from *text* and return the clean version."""
    for pattern, label in _DANGEROUS_PATTERNS:
        if pattern.search(text):
            logger.warning("Sanitizing dangerous pattern '%s' from text.", label)
            text = pattern.sub(f"[REMOVED:{label}]", text)
    return text
