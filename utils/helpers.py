"""
General-purpose utility helpers for the Kaggle MAS pipeline.

Functions:
    set_seed            — reproducibility seed for numpy/random/torch
    get_memory_usage    — current process RSS in MB
    timer_decorator     — function decorator that logs elapsed time
    safe_json_parse     — robust JSON extraction from raw LLM text
    flatten_dict        — recursively flatten nested dicts
    validate_dataframe  — DataFrame shape / column / constraint checker
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])

_log = logging.getLogger("kaggle-mas.helpers")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across the standard scientific stack.

    Sets seeds for:
    - Python's built-in :mod:`random` module
    - NumPy (if installed)
    - PyTorch CPU and CUDA (if installed)
    - ``PYTHONHASHSEED`` environment variable

    Args:
        seed: Integer seed value. Defaults to ``42``.

    Example::

        set_seed(42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _NUMPY_AVAILABLE:
        np.random.seed(seed)
        _log.debug("NumPy seed set to %d", seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _log.debug("PyTorch seed set to %d", seed)
    except ImportError:
        pass  # torch is optional

    _log.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Memory usage
# ---------------------------------------------------------------------------

def get_memory_usage() -> float:
    """
    Return the current process resident set size (RSS) in **megabytes**.

    Uses :mod:`psutil` when available; falls back to :mod:`resource`
    (POSIX only).  Returns ``0.0`` if neither is available.

    Returns:
        Memory usage in MB as a float.

    Example::

        mb = get_memory_usage()
        print(f"Using {mb:.1f} MB")
    """
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        pass
    try:
        import resource
        # ru_maxrss is in kilobytes on Linux, bytes on macOS
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import platform
        if platform.system() == "Darwin":
            return rss_kb / 1024 ** 2  # bytes → MB
        return rss_kb / 1024.0         # KB → MB
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Timer decorator
# ---------------------------------------------------------------------------

def timer_decorator(
    logger: Optional[logging.Logger] = None,
    level: str = "INFO",
) -> Callable[[F], F]:
    """
    Decorator factory that logs execution time of the wrapped function.

    Args:
        logger: Logger to write to.  If ``None``, uses ``kaggle-mas.helpers``.
        level:  Log level string (``"DEBUG"``, ``"INFO"``, etc.).

    Returns:
        A decorator that wraps the target function.

    Example::

        @timer_decorator(level="DEBUG")
        def expensive_operation(df):
            ...
    """
    _logger = logger or _log
    _level  = getattr(logging, level.upper(), logging.INFO)

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            mem_before = get_memory_usage()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                mem_after = get_memory_usage()
                _logger.log(
                    _level,
                    "%s completed in %.3fs | mem Δ%.1f MB",
                    fn.__qualname__, elapsed, mem_after - mem_before,
                )
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                _logger.error(
                    "%s FAILED after %.3fs: %s",
                    fn.__qualname__, elapsed, exc,
                )
                raise
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Safe JSON parsing
# ---------------------------------------------------------------------------

# Patterns to strip markdown fences and leading text before the JSON payload
_MD_FENCE_RE    = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
_LEADING_TEXT_RE = re.compile(r"^[^{[\n]*({.*)", re.DOTALL)


def safe_json_parse(text: str) -> Optional[Any]:
    """
    Robustly parse JSON from raw LLM output.

    Handles:
    - Plain JSON strings
    - Markdown fenced code blocks (`` ```json `` or `` ``` ``)
    - Leading prose before the JSON object
    - Trailing prose after the JSON object

    Args:
        text: Raw string output from an LLM.

    Returns:
        Parsed Python object, or ``None`` if parsing fails after all attempts.

    Example::

        result = safe_json_parse('Sure! Here is the data: {"key": "value"}')
        # → {"key": "value"}
    """
    if not text or not isinstance(text, str):
        return None

    candidates: List[str] = []

    # 1. Try extracting from markdown fences
    fence_matches = _MD_FENCE_RE.findall(text)
    candidates.extend(m.strip() for m in fence_matches)

    # 2. Raw text (trimmed)
    candidates.append(text.strip())

    # 3. Try finding the first '{' or '['
    for char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(char)
        if idx != -1:
            candidates.append(text[idx:])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Try to extract first JSON object / array with balanced braces
        extracted = _extract_first_json(candidate)
        if extracted is not None:
            return extracted

    _log.warning("safe_json_parse: could not parse JSON from text of length %d", len(text))
    return None


def _extract_first_json(text: str) -> Optional[Any]:
    """Find and return the first complete JSON object or array in *text*."""
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[start_idx:], start=start_idx):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx: i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Dictionary utilities
# ---------------------------------------------------------------------------

def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.

    Args:
        d:          The dictionary to flatten.
        parent_key: Prefix for keys in the current recursion level.
        sep:        Separator between levels. Defaults to ``"."``.

    Returns:
        A new flat dictionary.

    Example::

        flat = flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
        # → {"a.b": 1, "a.c.d": 2}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ---------------------------------------------------------------------------
# DataFrame validation
# ---------------------------------------------------------------------------

class DataFrameValidationError(ValueError):
    """Raised when :func:`validate_dataframe` detects a constraint violation."""


def validate_dataframe(
    df: "pd.DataFrame",
    expected_columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    min_rows: int = 1,
    *,
    allow_missing_columns: bool = False,
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate a :class:`pandas.DataFrame` against common constraints.

    Checks:
    - DataFrame is not ``None`` and is a pandas DataFrame
    - Row count is within ``[min_rows, max_rows]``
    - All *expected_columns* are present (unless *allow_missing_columns*)

    Args:
        df:                    The DataFrame to validate.
        expected_columns:      Column names that must be present.
        max_rows:              Maximum number of allowed rows.
        min_rows:              Minimum number of required rows. Defaults to ``1``.
        allow_missing_columns: If ``True``, missing columns produce warnings
                               instead of errors.
        raise_on_error:        If ``True`` (default), raise
                               :class:`DataFrameValidationError` on the first
                               violation. Otherwise return ``(False, [messages])``.

    Returns:
        ``(True, [])`` on success, or ``(False, [error_messages])`` when
        *raise_on_error* is ``False``.

    Raises:
        DataFrameValidationError: On first constraint violation when
                                  *raise_on_error* is ``True``.
        ImportError:              If pandas is not installed.

    Example::

        ok, errors = validate_dataframe(
            df,
            expected_columns=["target", "lat", "lon"],
            max_rows=500_000,
            raise_on_error=False,
        )
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError("pandas is required for validate_dataframe()")

    errors: List[str] = []

    def _fail(msg: str) -> None:
        if raise_on_error:
            raise DataFrameValidationError(msg)
        errors.append(msg)

    # Type check
    if df is None:
        _fail("DataFrame is None")
        return False, errors
    if not isinstance(df, pd.DataFrame):
        _fail(f"Expected pd.DataFrame, got {type(df).__name__}")
        return False, errors

    # Row count
    n_rows = len(df)
    if n_rows < min_rows:
        _fail(f"DataFrame has {n_rows} rows, expected at least {min_rows}")
    if max_rows is not None and n_rows > max_rows:
        _fail(f"DataFrame has {n_rows} rows, maximum allowed is {max_rows}")

    # Column check
    if expected_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            msg = f"DataFrame is missing expected columns: {missing}"
            if allow_missing_columns:
                _log.warning(msg)
            else:
                _fail(msg)

    if not errors:
        _log.debug(
            "DataFrame validation passed: %d rows × %d cols", n_rows, len(df.columns)
        )
    return len(errors) == 0, errors


__all__ = [
    "set_seed",
    "get_memory_usage",
    "timer_decorator",
    "safe_json_parse",
    "flatten_dict",
    "validate_dataframe",
    "DataFrameValidationError",
]
