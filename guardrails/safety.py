"""
Safety guardrails for the multi-agent ML pipeline.

Provides LLM response sanitization, resource limit checks, and config
integrity validation.
"""

from __future__ import annotations

import logging
import re
import shutil
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dangerous pattern catalogue (pattern, human-readable label, replacement)
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # OS / subprocess calls
    (re.compile(r"os\.system\s*\(.*?\)", re.I | re.S), "os.system()", "[BLOCKED:os.system]"),
    (re.compile(r"subprocess\.(?:run|call|Popen|check_output)\s*\(.*?\)", re.I | re.S),
     "subprocess call", "[BLOCKED:subprocess]"),
    # Code execution
    (re.compile(r"\beval\s*\(.*?\)", re.I | re.S), "eval()", "[BLOCKED:eval]"),
    (re.compile(r"\bexec\s*\(.*?\)", re.I | re.S), "exec()", "[BLOCKED:exec]"),
    (re.compile(r"__import__\s*\(.*?\)", re.I | re.S), "__import__()", "[BLOCKED:__import__]"),
    (re.compile(r"compile\s*\(.*?\)", re.I | re.S), "compile()", "[BLOCKED:compile]"),
    # File deletion
    (re.compile(r"os\.remove\s*\(.*?\)", re.I | re.S), "os.remove()", "[BLOCKED:os.remove]"),
    (re.compile(r"os\.unlink\s*\(.*?\)", re.I | re.S), "os.unlink()", "[BLOCKED:os.unlink]"),
    (re.compile(r"shutil\.rmtree\s*\(.*?\)", re.I | re.S), "shutil.rmtree()", "[BLOCKED:shutil.rmtree]"),
    (re.compile(r"pathlib\.Path.*?\.unlink\s*\(", re.I | re.S), "Path.unlink()", "[BLOCKED:Path.unlink]"),
    # Write to arbitrary files
    (re.compile(r"open\s*\([^)]+['\"]w[a-z]?['\"].*?\)", re.I | re.S),
     "open(..., 'w')", "[BLOCKED:open-write]"),
    # Prompt injection attempts
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
     "prompt injection (ignore previous)", "[BLOCKED:injection]"),
    (re.compile(r"disregard\s+(all\s+)?previous\s+instructions?", re.I),
     "prompt injection (disregard)", "[BLOCKED:injection]"),
    (re.compile(r"\bsystem\s+prompt\b", re.I), "system prompt reference", "[BLOCKED:system-prompt]"),
    (re.compile(r"\bDAN\b"), "DAN jailbreak", "[BLOCKED:jailbreak]"),
]


class SafetyGuard:
    """Safety guardrails: LLM sanitization, resource checks, config validation.

    Parameters
    ----------
    log_sanitizations:
        When ``True`` (default), every sanitization action is logged at
        WARNING level.
    """

    def __init__(self, log_sanitizations: bool = True) -> None:
        self.log_sanitizations = log_sanitizations
        self._sanitization_count = 0

    # ------------------------------------------------------------------
    # LLM response sanitization
    # ------------------------------------------------------------------

    def sanitize_llm_response(self, response: str) -> str:
        """Remove potentially harmful content from an LLM response.

        Actions performed
        -----------------
        * Strip ``os.system()``, ``subprocess.*``, ``eval()``, ``exec()``
          and similar code-execution calls.
        * Remove file-deletion commands.
        * Replace prompt injection attempts (e.g. "ignore previous instructions").
        * Log any sanitization actions.

        Parameters
        ----------
        response:
            Raw string returned by the LLM.

        Returns
        -------
        str
            Sanitized response.  Original is returned unchanged if no
            dangerous patterns are found.
        """
        if not isinstance(response, str):
            logger.warning("sanitize_llm_response: input is not a string (%s). Returning ''.", type(response))
            return ""

        sanitized = response
        actions_taken: List[str] = []

        for pattern, label, replacement in _DANGEROUS_PATTERNS:
            new_text, n_subs = pattern.subn(replacement, sanitized)
            if n_subs > 0:
                actions_taken.append(f"Removed {n_subs}x '{label}'")
                self._sanitization_count += n_subs
                sanitized = new_text

        if actions_taken and self.log_sanitizations:
            logger.warning(
                "LLM response sanitized — %d action(s): %s",
                len(actions_taken),
                "; ".join(actions_taken),
            )

        return sanitized

    # ------------------------------------------------------------------
    # Resource limit checks
    # ------------------------------------------------------------------

    def check_resource_limits(self, cfg: Any) -> bool:
        """Check available system resources and warn if running low.

        Parameters
        ----------
        cfg:
            Config object / dict.  Optional keys:

            * ``min_free_ram_mb`` – raise warning below this RAM threshold
              (default 500 MB).
            * ``min_free_disk_mb`` – raise warning below this disk threshold
              (default 1000 MB).

        Returns
        -------
        bool
            ``True`` if resources are sufficient, ``False`` if a warning
            was issued.
        """
        cfg_dict = _to_dict(cfg)
        min_ram_mb: float = cfg_dict.get("min_free_ram_mb", 500.0)
        min_disk_mb: float = cfg_dict.get("min_free_disk_mb", 1000.0)

        warnings: List[str] = []

        # RAM check via psutil (optional dependency)
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            free_ram_mb = vm.available / (1024 ** 2)
            if free_ram_mb < min_ram_mb:
                warnings.append(
                    f"Low RAM: {free_ram_mb:.0f} MB available (threshold {min_ram_mb:.0f} MB)."
                )
            else:
                logger.debug("RAM OK: %.0f MB available.", free_ram_mb)
        except ImportError:
            # psutil not installed — fall back to /proc/meminfo on Linux
            try:
                free_ram_mb = _read_proc_meminfo_free_mb()
                if free_ram_mb is not None and free_ram_mb < min_ram_mb:
                    warnings.append(
                        f"Low RAM: {free_ram_mb:.0f} MB available (threshold {min_ram_mb:.0f} MB)."
                    )
            except Exception:
                logger.debug("Could not read memory info; skipping RAM check.")

        # Disk check via shutil
        try:
            disk = shutil.disk_usage("/")
            free_disk_mb = disk.free / (1024 ** 2)
            if free_disk_mb < min_disk_mb:
                warnings.append(
                    f"Low disk space: {free_disk_mb:.0f} MB free (threshold {min_disk_mb:.0f} MB)."
                )
            else:
                logger.debug("Disk OK: %.0f MB free.", free_disk_mb)
        except Exception as exc:
            logger.debug("Could not check disk space: %s", exc)

        if warnings:
            for w in warnings:
                logger.warning("RESOURCE WARNING: %s", w)
            return False
        return True

    # ------------------------------------------------------------------
    # Config integrity validation
    # ------------------------------------------------------------------

    def validate_config(self, cfg: Any) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration integrity.

        Checks performed
        ----------------
        * Config is not ``None`` or empty.
        * Required keys are present (``target_col``, ``id_col``).
        * Numeric parameters are within sane ranges.
        * No obviously dangerous string values (shell commands, paths
          outside the workspace).

        Parameters
        ----------
        cfg:
            Config object / dict.

        Returns
        -------
        (is_valid, issues)
        """
        issues: List[str] = []

        cfg_dict = _to_dict(cfg)

        if not cfg_dict:
            return False, ["Config is empty or None."]

        # Required keys
        for key in ("target_col",):
            if key not in cfg_dict:
                issues.append(f"Required config key '{key}' is missing.")

        # Numeric range checks
        _check_numeric_range(cfg_dict, "n_folds", 2, 20, issues)
        _check_numeric_range(cfg_dict, "max_rows", 1, 10_000_000, issues)
        _check_numeric_range(cfg_dict, "max_cols", 1, 10_000, issues)
        _check_numeric_range(cfg_dict, "optuna_trials", 1, 500, issues)

        # String safety: check for shell injection in string values
        _SHELL_RE = re.compile(r"[;&|`$]")
        for key, val in cfg_dict.items():
            if isinstance(val, str) and _SHELL_RE.search(val):
                issues.append(
                    f"Config key '{key}' contains potentially unsafe shell characters: '{val}'."
                )

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning("Config validation failed: %s", issues)
        else:
            logger.info("Config validation passed.")
        return is_valid, issues

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def total_sanitizations(self) -> int:
        """Total number of sanitization replacements performed so far."""
        return self._sanitization_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(cfg: Any) -> dict:
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return cfg.__dict__
    return {}


def _check_numeric_range(
    cfg_dict: dict,
    key: str,
    min_val: float,
    max_val: float,
    issues: List[str],
) -> None:
    if key not in cfg_dict:
        return
    val = cfg_dict[key]
    if not isinstance(val, (int, float)):
        issues.append(f"Config key '{key}' must be numeric, got {type(val).__name__}.")
        return
    if not (min_val <= val <= max_val):
        issues.append(
            f"Config key '{key}' = {val} is outside valid range [{min_val}, {max_val}]."
        )


def _read_proc_meminfo_free_mb() -> float | None:
    """Read free memory from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return None
