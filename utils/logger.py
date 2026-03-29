"""
Structured logging, token tracking, and agent monitoring for the Kaggle MAS pipeline.

Provides:
- get_logger()      — structured logger with JSON file handler + human-readable console
- TokenTracker      — tracks LLM API token usage per agent/phase with cost estimation
- AgentMonitor      — logs agent lifecycle events (start, action, end) with metrics
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Cost table (USD per 1 000 tokens, input / output)
# Update as providers publish new pricing.
# ---------------------------------------------------------------------------
_COST_PER_1K: Dict[str, Dict[str, float]] = {
    # Groq
    "llama-3.3-70b-versatile":   {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant":      {"input": 0.00005, "output": 0.00008},
    # HuggingFace Serverless (approx)
    "Qwen/Qwen2.5-72B-Instruct": {"input": 0.00090, "output": 0.00090},
    "Qwen/Qwen2.5-7B-Instruct":  {"input": 0.00007, "output": 0.00007},
    # OpenRouter
    "qwen/qwen-2.5-72b-instruct":{"input": 0.00090, "output": 0.00090},
    "qwen/qwen-2.5-7b-instruct": {"input": 0.00007, "output": 0.00007},
    # Default fallback
    "__default__":               {"input": 0.00100, "output": 0.00100},
}


def _cost_for_model(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a given model and token counts."""
    rates = _COST_PER_1K.get(model, _COST_PER_1K["__default__"])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000.0


# ---------------------------------------------------------------------------
# JSON formatter for file handler
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "funcName":  record.funcName,
            "lineno":    record.lineno,
        }
        # Attach any extra fields that callers pass via `extra=`
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                log_obj[key] = val
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Console formatter — human-readable with colour
# ---------------------------------------------------------------------------
_LEVEL_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ConsoleFormatter(logging.Formatter):
    """Coloured, concise console format."""

    _FMT = "{colour}[{level}]{reset} {ts} | {name} | {msg}"

    def format(self, record: logging.LogRecord) -> str:
        colour  = _LEVEL_COLOURS.get(record.levelname, "")
        ts      = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return self._FMT.format(
            colour=colour,
            reset=_RESET,
            level=record.levelname[0],   # single char: I/W/E/D/C
            ts=ts,
            name=record.name,
            msg=message,
        )


# ---------------------------------------------------------------------------
# Module-level logger registry (one logger per name, thread-safe)
# ---------------------------------------------------------------------------
_loggers: Dict[str, logging.Logger] = {}
_logger_lock = threading.Lock()


def get_logger(
    name: str = "kaggle-mas",
    log_dir: str | Path = "./logs",
    level: str = "INFO",
    *,
    json_log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Return a named logger configured with:
    - A rotating JSON file handler writing to ``<log_dir>/<name>.jsonl``
    - A coloured console handler

    Subsequent calls with the same *name* return the cached instance.

    Args:
        name:          Logger name (typically agent or module name).
        log_dir:       Directory where ``.jsonl`` log files are stored.
        level:         Logging level string, e.g. ``"INFO"``, ``"DEBUG"``.
        json_log_file: Override the JSON log file path.

    Returns:
        Configured :class:`logging.Logger`.
    """
    with _logger_lock:
        if name in _loggers:
            return _loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False

        # --- Console handler ---
        ch = logging.StreamHandler()
        ch.setFormatter(_ConsoleFormatter())
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(ch)

        # --- JSON file handler ---
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fpath = json_log_file or str(log_path / f"{name}.jsonl")
        fh = logging.FileHandler(fpath, encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        fh.setLevel(logging.DEBUG)  # always capture DEBUG to file
        logger.addHandler(fh)

        _loggers[name] = logger
        return logger


# ---------------------------------------------------------------------------
# TokenTracker
# ---------------------------------------------------------------------------
@dataclass
class _TokenRecord:
    agent:        str
    phase:        str
    model:        str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd:     float
    timestamp:    str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())


class TokenTracker:
    """
    Thread-safe tracker for LLM API token consumption.

    Usage::

        tracker = TokenTracker()
        tracker.record(
            agent="DataAgent", phase="data_profiling",
            model="llama-3.3-70b-versatile",
            prompt_tokens=512, completion_tokens=256,
        )
        print(tracker.summary())
        df = tracker.to_dataframe()
    """

    def __init__(self) -> None:
        self._records: List[_TokenRecord] = []
        self._lock = threading.Lock()
        # Aggregated counters: {agent: {phase: {tokens/cost}}}
        self._by_agent: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0, "cost_usd": 0.0})
        )

    def record(
        self,
        agent: str,
        phase: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Record a single LLM call.

        Args:
            agent:             Agent name (e.g. ``"DataAgent"``).
            phase:             Pipeline phase (e.g. ``"data_profiling"``).
            model:             Model identifier used for this call.
            prompt_tokens:     Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.

        Returns:
            Estimated cost in USD for this call.
        """
        total  = prompt_tokens + completion_tokens
        cost   = _cost_for_model(model, prompt_tokens, completion_tokens)
        rec    = _TokenRecord(
            agent=agent, phase=phase, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=total, cost_usd=cost,
        )
        with self._lock:
            self._records.append(rec)
            agg = self._by_agent[agent][phase]
            agg["prompt"]      += prompt_tokens
            agg["completion"]  += completion_tokens
            agg["total"]       += total
            agg["cost_usd"]    += cost
        return cost

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        """Return total tokens across all agents and phases."""
        with self._lock:
            return sum(r.total_tokens for r in self._records)

    def total_cost_usd(self) -> float:
        """Return total estimated cost in USD."""
        with self._lock:
            return sum(r.cost_usd for r in self._records)

    def by_agent(self) -> Dict[str, Dict[str, Any]]:
        """Return aggregated stats keyed by agent name."""
        with self._lock:
            return {
                agent: {
                    "total_tokens": sum(p["total"] for p in phases.values()),
                    "total_cost_usd": sum(p["cost_usd"] for p in phases.values()),
                    "phases": dict(phases),
                }
                for agent, phases in self._by_agent.items()
            }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"{'Agent':<20} {'Phase':<25} {'Prompt':>8} {'Compl.':>8} "
            f"{'Total':>8} {'Cost $':>10}",
            "-" * 85,
        ]
        with self._lock:
            for agent, phases in self._by_agent.items():
                for phase, agg in phases.items():
                    lines.append(
                        f"{agent:<20} {phase:<25} {int(agg['prompt']):>8} "
                        f"{int(agg['completion']):>8} {int(agg['total']):>8} "
                        f"{agg['cost_usd']:>10.6f}"
                    )
            lines.append("-" * 85)
            total_t = sum(r.total_tokens for r in self._records)
            total_c = sum(r.cost_usd for r in self._records)
            lines.append(f"{'TOTAL':<46} {total_t:>8} {total_c:>10.6f}")
        return "\n".join(lines)

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Export all token records as a :class:`pandas.DataFrame`.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError("pandas is required for TokenTracker.to_dataframe()")
        with self._lock:
            return pd.DataFrame([asdict(r) for r in self._records])

    def reset(self) -> None:
        """Clear all recorded data."""
        with self._lock:
            self._records.clear()
            self._by_agent.clear()


# ---------------------------------------------------------------------------
# AgentMonitor
# ---------------------------------------------------------------------------
@dataclass
class _AgentEvent:
    event_type:   str        # "start" | "action" | "end" | "error"
    agent:        str
    phase:        str
    action:       str
    duration_s:   Optional[float]
    memory_mb:    Optional[float]
    token_usage:  Optional[Dict[str, Any]]
    result_status: str       # "ok" | "error" | "pending"
    detail:       Optional[str]
    timestamp:    str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())


class AgentMonitor:
    """
    Monitors and logs agent lifecycle events with timing, memory, and token metrics.

    Usage::

        monitor = AgentMonitor(logger=get_logger("monitor"), token_tracker=tracker)

        token = monitor.start(agent="DataAgent", phase="data_profiling", action="load_csv")
        # ... do work ...
        monitor.end(token=token, result_status="ok", detail="Loaded 50000 rows")

        df = monitor.to_dataframe()
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        token_tracker: Optional[TokenTracker] = None,
        log_dir: str | Path = "./logs",
    ) -> None:
        self._logger        = logger or get_logger("agent-monitor", log_dir=log_dir)
        self._token_tracker = token_tracker
        self._events: List[_AgentEvent] = []
        self._lock  = threading.Lock()
        self._active: Dict[str, float] = {}  # token_id -> start_time

    # ------------------------------------------------------------------
    # Context management helpers
    # ------------------------------------------------------------------

    def start(
        self,
        agent: str,
        phase: str,
        action: str,
        *,
        memory_mb: Optional[float] = None,
    ) -> str:
        """
        Signal the start of an agent action.

        Args:
            agent:     Agent class name.
            phase:     Pipeline phase name.
            action:    Short description of the action being performed.
            memory_mb: Current process memory in MB (optional; auto-measured if omitted).

        Returns:
            An opaque token string to pass to :meth:`end`.
        """
        if memory_mb is None:
            memory_mb = _get_mem_mb()

        token_id = f"{agent}:{phase}:{action}:{time.monotonic()}"
        evt = _AgentEvent(
            event_type="start", agent=agent, phase=phase, action=action,
            duration_s=None, memory_mb=memory_mb, token_usage=None,
            result_status="pending", detail=None,
        )
        with self._lock:
            self._events.append(evt)
            self._active[token_id] = time.monotonic()

        self._logger.info(
            "Agent START",
            extra={"agent": agent, "phase": phase, "action": action,
                   "memory_mb": memory_mb, "event_type": "start"},
        )
        return token_id

    def end(
        self,
        token: str,
        result_status: str = "ok",
        detail: Optional[str] = None,
        *,
        memory_mb: Optional[float] = None,
        token_usage: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Signal the end of an agent action.

        Args:
            token:         Token returned by :meth:`start`.
            result_status: ``"ok"`` or ``"error"``.
            detail:        Optional free-text detail (e.g. error message, row count).
            memory_mb:     Current memory usage in MB.
            token_usage:   Dict with keys ``prompt_tokens``, ``completion_tokens``, etc.

        Returns:
            Elapsed seconds since the matching :meth:`start` call.
        """
        if memory_mb is None:
            memory_mb = _get_mem_mb()

        with self._lock:
            start_t = self._active.pop(token, None)
        duration = (time.monotonic() - start_t) if start_t is not None else None

        parts  = token.split(":")
        agent  = parts[0] if len(parts) > 0 else "unknown"
        phase  = parts[1] if len(parts) > 1 else "unknown"
        action = parts[2] if len(parts) > 2 else "unknown"

        evt = _AgentEvent(
            event_type="end", agent=agent, phase=phase, action=action,
            duration_s=duration, memory_mb=memory_mb,
            token_usage=token_usage, result_status=result_status, detail=detail,
        )
        with self._lock:
            self._events.append(evt)

        log_fn = self._logger.error if result_status == "error" else self._logger.info
        log_fn(
            f"Agent END [{result_status}]",
            extra={
                "agent": agent, "phase": phase, "action": action,
                "duration_s": duration, "memory_mb": memory_mb,
                "token_usage": token_usage, "result_status": result_status,
                "detail": detail, "event_type": "end",
            },
        )
        return duration or 0.0

    def log_action(
        self,
        agent: str,
        phase: str,
        action: str,
        detail: Optional[str] = None,
        *,
        result_status: str = "ok",
    ) -> None:
        """Log a discrete (non-timed) agent action."""
        evt = _AgentEvent(
            event_type="action", agent=agent, phase=phase, action=action,
            duration_s=None, memory_mb=_get_mem_mb(), token_usage=None,
            result_status=result_status, detail=detail,
        )
        with self._lock:
            self._events.append(evt)
        self._logger.info(
            f"Agent ACTION: {action}",
            extra={"agent": agent, "phase": phase, "action": action,
                   "detail": detail, "event_type": "action",
                   "result_status": result_status},
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Export all recorded events as a :class:`pandas.DataFrame`.

        Raises:
            ImportError: If pandas is not installed.
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError("pandas is required for AgentMonitor.to_dataframe()")
        with self._lock:
            return pd.DataFrame([asdict(e) for e in self._events])

    def summary(self) -> str:
        """Return a human-readable summary of all completed agent events."""
        with self._lock:
            completed = [e for e in self._events if e.event_type == "end"]
        if not completed:
            return "No completed agent events recorded."
        lines = [
            f"{'Agent':<20} {'Phase':<25} {'Action':<30} "
            f"{'Status':<8} {'Dur(s)':>8} {'Mem(MB)':>10}",
            "-" * 105,
        ]
        for e in completed:
            dur = f"{e.duration_s:.2f}" if e.duration_s is not None else "N/A"
            mem = f"{e.memory_mb:.1f}" if e.memory_mb is not None else "N/A"
            lines.append(
                f"{e.agent:<20} {e.phase:<25} {e.action:<30} "
                f"{e.result_status:<8} {dur:>8} {mem:>10}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()
            self._active.clear()


# ---------------------------------------------------------------------------
# Internal helper (avoids circular import with helpers.py)
# ---------------------------------------------------------------------------
def _get_mem_mb() -> float:
    """Return current process RSS memory in MB (best-effort)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        try:
            import resource
            # RUSAGE_SELF returns KB on Linux
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Module-level singletons exposed for convenience
# ---------------------------------------------------------------------------
#: Global TokenTracker instance — import and use directly across the codebase.
global_token_tracker: TokenTracker = TokenTracker()

#: Global AgentMonitor instance wired to the root "kaggle-mas" logger.
global_monitor: AgentMonitor = AgentMonitor(
    logger=get_logger("kaggle-mas"),
    token_tracker=global_token_tracker,
)

__all__ = [
    "get_logger",
    "TokenTracker",
    "AgentMonitor",
    "global_token_tracker",
    "global_monitor",
]
