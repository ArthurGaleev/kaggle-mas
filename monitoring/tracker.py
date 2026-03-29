"""
Pipeline execution tracker for the multi-agent ML system.

Records phase timings, agent actions, LLM calls, model metrics, and memory
snapshots.  Produces a structured JSON report at the end of a run.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------
EVT_PHASE_START = "phase_start"
EVT_PHASE_END = "phase_end"
EVT_AGENT_ACTION = "agent_action"
EVT_LLM_CALL = "llm_call"
EVT_MODEL_METRIC = "model_metric"
EVT_MEMORY_SNAPSHOT = "memory_snapshot"
EVT_FEEDBACK_ITER = "feedback_iteration"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed(start_ts: float) -> float:
    return round(time.monotonic() - start_ts, 4)


class PipelineTracker:
    """Tracks the full lifecycle of a multi-agent ML pipeline run.

    All events are stored in :attr:`events` as dicts with at least
    ``type``, ``timestamp``, and ``elapsed_s`` fields.

    Usage example
    -------------
    >>> tracker = PipelineTracker()
    >>> tracker.start_phase("data_cleaning")
    >>> tracker.log_agent_action("DataAgent", "impute_missing", {"cols": 3})
    >>> tracker.end_phase("data_cleaning", "success", {"rows_cleaned": 1000})
    >>> report = tracker.get_summary()

    Loading a saved report
    ----------------------
    >>> tracker = PipelineTracker.load("outputs/pipeline_report.json")
    >>> summary = tracker.get_summary()
    """

    def __init__(self) -> None:
        self._start_ts: float = time.monotonic()
        self._start_wall: str = _now_iso()
        self.events: List[Dict[str, Any]] = []

        # Phase bookkeeping: phase_name -> monotonic start time
        self._phase_starts: Dict[str, float] = {}

        # Aggregate counters (updated incrementally for fast access)
        self._llm_calls: int = 0
        self._llm_prompt_tokens: int = 0
        self._llm_completion_tokens: int = 0
        self._llm_total_latency: float = 0.0

        # Caches populated only when loading from a saved report.
        # None means "not loaded from file — derive from events as usual".
        self._cached_model_metrics: Optional[Dict[str, Dict[str, List[float]]]] = None
        self._cached_feedback_history: Optional[List[Dict[str, Any]]] = None

        logger.info("PipelineTracker initialised at %s", self._start_wall)

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "PipelineTracker":
        """Load a tracker from a previously saved JSON report.

        Reconstructs the tracker object from the JSON file produced by
        :meth:`save_report`, restoring all events and aggregate LLM counters
        so that :meth:`get_summary` works correctly on the reloaded instance.

        Top-level ``model_metrics`` and ``feedback_history`` keys saved by
        :meth:`save_report` are cached on the instance and used as a fallback
        by :meth:`get_summary` when the ``events`` list does not contain the
        corresponding raw event entries.  This ensures that
        ``feedback_loop_progress.png`` and ``model_comparison.png`` are always
        populated when those fields were present in the original report.

        Parameters
        ----------
        path:
            File path of the JSON report produced by :meth:`save_report`.

        Returns
        -------
        PipelineTracker
            A new tracker instance pre-populated with the saved events.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.

        Example
        -------
        >>> tracker = PipelineTracker.load("outputs/pipeline_report.json")
        >>> print(tracker.get_summary()["phase_durations_s"])
        """
        out_path = Path(path)
        with open(out_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        tracker = cls.__new__(cls)
        tracker._start_ts = time.monotonic()
        tracker._start_wall = report.get("run_start", _now_iso())
        tracker.events = report.get("events", [])
        tracker._phase_starts = {}

        # Restore LLM aggregate counters by replaying events
        tracker._llm_calls = 0
        tracker._llm_prompt_tokens = 0
        tracker._llm_completion_tokens = 0
        tracker._llm_total_latency = 0.0
        for evt in tracker.events:
            if evt.get("type") == EVT_LLM_CALL:
                tracker._llm_calls += 1
                tracker._llm_prompt_tokens += evt.get("prompt_tokens", 0)
                tracker._llm_completion_tokens += evt.get("completion_tokens", 0)
                tracker._llm_total_latency += evt.get("latency_s", 0.0)

        # Cache top-level aggregates from the saved report so that
        # get_summary() can fall back to them when the events list does not
        # contain the raw metric / feedback events (e.g. summary-only reports).
        saved_model_metrics = report.get("model_metrics")
        tracker._cached_model_metrics = (
            saved_model_metrics if isinstance(saved_model_metrics, dict) else None
        )

        saved_feedback_history = report.get("feedback_history")
        tracker._cached_feedback_history = (
            saved_feedback_history if isinstance(saved_feedback_history, list) else None
        )

        logger.info(
            "PipelineTracker loaded from %s (%d events)", out_path, len(tracker.events)
        )
        return tracker

    # ------------------------------------------------------------------
    # Phase tracking
    # ------------------------------------------------------------------

    def start_phase(self, phase_name: str) -> None:
        """Record the start of a named pipeline phase.

        Parameters
        ----------
        phase_name:
            Human-readable phase identifier, e.g. ``"data_cleaning"``.
        """
        ts = time.monotonic()
        self._phase_starts[phase_name] = ts
        self.events.append(
            {
                "type": EVT_PHASE_START,
                "phase": phase_name,
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        logger.debug("Phase started: %s", phase_name)

    def end_phase(
        self,
        phase_name: str,
        status: str = "success",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the end of a named pipeline phase.

        Parameters
        ----------
        phase_name:
            Same identifier used in :meth:`start_phase`.
        status:
            Outcome string, e.g. ``"success"`` or ``"failed"``.
        metrics:
            Optional dict of phase-level metrics to attach (e.g.
            ``{"rows_processed": 50000}``).
        """
        now = time.monotonic()
        start = self._phase_starts.pop(phase_name, now)
        duration = round(now - start, 4)

        self.events.append(
            {
                "type": EVT_PHASE_END,
                "phase": phase_name,
                "status": status,
                "duration_s": duration,
                "metrics": metrics or {},
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        logger.debug("Phase ended: %s (%.2f s, status=%s)", phase_name, duration, status)

    # ------------------------------------------------------------------
    # Agent actions
    # ------------------------------------------------------------------

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an agent decision or action.

        Parameters
        ----------
        agent_name:
            Name of the agent, e.g. ``"FeatureAgent"``.
        action:
            Short description of the action, e.g. ``"add_geo_cluster"``.
        details:
            Optional additional context dict.
        """
        self.events.append(
            {
                "type": EVT_AGENT_ACTION,
                "agent": agent_name,
                "action": action,
                "details": details or {},
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        logger.debug("[%s] %s — %s", agent_name, action, details)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def log_llm_call(
        self,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        model: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Log a single LLM API call.

        Parameters
        ----------
        agent_name:
            Agent that made the call.
        prompt_tokens:
            Number of tokens in the prompt.
        completion_tokens:
            Number of tokens in the completion.
        latency:
            Wall-clock time in seconds for the call.
        model:
            Optional model identifier string.
        success:
            Whether the call completed without error.
        """
        total_tokens = prompt_tokens + completion_tokens
        self._llm_calls += 1
        self._llm_prompt_tokens += prompt_tokens
        self._llm_completion_tokens += completion_tokens
        self._llm_total_latency += latency

        self.events.append(
            {
                "type": EVT_LLM_CALL,
                "agent": agent_name,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_s": round(latency, 4),
                "success": success,
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        logger.debug(
            "LLM call [%s]: %d+%d tokens, %.2f s",
            agent_name, prompt_tokens, completion_tokens, latency,
        )

    # ------------------------------------------------------------------
    # Model metrics
    # ------------------------------------------------------------------

    def log_model_metric(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        fold: Optional[int] = None,
        split: str = "validation",
    ) -> None:
        """Log a training / validation metric for a named model.

        Parameters
        ----------
        model_name:
            Model identifier, e.g. ``"lgbm"``.
        metric_name:
            Metric name, e.g. ``"mse"`` or ``"rmse"``.
        value:
            Numeric metric value.
        fold:
            Optional CV fold index (0-based).
        split:
            Data split the metric applies to: ``"train"`` or ``"validation"``.
        """
        self.events.append(
            {
                "type": EVT_MODEL_METRIC,
                "model": model_name,
                "metric": metric_name,
                "value": value,
                "fold": fold,
                "split": split,
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        fold_str = f" fold={fold}" if fold is not None else ""
        logger.debug(
            "Metric [%s]%s %s/%s = %.6f",
            model_name, fold_str, split, metric_name, value,
        )

    # ------------------------------------------------------------------
    # Memory snapshots
    # ------------------------------------------------------------------

    def log_memory_snapshot(self, label: str = "") -> Optional[float]:
        """Record current memory usage.

        Requires ``psutil`` (optional).  Falls back to ``/proc/self/status``
        on Linux.

        Parameters
        ----------
        label:
            Optional label for the snapshot (e.g. ``"after_feature_eng"``).

        Returns
        -------
        float | None
            Current RSS memory in MB, or ``None`` if unavailable.
        """
        mem_mb: Optional[float] = None
        try:
            import psutil  # type: ignore
            import os
            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / (1024 ** 2)
        except Exception:
            mem_mb = _read_proc_rss_mb()

        self.events.append(
            {
                "type": EVT_MEMORY_SNAPSHOT,
                "label": label,
                "rss_mb": mem_mb,
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        if mem_mb is not None:
            logger.debug("Memory [%s]: %.1f MB RSS", label or "snapshot", mem_mb)
        return mem_mb

    # ------------------------------------------------------------------
    # Feedback loop
    # ------------------------------------------------------------------

    def log_feedback_iteration(
        self,
        iteration: int,
        best_mse: float,
        improvements: Optional[List[str]] = None,
    ) -> None:
        """Record a feedback-loop iteration and the best MSE achieved.

        Parameters
        ----------
        iteration:
            Feedback loop iteration number (1-based).
        best_mse:
            Best validation MSE at this iteration.
        improvements:
            Optional list of improvement actions taken.
        """
        self.events.append(
            {
                "type": EVT_FEEDBACK_ITER,
                "iteration": iteration,
                "best_mse": best_mse,
                "improvements": improvements or [],
                "timestamp": _now_iso(),
                "elapsed_s": _elapsed(self._start_ts),
            }
        )
        logger.info(
            "Feedback iteration %d — best MSE: %.6f, actions: %s",
            iteration, best_mse, improvements,
        )

    # ------------------------------------------------------------------
    # Summary & reporting
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the full pipeline run.

        Returns
        -------
        dict
            Includes wall times, phase durations, LLM aggregate stats, model
            metrics keyed by model name, and the raw event list.

        Notes
        -----
        When the tracker was created via :meth:`load`, ``model_metrics`` and
        ``feedback_history`` are first rebuilt by scanning ``self.events``.
        If that scan yields empty results (because the saved report did not
        include raw metric/feedback events), the method falls back to the
        top-level values cached from the JSON file during :meth:`load`.  This
        prevents ``model_comparison.png`` and ``feedback_loop_progress.png``
        from rendering as empty "no data" placeholders after a reload.
        """
        total_elapsed = _elapsed(self._start_ts)

        # Phase durations
        phase_durations: Dict[str, float] = {}
        phase_statuses: Dict[str, str] = {}
        for evt in self.events:
            if evt["type"] == EVT_PHASE_END:
                phase_durations[evt["phase"]] = evt.get("duration_s", 0.0)
                phase_statuses[evt["phase"]] = evt.get("status", "unknown")

        # Model metrics grouped by model → metric → list of values
        model_metrics: Dict[str, Dict[str, List[float]]] = {}
        for evt in self.events:
            if evt["type"] == EVT_MODEL_METRIC:
                m = evt["model"]
                k = f"{evt['split']}/{evt['metric']}"
                model_metrics.setdefault(m, {}).setdefault(k, []).append(evt["value"])

        # Fall back to the cached top-level dict when events yield nothing
        if not model_metrics and self._cached_model_metrics:
            model_metrics = self._cached_model_metrics

        # Feedback loop history
        feedback_history = [
            {"iteration": e["iteration"], "best_mse": e["best_mse"]}
            for e in self.events
            if e["type"] == EVT_FEEDBACK_ITER
        ]

        # Fall back to the cached top-level list when events yield nothing
        if not feedback_history and self._cached_feedback_history:
            feedback_history = self._cached_feedback_history

        return {
            "run_start": self._start_wall,
            "total_elapsed_s": total_elapsed,
            "phase_durations_s": phase_durations,
            "phase_statuses": phase_statuses,
            "llm_calls": {
                "count": self._llm_calls,
                "total_prompt_tokens": self._llm_prompt_tokens,
                "total_completion_tokens": self._llm_completion_tokens,
                "avg_latency_s": (
                    round(self._llm_total_latency / self._llm_calls, 4)
                    if self._llm_calls else 0.0
                ),
            },
            "model_metrics": model_metrics,
            "feedback_history": feedback_history,
            "total_events": len(self.events),
            "events": self.events,
        }

    def save_report(self, path: str) -> None:
        """Save the execution summary as a JSON file.

        Parameters
        ----------
        path:
            File path where the JSON report will be written.
        """
        report = self.get_summary()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Pipeline report saved to %s", out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_proc_rss_mb() -> Optional[float]:
    """Read RSS memory from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return None
