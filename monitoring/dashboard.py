"""
Metrics dashboard for the multi-agent ML pipeline.

Generates matplotlib figures for model comparison, feature importances,
residuals, pipeline timeline, and feedback-loop progress.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Use a non-interactive backend when running headlessly
matplotlib.use("Agg")


class MetricsDashboard:
    """Generates diagnostic plots for pipeline runs.

    All ``plot_*`` methods return a ``matplotlib.figure.Figure`` object.
    Use :meth:`generate_report` to produce and save all plots at once.

    Parameters
    ----------
    style:
        Matplotlib style string applied to all plots (default ``"seaborn-v0_8-whitegrid"``).
    figsize:
        Default figure size ``(width, height)`` in inches.
    dpi:
        Resolution for saved figures.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 120,
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        # Gracefully fall back if the requested style is unavailable
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                pass  # use matplotlib default

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def plot_model_comparison(
        self,
        tracker: Any,
        metric: str = "mse",
        split: str = "validation",
    ) -> plt.Figure:
        """Bar chart comparing mean validation MSE across models.

        Parameters
        ----------
        tracker:
            :class:`~monitoring.tracker.PipelineTracker` instance.
        metric:
            Metric name to compare (default ``"mse"``).
        split:
            Data split to use (default ``"validation"``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        summary = tracker.get_summary()
        model_metrics: Dict[str, Dict[str, List[float]]] = summary.get("model_metrics", {})

        key = f"{split}/{metric}"
        model_names: List[str] = []
        mean_vals: List[float] = []

        for model_name, metrics in model_metrics.items():
            if key in metrics and metrics[key]:
                model_names.append(model_name)
                mean_vals.append(float(np.mean(metrics[key])))

        fig, ax = plt.subplots(figsize=self.figsize)

        if not model_names:
            ax.text(0.5, 0.5, "No model metrics available.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Model Comparison (no data)")
            return fig

        # Sort by ascending metric value (lower MSE is better)
        order = np.argsort(mean_vals)
        model_names = [model_names[i] for i in order]
        mean_vals = [mean_vals[i] for i in order]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))
        bars = ax.barh(model_names, mean_vals, color=colors, edgecolor="white", height=0.5)

        # Value labels
        for bar, val in zip(bars, mean_vals):
            ax.text(
                val + max(mean_vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel(f"{split.capitalize()} {metric.upper()}", fontsize=11)
        ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        fig.tight_layout()
        logger.debug("plot_model_comparison: %d models plotted.", len(model_names))
        return fig

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
    ) -> plt.Figure:
        """Horizontal bar chart of feature importances.

        Parameters
        ----------
        importance_dict:
            Mapping of feature name → importance score.
        top_n:
            Number of top features to display.
        title:
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not importance_dict:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, "No importance data available.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return fig

        # Sort and select top N
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_n]
        features = [item[0] for item in reversed(top_items)]
        values = [item[1] for item in reversed(top_items)]

        height = max(5, len(features) * 0.35)
        fig, ax = plt.subplots(figsize=(self.figsize[0], height))

        norm_values = np.array(values, dtype=float)
        if norm_values.max() > 0:
            norm_values = norm_values / norm_values.max()

        cmap = plt.cm.Blues
        colors = cmap(0.4 + 0.5 * norm_values)
        bars = ax.barh(features, values, color=colors, edgecolor="white")

        # Value labels
        max_val = max(values) if values else 1
        for bar, val in zip(bars, values):
            ax.text(
                val + max_val * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )

        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(f"{title} (top {len(features)})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        logger.debug("plot_feature_importance: %d features plotted.", len(features))
        return fig

    # ------------------------------------------------------------------
    # Residual plot
    # ------------------------------------------------------------------

    def plot_residuals(
        self,
        y_true: Any,
        y_pred: Any,
        title: str = "Residual Plot",
    ) -> plt.Figure:
        """Scatter plot of residuals (y_true - y_pred) vs predicted values.

        Parameters
        ----------
        y_true:
            Ground-truth target values (array-like).
        y_pred:
            Model predictions (array-like).
        title:
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        residuals = y_true_arr - y_pred_arr

        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.6, self.figsize[1]))

        # Left: residuals vs predicted
        ax = axes[0]
        ax.scatter(y_pred_arr, residuals, alpha=0.4, s=15, color="steelblue", edgecolors="none")
        ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Predicted Values", fontsize=11)
        ax.set_ylabel("Residuals", fontsize=11)
        ax.set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")

        # Right: residual distribution
        ax2 = axes[1]
        ax2.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax2.axvline(0, color="red", linewidth=1.2, linestyle="--")
        ax2.set_xlabel("Residual", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold")

        mse = float(np.mean(residuals ** 2))
        fig.suptitle(f"{title}  |  MSE = {mse:.4f}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        logger.debug("plot_residuals: n=%d, MSE=%.4f", len(y_true_arr), mse)
        return fig

    # ------------------------------------------------------------------
    # Pipeline timeline (Gantt-like)
    # ------------------------------------------------------------------

    def plot_pipeline_timeline(
        self,
        tracker: Any,
        title: str = "Pipeline Phase Timeline",
    ) -> plt.Figure:
        """Gantt-like chart showing start time and duration of each phase.

        Parameters
        ----------
        tracker:
            :class:`~monitoring.tracker.PipelineTracker` instance.
        title:
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        events = tracker.events

        # Build (phase, start_elapsed, end_elapsed, status) tuples
        phase_start_map: Dict[str, float] = {}
        phase_data: List[Tuple[str, float, float, str]] = []

        for evt in events:
            if evt["type"] == "phase_start":
                phase_start_map[evt["phase"]] = evt.get("elapsed_s", 0.0)
            elif evt["type"] == "phase_end":
                phase = evt["phase"]
                start = phase_start_map.get(phase, evt.get("elapsed_s", 0.0))
                end = evt.get("elapsed_s", start)
                status = evt.get("status", "unknown")
                phase_data.append((phase, start, end, status))

        fig, ax = plt.subplots(figsize=(self.figsize[0] * 1.4, max(4, len(phase_data) * 0.7)))

        if not phase_data:
            ax.text(0.5, 0.5, "No phase timing data available.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return fig

        status_colors = {"success": "#4caf50", "failed": "#f44336", "unknown": "#9e9e9e"}
        yticks = []
        ylabels = []

        for i, (phase, start, end, status) in enumerate(phase_data):
            color = status_colors.get(status, "#9e9e9e")
            duration = max(end - start, 0.01)
            ax.barh(i, duration, left=start, height=0.5, color=color, edgecolor="white")
            ax.text(
                start + duration / 2, i, f"{duration:.1f}s",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )
            yticks.append(i)
            ylabels.append(phase)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Elapsed Time (s)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.invert_yaxis()

        # Legend
        legend_patches = [
            mpatches.Patch(color=c, label=s.capitalize())
            for s, c in status_colors.items()
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

        fig.tight_layout()
        logger.debug("plot_pipeline_timeline: %d phases plotted.", len(phase_data))
        return fig

    # ------------------------------------------------------------------
    # Feedback loop progress
    # ------------------------------------------------------------------

    def plot_feedback_loop_progress(
        self,
        tracker: Any,
        title: str = "Feedback Loop — MSE Improvement",
    ) -> plt.Figure:
        """Line chart showing MSE improvement across feedback iterations.

        Parameters
        ----------
        tracker:
            :class:`~monitoring.tracker.PipelineTracker` instance.
        title:
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        summary = tracker.get_summary()
        feedback_history = summary.get("feedback_history", [])

        fig, ax = plt.subplots(figsize=self.figsize)

        if not feedback_history:
            ax.text(0.5, 0.5, "No feedback loop data available.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return fig

        iterations = [entry["iteration"] for entry in feedback_history]
        mse_values = [entry["best_mse"] for entry in feedback_history]

        ax.plot(
            iterations, mse_values,
            marker="o", linewidth=2, color="#1565c0", markersize=7, markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.fill_between(iterations, mse_values, alpha=0.1, color="#1565c0")

        # Annotate each point
        for it, mse in zip(iterations, mse_values):
            ax.annotate(
                f"{mse:.4f}",
                (it, mse),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

        # Delta vs first iteration
        if len(mse_values) > 1:
            improvement = mse_values[0] - mse_values[-1]
            pct = improvement / mse_values[0] * 100
            ax.set_title(
                f"{title}\nTotal improvement: {improvement:.4f} ({pct:.1f}%)",
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax.set_title(title, fontsize=13, fontweight="bold")

        ax.set_xlabel("Feedback Iteration", fontsize=11)
        ax.set_ylabel("Best Validation MSE", fontsize=11)
        ax.set_xticks(iterations)
        fig.tight_layout()
        logger.debug("plot_feedback_loop_progress: %d iterations.", len(iterations))
        return fig

    # ------------------------------------------------------------------
    # Full report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        tracker: Any,
        output_dir: str,
        importance_dict: Optional[Dict[str, float]] = None,
        y_true: Optional[Any] = None,
        y_pred: Optional[Any] = None,
    ) -> List[str]:
        """Generate all available plots and save them to *output_dir*.

        Parameters
        ----------
        tracker:
            :class:`~monitoring.tracker.PipelineTracker` instance.
        output_dir:
            Directory where PNG files will be written.
        importance_dict:
            Optional feature importance dict for the feature importance plot.
        y_true:
            Optional ground-truth array for the residual plot.
        y_pred:
            Optional prediction array for the residual plot.

        Returns
        -------
        List[str]
            Paths of the saved plot files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved: List[str] = []

        def _save(fig: plt.Figure, name: str) -> None:
            fp = out_path / name
            fig.savefig(fp, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(fp))
            logger.info("Saved plot: %s", fp)

        _save(self.plot_model_comparison(tracker), "model_comparison.png")
        _save(self.plot_pipeline_timeline(tracker), "pipeline_timeline.png")
        _save(self.plot_feedback_loop_progress(tracker), "feedback_loop_progress.png")

        if importance_dict:
            _save(self.plot_feature_importance(importance_dict), "feature_importance.png")

        if y_true is not None and y_pred is not None:
            _save(self.plot_residuals(y_true, y_pred), "residuals.png")

        logger.info("Report generated: %d plots saved to %s.", len(saved), out_path)
        return saved
