"""
OrchestratorAgent — manages the iterative improvement feedback loop.

After each evaluation the orchestrator asks the LLM to decide whether to
ACCEPT the current results or IMPROVE with specific targeted actions. It
then routes the pipeline back to the appropriate agent with an improvement
plan injected into state, and enforces a hard iteration cap to prevent
infinite loops.

Threshold behaviour:
  The target MSE threshold is read from ``cfg.pipeline.target_mse_threshold``
  (default 1500.0, defined in configs/config.yaml).  When the ensemble OOF
  MSE is already at or below this value the orchestrator ACCEPT-rules are
  calibrated accordingly and the LLM is explicitly told the target is met.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from agents.base import BaseAgent
from utils.helpers import safe_json_parse


# Sentinel strings used by the LLM decision
_ACCEPT = "ACCEPT"
_IMPROVE = "IMPROVE"

# Agents that can be targeted for improvement
_IMPROVABLE_AGENTS = ("feature_agent", "model_agent", "data_agent")


class OrchestratorAgent(BaseAgent):
    """
    Feedback-loop controller for the multi-agent ML pipeline.

    Responsibilities:
    1. After EvaluatorAgent runs, ask the LLM whether results are acceptable.
    2. If IMPROVE: translate LLM reasoning into a structured improvement plan
       and inject it into state so the next agent can act on it.
    3. Route state to the correct agent (feature / model / data).
    4. Track iteration count and enforce cfg.max_iterations.
    5. Log all decisions and reasoning transparently.

    Expected state keys consumed:
        evaluation_report   (dict):  structured metrics from EvaluatorAgent.
        llm_interpretation  (str):   LLM interpretation from EvaluatorAgent.
        iteration           (int):   current loop iteration (0-indexed).
        decision_history    (list):  list of past decision dicts.

    State keys produced / updated:
        decision            (str):   "ACCEPT" or "IMPROVE".
        next_agent          (str):   which agent to run next (on IMPROVE).
        improvement_plan    (dict):  agent-specific hints for improvement.
        iteration           (int):   incremented iteration counter.
        decision_history    (list):  appended with latest decision record.
        pipeline_complete   (bool):  True when ACCEPT or max iterations reached.
    """

    SYSTEM_PROMPT = (
        "You are the lead ML engineer coordinating a Kaggle rental-property "
        "regression competition (MSE metric). You review evaluation results "
        "and decide whether to accept the current solution or specify targeted "
        "improvements. You are pragmatic: you only iterate when there is a "
        "realistic chance of a meaningful MSE reduction."
    )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _target_mse_threshold(self) -> float:
        """Return the target MSE threshold from config (pipeline.target_mse_threshold)."""
        return float(
            OmegaConf.select(self.cfg, "pipeline.target_mse_threshold", default=1500.0)
        )

    # ------------------------------------------------------------------
    # LLM decision request
    # ------------------------------------------------------------------

    def _request_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the LLM to decide ACCEPT or IMPROVE, and if IMPROVE, what to change.
        Returns a structured decision dict.
        """
        report = state.get("evaluation_report", {})
        interpretation = state.get("llm_interpretation", "")
        iteration = state.get("iteration", 0)
        max_iter = int(OmegaConf.select(self.cfg, "max_iterations", default=3))
        threshold = self._target_mse_threshold()
        history_summary = self._format_history(state.get("decision_history", []))

        ens = report.get("ensemble", {}).get("oof_metrics", {})
        per_algo = report.get("per_algorithm", {})
        best_algo = min(per_algo, key=lambda a: per_algo[a]["oof_metrics"]["mse"], default="N/A")
        best_mse = per_algo.get(best_algo, {}).get("oof_metrics", {}).get("mse", "N/A")

        current_mse = ens.get("mse", float("inf"))
        threshold_met = isinstance(current_mse, (int, float)) and current_mse <= threshold
        threshold_status = (
            f"TARGET MET (MSE={current_mse:.2f} <= threshold={threshold:.2f})"
            if threshold_met
            else f"target not yet met (MSE={current_mse} vs threshold={threshold:.2f})"
        )

        prompt = f"""
## Iteration {iteration + 1} / {max_iter}

### Current ensemble OOF metrics
MSE={ens.get('mse', 'N/A')}, RMSE={ens.get('rmse', 'N/A')}, R\u00b2={ens.get('r2', 'N/A')}

### Best single algorithm: {best_algo} (MSE={best_mse})

### Competition threshold: {threshold:.2f}  —  {threshold_status}

### EvaluatorAgent interpretation
{interpretation}

### Decision history (previous iterations)
{history_summary}

## Instructions
Decide whether to ACCEPT the current results or to IMPROVE the pipeline.

Rules:
- ACCEPT if: (a) MSE improvement from the last iteration was < 2%  OR
                 ensemble MSE is already <= competition threshold ({threshold:.2f}), OR
              (b) this is the last allowed iteration ({max_iter}), OR
              (c) results are already strong (e.g. R\u00b2 > 0.90).
- IMPROVE only if there is a specific, achievable change with realistic impact.
  When suggesting stacking, set model_hints to include the word 'stacking'.

If IMPROVE, specify which agent to re-run and exactly what to change.

Respond in **strict JSON only** (no markdown):
{{
  "decision": "ACCEPT" | "IMPROVE",
  "reasoning": "<one or two sentences>",
  "next_agent": "feature_agent" | "model_agent" | "data_agent",   // only if IMPROVE
  "improvement_plan": {{
    "summary": "<brief description of changes>",
    "feature_hints": "<if targeting feature_agent: e.g. add polynomial features for sum and min_days>",
    "model_hints":   "<if targeting model_agent: e.g. increase n_estimators, add CatBoost, try stacking>",
    "data_hints":    "<if targeting data_agent: e.g. revisit outlier thresholds for target>"
  }}
}}
Respond with JSON only.
"""
        default: Dict[str, Any] = {
            "decision": _ACCEPT,
            "reasoning": "Defaulting to ACCEPT (LLM parse failure).",
            "next_agent": None,
            "improvement_plan": {},
        }
        return self._ask_llm_json(prompt, default=default)

    # ------------------------------------------------------------------
    # History formatter
    # ------------------------------------------------------------------

    @staticmethod
    def _format_history(history: List[Dict[str, Any]]) -> str:
        """Produce a compact text summary of past decisions."""
        if not history:
            return "No previous iterations."
        lines = []
        for i, record in enumerate(history):
            lines.append(
                f"Iter {i+1}: decision={record.get('decision')} | "
                f"ensemble_mse={record.get('ensemble_mse', 'N/A')} | "
                f"target_agent={record.get('next_agent', 'N/A')} | "
                f"reason={record.get('reasoning', '')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State routing helpers
    # ------------------------------------------------------------------

    def _inject_improvement_plan(
        self, state: Dict[str, Any], plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge the LLM improvement plan into state so the target agent can
        inspect it and adjust its behaviour accordingly.
        """
        state["improvement_plan"] = plan
        return state

    def _clear_stale_results(
        self, state: Dict[str, Any], next_agent: str
    ) -> Dict[str, Any]:
        """
        Remove downstream results that will be regenerated on the next run.
        This prevents stale data from bleeding into the new evaluation.
        """
        keys_to_clear: Dict[str, List[str]] = {
            "data_agent": [
                "train_df", "test_df", "data_profile", "cleaning_plan",
                "train_feat", "test_feat", "feature_names", "feature_plan",
                "target_series", "test_ids",
                "models", "oof_predictions", "cv_scores", "feature_importances",
                "ensemble_weights", "ensemble_oof", "test_predictions",
                "ensemble_test", "submission_df", "ensemble_cv_mse", "model_plan",
                "evaluation_report", "llm_interpretation", "stacking_meta",
            ],
            "feature_agent": [
                "train_feat", "test_feat", "feature_names", "feature_plan",
                "target_series", "test_ids",
                "models", "oof_predictions", "cv_scores", "feature_importances",
                "ensemble_weights", "ensemble_oof", "test_predictions",
                "ensemble_test", "submission_df", "ensemble_cv_mse", "model_plan",
                "evaluation_report", "llm_interpretation", "stacking_meta",
            ],
            "model_agent": [
                "models", "oof_predictions", "cv_scores", "feature_importances",
                "ensemble_weights", "ensemble_oof", "test_predictions",
                "ensemble_test", "submission_df", "ensemble_cv_mse", "model_plan",
                "evaluation_report", "llm_interpretation", "stacking_meta",
            ],
        }
        for key in keys_to_clear.get(next_agent, []):
            state.pop(key, None)
        return state

    # ------------------------------------------------------------------
    # Full pipeline runner (convenience method)
    # ------------------------------------------------------------------

    def run_pipeline(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete multi-agent pipeline with the feedback loop.

        .. warning::
            **Standalone convenience path only.**  This method bypasses the
            LangGraph topology entirely and wires agents together with direct
            Python calls.  It exists solely for quick local testing and
            notebook experiments where LangGraph is not available or not
            desired.  Any structural change to the graph (new nodes, edges,
            guard rails, RAG injection, conditional routing) must be made in
            ``pipeline.build_pipeline()`` — this method will *not* reflect
            those changes automatically.  Do **not** use ``run_pipeline()``
            as the authoritative execution path in production.

        This convenience method orchestrates DataAgent → FeatureAgent →
        ModelAgent → EvaluatorAgent → (OrchestratorAgent feedback loop).

        Args:
            initial_state: Initial state dict (must include 'data_dir').

        Returns:
            Final state after pipeline completion.
        """
        from agents.data_agent import DataAgent
        from agents.feature_agent import FeatureAgent
        from agents.model_agent import ModelAgent
        from agents.evaluator_agent import EvaluatorAgent

        max_iter = int(OmegaConf.select(self.cfg, "max_iterations", default=3))
        state = dict(initial_state)
        state.setdefault("iteration", 0)
        state.setdefault("decision_history", [])
        state.setdefault("pipeline_complete", False)

        data_agent = DataAgent(self.cfg, self.llm_client, self.logger)
        feature_agent = FeatureAgent(self.cfg, self.llm_client, self.logger)
        model_agent = ModelAgent(self.cfg, self.llm_client, self.logger)
        evaluator_agent = EvaluatorAgent(self.cfg, self.llm_client, self.logger)

        self._log("Starting initial pipeline run\u2026")
        state = data_agent._timed_execute(state)
        state = feature_agent._timed_execute(state)
        state = model_agent._timed_execute(state)
        state = evaluator_agent._timed_execute(state)

        while state["iteration"] < max_iter:
            state = self._timed_execute(state)

            if state.get("pipeline_complete"):
                self._log("Pipeline complete — exiting feedback loop.")
                break

            next_agent = state.get("next_agent")
            self._log(f"Re-running pipeline from: {next_agent}")

            if next_agent == "data_agent":
                state = data_agent._timed_execute(state)
                state = feature_agent._timed_execute(state)
            elif next_agent == "feature_agent":
                state = feature_agent._timed_execute(state)
            elif next_agent == "model_agent":
                pass
            else:
                self._log(f"Unknown next_agent '{next_agent}'; stopping.", level="warning")
                break

            state = model_agent._timed_execute(state)
            state = evaluator_agent._timed_execute(state)

        else:
            self._log(f"Maximum iterations ({max_iter}) reached. Accepting current results.")
            state["pipeline_complete"] = True
            state["decision"] = _ACCEPT

        self._log("Pipeline finished.")
        return state

    # ------------------------------------------------------------------
    # Main entry point (single iteration decision)
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single ACCEPT / IMPROVE decision for the current iteration.

        This is used by LangGraph when each agent is a node. For the full
        pipeline loop use run_pipeline().
        """
        iteration = state.get("iteration", 0)
        max_iter = int(OmegaConf.select(self.cfg, "max_iterations", default=3))
        threshold = self._target_mse_threshold()

        # Hard limit check
        if iteration >= max_iter:
            self._log(f"Max iterations ({max_iter}) reached. Forcing ACCEPT.")
            state["decision"] = _ACCEPT
            state["pipeline_complete"] = True
            state["reasoning"] = f"Hard limit of {max_iter} iterations reached."
            state["next_agent"] = None
            state["improvement_plan"] = {}
            return state

        # Early-exit if threshold already met (no LLM call needed)
        current_mse = (
            state.get("evaluation_report", {})
                 .get("ensemble", {})
                 .get("oof_metrics", {})
                 .get("mse", float("inf"))
        )
        if isinstance(current_mse, (int, float)) and current_mse <= threshold:
            self._log(
                f"MSE={current_mse:.2f} already meets threshold={threshold:.2f}. "
                "Auto-ACCEPT without LLM call."
            )
            state["decision"] = _ACCEPT
            state["pipeline_complete"] = True
            state["reasoning"] = (
                f"MSE={current_mse:.2f} <= target threshold={threshold:.2f}. "
                "Auto-accepted."
            )
            state["next_agent"] = None
            state["improvement_plan"] = {}
            state["iteration"] = iteration + 1
            history: List[Dict[str, Any]] = state.get("decision_history", [])
            history.append({
                "iteration": iteration,
                "decision": _ACCEPT,
                "reasoning": state["reasoning"],
                "next_agent": None,
                "ensemble_mse": current_mse,
            })
            state["decision_history"] = history
            return state

        # --- Ask LLM ---
        decision_dict = self._request_decision(state)
        decision = decision_dict.get("decision", _ACCEPT).upper()
        reasoning = decision_dict.get("reasoning", "")
        next_agent = decision_dict.get("next_agent")
        improvement_plan = decision_dict.get("improvement_plan", {})

        self._log(f"LLM decision: {decision} | reasoning: {reasoning}")

        # Validate next_agent
        if decision == _IMPROVE and next_agent not in _IMPROVABLE_AGENTS:
            self._log(
                f"LLM suggested invalid next_agent='{next_agent}'. "
                "Defaulting to 'model_agent'.",
                level="warning",
            )
            next_agent = "model_agent"

        # Record history
        history_entry = {
            "iteration": iteration,
            "decision": decision,
            "reasoning": reasoning,
            "next_agent": next_agent if decision == _IMPROVE else None,
            "ensemble_mse": current_mse,
        }
        history = state.get("decision_history", [])
        history.append(history_entry)

        # Update state
        state["decision"] = decision
        state["reasoning"] = reasoning
        state["decision_history"] = history
        state["iteration"] = iteration + 1

        if decision == _ACCEPT:
            state["pipeline_complete"] = True
            state["next_agent"] = None
            state["improvement_plan"] = {}
            self._log("Decision: ACCEPT. Pipeline complete.")
        else:
            state["pipeline_complete"] = False
            state["next_agent"] = next_agent
            state = self._inject_improvement_plan(state, improvement_plan)
            state = self._clear_stale_results(state, next_agent)
            self._log(
                f"Decision: IMPROVE \u2192 {next_agent}. "
                f"Plan: {improvement_plan.get('summary', 'no summary')}"
            )

        return state
