"""
Multi-Agent Pipeline using LangGraph.

Architecture:
  ┌─────────────┐
  │   START      │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  DataAgent   │ (profile + clean)
  │  + RAG ctx   │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  InputGuard  │ (validate cleaned data)
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ FeatureAgent │ (engineer features)
  │  + RAG ctx   │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  ModelAgent  │ (train models)
  │  + RAG ctx   │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  Evaluator   │ (evaluate + compare)
  │  + RAG ctx   │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  OutputGuard │ (validate predictions)
  └──────┬───────┘
         │
  ┌──────▼───────────────┐
  │  Orchestrator        │──── IMPROVE ──→ (back to FeatureAgent or ModelAgent)
  │  (feedback decision) │
  └──────┬───────────────┘
         │ ACCEPT
  ┌──────▼───────┐
  │    END       │
  └──────────────┘

Install dependencies:
  pip install langgraph omegaconf hydra-core
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from omegaconf import DictConfig, OmegaConf

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.state import CompiledStateGraph
    _LANGGRAPH_AVAILABLE = True
except ImportError as _lg_err:
    _LANGGRAPH_AVAILABLE = False
    _lg_import_error = str(_lg_err)

logger = logging.getLogger("kaggle-mas.pipeline")


# ---------------------------------------------------------------------------
# Pipeline State TypedDict
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Full pipeline state passed between LangGraph nodes."""

    # --- Config & bookkeeping ---
    config: Any                          # OmegaConf DictConfig
    data_dir: str                        # path to raw data files
    output_dir: str                      # path for outputs
    iteration: int                       # current feedback-loop iteration
    pipeline_complete: bool              # True when done
    decision: str                        # ACCEPT or IMPROVE
    next_agent: Optional[str]            # which agent to revisit
    reasoning: str                       # orchestrator reasoning text
    decision_history: List[Dict]         # list of past decisions
    improvement_plan: Dict               # hints for the target agent
    agent_timings: Dict[str, float]      # per-agent elapsed seconds
    errors: List[str]                    # non-fatal errors logged per node

    # --- Data phase ---
    train_df: Any                        # pd.DataFrame (raw training data)
    test_df: Any                         # pd.DataFrame (raw test data)
    data_profile: Dict                   # statistics from DataAgent
    cleaning_plan: Dict                  # cleaning instructions from LLM
    data_validation_issues: List[str]    # issues from InputValidator

    # --- Feature phase ---
    train_feat: Any                      # pd.DataFrame (feature matrix, train)
    test_feat: Any                       # pd.DataFrame (feature matrix, test)
    feature_names: List[str]             # column names after feature engineering
    target_series: Any                   # pd.Series (target values for train)
    test_ids: Any                        # pd.Series (test set IDs)
    feature_plan: Dict                   # feature engineering instructions from LLM
    feature_validation_issues: List[str] # issues from InputValidator (features)

    # --- Model phase ---
    models: Dict                         # trained model objects keyed by name
    oof_predictions: Any                 # np.ndarray (out-of-fold predictions)
    cv_scores: Dict                      # per-fold metrics by model name
    feature_importances: Dict            # feature importance dict
    ensemble_weights: Dict               # weights for ensemble
    ensemble_oof: Any                    # np.ndarray (ensemble OOF predictions)
    test_predictions: Dict               # per-model test predictions
    ensemble_test: Any                   # np.ndarray (ensemble test preds)
    submission_df: Any                   # pd.DataFrame ready for submission
    ensemble_cv_mse: float               # best ensemble CV MSE
    model_plan: Dict                     # model training hints from LLM

    # --- Evaluation phase ---
    evaluation_report: Dict              # structured metrics from EvaluatorAgent
    llm_interpretation: str             # LLM narrative of evaluation results
    output_validation_issues: List[str] # issues from OutputValidator

    # --- RAG context (injected per phase) ---
    rag_context_data: str
    rag_context_feature: str
    rag_context_model: str
    rag_context_evaluator: str
    rag_context_orchestrator: str


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

# Explicit mapping from agent class name to RAG context key suffix.
# Using a dict avoids brittle string manipulation (e.g. 'Agent'.replace...).
_AGENT_RAG_KEY: Dict[str, str] = {
    "DataAgent":          "data",
    "FeatureAgent":       "feature",
    "ModelAgent":         "model",
    "EvaluatorAgent":     "evaluator",
    "OrchestratorAgent":  "orchestrator",
}


def _make_rag_node(agent_name: str, rag_retriever: Any) -> Any:
    """Return a node function that injects RAG context into state."""
    ctx_key = f"rag_context_{_AGENT_RAG_KEY.get(agent_name, agent_name.lower())}"

    def _node(state: PipelineState) -> PipelineState:  # type: ignore[type-arg]
        try:
            profile = state.get("data_profile") if agent_name != "DataAgent" else None
            ctx = rag_retriever.get_context_for_agent(
                agent_name,
                data_profile=profile,
            )
            state[ctx_key] = ctx  # type: ignore[literal-required]
            logger.debug("[RAG] Retrieved context for %s (%d chars).", agent_name, len(ctx))
        except Exception as exc:
            logger.warning("[RAG] Failed to retrieve context for %s: %s", agent_name, exc)
            state[ctx_key] = ""  # type: ignore[literal-required]
        return state

    _node.__name__ = f"rag_{agent_name.lower()}"
    return _node


def _make_agent_node(agent: Any, tracker: Any) -> Any:
    """Return a node function wrapping agent.execute(state).

    After ModelAgent finishes, every per-fold validation MSE stored in
    ``state["cv_scores"]`` is forwarded to ``tracker.log_model_metric`` so
    that ``model_comparison.png`` is populated correctly.

    After OrchestratorAgent finishes, the latest entry appended to
    ``state["decision_history"]`` is forwarded to
    ``tracker.log_feedback_iteration`` so that
    ``feedback_loop_progress.png`` is populated correctly.
    """

    def _node(state: PipelineState) -> PipelineState:  # type: ignore[type-arg]
        phase = agent.name
        tracker.start_phase(phase)
        tracker.log_memory_snapshot(f"before_{phase}")
        try:
            state = agent._timed_execute(state)
            tracker.end_phase(phase, "success")
        except Exception as exc:
            logger.error("[%s] Execution failed: %s", phase, exc, exc_info=True)
            tracker.end_phase(phase, "failed", {"error": str(exc)})
            errors: List[str] = list(state.get("errors") or [])
            errors.append(f"{phase}: {exc}")
            state["errors"] = errors  # type: ignore[typeddict-unknown-key]
            state["pipeline_complete"] = True  # type: ignore[typeddict-unknown-key]
            raise
        tracker.log_memory_snapshot(f"after_{phase}")

        # ----------------------------------------------------------------
        # Forward model CV metrics to the tracker so the dashboard plots
        # can render model_comparison.png after a reload.
        # ----------------------------------------------------------------
        if phase == "ModelAgent":
            cv_scores: Dict[str, Any] = state.get("cv_scores") or {}
            for model_name, scores in cv_scores.items():
                fold_mses = scores.get("fold_mses", [])
                for fold_idx, mse_val in enumerate(fold_mses):
                    try:
                        tracker.log_model_metric(
                            model_name=model_name,
                            metric_name="mse",
                            value=float(mse_val),
                            fold=fold_idx,
                            split="validation",
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to log model metric [%s] fold %d: %s",
                            model_name, fold_idx, exc,
                        )
            # Also log ensemble MSE as a separate model entry
            ensemble_mse = state.get("ensemble_cv_mse")
            if ensemble_mse is not None:
                try:
                    tracker.log_model_metric(
                        model_name="ensemble",
                        metric_name="mse",
                        value=float(ensemble_mse),
                        fold=None,
                        split="validation",
                    )
                except Exception as exc:
                    logger.warning("Failed to log ensemble metric: %s", exc)

        # ----------------------------------------------------------------
        # Forward the latest orchestrator feedback iteration to the tracker
        # so that feedback_loop_progress.png is populated after a reload.
        # ----------------------------------------------------------------
        if phase == "OrchestratorAgent":
            decision_history: List[Dict[str, Any]] = state.get("decision_history") or []
            if decision_history:
                latest = decision_history[-1]
                ensemble_mse = latest.get("ensemble_mse")
                if ensemble_mse is not None:
                    try:
                        tracker.log_feedback_iteration(
                            iteration=int(latest.get("iteration", len(decision_history) - 1)) + 1,
                            best_mse=float(ensemble_mse),
                            improvements=(
                                [latest.get("reasoning", "")]
                                if latest.get("decision") == "IMPROVE"
                                else []
                            ),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to log feedback iteration: %s", exc
                        )

        return state

    _node.__name__ = f"agent_{agent.name.lower()}"
    return _node


def _make_input_guard_node(
    input_validator: Any,
    safety_guard: Any,
    cfg: Any,
    tracker: Any,
) -> Any:
    """Validate the cleaned data produced by DataAgent."""

    def _node(state: PipelineState) -> PipelineState:  # type: ignore[type-arg]
        tracker.start_phase("input_validation")
        issues: List[str] = []

        train_df = state.get("train_df")
        if train_df is not None:
            try:
                _valid, _issues = input_validator.validate_dataset(train_df, cfg)
                if not _valid:
                    issues.extend([f"train: {i}" for i in _issues])
            except Exception as exc:
                logger.warning("[InputGuard] validate_dataset failed: %s", exc)

        # Safety guard on the cleaning plan
        cleaning_plan = state.get("cleaning_plan", {})
        if cleaning_plan:
            try:
                text = json.dumps(cleaning_plan)
                sanitized = safety_guard.sanitize_llm_response(text)
                if sanitized != text:
                    issues.append("Cleaning plan was sanitized by SafetyGuard.")
            except Exception as exc:
                logger.warning("[InputGuard] safety scan failed: %s", exc)

        state["data_validation_issues"] = issues  # type: ignore[typeddict-unknown-key]
        if issues:
            logger.warning("[InputGuard] Issues found: %s", issues)
        else:
            logger.info("[InputGuard] Input validation passed.")
        tracker.end_phase("input_validation", "success", {"issues": len(issues)})
        return state

    return _node


def _make_feature_guard_node(
    input_validator: Any,
    cfg: Any,
    tracker: Any,
) -> Any:
    """Validate the feature matrix produced by FeatureAgent."""

    def _node(state: PipelineState) -> PipelineState:  # type: ignore[type-arg]
        tracker.start_phase("feature_validation")
        issues: List[str] = []

        train_feat = state.get("train_feat")
        if train_feat is not None:
            try:
                _valid, _issues = input_validator.validate_features(train_feat, cfg)
                if not _valid:
                    issues.extend(_issues)
            except Exception as exc:
                logger.warning("[FeatureGuard] validate_features failed: %s", exc)

        state["feature_validation_issues"] = issues  # type: ignore[typeddict-unknown-key]
        tracker.end_phase("feature_validation", "success", {"issues": len(issues)})
        return state

    return _node


def _make_output_guard_node(
    output_validator: Any,
    tracker: Any,
) -> Any:
    """Validate predictions produced by ModelAgent."""

    def _node(state: PipelineState) -> PipelineState:  # type: ignore[type-arg]
        tracker.start_phase("output_validation")
        issues: List[str] = []

        ensemble_test = state.get("ensemble_test")
        if ensemble_test is not None:
            try:
                _valid, _issues = output_validator.validate_predictions(
                    ensemble_test, state.get("config")
                )
                if not _valid:
                    issues.extend(_issues)
            except Exception as exc:
                logger.warning("[OutputGuard] validate_predictions failed: %s", exc)

        state["output_validation_issues"] = issues  # type: ignore[typeddict-unknown-key]
        if issues:
            logger.warning("[OutputGuard] Issues: %s", issues)
        else:
            logger.info("[OutputGuard] Output validation passed.")
        tracker.end_phase("output_validation", "success", {"issues": len(issues)})
        return state

    return _node


# ---------------------------------------------------------------------------
# Routing logic (conditional edge after Orchestrator node)
# ---------------------------------------------------------------------------

def _orchestrator_router(state: PipelineState) -> str:  # type: ignore[type-arg]
    """
    Inspect orchestrator decision and return the next node name.

    Returns
    -------
    str
        One of: ``"__end__"`` (accept), ``"feature_agent"`` (improve features),
        ``"model_agent"`` (improve model), ``"data_agent"`` (improve data).
    """
    if state.get("pipeline_complete", False):
        return END  # type: ignore[return-value]

    next_agent = state.get("next_agent", "")
    if next_agent == "feature_agent":
        return "feature_rag"
    if next_agent == "model_agent":
        return "model_rag"
    if next_agent == "data_agent":
        return "data_rag"

    # Fallback: accept
    return END  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline(
    cfg: DictConfig,
    llm_client: Any,
    tracker: Any,
    rag_retriever: Any,
    input_validator: Any,
    output_validator: Any,
    safety_guard: Any,
) -> "CompiledStateGraph":
    """
    Build and compile the LangGraph multi-agent pipeline.

    Parameters
    ----------
    cfg:              Full Hydra OmegaConf config.
    llm_client:       Configured :class:`~utils.llm_client.LLMClient`.
    tracker:          :class:`~monitoring.tracker.PipelineTracker` instance.
    rag_retriever:    :class:`~rag.retriever.RAGRetriever` instance.
    input_validator:  :class:`~guardrails.input_validator.InputValidator`.
    output_validator: :class:`~guardrails.output_validator.OutputValidator`.
    safety_guard:     :class:`~guardrails.safety.SafetyGuard`.

    Returns
    -------
    CompiledStateGraph
        A compiled LangGraph ``StateGraph`` ready to invoke.

    Raises
    ------
    ImportError
        If ``langgraph`` is not installed.
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            f"langgraph is required but not installed: {_lg_import_error}\n"
            "Install with: pip install langgraph"
        )

    # ------------------------------------------------------------------
    # Import agents
    # ------------------------------------------------------------------
    from agents.data_agent import DataAgent
    from agents.feature_agent import FeatureAgent
    from agents.model_agent import ModelAgent
    from agents.evaluator_agent import EvaluatorAgent
    from agents.orchestrator import OrchestratorAgent

    # Pipeline sub-config (for limits passed to guards)
    pipeline_cfg = getattr(cfg, "pipeline", cfg)

    # ------------------------------------------------------------------
    # Instantiate agents
    # ------------------------------------------------------------------
    data_agent        = DataAgent(cfg, llm_client)
    feature_agent     = FeatureAgent(cfg, llm_client)
    model_agent       = ModelAgent(cfg, llm_client)
    evaluator_agent   = EvaluatorAgent(cfg, llm_client)
    orchestrator      = OrchestratorAgent(cfg, llm_client)

    # ------------------------------------------------------------------
    # Create node functions
    # ------------------------------------------------------------------
    # RAG nodes
    data_rag_node      = _make_rag_node("DataAgent",      rag_retriever)
    feature_rag_node   = _make_rag_node("FeatureAgent",   rag_retriever)
    model_rag_node     = _make_rag_node("ModelAgent",     rag_retriever)
    evaluator_rag_node = _make_rag_node("EvaluatorAgent", rag_retriever)
    orch_rag_node      = _make_rag_node("OrchestratorAgent", rag_retriever)

    # Agent nodes
    data_node        = _make_agent_node(data_agent,      tracker)
    feature_node     = _make_agent_node(feature_agent,   tracker)
    model_node       = _make_agent_node(model_agent,     tracker)
    evaluator_node   = _make_agent_node(evaluator_agent, tracker)
    orchestrator_node = _make_agent_node(orchestrator,   tracker)

    # Guard nodes
    guardrails_cfg = getattr(cfg, "guardrails", {})
    input_guard_node   = _make_input_guard_node(
        input_validator, safety_guard, guardrails_cfg, tracker
    )
    feature_guard_node = _make_feature_guard_node(input_validator, guardrails_cfg, tracker)
    output_guard_node  = _make_output_guard_node(output_validator, tracker)

    # ------------------------------------------------------------------
    # Build the graph
    # ------------------------------------------------------------------
    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("data_rag",         data_rag_node)
    graph.add_node("data_agent",       data_node)
    graph.add_node("input_guard",      input_guard_node)
    graph.add_node("feature_rag",      feature_rag_node)
    graph.add_node("feature_agent",    feature_node)
    graph.add_node("feature_guard",    feature_guard_node)
    graph.add_node("model_rag",        model_rag_node)
    graph.add_node("model_agent",      model_node)
    graph.add_node("evaluator_rag",    evaluator_rag_node)
    graph.add_node("evaluator_agent",  evaluator_node)
    graph.add_node("output_guard",     output_guard_node)
    graph.add_node("orchestrator_rag", orch_rag_node)
    graph.add_node("orchestrator",     orchestrator_node)

    # Linear forward edges
    graph.add_edge(START,              "data_rag")
    graph.add_edge("data_rag",         "data_agent")
    graph.add_edge("data_agent",       "input_guard")
    graph.add_edge("input_guard",      "feature_rag")
    graph.add_edge("feature_rag",      "feature_agent")
    graph.add_edge("feature_agent",    "feature_guard")
    graph.add_edge("feature_guard",    "model_rag")
    graph.add_edge("model_rag",        "model_agent")
    graph.add_edge("model_agent",      "evaluator_rag")
    graph.add_edge("evaluator_rag",    "evaluator_agent")
    graph.add_edge("evaluator_agent",  "output_guard")
    graph.add_edge("output_guard",     "orchestrator_rag")
    graph.add_edge("orchestrator_rag", "orchestrator")

    # Conditional edge: orchestrator decides ACCEPT or IMPROVE
    graph.add_conditional_edges(
        "orchestrator",
        _orchestrator_router,
        {
            END:             END,
            "feature_rag":   "feature_rag",
            "model_rag":     "model_rag",
            "data_rag":      "data_rag",
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_pipeline(cfg: DictConfig) -> Dict[str, Any]:
    """
    Initialize all components, build and run the LangGraph pipeline.

    Parameters
    ----------
    cfg:
        Full Hydra/OmegaConf config as loaded by ``@hydra.main``.

    Returns
    -------
    dict
        Final pipeline state containing all results (models, predictions,
        evaluation_report, submission_df, etc.).

    Usage
    -----
    ::

        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/config.yaml")
        final_state = run_pipeline(cfg)
        submission = final_state["submission_df"]
    """
    from utils.llm_client import LLMClient
    from rag.knowledge_base import KnowledgeBase
    from rag.retriever import RAGRetriever
    from guardrails.input_validator import InputValidator
    from guardrails.output_validator import OutputValidator
    from guardrails.safety import SafetyGuard
    from monitoring.tracker import PipelineTracker

    logger.info("=== Initializing Multi-Agent Pipeline ===")

    # ------------------------------------------------------------------
    # 1. LLM Client
    # ------------------------------------------------------------------
    llm_client = LLMClient(cfg)

    # ------------------------------------------------------------------
    # 2. Monitoring tracker
    # ------------------------------------------------------------------
    tracker = PipelineTracker()

    # ------------------------------------------------------------------
    # 3. RAG
    # ------------------------------------------------------------------
    enable_rag = bool(OmegaConf.select(cfg, "pipeline.enable_rag", default=True))
    if enable_rag:
        embedding_model = OmegaConf.select(
            cfg, "rag.embedding_model", default="all-MiniLM-L6-v2",
        )
        knowledge_base = KnowledgeBase(model_name=embedding_model)
        knowledge_base.load_builtin_knowledge()
        # Also load custom knowledge files if the directory exists
        kb_path = OmegaConf.select(
            cfg, "rag.knowledge_base_path", default="./rag/knowledge_base",
        )
        if os.path.isdir(kb_path):
            knowledge_base.load_from_directory(kb_path)
        rag_retriever = RAGRetriever(knowledge_base, llm_client)
        logger.info("RAG enabled — knowledge base size: %d chunks.", len(knowledge_base))
    else:
        # Stub retriever that always returns empty strings
        class _NoopRAG:
            def get_context_for_agent(self, *a, **kw):
                return ""
            def retrieve_and_augment(self, *a, **kw):
                return ""

        rag_retriever = _NoopRAG()  # type: ignore[assignment]
        logger.info("RAG disabled.")

    # ------------------------------------------------------------------
    # 4. Guardrails
    # ------------------------------------------------------------------
    enable_guardrails = bool(
        OmegaConf.select(cfg, "pipeline.enable_guardrails", default=True)
    )
    if enable_guardrails:
        input_validator  = InputValidator()
        output_validator = OutputValidator()
        safety_guard     = SafetyGuard()
        logger.info("Guardrails enabled.")
    else:
        # Stub validators that always pass
        class _PassValidator:
            def validate_dataset(self, df, cfg):
                return True, []
            def validate_features(self, df, cfg):
                return True, []
            def validate_predictions(self, preds, cfg):
                return True, []

        class _PassSafety:
            def sanitize_llm_response(self, text):
                return text

        input_validator  = _PassValidator()   # type: ignore[assignment]
        output_validator = _PassValidator()   # type: ignore[assignment]
        safety_guard     = _PassSafety()      # type: ignore[assignment]
        logger.info("Guardrails disabled (stubs active).")

    # ------------------------------------------------------------------
    # 5. Build graph
    # ------------------------------------------------------------------
    compiled_graph = build_pipeline(
        cfg=cfg,
        llm_client=llm_client,
        tracker=tracker,
        rag_retriever=rag_retriever,
        input_validator=input_validator,
        output_validator=output_validator,
        safety_guard=safety_guard,
    )
    logger.info("LangGraph compiled successfully.")

    # ------------------------------------------------------------------
    # 6. Initial state
    # ------------------------------------------------------------------
    data_dir   = str(OmegaConf.select(cfg, "project.data_dir",   default="./data"))
    output_dir = str(OmegaConf.select(cfg, "project.output_dir", default="./outputs"))
    max_loops  = int(OmegaConf.select(cfg, "pipeline.max_feedback_loops", default=3))

    initial_state: PipelineState = {  # type: ignore[typeddict-item]
        "config":           cfg,
        "data_dir":         data_dir,
        "output_dir":       output_dir,
        "iteration":        0,
        "pipeline_complete": False,
        "decision":         "",
        "next_agent":       None,
        "reasoning":        "",
        "decision_history": [],
        "improvement_plan": {},
        "agent_timings":    {},
        "errors":           [],
    }

    # Inject max_iterations into cfg namespace expected by OrchestratorAgent
    # (it reads cfg.max_iterations or cfg.pipeline.max_feedback_loops)
    try:
        OmegaConf.update(cfg, "max_iterations", max_loops, merge=True)
    except Exception:
        pass  # read-only struct — OrchestratorAgent will fall back to its default

    # ------------------------------------------------------------------
    # 7. Run the graph
    # ------------------------------------------------------------------
    tracker.start_phase("full_pipeline")
    logger.info("Starting graph execution (max feedback loops: %d).", max_loops)

    try:
        final_state: Dict[str, Any] = compiled_graph.invoke(
            initial_state,
            config={"recursion_limit": max(100, (max_loops + 1) * 20)},
        )
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        tracker.end_phase("full_pipeline", "failed", {"error": str(exc)})
        raise

    tracker.end_phase("full_pipeline", "success")

    # ------------------------------------------------------------------
    # 8. Attach tracker to state for downstream use
    # ------------------------------------------------------------------
    final_state["tracker"] = tracker  # type: ignore[assignment]

    # Save tracker report
    os.makedirs(output_dir, exist_ok=True)
    tracker_path = os.path.join(output_dir, "pipeline_report.json")
    tracker.save_report(tracker_path)
    logger.info("Pipeline tracker report saved to %s", tracker_path)

    logger.info("=== Pipeline complete ===")
    return final_state


# ---------------------------------------------------------------------------
# Module-level __all__
# ---------------------------------------------------------------------------

__all__ = [
    "PipelineState",
    "build_pipeline",
    "run_pipeline",
]
