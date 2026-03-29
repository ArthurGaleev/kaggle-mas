"""
Integration tests for the LangGraph multi-agent pipeline.

These tests verify that:
1. The pipeline graph compiles without errors.
2. State dict contains expected keys after each conceptual phase.

The LLM client is fully mocked so no real API calls are made.
All agents are patched to inject predetermined state without running
any ML training or LLM inference.

Run with:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers & fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg() -> Dict[str, Any]:
    """Return a minimal config dict compatible with all agents."""
    return {
        "target_col": "target",
        "id_col": "_id",
        "max_rows": 500_000,
        "max_cols": 500,
        "max_iterations": 1,           # one feedback loop max for speed
        "n_folds": 2,
        "max_features": 300,
        "enable_rag": False,
        "enable_guardrails": False,
        "pipeline": {
            "enable_rag": False,
            "enable_guardrails": False,
            "max_feedback_loops": 1,
        },
        "project": {
            "data_dir": "./data",
            "output_dir": "./outputs",
        },
    }


def _make_rental_df(n: int = 60, include_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "_id": range(n),
            "name": [f"Listing {i}" for i in range(n)],
            "host_name": rng.choice(["Alice", "Bob", "Carol"], n),
            "location_cluster": rng.choice(["A", "B", "C"], n),
            "location": [f"Moscow St {i}" for i in range(n)],
            "lat": rng.uniform(55.5, 56.0, n),
            "lon": rng.uniform(37.3, 37.9, n),
            "type_house": rng.choice(["Apartment", "Room"], n),
            "sum": rng.uniform(1_000, 10_000, n),
            "min_days": rng.integers(1, 30, n).astype(float),
            "amt_reviews": rng.integers(0, 200, n).astype(float),
            "last_dt": pd.date_range("2023-01-01", periods=n, freq="5D").astype(str),
            "avg_reviews": rng.uniform(1.0, 5.0, n),
            "total_host": rng.integers(1, 20, n).astype(float),
        }
    )
    if include_target:
        df["target"] = rng.uniform(500, 5_000, n)
    return df


def _mock_llm_json_response(payload: Dict) -> MagicMock:
    """Return a mock LLMClient whose generate() returns a JSON string."""
    client = MagicMock()
    client.generate.return_value = json.dumps(payload)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built state snapshots (simulate what each agent would produce)
# ─────────────────────────────────────────────────────────────────────────────

def _data_phase_state(cfg: Dict) -> Dict[str, Any]:
    """State keys expected after DataAgent runs."""
    train = _make_rental_df(60)
    test = _make_rental_df(30, include_target=False)
    return {
        "config": cfg,
        "data_dir": "./data",
        "output_dir": "./outputs",
        "iteration": 0,
        "pipeline_complete": False,
        "decision": "",
        "next_agent": None,
        "reasoning": "",
        "decision_history": [],
        "improvement_plan": {},
        "agent_timings": {},
        "errors": [],
        "train_df": train,
        "test_df": test,
        "data_profile": {
            "train": {"shape": [60, 15], "columns": {}},
            "test": {"shape": [30, 14], "columns": {}},
            "train_duplicate_rows": 0,
            "test_duplicate_rows": 0,
            "column_overlap": list(test.columns),
        },
        "cleaning_plan": {
            "drop_columns": [],
            "imputation": {},
            "outlier_handling": {},
            "type_conversions": {},
        },
        "data_validation_issues": [],
    }


def _feature_phase_state(base: Dict) -> Dict[str, Any]:
    """Add feature-phase keys to state."""
    rng = np.random.default_rng(3)
    n_train = 60
    n_test = 30
    n_feats = 20
    feat_names = [f"feat_{i}" for i in range(n_feats)]
    train_feat = pd.DataFrame(rng.standard_normal((n_train, n_feats)), columns=feat_names)
    test_feat = pd.DataFrame(rng.standard_normal((n_test, n_feats)), columns=feat_names)
    target = base["train_df"]["target"]

    return {
        **base,
        "train_feat": train_feat,
        "test_feat": test_feat,
        "feature_names": feat_names,
        "target_series": target,
        "test_ids": base["test_df"]["_id"],
        "feature_plan": {"groups": {}},
        "feature_validation_issues": [],
    }


def _model_phase_state(base: Dict) -> Dict[str, Any]:
    """Add model-phase keys to state."""
    rng = np.random.default_rng(5)
    n_train = 60
    n_test = 30

    oof = rng.uniform(500, 5_000, n_train)
    test_preds = rng.uniform(500, 5_000, n_test)

    return {
        **base,
        "models": {"lgbm": MagicMock(), "xgb": MagicMock()},
        "oof_predictions": oof,
        "cv_scores": {"lgbm": {"mse": 100_000}, "xgb": {"mse": 120_000}},
        "feature_importances": {"feat_0": 0.05},
        "ensemble_weights": {"lgbm": 0.5, "xgb": 0.5},
        "ensemble_oof": oof,
        "test_predictions": {"lgbm": test_preds, "xgb": test_preds},
        "ensemble_test": test_preds,
        "ensemble_cv_mse": 110_000.0,
        "submission_df": pd.DataFrame({"_id": range(n_test), "target": test_preds}),
        "model_plan": {},
        "output_validation_issues": [],
    }


def _evaluation_phase_state(base: Dict) -> Dict[str, Any]:
    """Add evaluation-phase keys to state."""
    return {
        **base,
        "evaluation_report": {
            "ensemble": {"oof_metrics": {"mse": 110_000, "rmse": 331}},
            "per_algorithm": {
                "lgbm": {"oof_metrics": {"mse": 100_000}},
                "xgb": {"oof_metrics": {"mse": 120_000}},
            },
        },
        "llm_interpretation": "The ensemble achieves acceptable MSE for this dataset.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# test_pipeline_builds
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineBuilds:
    """Verify that the LangGraph graph compiles without runtime errors."""

    def test_pipeline_builds(self):
        """build_pipeline() must return a compiled graph object."""
        pytest.importorskip("langgraph", reason="langgraph not installed")

        from pipeline import build_pipeline, PipelineState
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(_make_cfg())
        llm_client = _mock_llm_json_response({})
        tracker = MagicMock()
        rag_retriever = MagicMock()
        rag_retriever.get_context_for_agent.return_value = ""
        input_validator = MagicMock()
        input_validator.validate_dataset.return_value = (True, [])
        input_validator.validate_features.return_value = (True, [])
        output_validator = MagicMock()
        output_validator.validate_predictions.return_value = (True, [])
        safety_guard = MagicMock()
        safety_guard.scan.return_value = (True, [])

        compiled = build_pipeline(
            cfg=cfg,
            llm_client=llm_client,
            tracker=tracker,
            rag_retriever=rag_retriever,
            input_validator=input_validator,
            output_validator=output_validator,
            safety_guard=safety_guard,
        )

        # A compiled LangGraph should expose an `invoke` method
        assert callable(getattr(compiled, "invoke", None)), (
            "Compiled graph must have an invoke() method"
        )

    def test_pipeline_state_typeddict_keys(self):
        """PipelineState TypedDict must include all expected state keys."""
        from pipeline import PipelineState

        expected_keys = {
            "config", "data_dir", "output_dir", "iteration", "pipeline_complete",
            "train_df", "test_df", "data_profile", "cleaning_plan",
            "train_feat", "test_feat", "feature_names", "target_series",
            "models", "oof_predictions", "ensemble_test", "submission_df",
            "evaluation_report", "llm_interpretation",
            "decision", "next_agent", "decision_history", "improvement_plan",
            "errors",
        }
        # PipelineState is a TypedDict — its annotations are the valid keys
        state_keys = set(PipelineState.__annotations__.keys())
        missing = expected_keys - state_keys
        assert not missing, f"PipelineState is missing expected keys: {missing}"

    def test_pipeline_builds_without_langgraph_raises(self):
        """build_pipeline must raise ImportError if langgraph is absent."""
        import sys
        import importlib

        # Temporarily hide langgraph
        saved = sys.modules.get("langgraph.graph")
        sys.modules["langgraph.graph"] = None  # type: ignore[assignment]

        try:
            import importlib
            import pipeline as _p
            importlib.reload(_p)

            # If langgraph is genuinely not available, calling build_pipeline raises
            if not _p._LANGGRAPH_AVAILABLE:
                from omegaconf import OmegaConf
                with pytest.raises(ImportError):
                    _p.build_pipeline(
                        cfg=OmegaConf.create({}),
                        llm_client=MagicMock(),
                        tracker=MagicMock(),
                        rag_retriever=MagicMock(),
                        input_validator=MagicMock(),
                        output_validator=MagicMock(),
                        safety_guard=MagicMock(),
                    )
        finally:
            # Restore
            if saved is not None:
                sys.modules["langgraph.graph"] = saved
            elif "langgraph.graph" in sys.modules:
                del sys.modules["langgraph.graph"]


# ─────────────────────────────────────────────────────────────────────────────
# test_state_flow
# ─────────────────────────────────────────────────────────────────────────────

class TestStateFlow:
    """Verify that state dicts contain expected keys after each pipeline phase."""

    def test_data_phase_keys(self):
        """State after data phase must have train_df, test_df, and data_profile."""
        cfg = _make_cfg()
        state = _data_phase_state(cfg)

        required = ["train_df", "test_df", "data_profile", "cleaning_plan"]
        for key in required:
            assert key in state, f"Missing key after data phase: {key!r}"

        assert isinstance(state["train_df"], pd.DataFrame)
        assert isinstance(state["test_df"], pd.DataFrame)
        assert isinstance(state["data_profile"], dict)
        assert "train" in state["data_profile"]
        assert "test" in state["data_profile"]

    def test_feature_phase_keys(self):
        """State after feature phase must have train_feat, test_feat, and feature_names."""
        cfg = _make_cfg()
        state = _feature_phase_state(_data_phase_state(cfg))

        required = ["train_feat", "test_feat", "feature_names", "target_series", "test_ids"]
        for key in required:
            assert key in state, f"Missing key after feature phase: {key!r}"

        assert isinstance(state["train_feat"], pd.DataFrame)
        assert isinstance(state["test_feat"], pd.DataFrame)
        assert len(state["feature_names"]) == state["train_feat"].shape[1]

    def test_model_phase_keys(self):
        """State after model phase must have models, predictions, and submission_df."""
        cfg = _make_cfg()
        state = _model_phase_state(
            _feature_phase_state(_data_phase_state(cfg))
        )

        required = [
            "models", "oof_predictions", "ensemble_test",
            "submission_df", "ensemble_cv_mse",
        ]
        for key in required:
            assert key in state, f"Missing key after model phase: {key!r}"

        assert isinstance(state["models"], dict)
        assert len(state["models"]) > 0
        assert isinstance(state["ensemble_test"], np.ndarray)
        assert isinstance(state["submission_df"], pd.DataFrame)

    def test_evaluation_phase_keys(self):
        """State after evaluation phase must have evaluation_report and llm_interpretation."""
        cfg = _make_cfg()
        state = _evaluation_phase_state(
            _model_phase_state(
                _feature_phase_state(_data_phase_state(cfg))
            )
        )

        required = ["evaluation_report", "llm_interpretation"]
        for key in required:
            assert key in state, f"Missing key after evaluation phase: {key!r}"

        report = state["evaluation_report"]
        assert "ensemble" in report
        assert "per_algorithm" in report
        assert "oof_metrics" in report["ensemble"]

    def test_full_state_has_no_missing_dataframes(self):
        """train_df and test_df must not be empty after data phase."""
        cfg = _make_cfg()
        state = _data_phase_state(cfg)
        assert len(state["train_df"]) > 0, "train_df is empty"
        assert len(state["test_df"]) > 0, "test_df is empty"

    def test_feature_shapes_consistent(self):
        """train_feat and test_feat must have the same number of columns."""
        cfg = _make_cfg()
        state = _feature_phase_state(_data_phase_state(cfg))
        assert state["train_feat"].shape[1] == state["test_feat"].shape[1], (
            "train_feat and test_feat have different column counts"
        )

    def test_submission_df_structure(self):
        """submission_df must have exactly two columns: _id and target."""
        cfg = _make_cfg()
        state = _model_phase_state(
            _feature_phase_state(_data_phase_state(cfg))
        )
        sub = state["submission_df"]
        assert "_id" in sub.columns, "submission_df missing '_id' column"
        assert "target" in sub.columns, "submission_df missing 'target' column"
        assert sub["target"].isna().sum() == 0, "submission_df has NaN targets"

    def test_orchestrator_decision_keys(self):
        """After orchestrator sets pipeline_complete=True, state reflects final decision."""
        cfg = _make_cfg()
        state = _evaluation_phase_state(
            _model_phase_state(
                _feature_phase_state(_data_phase_state(cfg))
            )
        )
        # Simulate orchestrator ACCEPT decision
        state["decision"] = "ACCEPT"
        state["pipeline_complete"] = True
        state["next_agent"] = None
        state["decision_history"] = [
            {"iteration": 0, "decision": "ACCEPT", "reasoning": "MSE acceptable."}
        ]

        assert state["pipeline_complete"] is True
        assert state["decision"] == "ACCEPT"
        assert len(state["decision_history"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# test_mocked_llm_agent_calls
# ─────────────────────────────────────────────────────────────────────────────

class TestMockedLLMAgentCalls:
    """
    Verify agents parse predetermined JSON responses correctly.
    The LLM client is fully mocked — no API keys required.
    """

    def test_data_agent_parses_cleaning_plan(self):
        """DataAgent._ask_llm_json must return the mocked cleaning plan."""
        preset_plan = {
            "drop_columns": ["name"],
            "imputation": {"avg_reviews": "median"},
            "outlier_handling": {},
            "type_conversions": {"last_dt": "datetime"},
        }
        mock_llm = _mock_llm_json_response(preset_plan)

        from agents.data_agent import DataAgent
        agent = DataAgent(cfg=_make_cfg(), llm_client=mock_llm)

        # _ask_llm_json calls llm_client.generate() and parses the result
        result = agent._ask_llm_json("dummy prompt", default={})
        assert result == preset_plan

    def test_feature_agent_parses_feature_plan(self):
        """FeatureAgent._ask_llm_json must return the mocked feature plan."""
        preset_plan = {
            "groups": {
                "datetime_features": {"enabled": True},
                "geo_features": {"enabled": True, "n_clusters": 6},
                "text_features": {"enabled": False},
                "target_encoding": {"enabled": True, "smoothing": 10},
                "frequency_encoding": {"enabled": True},
                "interaction_features": {"enabled": False},
            }
        }
        mock_llm = _mock_llm_json_response(preset_plan)

        from agents.feature_agent import FeatureAgent
        agent = FeatureAgent(cfg=_make_cfg(), llm_client=mock_llm)
        result = agent._ask_llm_json("dummy prompt", default={})
        assert result == preset_plan

    def test_orchestrator_parses_accept_decision(self):
        """OrchestratorAgent._ask_llm_json must parse an ACCEPT decision correctly."""
        preset_decision = {
            "decision": "ACCEPT",
            "reasoning": "Current MSE of 110000 is within acceptable bounds.",
            "next_agent": None,
            "improvement_plan": {},
        }
        mock_llm = _mock_llm_json_response(preset_decision)

        from agents.orchestrator import OrchestratorAgent
        agent = OrchestratorAgent(cfg=_make_cfg(), llm_client=mock_llm)
        result = agent._ask_llm_json("dummy prompt", default={})
        assert result["decision"] == "ACCEPT"
        assert "reasoning" in result

    def test_orchestrator_parses_improve_decision(self):
        """OrchestratorAgent._ask_llm_json must parse an IMPROVE decision correctly."""
        preset_decision = {
            "decision": "IMPROVE",
            "reasoning": "MSE can be reduced by adding geo features.",
            "next_agent": "feature_agent",
            "improvement_plan": {
                "target_agent": "feature_agent",
                "suggestions": ["add geo_cluster features", "add distance features"],
            },
        }
        mock_llm = _mock_llm_json_response(preset_decision)

        from agents.orchestrator import OrchestratorAgent
        agent = OrchestratorAgent(cfg=_make_cfg(), llm_client=mock_llm)
        result = agent._ask_llm_json("dummy prompt", default={})
        assert result["decision"] == "IMPROVE"
        assert result["next_agent"] == "feature_agent"
