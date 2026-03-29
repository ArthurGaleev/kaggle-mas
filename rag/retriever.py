"""
RAG retriever — augments agent prompts with relevant knowledge-base context.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-specific query templates
# ---------------------------------------------------------------------------
_AGENT_QUERIES: Dict[str, List[str]] = {
    "DataAgent": [
        "missing value imputation strategies for rental data",
        "data cleaning outlier handling",
        "data type conversion date features",
    ],
    "FeatureAgent": [
        "feature engineering rental property prediction",
        "encoding high-cardinality categorical variables",
        "geographic clustering latitude longitude",
        "text features TF-IDF dimensionality reduction",
        "feature interaction terms non-linear relationships",
    ],
    "ModelAgent": [
        "model selection LightGBM XGBoost CatBoost tabular data",
        "hyperparameter tuning Optuna trials",
        "ensemble methods inverse-MSE weighting",
        "cross-validation folds regression",
    ],
    "EvaluatorAgent": [
        "overfitting detection train validation MSE ratio",
        "evaluation metrics regression MSE",
        "feature importance selection gradient boosting",
    ],
    "OrchestratorAgent": [
        "Kaggle competition strategy ML pipeline",
        "memory optimisation float32 gc collect",
    ],
}


class RAGRetriever:
    """Retrieves domain knowledge and augments agent prompts.

    Parameters
    ----------
    knowledge_base:
        Populated :class:`~rag.knowledge_base.KnowledgeBase` instance.
    llm_client:
        LLM client reference (kept for future summarisation use; not required
        for current functionality).
    """

    def __init__(self, knowledge_base: KnowledgeBase, llm_client=None) -> None:
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve_and_augment(self, query: str, top_k: int = 5) -> str:
        """Search the knowledge base and format results as a context block.

        Parameters
        ----------
        query:
            Free-text query describing what context is needed.
        top_k:
            Number of knowledge chunks to include.

        Returns
        -------
        str
            A formatted multi-line string that can be prepended to an agent
            prompt.  Returns an empty string if the knowledge base is empty.
        """
        if len(self.knowledge_base) == 0:
            logger.warning("Knowledge base is empty; no context retrieved.")
            return ""

        results = self.knowledge_base.search(query, top_k=top_k)
        if not results:
            return ""

        lines = ["=== Relevant Domain Knowledge ==="]
        for i, doc in enumerate(results, start=1):
            topic = doc["metadata"].get("topic", "general")
            lines.append(f"[{i}] ({topic}) {doc['text']}")
        lines.append("=== End of Domain Knowledge ===")

        context = "\n".join(lines)
        logger.debug("Retrieved %d chunks for query: %s", len(results), query[:80])
        return context

    # ------------------------------------------------------------------
    # Agent-specific context
    # ------------------------------------------------------------------

    def get_context_for_agent(
        self,
        agent_name: str,
        data_profile: Optional[Dict] = None,
        top_k_per_query: int = 3,
    ) -> str:
        """Build a rich context block tailored to a specific agent role.

        Parameters
        ----------
        agent_name:
            One of ``DataAgent``, ``FeatureAgent``, ``ModelAgent``,
            ``EvaluatorAgent``, or ``OrchestratorAgent``.
        data_profile:
            Optional dict with dataset statistics (e.g. ``n_rows``,
            ``n_cols``, ``missing_pct``, ``target_skew``).  When provided,
            extra targeted queries are added automatically.
        top_k_per_query:
            Number of chunks retrieved per query.  Results are deduplicated.

        Returns
        -------
        str
            Formatted context string ready to prepend to an agent system
            prompt.
        """
        queries = list(_AGENT_QUERIES.get(agent_name, ["ML best practices tabular data"]))

        # Supplement with data-profile-driven queries
        if data_profile:
            if data_profile.get("missing_pct", 0) > 0.1:
                queries.append("handling missing values imputation")
            if data_profile.get("target_skew", 0) > 1.0:
                queries.append("log transform skewed regression target")
            if data_profile.get("n_rows", 0) > 100_000:
                queries.append("memory optimisation large dataset float32")

        # Gather and deduplicate results
        seen_texts: set = set()
        all_results: List[Dict] = []
        for q in queries:
            for doc in self.knowledge_base.search(q, top_k=top_k_per_query):
                if doc["text"] not in seen_texts:
                    seen_texts.add(doc["text"])
                    all_results.append(doc)

        if not all_results:
            return ""

        # Sort by score descending and cap at a reasonable total
        all_results.sort(key=lambda d: d["score"], reverse=True)
        all_results = all_results[:12]  # at most 12 unique chunks per agent

        lines = [
            f"=== Domain Knowledge for {agent_name} ===",
        ]
        for i, doc in enumerate(all_results, start=1):
            topic = doc["metadata"].get("topic", "general")
            lines.append(f"[{i}] ({topic}) {doc['text']}")
        lines.append(f"=== End of Knowledge for {agent_name} ===")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RAGRetriever(kb_size={len(self.knowledge_base)}, "
            f"llm_client={'set' if self.llm_client else 'None'})"
        )
