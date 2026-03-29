"""
Knowledge base for the RAG module.

Stores ML/domain knowledge as vector embeddings using sentence-transformers
and FAISS for efficient similarity search.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Builtin domain knowledge chunks
# ---------------------------------------------------------------------------
BUILTIN_KNOWLEDGE: List[Dict] = [
    {
        "text": (
            "For rental property prediction, key features include location, property type, "
            "number of reviews (proxy for demand), and pricing. High review counts often "
            "correlate with popular listings."
        ),
        "metadata": {"topic": "feature_engineering", "domain": "rental"},
    },
    {
        "text": (
            "Missing values in review-related columns often indicate new listings with no "
            "reviews. Imputing with 0 is semantically correct."
        ),
        "metadata": {"topic": "missing_values", "domain": "rental"},
    },
    {
        "text": (
            "last_dt (last review date) can be converted to 'days since last review' — a "
            "strong signal for property activity. Inactive properties tend to have more idle days."
        ),
        "metadata": {"topic": "feature_engineering", "domain": "rental"},
    },
    {
        "text": (
            "Geographic clustering using lat/lon with KMeans (5-15 clusters) captures "
            "neighborhood effects. Haversine distance to city center is also informative."
        ),
        "metadata": {"topic": "feature_engineering", "domain": "geo"},
    },
    {
        "text": (
            "For text features like property names, TF-IDF with SVD dimensionality reduction "
            "(5-10 components) captures naming patterns without high dimensionality."
        ),
        "metadata": {"topic": "text_features", "domain": "nlp"},
    },
    {
        "text": (
            "Target encoding for high-cardinality categoricals (host_name) must use "
            "cross-validation to prevent leakage. Smoothing with global mean is recommended."
        ),
        "metadata": {"topic": "encoding", "domain": "feature_engineering"},
    },
    {
        "text": (
            "LightGBM typically outperforms XGBoost on tabular data with categorical features. "
            "CatBoost handles categoricals natively."
        ),
        "metadata": {"topic": "model_selection", "domain": "ml"},
    },
    {
        "text": (
            "For regression with MSE, log-transform of the target can help if it's "
            "right-skewed. Check target distribution first."
        ),
        "metadata": {"topic": "target_transform", "domain": "ml"},
    },
    {
        "text": (
            "Ensemble of LightGBM + XGBoost + CatBoost with inverse-MSE weighting typically "
            "improves results by 2-5%."
        ),
        "metadata": {"topic": "ensemble", "domain": "ml"},
    },
    {
        "text": (
            "Feature interaction terms (e.g., price * min_days, reviews * avg_rating) often "
            "capture non-linear relationships that tree models miss at low depth."
        ),
        "metadata": {"topic": "feature_engineering", "domain": "ml"},
    },
    {
        "text": (
            "Optuna with 20-30 trials is sufficient for hyperparameter tuning on Kaggle. "
            "Use pruning callbacks to save time."
        ),
        "metadata": {"topic": "hyperparameter_tuning", "domain": "ml"},
    },
    {
        "text": (
            "Cross-validation with 5 folds is standard. Use StratifiedKFold for classification "
            "or KFold for regression with shuffle=True."
        ),
        "metadata": {"topic": "cross_validation", "domain": "ml"},
    },
    {
        "text": (
            "Overfitting detection: if train MSE is much lower than validation MSE "
            "(ratio > 2), reduce model complexity or add regularization."
        ),
        "metadata": {"topic": "overfitting", "domain": "ml"},
    },
    {
        "text": (
            "For rental/housing data, location_cluster often has high predictive power. "
            "Frequency encoding preserves ordinal information."
        ),
        "metadata": {"topic": "feature_engineering", "domain": "rental"},
    },
    {
        "text": (
            "When RAM is limited (16 GB), process features in chunks, use float32 instead of "
            "float64, and delete intermediate DataFrames with gc.collect()."
        ),
        "metadata": {"topic": "memory_optimization", "domain": "engineering"},
    },
    {
        "text": (
            "Standard scaling is important for distance-based features but optional for tree "
            "models. Apply it only for potential neural network models."
        ),
        "metadata": {"topic": "preprocessing", "domain": "ml"},
    },
    {
        "text": (
            "Outlier clipping for the target variable at reasonable bounds (e.g., 0 to 365 "
            "for days) prevents extreme predictions."
        ),
        "metadata": {"topic": "outlier_handling", "domain": "ml"},
    },
    {
        "text": (
            "feature_importances from gradient boosted trees can guide feature selection. "
            "Drop features with near-zero importance to reduce noise."
        ),
        "metadata": {"topic": "feature_selection", "domain": "ml"},
    },
    {
        "text": (
            "Label encoding ordinal categoricals (room_type: shared < private < entire) "
            "before tree models preserves the semantic order and improves splits."
        ),
        "metadata": {"topic": "encoding", "domain": "feature_engineering"},
    },
    {
        "text": (
            "For Kaggle competitions, always validate that your local CV score correlates "
            "with the public leaderboard before heavy tuning to avoid overfitting the holdout."
        ),
        "metadata": {"topic": "competition_strategy", "domain": "ml"},
    },
]


class KnowledgeBase:
    """In-memory vector store for ML domain knowledge.

    Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings and FAISS
    for efficient approximate nearest-neighbour search.

    Attributes
    ----------
    model_name : str
        Name of the sentence-transformers model.
    documents : List[dict]
        List of stored document dicts (text, embedding, metadata).
    index : faiss.Index | None
        FAISS flat-L2 index built lazily on first search.
    _model : SentenceTransformer | None
        Lazy-loaded embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialise the knowledge base.

        Parameters
        ----------
        model_name:
            Sentence-transformers model identifier.  Defaults to the compact
            ``all-MiniLM-L6-v2`` (~80 MB, CPU-friendly).
        """
        self.model_name = model_name
        self.documents: List[Dict] = []
        self.index = None  # built lazily
        self._model = None  # loaded lazily
        self._embedding_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                import warnings
                logger.info("Loading sentence-transformers model: %s", self.model_name)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*position_ids.*",
                        category=FutureWarning,
                    )
                    self._model = SentenceTransformer(self.model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for KnowledgeBase. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        """Return a unit-normalised embedding for *text*."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from all stored embeddings."""
        if not self.documents:
            self.index = None
            return
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for KnowledgeBase. "
                "Install it with: pip install faiss-cpu"
            ) from exc

        embeddings = np.stack(
            [doc["embedding"] for doc in self.documents], axis=0
        ).astype(np.float32)
        dim = embeddings.shape[1]
        self._embedding_dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalised vecs = cosine sim
        self.index.add(embeddings)
        logger.debug("FAISS index rebuilt: %d vectors, dim=%d", len(self.documents), dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add a single text chunk to the knowledge base.

        Parameters
        ----------
        text:
            The document text to store.
        metadata:
            Optional key-value metadata attached to the document (e.g. topic,
            source file path).
        """
        if not text or not text.strip():
            logger.warning("Skipping empty document.")
            return

        embedding = self._embed(text)
        self.documents.append(
            {
                "text": text.strip(),
                "embedding": embedding,
                "metadata": metadata or {},
            }
        )
        # Invalidate index so it is rebuilt on next search
        self.index = None
        logger.debug("Added document (total: %d)", len(self.documents))

    def load_from_directory(self, path: str) -> int:
        """Load all ``.txt`` and ``.md`` files from *path* into the knowledge base.

        Each file is treated as a single document.

        Parameters
        ----------
        path:
            Directory to scan for knowledge files.

        Returns
        -------
        int
            Number of documents loaded.
        """
        directory = Path(path)
        if not directory.is_dir():
            logger.warning("Directory not found: %s", path)
            return 0

        loaded = 0
        for file_path in sorted(directory.glob("**/*")):
            if file_path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    self.add_document(
                        text,
                        metadata={"source": str(file_path), "filename": file_path.name},
                    )
                    loaded += 1
            except Exception as exc:
                logger.warning("Could not read %s: %s", file_path, exc)

        logger.info("Loaded %d documents from %s", loaded, path)
        return loaded

    def load_builtin_knowledge(self) -> int:
        """Load the hardcoded ML / rental-domain knowledge chunks.

        Returns
        -------
        int
            Number of chunks loaded.
        """
        logger.info("Loading %d builtin knowledge chunks…", len(BUILTIN_KNOWLEDGE))
        for chunk in BUILTIN_KNOWLEDGE:
            self.add_document(chunk["text"], metadata=chunk.get("metadata", {}))
        logger.info("Builtin knowledge loaded.")
        return len(BUILTIN_KNOWLEDGE)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search over the knowledge base.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        List[dict]
            Up to *top_k* documents sorted by descending cosine similarity.
            Each entry includes the original ``text``, ``metadata``, and a
            ``score`` field (float, higher is better).
        """
        if not self.documents:
            logger.warning("Knowledge base is empty — no results returned.")
            return []

        if self.index is None:
            self._rebuild_index()

        query_embedding = self._embed(query).reshape(1, -1)

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(score),
                }
            )

        return results

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return f"KnowledgeBase(documents={len(self.documents)}, model='{self.model_name}')"
