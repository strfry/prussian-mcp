"""Embedding backends for semantic search.

Two backends are supported:

- ``model2vec``  – local, CPU-only static embeddings (no API, no GPU). This is
  the default and makes the search self-contained so it can run in the cloud
  without a separate embedding server.
- ``api``        – remote OpenAI-/Jina-compatible embedding endpoint via
  :class:`~prussian.engine.embeddings.client.EmbeddingClient` (legacy behaviour).

Both backends expose the same tiny interface used by the search engine and the
embedding generation script::

    embedder.dim                      # int, embedding dimension
    embedder.get_embeddings(texts)    # -> np.ndarray (n, dim), float32, L2-normalized
    embedder.get_embedding(text)      # -> np.ndarray (dim,),   float32, L2-normalized
"""

import sys
from typing import List, Optional

import numpy as np

from prussian.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    API_KEY,
    EMBEDDING_DIM,
)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safe for zero vectors)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-10, a_max=None)
    return matrix / norms


class Model2VecEmbedder:
    """Local static embeddings via `model2vec` (CPU-only, no network at runtime).

    The model can be referenced either by a HuggingFace id
    (e.g. ``minishlab/potion-multilingual-128M``) or by a local directory that
    was produced with ``StaticModel.save_pretrained(...)``. Vendoring the model
    into the repo lets the search run fully offline.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or EMBEDDING_MODEL
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "model2vec is required for the 'model2vec' embedding backend. "
                "Install it with `pip install model2vec`."
            ) from exc

        print(f"Loading embedding model: {self.model_name}...", file=sys.stderr)
        self.model = StaticModel.from_pretrained(self.model_name)
        self.dim = int(self.model.dim)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns an (n, dim) L2-normalized float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self.model.encode(texts)
        return _l2_normalize(embeddings)

    def get_embedding(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a (dim,) L2-normalized float32 array."""
        return self.get_embeddings([text])[0]


class ApiEmbedder:
    """Remote embedding backend wrapping :class:`EmbeddingClient`."""

    def __init__(self):
        from prussian.engine.embeddings.client import EmbeddingClient

        if not API_KEY:
            raise ValueError(
                "API_KEY environment variable is required for the 'api' "
                "embedding backend"
            )
        self.client = EmbeddingClient()
        self.dim = EMBEDDING_DIM

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.client.get_embeddings(texts)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.client.get_embedding(text)


def get_embedder(backend: Optional[str] = None):
    """Return an embedder for the configured (or requested) backend."""
    backend = (backend or EMBEDDING_BACKEND or "model2vec").lower()
    if backend == "model2vec":
        return Model2VecEmbedder()
    if backend == "api":
        return ApiEmbedder()
    raise ValueError(
        f"Unknown EMBEDDING_BACKEND: {backend!r} (expected 'model2vec' or 'api')"
    )
