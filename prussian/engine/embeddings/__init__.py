"""Embedding backends, HTTP client, and reranked search.

- :mod:`.backend` — ``get_embedder`` factory (local model2vec / remote API).
- :mod:`.client` — ``EmbeddingClient`` (embedding + rerank HTTP endpoints).
- :mod:`.rerank` — ``RerankedSearchEngine`` (retrieval + cross-encoder rerank).
"""

from .backend import get_embedder
from .client import EmbeddingClient

__all__ = ["get_embedder", "EmbeddingClient"]
