"""Reranked search with cross-encoder reranking.

- :mod:`.rerank` — ``RerankedSearchEngine`` (retrieval + cross-encoder rerank).

Embedding backends and HTTP client now sourced from prussian_embeddings package.
"""

from prussian_embeddings import get_embedder, EmbeddingClient

__all__ = ["get_embedder", "EmbeddingClient"]
