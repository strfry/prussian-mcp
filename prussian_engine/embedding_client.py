"""Embedding and reranking API client."""

import httpx
import numpy as np
from typing import List, Dict, Any

from .config import (
    RERANK_API_KEY,
    RERANK_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    RERANKER_MODEL,
)


class EmbeddingClient:
    """Client for embedding and reranking APIs (e.g., Jina AI)."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        embedding_model: str = None,
        embedding_dim: int = None,
        reranker_model: str = None,
    ):
        self.api_key = api_key or RERANK_API_KEY
        self.base_url = (base_url or RERANK_BASE_URL).rstrip("/")
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.embedding_dim = embedding_dim or EMBEDDING_DIM
        self.reranker_model = reranker_model or RERANKER_MODEL

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.embedding_model,
            "input": texts,
            "normalized": True,
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/v1/embeddings", headers=headers, json=payload
            )

            if response.status_code != 200:
                raise Exception(
                    f"Embedding API error: {response.status_code} - {response.text}"
                )

            try:
                data = response.json()
            except Exception:
                raise Exception(
                    f"Embedding API returned invalid JSON: {response.text[:200]}"
                )

            embeddings = []
            for item in data["data"]:
                vec = item["embedding"]
                if not vec or not any(isinstance(v, (int, float)) and v != 0 for v in vec):
                    raise Exception(
                        f"Embedding API returned empty/null embedding vector "
                        f"(len={len(vec) if vec else 0}). "
                        f"Check if the model '{self.embedding_model}' is loaded correctly on {self.base_url}"
                    )
                embeddings.append(vec)
            return np.array(embeddings, dtype=np.float32)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        embeddings = self.get_embeddings([text])
        return embeddings[0]

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return
            return_documents: Whether to include full documents in response

        Returns:
            List of dicts with index, relevance_score, and optionally document
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/rerank", headers=headers, json=payload
            )

            if response.status_code != 200:
                raise Exception(
                    f"Rerank API error: {response.status_code} - {response.text}"
                )

            data = response.json()
            return data.get("results", [])
