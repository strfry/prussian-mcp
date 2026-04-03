"""Reranker-enhanced search combining embedding retrieval with semantic reranking."""

import asyncio
import re
from typing import Dict, Any, List

from .search import SearchEngine
from .embedding_client import EmbeddingClient
from .config import RERANK_API_KEY


def get_word_type(entry: dict) -> str:
    desc = entry.get("desc", "")
    if desc:
        match = re.match(r"^\s*(\w+)", desc)
        if match:
            return match.group(1).lower()
    return ""


def format_entry_multilang(entry: dict) -> str:
    """Format entry for reranker - no language names, just translations.

    Format: "Haus | house | namas namai | nms | dom | дом"
    Best format according to cross-matrix testing (62% accuracy).
    """
    translations = entry.get("translations", {})
    trans_parts = []

    for lang_key in ["miks", "engl", "leit", "latt", "pols", "mask"]:
        trans = translations.get(lang_key, [])
        if trans and trans[0]:
            trans_parts.append(trans[0])

    return " | ".join(trans_parts) if trans_parts else ""


class RerankedSearchEngine:
    """Two-stage search: embedding retrieval + reranking."""

    def __init__(self, use_reranker: bool = True):
        self.base_engine = SearchEngine()
        self.use_reranker = use_reranker
        if RERANK_API_KEY:
            self.rerank_client = EmbeddingClient()
        else:
            raise ValueError("RERANK_API_KEY environment variable is required")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_candidates: int = 100,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Search with optional reranking.

        Args:
            query: Search query in any language
            top_k: Final number of results
            rerank_candidates: Number of candidates to rerank (0 = no reranking)
            batch_size: Batch size for reranker

        Returns:
            List of results with word, de, en, score, word_type
        """
        results = self.base_engine.query(query, top_k=rerank_candidates)

        if not results or not self.use_reranker or rerank_candidates == 0:
            return results[:top_k]

        reranked = await self._rerank_results(query, results, batch_size)
        return reranked[:top_k]

    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Rerank results using reranking API."""
        entries = [self._get_entry(r["word"]) for r in results]
        entries = [e for e in entries if e]

        combined_scores: Dict[int, float] = {}

        for batch_idx in range(0, len(entries), batch_size):
            batch = entries[batch_idx : batch_idx + batch_size]
            documents = [format_entry_multilang(e) for e in batch]

            try:
                rerank_results = await self.rerank_client.rerank(
                    query=query,
                    documents=documents,
                    top_n=len(documents),
                    return_documents=False,
                )

                for item in rerank_results:
                    idx = item.get("index", 0) + batch_idx
                    score = item.get("relevance_score", 0)
                    combined_scores[idx] = combined_scores.get(idx, 0) + score
            except Exception as e:
                print(f"Reranking error: {e}")
                break

        sorted_indices = sorted(
            combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True
        )

        reranked = []
        for idx in sorted_indices:
            if idx < len(results):
                result = results[idx].copy()
                result["rerank_score"] = combined_scores[idx]
                result["word_type"] = get_word_type(entries[idx])
                reranked.append(result)

        return reranked

    def _get_entry(self, word: str) -> Dict[str, Any]:
        """Get full entry by word."""
        word_lower = word.lower()
        if word_lower in self.base_engine.word_to_entry:
            return self.base_engine.word_to_entry[word_lower]
        return {}

    def lookup(self, prussian_word: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Lookup a Prussian word (lemma or inflected form)."""
        return self.base_engine.lookup(prussian_word, fuzzy=fuzzy)

    def get_word_forms(self, lemma: str) -> Dict[str, Any]:
        """Get all declension or conjugation forms for a lemma."""
        return self.base_engine.get_word_forms(lemma)


def search_reranked(
    query: str,
    top_k: int = 10,
    rerank_candidates: int = 100,
    use_reranker: bool = True,
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for async search."""
    engine = RerankedSearchEngine(use_reranker=use_reranker)
    return asyncio.run(
        engine.search(query, top_k=top_k, rerank_candidates=rerank_candidates)
    )
