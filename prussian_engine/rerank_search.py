"""Reranker-enhanced search combining embedding retrieval with semantic reranking."""

import asyncio
import re
import sys
from typing import Dict, Any, List

from .search import SearchEngine
from .embedding_client import EmbeddingClient
from .config import API_KEY


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
        self.use_reranker = use_reranker
        if use_reranker:
            if not API_KEY:
                raise ValueError("API_KEY environment variable is required for reranking")
            print("Initializing reranker...", file=sys.stderr)
            self.rerank_client = EmbeddingClient()
        else:
            self.rerank_client = None
        print("Loading search engine...", file=sys.stderr)
        self.base_engine = SearchEngine()

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
        """Rerank results using reranking API - handles homonymes via desc field."""
        # Collect all homonymes, indexed by desc field for tracking
        entries_by_desc: Dict[str, tuple] = {}  # desc -> (result_idx, entry)
        entries_list: List[Dict[str, Any]] = []

        for result_idx, result in enumerate(results):
            word = result.get("word", "")
            word_lower = word.lower()
            if word_lower in self.base_engine.word_to_entry:
                for entry in self.base_engine.word_to_entry[word_lower]:
                    desc = entry.get("desc", "")
                    entries_by_desc[desc] = (result_idx, entry)
                    entries_list.append(entry)

        if not entries_list:
            return []

        combined_scores: Dict[str, float] = {}

        for batch_idx in range(0, len(entries_list), batch_size):
            batch = entries_list[batch_idx : batch_idx + batch_size]
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
                    if idx < len(entries_list):
                        desc = entries_list[idx].get("desc", "")
                        combined_scores[desc] = score
            except Exception as e:
                print(f"Reranking error: {e}")
                break

        # Group by result, keep best homonyme per word
        result_scores: Dict[int, float] = {}
        result_best: Dict[int, Dict[str, Any]] = {}
        for desc, score in combined_scores.items():
            result_idx, entry = entries_by_desc[desc]
            if result_idx not in result_scores or score > result_scores[result_idx]:
                result_scores[result_idx] = score
                result_best[result_idx] = entry

        # Sort and return
        sorted_indices = sorted(result_scores.keys(), key=lambda i: result_scores[i], reverse=True)
        reranked = []
        for result_idx in sorted_indices:
            if result_idx < len(results):
                result = results[result_idx].copy()
                result["rerank_score"] = result_scores[result_idx]
                result["word_type"] = get_word_type(result_best[result_idx])
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
