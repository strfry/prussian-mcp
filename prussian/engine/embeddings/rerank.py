"""Stateless reranking helpers for entry-mode search results.

``rerank_results`` groups homonyms by desc, sends formatted entries to
the cross-encoder, and adds ``rerank_score`` / ``word_type`` to each
result.  ``build_reranker`` is a lazy factory that returns the
cross-encoder or ``None`` when the API is unavailable.
"""

import re
from typing import Any, Dict, List, Optional


def get_word_type(entry: dict) -> str:
    desc = entry.get("desc", "")
    if desc:
        match = re.match(r"^\s*(\w+)", desc)
        if match:
            return match.group(1).lower()
    return ""


def format_entry_multilang(entry: dict) -> str:
    """Format entry for reranker — no language names, just translations.

    Format: "Haus | house | namas namai | nms | dom | дом"
    """
    translations = entry.get("translations", {})
    trans_parts = []
    for lang_key in ["miks", "engl", "leit", "latt", "pols", "mask"]:
        trans = translations.get(lang_key, [])
        if trans and trans[0]:
            trans_parts.append(trans[0])
    return " | ".join(trans_parts) if trans_parts else ""


def build_reranker():
    """Lazy factory: returns a reranker object or ``None`` when unavailable."""
    try:
        from prussian_embeddings import get_reranker
        return get_reranker()
    except Exception:
        return None


def rerank_results(
    engine,
    rerank_query: str,
    results: list,
    reranker,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Rerank entry-mode search results using a cross-encoder.

    Groups homonyms by ``desc``, sends formatted entries to the reranker,
    and adds ``rerank_score`` / ``word_type`` to each result dict.
    """
    if not results or not reranker:
        return results

    entries_by_desc: Dict[str, tuple] = {}
    entries_list: List[Dict[str, Any]] = []

    for result_idx, result in enumerate(results):
        word = result.get("word", "")
        word_lower = word.lower()
        if word_lower in engine.word_to_entry:
            for entry in engine.word_to_entry[word_lower]:
                desc = entry.get("desc", "")
                entries_by_desc[desc] = (result_idx, entry)
                entries_list.append(entry)

    if not entries_list:
        return results

    combined_scores: Dict[str, float] = {}

    for batch_idx in range(0, len(entries_list), batch_size):
        batch = entries_list[batch_idx : batch_idx + batch_size]
        documents = [format_entry_multilang(e) for e in batch]
        try:
            reranked = reranker.rerank(rerank_query, documents,
                                       top_n=len(documents))
            for item in reranked:
                idx = item.get("index", 0) + batch_idx
                score = item.get("relevance_score", 0)
                if idx < len(entries_list):
                    desc = entries_list[idx].get("desc", "")
                    combined_scores[desc] = score
        except Exception as e:
            print(f"Reranking error: {e}")
            break

    result_scores: Dict[int, float] = {}
    result_best: Dict[int, Dict[str, Any]] = {}
    for desc, score in combined_scores.items():
        result_idx, entry = entries_by_desc[desc]
        if result_idx not in result_scores or score > result_scores[result_idx]:
            result_scores[result_idx] = score
            result_best[result_idx] = entry

    sorted_indices = sorted(result_scores.keys(),
                            key=lambda i: result_scores[i], reverse=True)
    reranked_out = []
    for result_idx in sorted_indices:
        if result_idx < len(results):
            result = results[result_idx].copy()
            result["rerank_score"] = result_scores[result_idx]
            result["word_type"] = get_word_type(result_best[result_idx])
            reranked_out.append(result)

    return reranked_out
