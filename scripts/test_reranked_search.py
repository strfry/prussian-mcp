#!/usr/bin/env python3
"""Test reranked search."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prussian_engine.rerank_search import RerankedSearchEngine, search_reranked


def test_sync_search():
    """Test synchronous wrapper."""
    print("Testing synchronous search_reranked...")
    results = search_reranked(
        "family Familie šeima ģimene", top_k=10, rerank_candidates=100
    )

    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(
            f"{i:2d}. {r['word']:15s} | {r['de'][:30]:30s} | rerank: {r.get('rerank_score', 0):.4f}"
        )


if __name__ == "__main__":
    test_sync_search()
