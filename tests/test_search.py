"""Tests for the single (chunk) search path: query, hybrid toggle, context
rerank, dictionary-sourced enrichment.  Fake store/embedder/reranker — no
model downloads.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np


def _fake_chunk_records():
    return [
        {"lemma": "bērzi", "members": ["bērzi", "berzin"], "pos": "n",
         "text": "bērzi: Birke\nberzin: Birke (Akk)"},
        {"lemma": "dēiws", "members": ["dēiws", "deiwan"], "pos": "n",
         "text": "dēiws: Gott\ndeiwan: Gott (Akk)"},
        {"lemma": "wīdātun", "members": ["wīdātun", "widdai"], "pos": "v",
         "text": "wīdātun: sehen\nwiddai: sehen (Inf)"},
    ]


def _fake_dictionary():
    return [
        {"word": "bērzi", "gender": "fem", "translations": {"de": ["Birke"]},
         "desc": "n [bērzi]", "forms": {}},
        {"word": "dēiws", "gender": "masc", "translations": {"de": ["Gott"]},
         "desc": "n [dēiws]", "forms": {}},
        {"word": "wīdātun", "gender": "", "translations": {"de": ["sehen"]},
         "desc": "v [widdai 113]", "forms": {}},
    ]


def _make_engine(hybrid=True):
    """Build a SearchEngine bypassing disk __init__."""
    from prussian.engine.search import SearchEngine

    engine = SearchEngine.__new__(SearchEngine)
    engine.word_to_entry = {}
    engine.form_to_lemma = {}
    engine.form_to_pgr = {}

    records = _fake_chunk_records()
    engine.store = MagicMock()
    engine.store.records = records
    engine.store.embeddings = np.random.randn(len(records), 8).astype(np.float32)
    engine.store.query = MagicMock(
        return_value=[(r, 0.9 - i * 0.1) for i, r in enumerate(records)]
    )
    engine.store.top_k = MagicMock(
        return_value=[(i, 0.9 - i * 0.1) for i in range(len(records))]
    )

    engine.bm25 = None
    if hybrid:
        engine.bm25 = MagicMock()
        engine.bm25.query = MagicMock(
            return_value=[(i, 0.8 - i * 0.1) for i in range(len(records))]
        )

    # query() reads this (store-meta query prefix, commit 5f0e298); the real
    # __init__ sets it from store.meta and this fixture bypasses __init__.
    engine.query_prefix = ""

    engine.embedder = MagicMock()
    engine.embedder.get_embedding = MagicMock(
        return_value=np.random.randn(8).astype(np.float32)
    )

    engine.entries = _fake_dictionary()
    engine._build_indices()
    return engine


class TestChunkQuery(unittest.TestCase):
    def test_query_returns_chunks(self):
        engine = _make_engine()
        results = engine.query("Birke", top_k=3)
        self.assertTrue(results)
        for key in ("lemma", "members", "entries", "score", "text"):
            self.assertIn(key, results[0])

    def test_entries_enriched_from_dictionary(self):
        engine = _make_engine()
        results = engine.query("Birke", top_k=3)
        words = {e["word"] for c in results for e in c["entries"]}
        self.assertIn("bērzi", words)


class TestHybridToggle(unittest.TestCase):
    def test_dense_path_when_bm25_none(self):
        engine = _make_engine(hybrid=False)
        results = engine.query("Birke", top_k=3)
        self.assertTrue(results)
        engine.store.query.assert_called_once()

    def test_hybrid_path_when_bm25_set(self):
        engine = _make_engine(hybrid=True)
        results = engine.query("Birke", top_k=3)
        self.assertTrue(results)
        engine.bm25.query.assert_called()


class TestContextRerank(unittest.TestCase):
    def test_context_annotates_chunks(self):
        from prussian.tools.search import search_tool

        engine = _make_engine()
        reranker = MagicMock()
        reranker.rerank = MagicMock(
            return_value=[
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
            ]
        )
        results = search_tool(engine, "Birke", top_k=3,
                              reranker=reranker, context="Der Abendstern")
        self.assertIn("lines", results[0])
        self.assertIn("best_line", results[0])


if __name__ == "__main__":
    unittest.main()
