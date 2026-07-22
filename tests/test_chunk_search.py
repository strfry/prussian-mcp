"""Tests for chunk-mode search, hybrid recall, and context reranking.

Uses fake store / fake embedder / fake reranker — no model downloads.
Follows the _make_engine() construction pattern from test_fst_tools.py.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np


def _fake_chunk_records():
    return [
        {
            "lemma": "bērzi",
            "members": ["bērzi", "berzin"],
            "pos": "n",
            "text": "bērzi: Birke\nberzin: Birke (Akk)",
        },
        {
            "lemma": "dēiws",
            "members": ["dēiws", "deiwan"],
            "pos": "n",
            "text": "dēiws: Gott\ndeiwan: Gott (Akk)",
        },
        {
            "lemma": "wīdātun",
            "members": ["wīdātun", "widdai"],
            "pos": "v",
            "text": "wīdātun: sehen\nwiddai: sehen (Inf)",
        },
    ]


def _fake_entry_records():
    return [
        {
            "word": "bērzi",
            "gender": "fem",
            "translations": {"de": ["Birke"], "engl": ["birch"]},
            "desc": "n [bērzi]",
            "forms": {},
        },
        {
            "word": "dēiws",
            "gender": "masc",
            "translations": {"de": ["Gott"], "engl": ["god"]},
            "desc": "n [dēiws]",
            "forms": {},
        },
        {
            "word": "wīdātun",
            "gender": "",
            "translations": {"de": ["sehen"], "engl": ["to see"]},
            "desc": "v [widdai 113]",
            "forms": {},
        },
    ]


def _make_chunk_engine():
    """Build a SearchEngine with chunk mode, bypassing disk __init__."""
    from prussian.engine.search import SearchEngine

    engine = SearchEngine.__new__(SearchEngine)
    engine.chunk_mode = True
    engine.word_to_entry = {}
    engine.form_to_lemma = {}
    engine.form_to_pgr = {}

    records = _fake_chunk_records()
    embeddings = np.random.randn(len(records), 8).astype(np.float32)
    engine.store = MagicMock()
    engine.store.records = records
    engine.store.embeddings = embeddings
    # Make store.query return (record, score) tuples
    engine.store.query = MagicMock(
        return_value=[(r, 0.9 - i * 0.1) for i, r in enumerate(records)]
    )
    # Make store.top_k return (idx, score) tuples for hybrid_query
    engine.store.top_k = MagicMock(
        return_value=[(i, 0.9 - i * 0.1) for i in range(len(records))]
    )

    engine.entries = _fake_entry_records()

    # Fake BM25Index
    engine.bm25 = MagicMock()
    engine.bm25.query = MagicMock(
        return_value=[(i, 0.8 - i * 0.1) for i in range(len(records))]
    )

    # Fake embedder
    engine.embedder = MagicMock()
    engine.embedder.get_embedding = MagicMock(
        return_value=np.random.randn(8).astype(np.float32)
    )

    engine.chunk_mode = True
    engine.word_to_entry = {}
    engine.form_to_lemma = {}
    engine.form_to_pgr = {}
    engine._build_indices()
    return engine


class TestChunkQuery(unittest.TestCase):
    """Chunk-mode query returns entries per member."""

    def test_chunk_query_returns_entries(self):
        engine = _make_chunk_engine()
        results = engine.query("Birke", top_k=3)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        chunk = results[0]
        self.assertIn("lemma", chunk)
        self.assertIn("members", chunk)
        self.assertIn("entries", chunk)
        self.assertIn("score", chunk)
        self.assertIn("text", chunk)

    def test_members_without_translations_kept(self):
        engine = _make_chunk_engine()
        results = engine.query("Birke", top_k=3)
        for chunk in results:
            self.assertIsInstance(chunk["entries"], list)

    def test_no_translation_filter_in_chunk_mode(self):
        engine = _make_chunk_engine()
        results = engine.query("Birke", top_k=10)
        self.assertEqual(len(results), len(engine.store.records))


class TestChunkContextRerank(unittest.TestCase):
    """Context -> each top chunk gets best_line set and lines ranked."""

    def test_context_annotates_chunks(self):
        from prussian.tools.search import search_tool
        engine = _make_chunk_engine()
        fake_reranker = MagicMock()
        fake_reranker.rerank = MagicMock(
            return_value=[
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
            ]
        )
        results = search_tool(
            engine, "Birke", top_k=3,
            reranker=fake_reranker,
            context="Der Abendstern leuchtet am Himmel",
        )
        self.assertTrue(len(results) > 0)
        chunk = results[0]
        self.assertIn("lines", chunk)
        self.assertIn("best_line", chunk)
        lines = chunk["lines"]
        self.assertTrue(len(lines) > 0)
        best_rank = next(ln["rank"] for ln in lines
                         if ln["text"] == chunk["best_line"])
        self.assertEqual(best_rank, 0)


class TestHybridSearchToggle(unittest.TestCase):
    """HYBRID_SEARCH=0 -> dense path used instead of hybrid_query."""

    def test_dense_path_when_bm25_none(self):
        engine = _make_chunk_engine()
        engine.bm25 = None
        results = engine.query("Birke", top_k=3)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        engine.store.query.assert_called_once()

    def test_hybrid_path_when_bm25_set(self):
        engine = _make_chunk_engine()
        results = engine.query("Birke", top_k=3)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        engine.bm25.query.assert_called()


class TestEntryModeRegression(unittest.TestCase):
    """Entry-mode regression: existing behaviour unchanged."""

    def test_entry_mode_translation_filter(self):
        from prussian.engine.search import SearchEngine

        engine = SearchEngine.__new__(SearchEngine)
        engine.chunk_mode = False
        engine.bm25 = None
        engine.word_to_entry = {}
        engine.form_to_lemma = {}
        engine.form_to_pgr = {}

        records = [
            {"word": "bērzi", "translations": {"de": ["Birke"]}},
            {"word": "no_trans", "translations": {}},
        ]
        engine.store = MagicMock()
        engine.store.query = MagicMock(
            return_value=[(records[0], 0.9), (records[1], 0.8)]
        )
        engine.embedder = MagicMock()
        engine.entries = records
        engine._build_indices()

        results = engine.query("Birke", top_k=10)
        words = [r["word"] for r in results]
        self.assertIn("bērzi", words)
        self.assertNotIn("no_trans", words)

    def test_entry_mode_search_tool_signature(self):
        from prussian.tools.search import search_tool

        engine = MagicMock()
        engine.chunk_mode = False
        engine.query = MagicMock(return_value=[
            {"word": "test", "translations": {"de": ["test"]}},
        ])
        engine.word_to_entry = {}

        results = search_tool(engine, "test", top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["word"], "test")


if __name__ == "__main__":
    unittest.main()
