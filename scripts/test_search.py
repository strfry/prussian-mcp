#!/usr/bin/env python3
"""Integration test for Prussian MCP features — back-to-back pipeline check.

Tests all three MCP tools end-to-end:
  1. search_dictionary  — semantic search (triggers embedding API)
  2. lookup_prussian_word — reverse lookup (all form categories)
  3. get_word_forms        — structured forms + PGR filtering

Also tests optional reranker backend if API_BASE_URL is configured.

Usage:
    source env.local.sh && python scripts/test_search.py
    source env.jina.sh  && python scripts/test_search.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prussian_engine import SearchEngine
from prussian_engine.config import (
    API_BASE_URL,
    API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    PASSAGE_PREFIX,
    EMBEDDINGS_PATH,
    RERANKER_MODEL,
)

# ── Test framework ──────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"

results = []


def record(status, name, elapsed, detail=""):
    results.append((status, name, elapsed, detail))


def fmt_status(status):
    c = {"PASS": GREEN, "FAIL": RED, "SKIP": YELLOW}[status]
    return f"{c}[{status}]{RESET}"


# ── Connectivity tests ──────────────────────────────────────────────────────

BASE_URL = (API_BASE_URL or "(unset)").rstrip("/")


def test_embedding_connectivity():
    """Ping the embedding endpoint with a minimal request."""
    from prussian_engine.embedding_client import EmbeddingClient

    t0 = time.time()
    client = EmbeddingClient()
    vec = client.get_embedding("connectivity test")
    elapsed = time.time() - t0
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(f"Expected dim={EMBEDDING_DIM}, got {len(vec)}")
    return elapsed, f"dim={len(vec)}, model={EMBEDDING_MODEL}"


def test_reranker_connectivity():
    """Ping the reranker endpoint with a minimal request."""
    from prussian_engine.embedding_client import EmbeddingClient

    async def _run():
        client = EmbeddingClient()
        return await client.rerank("test", ["doc a", "doc b"], top_n=2)

    t0 = time.time()
    r = asyncio.run(_run())
    elapsed = time.time() - t0
    if not r:
        raise ValueError("Empty reranker response")
    return elapsed, f"{len(r)} results, model={RERANKER_MODEL}"


# ── Semantic search tests ───────────────────────────────────────────────────
#
# Each test checks that at least one result in the top-K contains any of the
# expected terms in its German ("miks") or English ("engl") translations.
# This is model-agnostic: different embedding backends rank differently but
# the relevant words should appear in the top results regardless.

SEARCH_QUERIES = [
    ("Haus Gebäude Wohnung", ["Haus", "Gebäude", "Wohnung", "home", "building"], "house"),
    ("Gott Himmel", ["Gott", "Himmel", "god", "heaven"], "god/heaven"),
    ("Familie Verwandte", ["Familie", "Verwandt", "family", "relative"], "family"),
]

TOP_K = 10


def _any_term_in_translations(hits, terms):
    """Check if any hit's translations contain any of the given terms."""
    for hit in hits:
        translations = hit.get("translations", {})
        for lang_key in ("miks", "engl"):
            for trans in translations.get(lang_key, []):
                trans_lower = trans.lower()
                if any(t.lower() in trans_lower for t in terms):
                    return hit["word"]
    return None


def test_semantic_search(engine, query, terms, label):
    """Semantic search: query must return at least one relevant result."""
    t0 = time.time()
    hits = engine.query(query, top_k=TOP_K)
    elapsed = time.time() - t0
    if not hits:
        raise ValueError(f"No results for query '{query}'")
    top_word = hits[0]["word"]
    match = _any_term_in_translations(hits, terms)
    if not match:
        words = [(h["word"], h.get("translations", {}).get("miks", [""])[0]) for h in hits[:3]]
        raise ValueError(
            f"No relevant result for query '{query}' in top-{TOP_K}. "
            f"Top-3: {words}"
        )
    return elapsed, f"top: {top_word} (score={hits[0]['score']:.3f}), match: {match}"


def test_reranked_search(rs_engine, query, terms, label, rerank_candidates=30, base_engine=None):
    """Two-stage search: compare pure embedding vs reranked ordering."""

    t0 = time.time()

    engine = base_engine or rs_engine.base_engine
    emb_hits = engine.query(query, top_k=TOP_K)
    emb_top = emb_hits[0]["word"] if emb_hits else "?"
    emb_score = emb_hits[0].get("score", 0) if emb_hits else 0

    reranked = asyncio.run(
        rs_engine.search(query, top_k=TOP_K, rerank_candidates=rerank_candidates)
    )

    elapsed = time.time() - t0
    if not reranked:
        raise ValueError(f"No reranked results for '{query}'")

    reranked_top = reranked[0]["word"]
    reranked_score = reranked[0].get("rerank_score", 0)

    changed = "Δ " if reranked_top != emb_top else "= "
    top_info = f"{changed}emb:#1 {emb_top}({emb_score:.3f}) → rerank:#1 {reranked_top}({reranked_score:.4f})"

    match = _any_term_in_translations(reranked, terms)
    if not match:
        words = [(h["word"], h.get("translations", {}).get("miks", [""])[0]) for h in reranked[:3]]
        raise ValueError(
            f"No relevant reranked result for '{query}' in top-{TOP_K}. "
            f"Top-3: {words}"
        )

    return elapsed, f"{top_info}, match: {match}"


# ── Lookup tests ────────────────────────────────────────────────────────────

LOOKUP_TESTS = [
    # (word, expected_lemma, expected_pgr_substr, label)
    ("Dēiwan", "Dēiws", "ACC", "declension"),
    ("būlai", "būtwei", "SUBJ", "subjunctive (NEW)"),
    ("seīsei", "būtwei", "OPT", "optative (NEW)"),
    ("sēnts", "būtwei", "PC", "participle (NEW)"),
    ("seīs", "būtwei", "IMP", "imperative"),
    ("abzōlutai", "abzōluts", "POS", "adverb (NEW)"),
    ("dāsei", "dātwei", "OPT", "ambiguous: IND|OPT (NEW)"),
]


def test_lookup(engine, word, expected_lemma, expected_pgr, label):
    """Reverse lookup: inflected form must resolve to correct lemma + PGR."""
    t0 = time.time()
    hits = engine.lookup(word, fuzzy=False)
    elapsed = time.time() - t0
    if not hits:
        raise ValueError(f"'{word}' not found in any form category")
    match = None
    for h in hits:
        if h["word"].lower() == expected_lemma.lower():
            match = h
            break
    if not match:
        lemmas = [h["word"] for h in hits[:3]]
        raise ValueError(
            f"'{word}' → {lemmas}, expected lemma '{expected_lemma}' not found"
        )
    pgr = match.get("pgr", "")
    if expected_pgr not in pgr:
        raise ValueError(
            f"'{word}' → {match['word']} pgr='{pgr}', expected to contain '{expected_pgr}'"
        )
    return elapsed, f"→ {match['word']} ({pgr})"


# ── get_word_forms tests ────────────────────────────────────────────────────

REQUIRED_CATEGORIES = {
    "indicative",
    "subjunctive",
    "optative",
    "imperative",
    "participles",
    "declension",
}


def test_get_word_forms_categories(engine):
    """get_word_forms must return all expected categories."""
    t0 = time.time()
    data = engine.get_word_forms("būtwei")
    elapsed = time.time() - t0
    if not data:
        raise ValueError("būtwei not found")
    entry = data[0]
    forms = entry.get("forms", {})
    categories = set(forms.keys())
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        raise ValueError(f"Missing categories: {missing}")
    return elapsed, f"{len(categories)} categories: {sorted(categories)}"


def test_get_word_forms_filter(engine):
    """get_word_forms with filter must return only matching forms."""
    t0 = time.time()
    data = engine.get_word_forms("būtwei", filter_pgr="SUBJ")
    elapsed = time.time() - t0
    entry = data[0]
    filtered = entry.get("filtered_forms", [])
    if not filtered:
        raise ValueError("SUBJ filter returned no forms")
    for ff in filtered:
        if "SUBJ" not in ff["pgr"]:
            raise ValueError(f"Filter mismatch: {ff['pgr']} does not contain SUBJ")
    return elapsed, f"{len(filtered)} subjunctive forms"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"{BOLD}=== Integration Test: Prussian MCP Features ==={RESET}")
    print(f"  Embedding: {BASE_URL}/v1/embeddings  ({EMBEDDING_MODEL}, dim={EMBEDDING_DIM})")
    print(f"  Embedding files: {EMBEDDINGS_PATH}.{{entries.json,embeddings.npy}}  (prefix={PASSAGE_PREFIX})")
    reranker_url = f"{BASE_URL}/v1/rerank" if API_BASE_URL else "(unset)"
    reranker_status = f"  Reranker:  {reranker_url}  ({RERANKER_MODEL})"
    if API_KEY:
        reranker_status += "  [key set]"
    else:
        reranker_status += f"  {YELLOW}[no key — reranked search will SKIP]{RESET}"
    print(reranker_status)
    print()

    reranker_available = bool(API_BASE_URL) and bool(API_KEY)

    # ── Phase 1: Connectivity ──────────────────────────────────────────────
    print(f"{BOLD}── Connectivity{'' if API_BASE_URL else ' (no API_BASE_URL set — tests will SKIP)'} ──{RESET}")

    try:
        elapsed, detail = test_embedding_connectivity()
        record(PASS, "Embedding API", elapsed, detail)
    except Exception as e:
        record(FAIL, "Embedding API", 0, str(e))

    if reranker_available:
        try:
            elapsed, detail = test_reranker_connectivity()
            record(PASS, "Reranker API", elapsed, detail)
        except Exception as e:
            record(FAIL, "Reranker API", 0, str(e))
    else:
        record(SKIP, "Reranker API", 0, "API_BASE_URL or API_KEY not set")

    # ── Phase 2: Search Engine init ────────────────────────────────────────
    print(f"\n{BOLD}── Loading Search Engine{'' if API_BASE_URL else ' (embedding server needed for semantic search)'} ──{RESET}")
    try:
        t0 = time.time()
        engine = SearchEngine()
        elapsed = time.time() - t0
        record(PASS, "SearchEngine init", elapsed, f"{len(engine.entries)} entries, {len(engine.form_to_lemma)} forms indexed")
        embedding_ok = True
    except Exception as e:
        record(FAIL, "SearchEngine init", 0, str(e))
        embedding_ok = False
        engine = None

    # ── Phase 3: Semantic Search ───────────────────────────────────────────
    print(f"\n{BOLD}── Semantic Search (search_dictionary){'' if embedding_ok else ' [SKIPPED — no engine]'} ──{RESET}")

    if embedding_ok:
        try:
            for query, terms, label in SEARCH_QUERIES:
                name = f'search "{query}"'
                try:
                    elapsed, detail = test_semantic_search(engine, query, terms, label)
                    record(PASS, name, elapsed, detail)
                except Exception as e:
                    record(FAIL, name, 0, str(e))
        except Exception as e:
            record(FAIL, "Semantic search (all)", 0, f"Embedding server unreachable: {e}")
    else:
        for query, terms, label in SEARCH_QUERIES:
            record(SKIP, f'search "{query}"', 0, "no engine")

    # ── Phase 4: Reranked Search ───────────────────────────────────────────
    print(f"\n{BOLD}── Reranked Search (vs pure embedding){'' if reranker_available and embedding_ok else ' [SKIPPED]'} ──{RESET}")

    if reranker_available and embedding_ok:
        from prussian_engine.rerank_search import RerankedSearchEngine

        rs_engine = RerankedSearchEngine(use_reranker=True)

        for query, terms, label in SEARCH_QUERIES:
            name = f'rerank "{query}"'
            try:
                elapsed, detail = test_reranked_search(
                    rs_engine, query, terms, label, base_engine=engine,
                )
                record(PASS, name, elapsed, detail)
            except Exception as e:
                record(FAIL, name, 0, str(e))

        try:
            elapsed, detail = test_reranked_search(
                rs_engine, "Familie Verwandte",
                ["Familie", "family", "Verwandt", "relative"], "family",
                rerank_candidates=50, base_engine=engine,
            )
            record(PASS, 'rerank (candidates=50)', elapsed, detail)
        except Exception as e:
            record(FAIL, 'rerank (candidates=50)', 0, str(e))
    else:
        reason = "reranker not configured" if not reranker_available else "no engine"
        for query, terms, label in SEARCH_QUERIES:
            record(SKIP, f'rerank "{query}"', 0, reason)
        record(SKIP, 'rerank (candidates=50)', 0, reason)

    # ── Phase 5: Lookup (all form categories) ──────────────────────────────
    print(f"\n{BOLD}── Reverse Lookup (lookup_prussian_word){'' if engine else ' [SKIPPED]'} ──{RESET}")

    if engine:
        for word, lemma, pgr, label in LOOKUP_TESTS:
            name = f'lookup "{word}" ({label})'
            try:
                elapsed, detail = test_lookup(engine, word, lemma, pgr, label)
                record(PASS, name, elapsed, detail)
            except Exception as e:
                record(FAIL, name, 0, str(e))
    else:
        for word, lemma, pgr, label in LOOKUP_TESTS:
            record(SKIP, f'lookup "{word}"', 0, "no engine")

    # ── Phase 6: get_word_forms ────────────────────────────────────────────
    print(f"\n{BOLD}── get_word_forms{'' if engine else ' [SKIPPED]'} ──{RESET}")

    if engine:
        try:
            elapsed, detail = test_get_word_forms_categories(engine)
            record(PASS, "get_word_forms categories", elapsed, detail)
        except Exception as e:
            record(FAIL, "get_word_forms categories", 0, str(e))

        try:
            elapsed, detail = test_get_word_forms_filter(engine)
            record(PASS, "get_word_forms + filter SUBJ", elapsed, detail)
        except Exception as e:
            record(FAIL, "get_word_forms + filter SUBJ", 0, str(e))
    else:
        record(SKIP, "get_word_forms", 0, "no engine")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}=== Summary ==={RESET}")
    passed = sum(1 for s, _, _, _ in results if s == PASS)
    failed = sum(1 for s, _, _, _ in results if s == FAIL)
    skipped = sum(1 for s, _, _, _ in results if s == SKIP)
    total_time = sum(t for _, _, t, _ in results)

    for status, name, elapsed, detail in results:
        print(f"  {fmt_status(status)} {name:<50s} {elapsed:5.2f}s  {detail}")

    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped  ({total_time:.1f}s total)")

    if failed:
        print(f"\n{RED}Some tests FAILED — check backend connectivity and env config.{RESET}")
        sys.exit(1)
    elif skipped and not passed:
        print(f"\n{YELLOW}No backend available — all tests skipped.{RESET}")
        print("  Set API_BASE_URL and API_KEY via env config (env.local.sh / env.jina.sh).")
        sys.exit(2)
    else:
        print(f"\n{GREEN}All tests passed.{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
