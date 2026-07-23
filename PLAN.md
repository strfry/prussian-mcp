# Plan: Chunk-Mode SearchEngine Integration

## Overview
Integrate chunk-based hybrid search (BM25 + dense RRF) into the MCP server's SearchEngine, with optional cross-encoder line annotation when context is provided. Entry-Mode stays unchanged as default.

**Reranker policy:** The old `RerankedSearchEngine` (Jina API) is obsolete. All reranking
goes through `prussian_embeddings.get_reranker()` (local fastembed cross-encoder, no API
key required). The `reranked_engine` parameter is removed from `search_tool`.

---

## Phase 0: Build Chunk Artifacts (embeddings repo)

The chunk store (`embeddings_st_e5large_chunks.*`) doesn't exist yet. Must be built **before** Phase 1.

### Step 0.1: Build chunk JSONL
```bash
cd /home/strfry/projekte/prussian/embeddings
uv run prussian-embeddings-build-chunks \
  --dictionary ../corpus/parsed/twanksta_entries.json \
  --links ../fst/build/links.json \
  --tags ../fst/build/tags.json \
  --out data/embedding_chunks.jsonl \
  --langs engl,miks,leit,latt
```
Produces `data/embedding_chunks.jsonl` (~10k chunks, one JSON dict per line).

### Step 0.2: Embed chunks
```bash
uv run python -m prussian_embeddings.generate \
  --chunks data/embedding_chunks.jsonl \
  --backend sentence-transformers \
  --passage-prefix "" \
  --query-prefix "query: " \
  --out data/embeddings_st_e5large_chunks
```
Produces:
- `data/embeddings_st_e5large_chunks.embeddings.npy`
- `data/embeddings_st_e5large_chunks.entries.json`
- `data/embeddings_st_e5large_chunks.meta.json`

**Verify:** `python -c "from prussian_embeddings import EmbeddingStore; s = EmbeddingStore.load('data/embeddings_st_e5large_chunks'); print(len(s.records), s.embeddings.shape)"` → `~10017 (10017, 1024)`

---

## Phase 1: Config (prussian/config.py)

Add two new env vars:

```python
# Chunk-Mode: when set (e.g. "embeddings_st_e5large_chunks"), load chunk
# embeddings store; empty = Entry-Mode (current behavior).
CHUNK_EMBEDDINGS_NAME = os.getenv("CHUNK_EMBEDDINGS_NAME", "")

# Hybrid search toggle (BM25 + RRF). "1" = on, "0" = dense-only.
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1")
```

Derive `CHUNK_EMBEDDINGS_PATH`:
```python
CHUNK_EMBEDDINGS_PATH = EMBEDDINGS_DIR / CHUNK_EMBEDDINGS_NAME if CHUNK_EMBEDDINGS_NAME else None
```

**Test:** import succeeds; `CHUNK_EMBEDDINGS_NAME=""` by default.

---

## Phase 2: SearchEngine Chunk-Mode (prussian/engine/search.py)

### 2.1: Imports
```python
from prussian.config import (
    EMBEDDINGS_PATH, QUERY_PREFIX,
    CHUNK_EMBEDDINGS_NAME, CHUNK_EMBEDDINGS_PATH, HYBRID_SEARCH,
    DICTIONARY_PATH,
)
```

Add conditional imports (lazy, inside chunk-mode branches):
```python
if CHUNK_EMBEDDINGS_NAME:
    from prussian_embeddings import BM25Index, hybrid_query
```

### 2.2: `__init__` branching
```
if CHUNK_EMBEDDINGS_PATH:
    # Chunk-Mode
    self.store = EmbeddingStore.load(str(CHUNK_EMBEDDINGS_PATH))
    self.bm25 = BM25Index([r["text"] for r in self.store.records])
    self.entries = _load_dictionary_entries(DICTIONARY_PATH)  # from twanksta_entries.json
    self._chunk_mode = True
else:
    # Entry-Mode (unchanged)
    self.store = EmbeddingStore.load(str(EMBEDDINGS_PATH))
    self.entries = self.store.records
    self.bm25 = None
    self._chunk_mode = False
```

### 2.3: `_load_dictionary_entries(path)` — new static helper
```python
def _load_dictionary_entries(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    return data
```

### 2.4: `_build_indices()` — unchanged
Fed by `self.entries` which is always entry records (from store in Entry-Mode, from dictionary in Chunk-Mode). No changes needed.

### 2.5: `query()` — Chunk-Mode branch
```python
def query(self, query: str, top_k: int = 10) -> list[dict]:
    if self._chunk_mode:
        return self._query_chunks(query, top_k)
    return self._query_entries(query, top_k)  # existing logic (extracted)
```

New `_query_chunks()`:
```python
def _query_chunks(self, query: str, top_k: int) -> list[dict]:
    if HYBRID_SEARCH == "1" and self.bm25:
        hits = hybrid_query(
            self.store, self.embedder, self.bm25, query,
            k=top_k, query_prefix=QUERY_PREFIX,
        )
    else:
        hits = self.store.query(self.embedder, query, k=top_k, query_prefix=QUERY_PREFIX)

    results = []
    for record, score in hits:
        members = record.get("members", [])
        entries = []
        for m in members:
            entry_hits = self.word_to_entry.get(m.lower(), [])
            if entry_hits:
                entries.append({
                    "word": entry_hits[0].get("word", m),
                    "translations": entry_hits[0].get("translations", {}),
                })
            else:
                entries.append({"word": m, "translations": {}})
        results.append({
            "lemma": record.get("lemma", ""),
            "members": members,
            "pos": record.get("pos"),
            "score": float(score),
            "entries": entries,
        })
    return results
```

**Note:** Members without translations stay in the result (that's the point of clustering).

### 2.6: Extract Entry-Mode query to `_query_entries()`
Move current `query()` body to `_query_entries(self, query, top_k)` returning the same format as before (`[{word, translations, score}]`). The existing format stays unchanged so downstream tools don't break.

---

## Phase 3: search_tool (prussian/tools/search.py)

### 3.1: Remove `reranked_engine`, add `context` + `reranker`
```python
def search_tool(
    engine,
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    filter_tags: str | None = None,
    context: str | None = None,        # NEW — triggers line annotation in chunk mode
    reranker=None,                      # NEW — prussian_embeddings reranker (fastembed)
) -> list[dict[str, Any]]:
```

`reranked_engine` parameter is removed entirely. The old `RerankedSearchEngine` (Jina
API) is dead code — all reranking goes through `prussian_embeddings.get_reranker()`.

### 3.2: Entry-Mode path — existing logic preserved
When `engine._chunk_mode` is False:
- Without reranker: `engine.query(query, top_k)` (unchanged)
- With reranker (`use_reranker=True` and `reranker` provided): use the local reranker
  to rerank entry translation strings, same two-stage pattern as before but via
  `prussian_embeddings` instead of Jina API

```python
if not engine._chunk_mode:
    results = engine.query(query, top_k)
    if use_reranker and reranker is not None:
        results = _rerank_entries(query, results, reranker)
    # ... filter_tags + output formatting (unchanged)
```

New helper `_rerank_entries()`:
```python
def _rerank_entries(query, results, reranker):
    """Rerank entry results using the local fastembed cross-encoder."""
    docs = []
    for r in results:
        trans = r.get("translations", {})
        doc = " | ".join(
            v[0] for lang in ["engl", "miks", "leit", "latt", "pols", "mask"]
            if (v := trans.get(lang)) and v[0]
        )
        docs.append(doc)
    ranked = reranker.rerank(query, docs, top_n=len(docs))
    for rank, item in enumerate(ranked):
        idx = item["index"]
        results[idx]["rerank_score"] = float(item["relevance_score"])
    results.sort(key=lambda r: r.get("rerank_score", 0), reverse=True)
    return results
```

### 3.3: Chunk-Mode path
When `engine._chunk_mode` is True:
```python
raw = engine.query(query, top_k=top_k)

if context and use_reranker and reranker is not None:
    from prussian_embeddings import annotate_chunk
    annotated = [annotate_chunk(c, context, reranker) for c in raw]
    output = _format_chunk_results(annotated)
else:
    output = _format_chunk_results(raw)

if filter_tags:
    output = _apply_chunk_filter_tags(output, engine, filter_tags)
return output
```

### 3.4: `_format_chunk_results()` — new helper
```python
def _format_chunk_results(chunks: list[dict]) -> list[dict]:
    out = []
    for c in chunks:
        item = {
            "lemma": c.get("lemma", ""),
            "members": c.get("members", []),
            "pos": c.get("pos"),
            "score": c.get("score", 0.0),
            "entries": c.get("entries", []),
        }
        if "best_line" in c:
            item["best_line"] = c["best_line"]
        if "lines" in c:
            item["lines"] = c["lines"]
        out.append(item)
    return out
```

### 3.5: `_apply_chunk_filter_tags()` — new helper
Same `forms_with_tags` pattern but iterating over member entries:
```python
def _apply_chunk_filter_tags(results, engine, filter_tags):
    from prussian.engine.fst.tags import forms_with_tags, fst_available, match_tags
    if not fst_available():
        return results
    filtered = []
    for r in results:
        for entry_dict in r.get("entries", []):
            word = entry_dict.get("word", "")
            for entry in engine.word_to_entry.get(word.lower(), []):
                fw = forms_with_tags(engine, entry)
                if any(match_tags(f.get("tags", []), filter_tags) for f in fw):
                    filtered.append(r)
                    break
            else:
                continue
            break
    return filtered
```

### 3.6: Update CLI main()
- Remove `--no-reranker` → replace with `--reranker` (opt-in, since fastembed needs first-use download)
- Add `--context` argument (string, optional)
- When `--context` is set: auto-enable reranker, load via `prussian_embeddings.get_reranker()`
- Chunk-mode output: `lemma (pos) [score]`, indented lines, `→` best_line marker

### 3.7: Remove `RerankedSearchEngine` reference from CLI
Delete the `from prussian.engine.embeddings.rerank import RerankedSearchEngine` import and
the lazy-loading block. Replace with:
```python
reranker = None
if not args.no_reranker or args.context:
    from prussian_embeddings import get_reranker
    reranker = get_reranker()
```

---

## Phase 4: MCP Adapter (prussian/adapters/mcp.py)

### 4.1: `search_dictionary` tool — add `context`, remove `reranked_engine` dependency
```python
@mcp.tool()
def search_dictionary(
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    filter_tags: str | None = None,
    context: str | None = None,     # NEW
) -> list[dict[str, Any]]:
```

Docstring update: document new `context` param and chunk-mode output format (`{lemma, members, pos, score, entries, best_line?, lines?}`).

### 4.2: Remove `_reranked_engine` global, add `_reranker` lazy singleton
```python
# Old (dead code):
# _reranked_engine: RerankedSearchEngine | None = None

# New: prussian_embeddings reranker (local fastembed cross-encoder)
_reranker_instance = None

def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        from prussian_embeddings import get_reranker
        _reranker_instance = get_reranker()
    return _reranker_instance
```

### 4.3: Updated `search_dictionary` body
```python
@mcp.tool()
def search_dictionary(
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    filter_tags: str | None = None,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """..."""
    reranker = None
    if use_reranker and _engine()._chunk_mode and context:
        reranker = _get_reranker()
    elif use_reranker and not _engine()._chunk_mode:
        reranker = _get_reranker()

    return search_tool(
        _engine(),
        query,
        top_k=top_k,
        use_reranker=use_reranker,
        filter_tags=filter_tags,
        context=context,
        reranker=reranker,
    )
```

### 4.4: Remove dead code
- Delete `from prussian.engine.embeddings.rerank import RerankedSearchEngine` import
- Delete `_reranked_engine` global and its lazy init in `search_dictionary`
- Update `prussian/engine/embeddings/__init__.py` docstring (remove `RerankedSearchEngine` mention)
- Update `ARCHITECTURE.md` line 81 (remove `rerank.py` reference or mark deprecated)
- Delete `prussian/engine/embeddings/rerank.py` (or rename to `_rerank_deprecated.py`)

### 4.5: Clean up scripts
- `scripts/test_search.py` line 325: remove `RerankedSearchEngine` usage
- `scripts/test_reranked_search.py`: delete entirely (tests obsolete code)

---

## Phase 5: Tests (tests/test_chunk_search.py)

### Test infrastructure
- `FakeEmbedder` — returns deterministic random vectors (seeded)
- `FakeReranker` — returns lines sorted by simple keyword match
- `FakeBM25` — thin wrapper returning pre-set rankings
- Fixtures: 3 chunk records (cluster with 2 members, singleton with translation, singleton without)

### Test cases

1. **Chunk-Mode query returns entries per member**
   - Fake store with 3 chunk records, fake dictionary with matching entries
   - `engine.query("test")` returns `[{lemma, members, pos, score, entries}]`
   - Each member has an entry with translations (or empty translations if not in dict)

2. **Entry-Mode regression**
   - Engine with `CHUNK_EMBEDDINGS_NAME=""` (monkeypatch)
   - `engine.query("test")` returns old format `[{word, translations, score}]`

3. **Entry-Mode reranking via embeddings repo**
   - `search_tool` with `use_reranker=True` + `reranker` provided
   - Results have `rerank_score` field, sorted by rerank score

4. **search_tool chunk-mode without context**
   - Returns `{lemma, members, pos, score, entries}` per chunk
   - No `best_line` or `lines` keys

5. **search_tool chunk-mode with context + reranker**
   - Returns `best_line` and `lines` (with rank/score fields)
   - `best_line` has rank 0

6. **search_tool chunk-mode with filter_tags**
   - Only chunks whose members have matching forms are returned

7. **HYBRID_SEARCH=0 disables BM25**
   - Monkeypatch `HYBRID_SEARCH` to "0"
   - Verify only dense results (BM25 not called)

8. **_build_indices from DICTIONARY_PATH in chunk-mode**
   - `engine.word_to_entry` is populated from twanksta_entries.json
   - `lookup_tool` and `wordforms_tool` work correctly on chunk-mode engine

### Test approach
- All tests use monkeypatch to control `CHUNK_EMBEDDINGS_NAME` and `HYBRID_SEARCH`
- No model downloads — FakeEmbedder/FakeReranker everywhere
- Fake store saved to tmp_path, loaded by EmbeddingStore.load()

---

## Phase 6: Verification

1. `uv run pytest tests/test_chunk_search.py -v` — all new tests pass
2. `uv run pytest tests/ -v` — all existing tests still pass (regression)
3. Manual smoke test (requires real models):
   ```bash
   CHUNK_EMBEDDINGS_NAME=embeddings_st_e5large_chunks \
   EMBEDDING_BACKEND=sentence-transformers \
   uv run search "Abendstern" --context "Der Abendstern leuchtet am Himmel"
   ```
4. MCP smoke: `search_dictionary` with and without `context`; `lookup_prussian_word` regression

---

## File Changes Summary

| File | Change |
|------|--------|
| `prussian/config.py` | Add `CHUNK_EMBEDDINGS_NAME`, `CHUNK_EMBEDDINGS_PATH`, `HYBRID_SEARCH` |
| `prussian/engine/search.py` | Chunk-Mode init, `_load_dictionary_entries`, `_query_chunks`, `_query_entries`, BM25 index |
| `prussian/tools/search.py` | Remove `reranked_engine`, add `context`+`reranker`, `_rerank_entries`, chunk-mode branch, `_format_chunk_results`, `_apply_chunk_filter_tags`, CLI overhaul |
| `prussian/adapters/mcp.py` | `context` param, remove `_reranked_engine`, lazy `_get_reranker()` via embeddings repo |
| `prussian/engine/embeddings/rerank.py` | Delete (dead code — RerankedSearchEngine) |
| `prussian/engine/embeddings/__init__.py` | Remove RerankedSearchEngine docstring reference |
| `ARCHITECTURE.md` | Update line 81 (remove rerank.py reference) |
| `scripts/test_search.py` | Remove RerankedSearchEngine usage |
| `scripts/test_reranked_search.py` | Delete (tests obsolete code) |
| `tests/test_chunk_search.py` | New test file (8 test cases) |
