"""Search backends — the entry-mode / chunk-mode strategy.

"Chunk mode" (German *Modus*) is a store representation: one embedding
vector per lemma-cluster instead of one per headword.  Historically both the
:class:`~prussian.engine.search.SearchEngine` and the ``search`` tool carried
parallel ``if self.chunk_mode`` branches for retrieval, reranking,
``filter_tags`` and rendering.  Those branches live here now, one class per
mode, so callers dispatch through a single interface and never sniff
``engine.chunk_mode`` again.

Each backend has two responsibilities:

* :meth:`query` — the *raw* mode-specific retrieval.  ``SearchEngine.query``
  delegates to it, so external callers keep the same result shape (entry
  dicts vs chunk dicts).
* :meth:`run` — the full ``search_dictionary`` pipeline (retrieval +
  rerank + ``filter_tags`` + output shaping).  It retrieves via
  ``engine.query(...)`` so the two levels share one code path and remain
  independently mockable.

Backends are stateless; module-level singletons are reused via
:func:`backend_for`.
"""

from __future__ import annotations

import sys
from typing import Any

from prussian.config import CHUNK_RERANK_TOPN, QUERY_PREFIX
from prussian_embeddings import hybrid_query


def _infer_gender(engine, word: str) -> str:
    """Get gender from the first matching entry."""
    entries = engine.word_to_entry.get(word.lower(), [])
    for e in entries:
        g = e.get("gender", "")
        if g:
            return g
    return ""


class EntryBackend:
    """One embedding vector per headword (the default representation)."""

    chunk_mode = False

    # ── retrieval ───────────────────────────────────────────────────────────
    def query(self, engine, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        print(f'Searching: "{query}"', file=sys.stderr)

        # Over-fetch for filtering, keep only entries with translations.
        hits = engine.store.query(
            engine.embedder, query, k=top_k * 2, query_prefix=QUERY_PREFIX
        )
        results: list[dict[str, Any]] = []
        for record, score in hits:
            translations = record.get("translations", {})
            if translations:
                results.append({
                    "word": record.get("word", ""),
                    "translations": translations,
                    "score": float(score),
                })
            if len(results) >= top_k:
                break
        return results

    # ── full tool pipeline ──────────────────────────────────────────────────
    def run(
        self, engine, query: str, top_k: int,
        *, filter_tags: str | None, reranker, context: str | None,
    ) -> list[dict[str, Any]]:
        if context and reranker:
            candidates = engine.query(query, 100)
            from prussian.engine.embeddings.rerank import rerank_results
            results = rerank_results(engine, context, candidates, reranker)[:top_k]
        else:
            results = engine.query(query, top_k)

        if filter_tags:
            self._filter_by_tags(engine, results, filter_tags)

        return self._finalize(results)

    def _filter_by_tags(self, engine, results, filter_tags) -> None:
        """Attach ``forms`` / ``gender`` to each result whose paradigm matches."""
        from prussian.engine.fst.tags import forms_with_tags, match_tags, fst_available
        if not fst_available():
            return

        forms_cache: dict[str, list[dict]] = {}
        for r in results:
            entries = engine.word_to_entry.get(r["word"].lower(), [])
            for entry in entries:
                fw = forms_with_tags(engine, entry)
                if fw:
                    forms_cache[r["word"].lower()] = fw
                    break

        for r in results:
            fw = forms_cache.get(r["word"].lower(), [])
            if fw:
                filtered = [f for f in fw
                            if match_tags(f.get("tags", []), filter_tags)]
                if filtered:
                    r["forms"] = [
                        {"form": f["form"],
                         "tags": ("+".join(f["tags"]) if f["tags"]
                                  else f.get("pgr", ""))}
                        for f in filtered
                    ]
                    r["gender"] = _infer_gender(engine, r["word"])

    def _finalize(self, results) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for r in results:
            entry: dict[str, Any] = {
                "word": r["word"],
                "translations": r["translations"],
            }
            if "forms" in r:
                entry["forms"] = r["forms"]
            if "gender" in r:
                entry["gender"] = r["gender"]
            if "rerank_score" in r:
                entry["rerank_score"] = r["rerank_score"]
            if "word_type" in r:
                entry["word_type"] = r["word_type"]
            output.append(entry)
        return output

    # ── rendering ───────────────────────────────────────────────────────────
    def render(self, i: int, e: dict, *, context: str | None = None) -> None:
        word = e["word"]
        trans = e["translations"]
        trans_str = ", ".join(
            f"{lang}: {t}" for lang, t in trans.items()
        ) if isinstance(trans, dict) else str(trans)
        print(f"  {i}. {word} — {trans_str}")
        if e.get("forms"):
            for f in e["forms"][:5]:
                tags_str = f.get("tags", "")
                print(f"     {tags_str}: {f.get('form', '')}")


class ChunkBackend:
    """One embedding vector per lemma-cluster (BM25+dense RRF or dense-only)."""

    chunk_mode = True

    # ── retrieval ───────────────────────────────────────────────────────────
    def query(self, engine, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if engine.bm25 is not None:
            hits = hybrid_query(engine.store, engine.embedder, engine.bm25, query,
                                k=top_k, query_prefix=QUERY_PREFIX)
        else:
            hits = engine.store.query(engine.embedder, query,
                                      k=top_k, query_prefix=QUERY_PREFIX)

        results: list[dict[str, Any]] = []
        for record, score in hits:
            members = record.get("members", [])
            entries = [
                {"word": e["word"], "translations": e.get("translations", {})}
                for m in members
                for e in engine.word_to_entry.get(m.lower(), [])
            ]
            results.append({
                "lemma": record.get("lemma", ""),
                "members": members,
                "pos": record.get("pos", ""),
                "score": float(score),
                "text": record["text"],
                "entries": entries,
            })
        return results

    # ── full tool pipeline ──────────────────────────────────────────────────
    def run(
        self, engine, query: str, top_k: int,
        *, filter_tags: str | None, reranker, context: str | None,
    ) -> list[dict[str, Any]]:
        results = engine.query(query, top_k)

        # Context reranking: annotate top N chunks with best_line / lines.
        if context and reranker:
            from prussian_embeddings import annotate_chunk
            for i, chunk in enumerate(results[:CHUNK_RERANK_TOPN]):
                results[i] = annotate_chunk(chunk, context, reranker)

        if filter_tags:
            self._filter_by_tags(engine, results, filter_tags)

        return results

    def _filter_by_tags(self, engine, results, filter_tags) -> None:
        """Attach ``filtered_entries`` to each chunk (per-member form filter)."""
        from prussian.engine.fst.tags import forms_with_tags, match_tags, fst_available
        if not fst_available():
            return

        for chunk in results:
            filtered_entries = []
            for entry_dict in chunk.get("entries", []):
                word = entry_dict.get("word", "")
                fw_list = engine.word_to_entry.get(word.lower(), [])
                for fe in fw_list:
                    fw = forms_with_tags(engine, fe)
                    if fw:
                        matched = [f for f in fw
                                   if match_tags(f.get("tags", []), filter_tags)]
                        if matched:
                            filtered_entries.append({
                                **entry_dict,
                                "forms": [
                                    {"form": f["form"],
                                     "tags": ("+".join(f["tags"])
                                              if f["tags"]
                                              else f.get("pgr", ""))}
                                    for f in matched
                                ],
                                "gender": _infer_gender(engine, word),
                            })
                        break
            chunk["filtered_entries"] = filtered_entries

    # ── rendering ───────────────────────────────────────────────────────────
    def render(self, i: int, chunk: dict, *, context: str | None = None) -> None:
        lemma = chunk.get("lemma", "")
        pos = chunk.get("pos", "")
        score = chunk.get("score", 0)
        members = chunk.get("members", [])
        header = f"  {i}. {lemma}"
        if pos:
            header += f" ({pos})"
        header += f"  [{score:.3f}]"
        if members:
            header += f"  members: {', '.join(members)}"
        print(header)
        lines = chunk.get("lines")
        if lines:
            best = chunk.get("best_line", "")
            for ln in sorted(lines, key=lambda x: x["rank"]):
                marker = " →" if ln["text"] == best else ""
                print(f"     [{ln['rank']}] {ln['text']}{marker}")
        entries = chunk.get("entries", [])
        if entries:
            for ed in entries[:3]:
                trans = ed.get("translations", {})
                trans_str = ", ".join(
                    f"{lang}: {t}" for lang, t in trans.items()
                ) if isinstance(trans, dict) else str(trans)
                print(f"     entry: {ed['word']} — {trans_str}")
        fe = chunk.get("filtered_entries", [])
        if fe:
            for fed in fe[:5]:
                forms_str = "; ".join(
                    f"{f['tags']}: {f['form']}" for f in fed.get("forms", [])
                )
                print(f"     filtered: {fed['word']} — {forms_str}")


# Stateless singletons — reuse instead of reallocating per call.
_ENTRY = EntryBackend()
_CHUNK = ChunkBackend()


def backend_for(engine) -> EntryBackend | ChunkBackend:
    """Return the backend for ``engine`` based on its ``chunk_mode`` flag.

    Derived from ``chunk_mode`` (not a stored attribute) so engines built via
    ``SearchEngine.__new__`` in tests — which set ``chunk_mode`` directly —
    resolve correctly without extra wiring.
    """
    return _CHUNK if getattr(engine, "chunk_mode", False) else _ENTRY
