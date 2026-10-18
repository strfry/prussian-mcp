"""search_dictionary — semantic dictionary search (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any

from prussian.tools._format import head, translations


def search_tool(
    engine,
    query: str,
    top_k: int = 10,
    filter_tags: str | None = None,
    reranker=None,
    context: str | None = None,
) -> str:
    """Semantic dictionary search core; returns compact text.

    The model-facing tool text lives in :data:`prussian.tools.spec.SEARCH` —
    do not duplicate it here.

    With ``context`` (and a reranker), a candidate pool of
    ``max(top_k, CHUNK_RERANK_TOPN)`` chunks is re-ordered by cross-encoder
    relevance of the whole chunk text to the context, and the best entry
    within the top chunks is marked.  Chunks holding an exact translation
    match (``exact`` entries, see ``SearchEngine.query``) stay in front
    either way.
    """
    if context and reranker:
        from prussian_embeddings import annotate_chunk
        from prussian.config import CHUNK_RERANK_TOPN

        results = engine.query(query, max(top_k, CHUNK_RERANK_TOPN))
        ranked = reranker.rerank(context, [c["text"] for c in results],
                                 top_n=len(results))
        order = [item["index"] for item in ranked]
        scores = {item["index"]: item["relevance_score"] for item in ranked}
        order += [i for i in range(len(results)) if i not in scores]
        results = [{**results[i], "rerank_score": scores.get(i, 0.0)}
                   for i in order]
        results = _exact_first(results)[:top_k]
        for i, chunk in enumerate(results[:CHUNK_RERANK_TOPN]):
            results[i] = annotate_chunk(chunk, context, reranker)
    else:
        results = engine.query(query, top_k)

    if filter_tags:
        from prussian.engine.fst.tags import generate_paradigm, match_tags, fst_available
        if fst_available():
            for chunk in results:
                filtered_entries = []
                for entry_dict in chunk.get("entries", []):
                    word = entry_dict.get("word", "")
                    for fe in engine.word_to_entry.get(word.lower(), []):
                        matched = [f for f in generate_paradigm(fe)
                                   if match_tags(f["tags"], filter_tags)]
                        if matched:
                            filtered_entries.append({
                                **entry_dict,
                                "forms": [{"form": f["form"], "tags": f["tags"][0]}
                                          for f in matched],
                            })
                            break
                chunk["filtered_entries"] = filtered_entries

    for chunk in results:
        for e in chunk.get("entries", []):
            e.setdefault("gender", _infer_gender(engine, e.get("word", "")))
    return _format(results)


def _format(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no results."
    out: list[str] = []
    for i, chunk in enumerate(results, 1):
        entries = chunk.get("entries", [])
        best = (chunk.get("best_line") or "").split(":", 1)[0].strip()
        filtered = {fe.get("word"): fe.get("forms", [])
                    for fe in chunk.get("filtered_entries", [])}

        if len(entries) == 1:
            prefix, indent = f"{i}. ", "  "
        else:
            pos = chunk.get("pos")
            out.append(f"{i}. cluster{f' ({pos})' if pos else ''}:")
            prefix, indent = "  - ", "    "

        for e in entries:
            word = e.get("word", "")
            marks = []
            if e.get("exact"):
                marks.append("exact match")
            if best and word == best:
                marks.append("best for context")
            mark = f"  ← {', '.join(marks)}" if marks else ""
            out.append(f"{prefix}{head(word, e.get('gender', ''))}{mark}")
            out.extend(translations(e.get("translations", {}), indent))
            for f in filtered.get(word, []):
                out.append(f"{indent}= {f['form']}  {f['tags']}")
        if best and best not in {e.get("word") for e in entries}:
            out.append(f"  context match: {chunk['best_line']}")
        if "filtered_entries" in chunk and not filtered:
            out.append("  (no forms match filter_tags)")
    return "\n".join(out)


def _exact_first(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable partition: chunks with an exact translation match first."""
    def is_exact(c):
        return any(e.get("exact") for e in c.get("entries", []))
    return [c for c in chunks if is_exact(c)] + [c for c in chunks if not is_exact(c)]


def _infer_gender(engine, word: str) -> str:
    """Get gender from the first matching dictionary entry."""
    entries = engine.word_to_entry.get(word.lower(), [])
    for e in entries:
        g = e.get("gender", "")
        if g:
            return g
    return ""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="search",
        description="Semantic search in the Prussian dictionary.",
    )
    ap.add_argument("query", help="Search query (German, English, etc.).")
    ap.add_argument("--top-k", type=int, default=10,
                    help="number of results (default: 10).")
    ap.add_argument("--context", default=None,
                    help="usage context for reranking (enables reranker when set).")
    ap.add_argument("--filter-tags", default=None,
                    help="FST tag filter, e.g. 'Acc+Sg', 'PrfPrc+Pss', 'Opt'.")
    ap.add_argument("--verbose", action="store_true",
                    help="print full traceback on errors.")
    args = ap.parse_args(argv)

    try:
        from prussian.engine.search import SearchEngine
        engine = SearchEngine()
    except Exception as e:
        print(f"error loading engine: {type(e).__name__}: {e}",
              file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    reranker = None
    if args.context:
        from prussian.engine.embeddings.rerank import build_reranker
        reranker = build_reranker()
        if reranker is None:
            print("warning: reranker unavailable — skipping context reranking",
                  file=sys.stderr)

    try:
        output = search_tool(
            engine,
            args.query,
            top_k=args.top_k,
            filter_tags=args.filter_tags,
            reranker=reranker,
            context=args.context,
        )
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    print(output)
    return 0
