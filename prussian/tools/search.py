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
    """Semantic search in the Prussian dictionary.

    Returns co-embedded lemma-cluster chunks; use it to find the Prussian
    equivalent of a concept.  Do NOT use for known Prussian forms — use
    ``lookup_tool`` instead.

    Args:
        engine: ``SearchEngine`` instance.
        query: search query (German, English, Lithuanian, Latvian, Polish,
            Russian).
        top_k: number of chunks to return.
        filter_tags: optional FST tag filter, e.g. ``"Acc+Sg"``.  When set,
            each entry lists its forms matching the tags.
        reranker: cross-encoder reranker (from ``build_reranker()``).  Used
            only when ``context`` is set.
        context: optional usage context; within each of the top
            ``CHUNK_RERANK_TOPN`` chunks the best-matching entry is marked.

    Returns:
        Compact text: one numbered block per chunk, each entry as
        ``lemma (gender)`` plus its translations in all six languages.
    """
    results = engine.query(query, top_k)

    if context and reranker:
        from prussian_embeddings import annotate_chunk
        from prussian.config import CHUNK_RERANK_TOPN
        for i, chunk in enumerate(results[:CHUNK_RERANK_TOPN]):
            results[i] = annotate_chunk(chunk, context, reranker)

    if filter_tags:
        from prussian.engine.fst.tags import forms_with_tags, match_tags, fst_available
        if fst_available():
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
                                         "tags": (" | ".join(f["tags"]) if f["tags"]
                                                  else f.get("pgr", ""))}
                                        for f in matched
                                    ],
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
