"""search_dictionary — semantic dictionary search (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def search_tool(
    engine,
    query: str,
    top_k: int = 10,
    filter_tags: str | None = None,
    reranker=None,
    context: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search in the Prussian dictionary.

    Use this when you have a concept or modern-language word and want to
    find the Prussian equivalent.  Do NOT use for looking up known
    Prussian forms — use ``lookup_tool`` instead.

    Args:
        engine: ``SearchEngine`` instance.
        query: search query (German, English, Lithuanian, Latvian,
            Polish, Russian).
        top_k: number of results to return.
        filter_tags: optional FST tag filter, e.g. ``"Akk+Sg"``,
            ``"Part+Pass"``, ``"Opt"``.  When set, each entry's forms
            are filtered to those whose FST tags contain all wanted tags.
        reranker: cross-encoder reranker object (from ``build_reranker()``).
            When ``None`` or context is empty, reranking is skipped.
        context: optional usage context that enables context-aware
            reranking.  In chunk mode each top chunk is annotated with
            ``best_line`` / ``lines``.  In entry mode the cross-encoder
            reranks candidate entries.

    Returns:
        List of entries.  Entry mode: ``{word, translations, forms?, gender?}``.
        Chunk mode: ``{lemma, members, pos, score, text, entries,
        best_line?, lines?, filtered_entries?}``.
    """
    from prussian.engine.backends import backend_for

    return backend_for(engine).run(
        engine, query, top_k,
        filter_tags=filter_tags, reranker=reranker, context=context,
    )


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
                    help="FST tag filter, e.g. 'Akk+Sg', 'Part+Pass', 'Opt'.")
    ap.add_argument("--json", action="store_true",
                    help="emit raw JSON.")
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

    if args.json:
        import json
        sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=2)
                         + "\n")
    else:
        if not output:
            print("no results.")
        from prussian.engine.backends import backend_for
        backend = backend_for(engine)
        for i, e in enumerate(output, 1):
            backend.render(i, e, context=args.context)

    return 0
