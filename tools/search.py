"""search_dictionary — semantic dictionary search (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def search_tool(
    engine,
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    filter_tags: str | None = None,
    reranked_engine=None,
) -> list[dict[str, Any]]:
    """Semantic search in the Prussian dictionary.

    Use this when you have a concept or modern-language word and want to
    find the Prussian equivalent.  Do NOT use for looking up known
    Prussian forms — use ``lookup_tool`` instead.

    Args:
        engine: ``SearchEngine`` instance (for fallback query + word_forms).
        query: search query (German, English, Lithuanian, Latvian,
            Polish, Russian).
        top_k: number of results to return.
        use_reranker: use ``reranked_engine`` when available.
        filter_tags: optional FST tag filter, e.g. ``"Akk+Sg"``,
            ``"Part+Pass"``, ``"Opt"``.  When set, each entry's forms
            are filtered to those whose FST tags contain all wanted tags.
        reranked_engine: optional ``RerankedSearchEngine``; when
            ``use_reranker=True`` and this is ``None``, the reranker
            branch is skipped (same as ``use_reranker=False``).

    Returns:
        List of entries ``{word, translations, forms?, gender?}``.
    """
    if use_reranker and reranked_engine is not None:
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    reranked_engine.search(query, top_k=top_k,
                                           rerank_candidates=100),
                )
                results = future.result()
        else:
            results = asyncio.run(
                reranked_engine.search(query, top_k=top_k,
                                       rerank_candidates=100)
            )
    else:
        results = engine.query(query, top_k)

    # Pre-compute forms_with_tags for all results if filter_tags is set
    forms_cache: dict[str, list[dict]] = {}
    if filter_tags:
        from prussian_engine.fst_tags import forms_with_tags, fst_available
        if fst_available():
            for r in results:
                entries = engine.word_to_entry.get(r["word"].lower(), [])
                for entry in entries:
                    fw = forms_with_tags(engine, entry)
                    if fw:
                        forms_cache[r["word"].lower()] = fw
                        break

    output: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "word": r["word"],
            "translations": r["translations"],
        }
        if filter_tags:
            from prussian_engine.fst_tags import match_tags
            fw = forms_cache.get(r["word"].lower(), [])
            if fw:
                filtered = [f for f in fw
                            if match_tags(f.get("tags", []), filter_tags)]
                if filtered:
                    entry["forms"] = [
                        {"form": f["form"],
                         "tags": "+".join(f["tags"]) if f["tags"] else f.get("pgr", "")}
                        for f in filtered
                    ]
                    entry["gender"] = _infer_gender(engine, r["word"])
        output.append(entry)
    return output


def _infer_gender(engine, word: str) -> str:
    """Get gender from the first matching entry."""
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
    ap.add_argument("--no-reranker", action="store_true",
                    help="skip reranker (faster, less accurate).")
    ap.add_argument("--filter-tags", default=None,
                    help="FST tag filter, e.g. 'Akk+Sg', 'Part+Pass', 'Opt'.")
    ap.add_argument("--json", action="store_true",
                    help="emit raw JSON.")
    ap.add_argument("--verbose", action="store_true",
                    help="print full traceback on errors.")
    args = ap.parse_args(argv)

    try:
        from prussian_engine.search import SearchEngine
        engine = SearchEngine()
    except Exception as e:
        print(f"error loading engine: {type(e).__name__}: {e}",
              file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    reranked_engine = None
    if not args.no_reranker:
        try:
            from prussian_engine.rerank_search import RerankedSearchEngine
            reranked_engine = RerankedSearchEngine(use_reranker=True)
        except ValueError:
            print("warning: RERANK_API_KEY not set — skipping reranker",
                  file=sys.stderr)

    try:
        output = search_tool(
            engine,
            args.query,
            top_k=args.top_k,
            use_reranker=not args.no_reranker,
            filter_tags=args.filter_tags,
            reranked_engine=reranked_engine,
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
        for i, e in enumerate(output, 1):
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

    return 0
