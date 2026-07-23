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

    Returns co-embedded lemma-cluster chunks; use it to find the Prussian
    equivalent of a concept.  Do NOT use for known Prussian forms — use
    ``lookup_tool`` instead.

    Args:
        engine: ``SearchEngine`` instance.
        query: search query (German, English, Lithuanian, Latvian, Polish,
            Russian).
        top_k: number of chunks to return.
        filter_tags: optional FST tag filter, e.g. ``"Akk+Sg"``.  When set,
            each chunk gets a ``filtered_entries`` list (per-member forms
            matching the tags).
        reranker: cross-encoder reranker (from ``build_reranker()``).  Used
            only when ``context`` is set.
        context: optional usage context; each of the top ``CHUNK_RERANK_TOPN``
            chunks is annotated with ``best_line`` / ``lines``.

    Returns:
        List of chunks ``{lemma, members, pos, score, text, entries,
        best_line?, lines?, filtered_entries?}``.
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
                                         "tags": ("+".join(f["tags"]) if f["tags"]
                                                  else f.get("pgr", ""))}
                                        for f in matched
                                    ],
                                    "gender": _infer_gender(engine, word),
                                })
                            break
                chunk["filtered_entries"] = filtered_entries

    return results


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
        for i, e in enumerate(output, 1):
            _print_chunk(i, e, context=args.context)

    return 0


def _print_chunk(i: int, chunk: dict, *, context: str | None = None) -> None:
    """Render a single chunk result."""
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
