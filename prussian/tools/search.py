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
    chunk_mode = getattr(engine, "chunk_mode", False)

    if chunk_mode:
        results = engine.query(query, top_k)
        # Chunk-mode context reranking: annotate top N chunks
        if context and reranker:
            from prussian_embeddings import annotate_chunk
            from prussian.config import CHUNK_RERANK_TOPN
            for i, chunk in enumerate(results[:CHUNK_RERANK_TOPN]):
                results[i] = annotate_chunk(chunk, context, reranker)
        # filter_tags: per-member entry forms
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
                                           if match_tags(f.get("tags", []),
                                                         filter_tags)]
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
    else:
        # Entry mode
        if context and reranker:
            candidates = engine.query(query, 100)
            from prussian.engine.embeddings.rerank import rerank_results
            results = rerank_results(engine, context, candidates,
                                     reranker)[:top_k]
        else:
            results = engine.query(query, top_k)

    # filter_tags for entry mode
    if not chunk_mode and filter_tags:
        from prussian.engine.fst.tags import forms_with_tags, match_tags, fst_available
        if fst_available():
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

    output: list[dict[str, Any]] = []
    for r in results:
        if chunk_mode:
            output.append(r)
        else:
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
        chunk_mode = getattr(engine, "chunk_mode", False)
        for i, e in enumerate(output, 1):
            if chunk_mode:
                _print_chunk(i, e, context=args.context)
            else:
                _print_entry(i, e)

    return 0


def _print_entry(i: int, e: dict) -> None:
    """Render a single entry-mode result."""
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


def _print_chunk(i: int, chunk: dict, *, context: bool = False) -> None:
    """Render a single chunk-mode result."""
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
