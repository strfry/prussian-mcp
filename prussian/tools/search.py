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
    """Semantic dictionary search core.

    The model-facing tool text lives in :data:`prussian.tools.spec.SEARCH` —
    do not duplicate it here.
    """
    if context and reranker:
        from prussian_embeddings import annotate_chunk
        from prussian.config import CHUNK_RERANK_TOPN

        candidate_count = max(top_k, CHUNK_RERANK_TOPN)
        results = engine.query(query, candidate_count)

        # Step 1: rerank chunks by their full text against context
        texts = [c["text"] for c in results]
        ranked = reranker.rerank(context, texts, top_n=len(texts))

        reranked = []
        seen = set()
        for item in ranked:
            idx = item["index"]
            seen.add(idx)
            chunk = results[idx].copy()
            chunk["rerank_score"] = item["relevance_score"]
            reranked.append(chunk)
        for i, chunk in enumerate(results):
            if i not in seen:
                chunk = chunk.copy()
                chunk["rerank_score"] = 0.0
                reranked.append(chunk)

        # Step 2: keep top_k after reranking
        results = reranked[:top_k]

        # Step 3: line-level annotation within top chunks
        for i, chunk in enumerate(results[:CHUNK_RERANK_TOPN]):
            results[i] = annotate_chunk(chunk, context, reranker)
    else:
        results = engine.query(query, top_k)

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
                            matched = [
                                f
                                for f in fw
                                if match_tags(f.get("tags", []), filter_tags)
                            ]
                            if matched:
                                filtered_entries.append(
                                    {
                                        **entry_dict,
                                        "forms": [
                                            {
                                                "form": f["form"],
                                                "tags": (
                                                    "+".join(f["tags"])
                                                    if f["tags"]
                                                    else f.get("pgr", "")
                                                ),
                                            }
                                            for f in matched
                                        ],
                                        "gender": _infer_gender(engine, word),
                                    }
                                )
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


# ── Compact text formatting ──────────────────────────────────────────────────

DISPLAY_LANGS = ("engl", "miks", "leit", "latt")


def format_search_results(chunks: list[dict]) -> str:
    """Format chunk search results as compact dictionary text.

    One block per chunk: multi-member chunks show a lemma/POS/score header
    followed by indented ``word: engl | miks | leit | latt`` lines; solo
    chunks (one member) collapse to a single line.  With ``context``
    reranking, ``best_line``'s matching entry is marked with ``→``.
    """
    if not chunks:
        return "no results."
    return "\n\n".join(_format_chunk(c) for c in chunks)


def _format_entry_line(entry: dict) -> str:
    """Format one entry as 'word: en | de | lt | lv'."""
    word = entry.get("word", "")
    translations = entry.get("translations", {})
    parts = []
    for lang in DISPLAY_LANGS:
        vals = translations.get(lang) or []
        if isinstance(vals, str):
            vals = [vals]
        parts.append("; ".join(vals))
    return f"{word}: {' | '.join(parts)}"


def _best_word(best_line: str | None) -> str | None:
    """Extract the headword from a best_line string like 'word: ...' or 'word (pos): ...'."""
    if not best_line:
        return None
    head = best_line.split(":", 1)[0]
    head = head.split("(", 1)[0]
    return head.strip()


def _filtered_forms_for(chunk: dict, word: str) -> list[dict]:
    for fe in chunk.get("filtered_entries", []) or []:
        if fe.get("word") == word:
            return fe.get("forms", [])
    return []


def _format_chunk(chunk: dict) -> str:
    lemma = chunk.get("lemma", "")
    pos = chunk.get("pos") or ""
    score = chunk.get("rerank_score", chunk.get("score", 0)) or 0
    members = chunk.get("members", [])
    entries = chunk.get("entries", [])
    bw = _best_word(chunk.get("best_line"))

    if len(members) <= 1:
        entry = entries[0] if entries else {"word": lemma, "translations": {}}
        word, rest = _format_entry_line(entry).split(":", 1)
        lines_out = [f"{word}, score={score:.3f}:{rest}"]
        for f in _filtered_forms_for(chunk, entry.get("word", "")):
            lines_out.append(f"  {f['tags']}: {f['form']}")
        return "\n".join(lines_out)

    header = f"{lemma} ({pos}), score={score:.3f}:" if pos else f"{lemma}, score={score:.3f}:"
    body = [header]
    for entry in entries:
        line = _format_entry_line(entry)
        headword = entry.get("word", "")
        marker = "→ " if bw and headword == bw else "  "
        body.append(f"{marker}{line}")
        for f in _filtered_forms_for(chunk, headword):
            body.append(f"    {f['tags']}: {f['form']}")
    return "\n".join(body)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="search",
        description="Semantic search in the Prussian dictionary.",
    )
    ap.add_argument("query", help="Search query (German, English, etc.).")
    ap.add_argument(
        "--top-k", type=int, default=10, help="number of results (default: 10)."
    )
    ap.add_argument(
        "--context",
        default=None,
        help="usage context for reranking (enables reranker when set).",
    )
    ap.add_argument(
        "--filter-tags",
        default=None,
        help="FST tag filter, e.g. 'Akk+Sg', 'Part+Pass', 'Opt'.",
    )
    ap.add_argument("--json", action="store_true", help="emit raw JSON.")
    ap.add_argument(
        "--verbose", action="store_true", help="print full traceback on errors."
    )
    args = ap.parse_args(argv)

    try:
        from prussian.engine.search import SearchEngine

        engine = SearchEngine()
    except Exception as e:
        print(f"error loading engine: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    reranker = None
    if args.context:
        from prussian.engine.embeddings.rerank import build_reranker

        reranker = build_reranker()
        if reranker is None:
            print(
                "warning: reranker unavailable — skipping context reranking",
                file=sys.stderr,
            )

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

        sys.stdout.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    else:
        print(format_search_results(output))

    return 0
