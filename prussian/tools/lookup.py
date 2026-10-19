"""lookup_prussian_word — sentence-level FST lookup (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any

from prussian.tools._format import head, translations


def lookup_tool(
    engine,
    text: str,
) -> str:
    """Look up a Prussian sentence: tokenize, FST-analyze, enrich from dictionary.

    Each token is analyzed via the FST cascade (surface → lowercase →
    titlecase → short_prep → lenient).  Analyses are grouped by lemma
    (``+Refl`` readings under the reflexive entry ``… si``) and enriched
    with translations and desc from the dictionary.  A token
    without FST analysis is reported as such — there is no dictionary
    fallback, so FST coverage gaps stay visible.

    Args:
        engine: ``SearchEngine`` instance.
        text: Prussian text (one or more sentences).

    Returns:
        Compact text: per token its analyses (lemma, tags, desc) with
        translations in all six languages.
    """
    from prussian.engine.fst.tags import (
        analyze_words, entry_lemma, fst_available, fst_error_message, tokenize,
    )

    if not fst_available():
        return f"error: FST not available: {fst_error_message()}"

    # 1. Tokenize, discard punctuation
    word_tokens = [t for t in tokenize(text) if t and t[0].isalpha()]
    if not word_tokens:
        return _format([])

    # 2. FST analysis batch (deduplicated)
    fst_results = analyze_words(list(dict.fromkeys(word_tokens)))

    # 3. Build output per token, grouping analyses by lemma
    results: list[dict[str, Any]] = []
    for token in word_tokens:
        info = fst_results.get(token, {})
        fst_analyses = info.get("analyses", [])
        via = info.get("via")
        matched_form = info.get("matched_form")

        if fst_analyses:
            # Group by dictionary headword: +Refl readings belong to the
            # reflexive entry (laikātun si), the rest to the bare verb.
            lemma_groups: dict[str, list[dict[str, Any]]] = {}
            bare_of: dict[str, str] = {}
            for lemma, tags in fst_analyses:
                key = entry_lemma(lemma, tags).lower()
                bare_of[key] = lemma.lower()
                if key not in lemma_groups:
                    lemma_groups[key] = []
                lemma_groups[key].append({
                    "tags": "+".join(tags),
                })

            analyses_out: list[dict[str, Any]] = []
            for lemma_lower, tag_list in lemma_groups.items():
                entries = (engine.word_to_entry.get(lemma_lower)
                           or engine.word_to_entry.get(bare_of[lemma_lower], []))
                if entries:
                    for entry in entries:
                        a: dict[str, Any] = {
                            "lemma": entry.get("word", lemma_lower),
                            "tags": [t["tags"] for t in tag_list],
                            "translations": entry.get("translations", {}),
                            "desc": entry.get("desc", ""),
                            "gender": entry.get("gender", ""),
                        }
                        analyses_out.append(a)
                else:
                    analyses_out.append({
                        "lemma": lemma_lower,
                        "tags": [t["tags"] for t in tag_list],
                        "translations": {},
                        "desc": "",
                        "gender": "",
                    })

            entry_out: dict[str, Any] = {
                "form": token,
                "method": "fst",
                "analyses": analyses_out,
            }
            if via and via != "surface":
                entry_out["adjustment"] = {
                    "via": via,
                    "matched_form": matched_form,
                }
            results.append(entry_out)
        else:
            results.append({"form": token, "method": "oov"})

    return _format(results)


def _format(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no words found."
    out: list[str] = []
    for r in results:
        form = r.get("form", "?")
        if r.get("method") == "fst":
            adj = r.get("adjustment")
            via = f"  (via {adj['via']}: {adj['matched_form']})" if adj else ""
            out.append(f"{form}{via}")
            for a in r.get("analyses", []):
                tags = " | ".join(a.get("tags", []))
                out.append(f"  {head(a.get('lemma', '?'), a.get('gender', ''), tags)}"
                           + (f"  — {a['desc']}" if a.get("desc") else ""))
                out.extend(translations(a.get("translations", {}), "    "))
            continue
        out.append(f"{form}  — not in FST (no analysis)")
    return "\n".join(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="lookup",
        description="Look up a Prussian sentence: FST analysis + dictionary.",
    )
    ap.add_argument("text", help="Prussian text (sentence or sentences).")
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

    try:
        results = lookup_tool(engine, args.text)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    print(results)
    return 0
