"""lookup_prussian_word — sentence-level FST lookup (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any

from prussian.tools._format import head, translations


def lookup_tool(
    engine,
    text: str,
    fuzzy: bool = False,
) -> str:
    """Look up a Prussian sentence: tokenize, FST-analyze, enrich from dictionary.

    Each token is analyzed via the FST cascade (surface → lowercase →
    titlecase → short_prep → lenient).  Analyses are grouped by lemma
    and enriched with translations and desc from the dictionary.  Tokens
    without FST analyses fall back to the dictionary ``engine.lookup()``.

    Args:
        engine: ``SearchEngine`` instance.
        text: Prussian text (one or more sentences).
        fuzzy: set ``True`` for Levenshtein fallback on OOV tokens.

    Returns:
        Compact text: per token its analyses (lemma, tags, desc) with
        translations in all six languages.
    """
    from prussian.engine.fst.tags import analyze_words, fst_available
    from prussian.engine.fst.validate import fst_api

    # 1. Tokenize, discard punctuation
    if fst_available():
        from prussian.engine.fst.tags import tokenize
        all_tokens = tokenize(text)
    else:
        import re
        all_tokens = re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", text)

    word_tokens = [t for t in all_tokens if t and t[0].isalpha()]
    if not word_tokens:
        return _format([])

    # 2. FST analysis batch (deduplicated)
    unique_types = list(dict.fromkeys(word_tokens))
    if fst_available():
        fst_results = analyze_words(unique_types)
    else:
        fst_results = {}

    # 3. Build output per token, grouping analyses by lemma
    results: list[dict[str, Any]] = []
    for token in word_tokens:
        info = fst_results.get(token, {})
        fst_analyses = info.get("analyses", [])
        via = info.get("via")
        matched_form = info.get("matched_form")

        if fst_analyses:
            # Group by lemma
            lemma_groups: dict[str, list[dict[str, Any]]] = {}
            for lemma, tags in fst_analyses:
                key = lemma.lower()
                if key not in lemma_groups:
                    lemma_groups[key] = []
                lemma_groups[key].append({
                    "tags": "+".join(tags),
                })

            analyses_out: list[dict[str, Any]] = []
            for lemma_lower, tag_list in lemma_groups.items():
                entries = engine.word_to_entry.get(lemma_lower, [])
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
            # OOV fallback → dictionary lookup
            dict_results = engine.lookup(token, fuzzy=fuzzy, apply_rules=True)
            matches: list[dict[str, Any]] = []
            for dr in dict_results:
                m: dict[str, Any] = {
                    "word": dr.get("word", ""),
                    "translations": dr.get("translations", {}),
                    "desc": dr.get("desc", ""),
                    "gender": dr.get("gender", ""),
                }
                if dr.get("form"):
                    m["form"] = dr["form"]
                if dr.get("pgr"):
                    m["pgr"] = dr["pgr"]
                if dr.get("method"):
                    m["method"] = dr["method"]
                if dr.get("rule_applied"):
                    m["rule_applied"] = dr["rule_applied"]
                if dr.get("matched_form"):
                    m["matched_form"] = dr["matched_form"]
                matches.append(m)
            results.append({
                "form": token,
                "method": "dictionary_fallback",
                "matches": matches,
            })

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
        matches = r.get("matches", [])
        if not matches:
            out.append(f"{form}  — not found (no FST analysis, no dictionary match)")
            continue
        out.append(f"{form}  (dictionary fallback, no FST analysis)")
        for m in matches:
            how = "/".join(x for x in (m.get("method"), m.get("rule_applied")) if x)
            info = " ".join(x for x in (
                f"form {m['form']}" if m.get("form") else "",
                m.get("pgr", ""),
                f"[{how}]" if how else "",
            ) if x)
            out.append(f"  {head(m.get('word', '?'), m.get('gender', ''), info)}"
                       + (f"  — {m['desc']}" if m.get("desc") else ""))
            out.extend(translations(m.get("translations", {}), "    "))
    return "\n".join(out)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="lookup",
        description="Look up a Prussian sentence: FST analysis + dictionary.",
    )
    ap.add_argument("text", help="Prussian text (sentence or sentences).")
    ap.add_argument("--fuzzy", action="store_true",
                    help="enable Levenshtein fallback for OOV tokens.")
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
        results = lookup_tool(engine, args.text, fuzzy=args.fuzzy)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    print(results)
    return 0
