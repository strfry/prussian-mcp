"""lookup_prussian_word — sentence-level FST lookup (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def lookup_tool(
    engine,
    text: str,
    fuzzy: bool = False,
) -> list[dict[str, Any]]:
    """Look up a Prussian sentence: tokenize, FST-analyze, enrich from dictionary.

    Each token is analyzed via the FST cascade (surface → lowercase →
    titlecase → short_prep → lenient).  Analyses are grouped by lemma
    and enriched with translations and desc from the dictionary.  Tokens
    without FST analyses fall back to the dictionary ``engine.lookup()``.

    Args:
        engine: ``SearchEngine`` instance.
        text: Prussian text (one or more sentences).
        fuzzy: set ``True`` for Levenshtein fallback on OOV tokens.
    """
    from prussian_engine.fst_tags import analyze_words, fst_available
    from prussian_engine.fsg_check import fst_api

    # 1. Tokenize, discard punctuation
    if fst_available():
        from prussian_engine.fst_tags import tokenize
        all_tokens = tokenize(text)
    else:
        import re
        all_tokens = re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", text)

    word_tokens = [t for t in all_tokens if t and t[0].isalpha()]
    if not word_tokens:
        return []

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

    return results


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

    try:
        results = lookup_tool(engine, args.text, fuzzy=args.fuzzy)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    if args.json:
        import json
        sys.stdout.write(json.dumps(results, ensure_ascii=False, indent=2)
                         + "\n")
    else:
        if not results:
            print("no words found.")
            return 1
        for r in results:
            form = r.get("form", "?")
            method = r.get("method", "")
            adjustment = r.get("adjustment")

            if method == "fst":
                analyses = r.get("analyses", [])
                parts = [f"{form}"]
                if adjustment:
                    via = adjustment.get("via", "")
                    mf = adjustment.get("matched_form", "")
                    parts.append(f"({via}: {mf})" if mf else f"({via})")
                parts.append(f"[{method}]")
                print(" ".join(parts))
                for a in analyses:
                    lemma = a.get("lemma", "?")
                    tags_str = " | ".join(a.get("tags", []))
                    trans = a.get("translations", {})
                    trans_str = ", ".join(
                        f"{lang}: {t}" for lang, t in trans.items()
                    ) if isinstance(trans, dict) else str(trans)
                    desc = a.get("desc", "")
                    line = f"  {lemma} [{tags_str}]"
                    if trans_str:
                        line += f" — {trans_str}"
                    if desc:
                        line += f"  {desc}"
                    print(line)
            elif method == "dictionary_fallback":
                matches = r.get("matches", [])
                if matches:
                    parts = [f"{form}"]
                    parts.append("[dictionary_fallback]")
                    print(" ".join(parts))
                    for m in matches:
                        word = m.get("word", "?")
                        trans = m.get("translations", {})
                        trans_str = ", ".join(
                            f"{lang}: {t}" for lang, t in trans.items()
                        ) if isinstance(trans, dict) else str(trans)
                        m_method = m.get("method", "")
                        rule = m.get("rule_applied", "")
                        line = f"  {word} — {trans_str}"
                        if m_method:
                            line += f"  ({m_method}"
                            if rule:
                                line += f"/{rule}"
                            line += ")"
                        print(line)
                else:
                    print(f"{form} [not found]")

    return 0
