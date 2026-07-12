"""get_word_forms — declension/conjugation paradigm (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def wordforms_tool(
    engine,
    lemma: str,
    filter_pgr: str | None = None,
) -> list[dict[str, Any]] | dict[str, str]:
    """Get all declension or conjugation forms for a Prussian lemma.

    Returns structured forms by category: indicative, optative,
    subjunctive, imperative, participles, declension, adverb,
    comparison.  Use this AFTER ``lookup_prussian_word`` has given
    you the base lemma.  Useful for translation INTO Prussian when
    you need a specific case or tense.

    Args:
        engine: ``SearchEngine`` instance.
        lemma: Prussian base form (from ``lookup_prussian_word``).
        filter_pgr: optional PGR filter, e.g. ``"GEN.PL"``,
            ``"PRS.1.SG"``.
    """
    return engine.get_word_forms(lemma, filter_pgr=filter_pgr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="wordforms",
        description="Get all declension/conjugation forms for a Prussian lemma.",
    )
    ap.add_argument("lemma", help="Prussian base form.")
    ap.add_argument("--filter", default=None,
                    help="PGR filter, e.g. 'GEN.PL', 'PRS.1.SG'.")
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
        results = wordforms_tool(engine, args.lemma, filter_pgr=args.filter)
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
        if isinstance(results, dict) and "error" in results:
            print(results["error"])
            return 1
        if not results:
            print(f"no forms found for: {args.lemma}")
            return 1
        for entry in results:
            lemma = entry.get("lemma", "?")
            trans = entry.get("translations", {})
            trans_str = ", ".join(
                f"{lang}: {t}" for lang, t in trans.items()
            ) if isinstance(trans, dict) else str(trans)
            gender = entry.get("gender", "")
            print(f"{lemma} — {trans_str}"
                  + (f" [{gender}]" if gender else ""))
            forms = entry.get("forms", {})
            for category, items in forms.items():
                if not items:
                    continue
                print(f"  {category}:")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            parts = []
                            for k in ("tense", "person", "number",
                                      "pronoun", "form", "case",
                                      "comparative", "positive"):
                                if k in item and item[k]:
                                    parts.append(f"{k}={item[k]}")
                            print(f"    {', '.join(parts)}")
                        else:
                            print(f"    {item}")
                elif isinstance(items, dict):
                    for k, v in items.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"    {items}")
            filtered = entry.get("filtered_forms")
            if filtered:
                print(f"  filtered ({args.filter}):")
                for f in filtered:
                    print(f"    {f.get('form', '')} [{f.get('pgr', '')}]")

    return 0
