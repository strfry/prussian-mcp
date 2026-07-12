"""lookup_prussian_word — reverse dictionary lookup (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def lookup_tool(
    engine,
    word: str,
    fuzzy: bool = False,
    apply_rules: bool = True,
) -> list[dict[str, Any]]:
    """Look up a specific Prussian word (lemma or inflected form).

    Searches all form categories: indicative, subjunctive, optative,
    imperative, participles, declensions.  Use this when you already
    have a Prussian word and need its meaning or base form.  For a
    full sentence, call once per word — never pass the whole sentence.

    Args:
        engine: ``SearchEngine`` instance.
        word: single Prussian word (lemma or inflected form).
        fuzzy: set ``True`` if exact lookup fails or the word may have
            spelling variants.
        apply_rules: when ``True`` and exact lookup fails, try prefix
            stripping (ni-, pa-, pra-, …) and orthographic
            transformations.
    """
    return engine.lookup(word, fuzzy=fuzzy, apply_rules=apply_rules)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="lookup",
        description="Look up a specific Prussian word (lemma or inflected form).",
    )
    ap.add_argument("word", help="Prussian word to look up.")
    ap.add_argument("--fuzzy", action="store_true",
                    help="enable Levenshtein fallback for misspellings.")
    ap.add_argument("--no-rules", action="store_true",
                    help="disable prefix-stripping and orthographic rules.")
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
        results = lookup_tool(
            engine,
            args.word,
            fuzzy=args.fuzzy,
            apply_rules=not args.no_rules,
        )
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
            print(f"not found: {args.word}")
            return 1
        for r in results:
            word = r.get("word", "?")
            trans = r.get("translations", {})
            trans_str = ", ".join(
                f"{lang}: {t}" for lang, t in trans.items()
            ) if isinstance(trans, dict) else str(trans)
            method = r.get("method", "")
            form = r.get("form", "")
            pgr = r.get("pgr", "")
            header = f"{word}"
            if form:
                header += f" (form: {form})"
            if pgr:
                header += f" [{pgr}]"
            if method:
                header += f" ({method})"
            print(f"  {header} — {trans_str}")
            gender = r.get("gender", "")
            if gender:
                print(f"    gender: {gender}")
            desc = r.get("desc", "")
            if desc:
                print(f"    {desc}")

    return 0
