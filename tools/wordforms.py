"""get_word_forms — declension/conjugation paradigm (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any


def wordforms_tool(
    engine,
    lemma: str,
    features: str | None = None,
) -> list[dict[str, Any]] | dict[str, str]:
    """Get all declension or conjugation forms for a Prussian lemma.

    Output is a flat list of forms with their FST tags.  For verbs
    (analyses contain ``V``), the default shows only indicative present
    forms plus a list of ``available_features``.  The ``features``
    argument selects specific morphological categories.

    Args:
        engine: ``SearchEngine`` instance.
        lemma: Prussian base form (from ``lookup_prussian_word``).
        features: optional comma-separated feature filter.  Accepts
            human-readable names (``participle``, ``conjunctive``,
            ``optative``, ``imperative``, ``present``, ``preterite``,
            ``infinitive``) or raw FST tags (``Part+Pass``, ``Ind``,
            ``Gen+Pl``).  Omit for the verb default (Ind+Pres) or
            full list (non-verbs).

    Returns:
        List of entries ``{lemma, translations, desc, gender, forms,
        available_features}``.  On unknown feature, returns an error
        dict with ``valid_features``.
    """
    from prussian_engine.fst_tags import (
        fst_available,
        resolve_features,
        match_tags,
        forms_with_tags,
    )

    word_lower = lemma.lower().strip()
    if word_lower not in engine.word_to_entry:
        return {"error": f"Word not found: {lemma}"}

    # Resolve feature filter
    wanted_tags: list[str] | None = None
    if features:
        wanted_tags = resolve_features(features)
        if wanted_tags is None:
            valid = sorted(set(
                list(_FEATURE_NAME_MAP.keys()) +
                ["Part", "Ind", "Opt", "Subj", "Imp", "Rel",
                 "Pres", "Pret", "Inf", "Pass", "Refl",
                 "Nom", "Gen", "Dat", "Akk",
                 "Sg", "Pl", "Masc", "Fem", "Neut",
                 "P1", "P2", "P3"]
            ))
            return {
                "error": f"Unknown feature: {features}",
                "valid_features": valid,
            }

    entries = engine.word_to_entry[word_lower]
    results: list[dict[str, Any]] = []

    for entry in entries:
        translations = entry.get("translations", {})
        fw = forms_with_tags(engine, entry) if fst_available() else []

        # Detect POS from tag inventory
        all_tag_strs = [f["tags"] for f in fw]
        is_verb = any(
            any(p.lower() == "v" for t in tags for p in t.split("+"))
            for tags in all_tag_strs
        )

        # Collect all unique tags present across all forms
        present_tags: set[str] = set()
        for f in fw:
            for t in f.get("tags", []):
                for part in t.split("+"):
                    present_tags.add(part.lower())

        # Build available_features from present tags
        available = _available_features(present_tags)

        # Filter forms
        if wanted_tags is not None:
            filtered = [
                f for f in fw
                if match_tags(f.get("tags", []), " ".join(wanted_tags))
            ]
        elif is_verb and not features:
            # Verb default: Ind+Pres only
            filtered = [
                f for f in fw
                if any(
                    all(
                        p.lower() in {t.lower() for t in tag.split("+")}
                        for p in ["Ind", "Pres"]
                    )
                    for tag in f.get("tags", [])
                )
            ]
        else:
            filtered = fw

        # Format output
        forms_out = [
            {"form": f["form"], "tags": f["tags"] if f["tags"] else []}
            for f in filtered
        ]

        # Include non-FST forms with PGR fallback (only when no filter)
        has_filter = wanted_tags is not None or (is_verb and not features)
        if not has_filter:
            for f in fw:
                if not f["tags"] and f.get("pgr"):
                    forms_out.append({
                        "form": f["form"],
                        "tags": [],
                        "pgr": f["pgr"],
                    })

        entry_out: dict[str, Any] = {
            "lemma": entry.get("word", ""),
            "translations": translations,
            "desc": entry.get("desc", ""),
            "gender": entry.get("gender", ""),
            "forms": forms_out,
            "available_features": available,
        }
        results.append(entry_out)

    return results


_FEATURE_NAME_MAP = {
    "participle": "Part",
    "indicative": "Ind",
    "optative": "Opt",
    "subjunctive": "Subj",
    "imperative": "Imp",
    "present": "Pres",
    "preterite": "Pret",
    "infinitive": "Inf",
}


def _available_features(present_tags: set[str]) -> list[str]:
    """Map present FST tag inventory to human-readable feature names."""
    features = []
    tag_checks = [
        ("Ind", "indicative"), ("Pres", "present"), ("Pret", "preterite"),
        ("Opt", "optative"), ("Subj", "subjunctive"), ("Imp", "imperative"),
        ("Inf", "infinitive"), ("Part", "participle"),
        ("Pass", "passive"), ("Refl", "reflexive"),
        ("Nom", "nominative"), ("Gen", "genitive"),
        ("Dat", "dative"), ("Akk", "accusative"),
        ("Sg", "singular"), ("Pl", "plural"),
        ("Masc", "masculine"), ("Fem", "feminine"), ("Neut", "neuter"),
    ]
    for tag, name in tag_checks:
        if tag.lower() in present_tags:
            features.append(name)
    return features


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="wordforms",
        description="Get all declension/conjugation forms for a Prussian lemma.",
    )
    ap.add_argument("lemma", help="Prussian base form.")
    ap.add_argument("--features", default=None,
                    help="Feature filter: 'participle', 'Ind+Pres', etc.")
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
        results = wordforms_tool(engine, args.lemma, features=args.features)
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
            vf = results.get("valid_features", [])
            if vf:
                print(f"  valid features: {', '.join(vf)}")
            return 1
        if not results:
            print(f"no forms found for: {args.lemma}")
            return 1
        for entry in results:
            lem = entry.get("lemma", "?")
            trans = entry.get("translations", {})
            trans_str = ", ".join(
                f"{lang}: {t}" for lang, t in trans.items()
            ) if isinstance(trans, dict) else str(trans)
            gender = entry.get("gender", "")
            desc = entry.get("desc", "")
            avail = entry.get("available_features", [])

            header = f"{lem} — {trans_str}"
            if gender:
                header += f" [{gender}]"
            if desc:
                header += f"  {desc}"
            print(header)

            forms = entry.get("forms", [])
            if not forms:
                print("  (no forms)")
            for f in forms:
                tags = f.get("tags", [])
                form = f.get("form", "")
                pgr = f.get("pgr", "")
                if tags:
                    print(f"  {form}  [{' + '.join(tags)}]")
                elif pgr:
                    print(f"  {form}  [{pgr}]")
                else:
                    print(f"  {form}")

            if avail:
                print(f"  [available: {', '.join(avail)}]")

    return 0
