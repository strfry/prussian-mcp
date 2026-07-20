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
            ``Gen+Pl``, ``Ind+Pres+P3``).  Omit for the verb default
            (Ind+Pres) or full list (non-verbs).

    Returns:
        List of entries ``{lemma, desc, gender, forms,
        available_features}`` where ``available_features`` is the set of
        FST tags occurring in the lemma's paradigm.  On unknown feature,
        returns an error dict with ``valid_features``.
    """
    from prussian.engine.fst.tags import (
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
        if wanted_tags is not None and "P3" in wanted_tags:
            # Finite 3rd-person forms carry no number tag (ast = is/are),
            # so a P3 request must not also require Sg/Pl.
            wanted_tags = [t for t in wanted_tags if t not in ("Sg", "Pl")]
        if wanted_tags is None:
            valid = sorted(set(
                list(_FEATURE_NAME_MAP.keys()) +
                ["Part", "Ind", "Opt", "Subj", "Imp", "Rel",
                 "Pres", "Pret", "Inf", "Pass", "Refl", "Adv",
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
        fw = forms_with_tags(engine, entry) if fst_available() else []

        # Detect POS from tag inventory
        all_tag_strs = [f["tags"] for f in fw]
        is_verb = any(
            any(p.lower() == "v" for t in tags for p in t.split("+"))
            for tags in all_tag_strs
        )

        # The set of individual tags occurring in the paradigm, in
        # canonical order — any combination of them is a `features` filter.
        present: set[str] = set()
        for f in fw:
            for t in f.get("tags", []):
                present.update(t.split("+"))
        available = [t for t in _TAG_ORDER if t in present]
        available += sorted(present - set(_TAG_ORDER))

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
            "desc": entry.get("desc", ""),
            "gender": entry.get("gender", ""),
            "forms": forms_out,
            "available_features": available,
        }
        if not forms_out and not available:
            entry_out["note"] = (
                "indeclinable — no inflected forms exist; use the lemma "
                "unchanged"
            )
        elif wanted_tags is not None and not forms_out:
            entry_out["note"] = (
                f"no forms match '{features}' — combine tags from "
                "available_features"
            )
        results.append(entry_out)

    return results


# Canonical presentation order: POS, mood, tense, person, number, case,
# gender, degree/misc.
_TAG_ORDER = [
    "N", "V", "Adj", "Adv", "Pron", "Num", "Part", "Prp", "Psp",
    "Cnj", "SCnj", "Pcl", "IJ", "PropN",
    "Ind", "Opt", "Subj", "Imp", "Rel", "Pres", "Pret", "Inf",
    "Pass", "Refl",
    "P1", "P2", "P3", "Sg", "Pl",
    "Nom", "Gen", "Dat", "Akk",
    "Masc", "Fem", "Neut",
    "Cmp", "Sup", "Card", "Ord", "Encl",
    "GovAkk", "GovDat", "GovGen",
]


_FEATURE_NAME_MAP = {
    "participle": "Part",
    "indicative": "Ind",
    "optative": "Opt",
    "subjunctive": "Subj",
    "imperative": "Imp",
    "present": "Pres",
    "preterite": "Pret",
    "infinitive": "Inf",
    "adverb": "Adv",
}


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
        from prussian.engine.search import SearchEngine
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
            gender = entry.get("gender", "")
            desc = entry.get("desc", "")
            avail = entry.get("available_features", [])

            header = lem
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
