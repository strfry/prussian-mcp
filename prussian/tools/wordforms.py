"""get_word_forms — declension/conjugation paradigm (shared core + CLI)."""

from __future__ import annotations

import sys
import traceback
from typing import Any

from prussian.tools._format import head


def wordforms_tool(
    engine,
    lemma: str,
    features: str | None = None,
) -> str:
    """Get all declension or conjugation forms for a Prussian lemma.

    Output is a flat list of forms with their FST tags.  For verbs
    (analyses contain ``V``), the default shows only indicative present
    forms plus the available features.  The ``features``
    argument selects specific morphological categories.

    Args:
        engine: ``SearchEngine`` instance.
        lemma: Prussian base form (from ``lookup_prussian_word``).
        features: optional comma-separated feature filter.  Accepts
            human-readable names (``participle``, ``conjunctive``,
            ``optative``, ``imperative``, ``present``, ``preterite``,
            ``infinitive``) or raw FST tags (``PrfPrc+Pss``, ``Ind``,
            ``Gen+Pl``, ``Ind+Pres+P3``).  Omit for the verb default
            (Ind+Pres) or full list (non-verbs).

    Returns:
        Compact text per entry: ``lemma (gender)  — desc``, one
        ``form: tags`` line per form, and the ``available features`` (the
        FST tags occurring in the paradigm).  Unknown lemma / feature →
        an ``error:`` line (plus the valid features).
    """
    from prussian.engine.fst.tags import (
        fst_available,
        resolve_features,
        match_tags,
        forms_with_tags,
    )

    word_lower = lemma.lower().strip()
    if word_lower not in engine.word_to_entry:
        return f"error: word not found: {lemma}"

    # Resolve feature filter
    wanted_tags: list[str] | None = None
    if features:
        wanted_tags = resolve_features(features)
        if wanted_tags is not None and ("P3" in wanted_tags
                                        or "SP3" in wanted_tags):
            # Finite 3rd-person forms carry no number tag (ast = is/are),
            # so a P3/SP3 request must not also require Sg/Pl.
            wanted_tags = [t for t in wanted_tags if t not in ("Sg", "Pl")]
        if wanted_tags is None:
            valid = sorted(set(
                list(_FEATURE_NAME_MAP.keys()) +
                ["PrsPrc", "PrfPrc", "Act", "Pss", "Refl",
                 "Ind", "Opt", "Subj", "Imprt", "Rel", "Prs", "Prt", "Inf", "Adv",
                 "Nom", "Gen", "Dat", "Acc", "Msc", "Fem", "Neu",
                 "Sg1", "Sg2", "SP3", "Pl1", "Pl2", "P1", "P2", "P3",
                 "Comp", "Superl", "Card", "Ord", "Encl"]
            ))
            return (f"error: unknown feature: {features}\n"
                    f"valid features: {', '.join(valid)}")

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
            # Verb default: Ind+Prs only
            filtered = [
                f for f in fw
                if any(
                    all(
                        p.lower() in {t.lower() for t in tag.split("+")}
                        for p in ["Ind", "Prs"]
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
        if wanted_tags is not None and not forms_out:
            entry_out["note"] = (
                f"no forms match '{features}' — combine tags from "
                "the available features"
            )
        results.append(entry_out)

    return _format(results)


def _format(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no forms found."
    out: list[str] = []
    for entry in results:
        desc = f"— {entry['desc']}" if entry.get("desc") else ""
        out.append(head(entry.get("lemma", "?"), entry.get("gender", ""), desc))
        for f in entry["forms"]:
            tags = " | ".join(f["tags"]) or f.get("pgr", "")
            out.append(f"  {f['form']}: {tags}" if tags else f"  {f['form']}")
        if not entry["forms"]:
            out.append("  (no forms)")
        if entry.get("note"):
            out.append(f"  note: {entry['note']}")
        if entry["available_features"]:
            out.append(f"  available features: {', '.join(entry['available_features'])}")
    return "\n".join(out)


# Canonical presentation order: POS, mood, tense, person, number, case,
# gender, degree/misc.
_TAG_ORDER = [
    "N", "A", "V", "Adv", "Pron", "Num", "Pr", "Adp", "Po",
    "CC", "CS", "Pcle", "Interj", "Prop",
    "Ind", "Opt", "Subj", "Imprt", "Rel", "Prs", "Prt", "Inf",
    "Pss", "Act", "Refl", "PrsPrc", "PrfPrc",
    "Sg1", "Sg2", "SP3", "Pl1", "Pl2", "P1", "P2", "P3", "Sg", "Pl",
    "Nom", "Gen", "Dat", "Acc",
    "Msc", "Fem", "Neu",
    "Comp", "Superl", "Card", "Ord", "Encl",
]


_FEATURE_NAME_MAP = {
    "participle": "PrfPrc",
    "present_participle": "PrsPrc",
    "indicative": "Ind",
    "optative": "Opt",
    "subjunctive": "Subj",
    "imperative": "Imprt",
    "present": "Prs",
    "preterite": "Prt",
    "infinitive": "Inf",
    "passive": "Pss",
    "active": "Act",
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

    print(results)
    return 1 if results.startswith("error:") else 0
