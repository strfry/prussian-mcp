"""FST-based tag analysis, tag matching, and form extraction.

Guarded import pattern (fsg_check.py): the server starts without FST;
the tools fall back to dictionary-only mode and emit a warning.
"""

from __future__ import annotations

import re
from typing import Any

# Guarded Import
try:
    from prussian_fst import api as fst_api
    from prussian_fst.cg3_pipeline import (
        PREP_SHORT,
        titlecase,
        tokenize,
    )

    _IMPORT_ERROR = None
except ImportError as e:
    fst_api = None
    tokenize = None  # type: ignore[assignment]
    PREP_SHORT = {}  # type: ignore[assignment]

    def titlecase(t: str) -> str:  # type: ignore[misc]
        return t[0].upper() + t[1:]

    _IMPORT_ERROR = (
        f"prussian_fst nicht importierbar ({e}) — im prussian-mcp-"
        "Checkout `uv sync` ausführen; der prussian-fst-Checkout muss "
        "am Pfad aus pyproject [tool.uv.sources] liegen "
        "(Default: ../prussian-fst)."
    )

try:
    from prussian_fst.fst_lookup import flookup_batch
except ImportError:
    from prussian_fst.cg3_pipeline import flookup_batch  # type: ignore[no-redef]

try:
    from prussian_fst.cg3_pipeline import DEFAULT_FST, DEFAULT_LENIENT
except ImportError:
    from pathlib import Path

    _FST_DIR = Path(__file__).resolve().parents[2] / "prussian-fst" / "fst"
    DEFAULT_FST = _FST_DIR / "build/base.hfstol"
    DEFAULT_LENIENT = _FST_DIR / "build/lenient.hfstol"


def fst_available() -> bool:
    """True if the FST pipeline is importable."""
    return fst_api is not None


def fst_error_message() -> str:
    """Human-readable message when FST is not available."""
    return _IMPORT_ERROR or ""


# ── Feature name mapping (human-readable → FST tag) ──────────────────────────

FEATURE_MAP: dict[str, str] = {
    "participle": "Part",
    "conjunctive": "Subj",
    "subjunctive": "Subj",
    "optative": "Opt",
    "imperative": "Imp",
    "indicative": "Ind",
    "present": "Pres",
    "preterite": "Pret",
    "past": "Pret",
    "infinitive": "Inf",
    "relative": "Rel",
    "passive": "Pass",
    "reflexive": "Refl",
    "nominative": "Nom",
    "genitive": "Gen",
    "dative": "Dat",
    "accusative": "Akk",
    "singular": "Sg",
    "plural": "Pl",
    "masculine": "Masc",
    "feminine": "Fem",
    "neuter": "Neut",
    "comparative": "Cmp",
    "superlative": "Sup",
    "adverb": "Adv",
    "cardinal": "Card",
    "ordinal": "Ord",
}


def resolve_features(spec: str) -> list[str] | None:
    """Resolve a comma-separated feature spec to FST tags.

    Handles human-readable names (``"participle"``, ``"conjunctive"``),
    raw FST tags (``"Part"``, ``"Ind+Pres"``), and mixed input.
    Returns ``None`` on unknown feature (caller should emit error dict).
    """
    if not spec:
        return None

    wanted: list[str] = []
    for part in re.split(r"[,+]", spec):
        part = part.strip()
        if not part:
            continue
        low = part.lower()
        if low in FEATURE_MAP:
            wanted.append(FEATURE_MAP[low])
        elif part in _KNOWN_FST_TAGS:
            wanted.append(part)
        elif low in _KNOWN_FST_TAGS_LOWER:
            wanted.append(_KNOWN_FST_TAGS_LOWER[low])
        else:
            return None  # unknown
    return wanted


# Build case-insensitive lookup for FST tags
_KNOWN_FST_TAGS = {
    "N", "Adj", "Pron", "Num", "V", "Part", "Adv", "Prp", "Psp",
    "Cnj", "SCnj", "Pcl", "IJ", "PropN",
    "Sg", "Pl", "Nom", "Gen", "Dat", "Akk",
    "Masc", "Fem", "Neut",
    "Pres", "Pret", "Inf", "Ind", "Opt", "Imp", "Subj", "Rel",
    "Pass", "Refl",
    "P1", "P2", "P3",
    "Cmp", "Sup", "Card", "Ord", "Encl",
    "GovAkk", "GovDat", "GovGen",
}
_KNOWN_FST_TAGS_LOWER = {t.lower(): t for t in _KNOWN_FST_TAGS}

# POS tags — to detect verbs for default filtering
_VERB_POS = {"V"}


# ── Tag matching ──────────────────────────────────────────────────────────────


def match_tags(tags: list[str], wanted: str) -> bool:
    """Check whether *tags* contain all wanted tags in at least one reading.

    ``wanted`` is split on ``+``, ``.``, ``,`` and whitespace; each piece
    is checked against at least one individual tag string (which may be
    ``+``-joined, e.g. ``"N+Sg+Akk+Fem"``).

    A form matches when at least one of its tag strings contains ALL
    wanted tags (case-insensitive).

    Examples::

        match_tags(["N+Sg+Akk+Fem"], "Akk+Sg")  → True
        match_tags(["N+Sg+Nom+Fem"], "Akk+Sg")  → False
        match_tags(["V+Ind+Pres+P1+Sg"], "Ind+Pres")  → True
        match_tags(["N+Sg+Gen+Fem", "N+Pl+Nom+Fem"], "Gen+Pl")  → False
        match_tags(["N+Pl+Gen+Fem"], "Gen+Pl")  → True
    """
    if not wanted:
        return True
    wanted_parts = {p.lower() for p in re.split(r"[+.\s,]+", wanted) if p}
    for t in tags:
        parts = {p.lower() for p in t.split("+")}
        if wanted_parts <= parts:
            return True
    return False


# ── Cascade analysis with provenance ─────────────────────────────────────────


def analyze_words(
    words: list[str],
    *,
    fst_path: str | None = None,
    lenient_path: str | None = None,
) -> dict[str, dict]:
    """Non-disambiguated FST analyses per word, with provenance.

    Cascade: surface → lowercase → titlecase → short_prep → lenient.
    Each word gets ``{"analyses": [(lemma, tags)], "via": <stage>,
    "matched_form": <form_or_None>}``.

    All FST access runs under ``fst_api._PIPELINE_LOCK`` when available.
    """
    if fst_api is None:
        return {w: {"analyses": [], "via": None, "matched_form": None}
                for w in words}
    if not words:
        return {}

    from pathlib import Path

    fst = Path(fst_path) if fst_path else DEFAULT_FST
    len_p = Path(lenient_path) if lenient_path else DEFAULT_LENIENT

    with fst_api._PIPELINE_LOCK:
        # Stage 1: surface
        surface = flookup_batch(words, fst)

        # Collect missing for further stages
        missing = {w for w in words if w not in surface}

        # Stage 2+3: lowercase + titlecase
        alt_forms: set[str] = set()
        for w in missing:
            lo = w.lower()
            if lo != w:
                alt_forms.add(lo)
            tc = titlecase(w)
            if tc != w:
                alt_forms.add(tc)
        alt = flookup_batch(sorted(alt_forms), fst) if alt_forms else {}

        # Stage 4: short prepositions
        short_prep_words = [w for w in missing
                            if w.lower() in PREP_SHORT and w not in surface]
        long_forms = {PREP_SHORT[w.lower()]: w
                      for w in short_prep_words}
        long_preps = (flookup_batch(sorted(set(long_forms.keys())), fst)
                      if long_forms else {})

        # Stage 5: lenient (orthographic correction layer)
        still_missing = set()
        result: dict[str, dict] = {}
        for w in words:
            if w in surface:
                result[w] = {"analyses": surface[w],
                             "via": "surface", "matched_form": None}
            elif w.lower() != w and w.lower() in alt:
                result[w] = {"analyses": alt[w.lower()],
                             "via": "lowercase", "matched_form": w.lower()}
            elif titlecase(w) != w and titlecase(w) in alt:
                result[w] = {"analyses": alt[titlecase(w)],
                             "via": "titlecase", "matched_form": titlecase(w)}
            elif w.lower() in PREP_SHORT:
                lp = long_preps.get(PREP_SHORT[w.lower()])
                if lp:
                    result[w] = {"analyses": lp,
                                 "via": "short_prep",
                                 "matched_form": PREP_SHORT[w.lower()]}
                else:
                    result[w] = {"analyses": [], "via": None,
                                 "matched_form": None}
                    still_missing.add(w)
            else:
                result[w] = {"analyses": [], "via": None,
                             "matched_form": None}
                still_missing.add(w)

        # Lenient pass for remaining unknowns
        if still_missing and len_p.exists():
            variants: set[str] = set()
            for w in still_missing:
                lo = w.lower()
                variants.add(lo)
                tc = titlecase(w)
                variants.add(tc)
            lenient = flookup_batch(sorted(variants), len_p)
            for w in still_missing:
                lo = w.lower()
                tc = titlecase(w)
                analyses = lenient.get(lo) or lenient.get(tc)
                if analyses:
                    matched = lo if lo in lenient and lenient[lo] else tc
                    result[w] = {"analyses": analyses,
                                 "via": "lenient", "matched_form": matched}

    return result


# ── Form extraction: dictionary forms → FST tags ─────────────────────────────


def forms_with_tags(engine: Any, entry: dict) -> list[dict]:
    """Collect all forms of *entry*, analyze via FST, filter by entry lemma.

    Returns a flat list of ``{"form": str, "tags": [str], "pgr": str}``.
    Forms whose FST analysis does not match the entry lemma get
    ``"tags": []`` (PGR kept as fallback).
    """
    from .pgr import extract_pgr_from_entry

    forms_pgr = extract_pgr_from_entry(entry)
    if not forms_pgr:
        return []

    entry_lemma = entry.get("word", "").lower()

    # Deduplicate forms
    unique_forms = list(dict.fromkeys(f for f, _ in forms_pgr))

    # Batch-analyze all unique forms
    if fst_api is not None:
        analyses = analyze_words(unique_forms)
    else:
        analyses = {}

    # Build PGR index for fallback
    pgr_by_form: dict[str, str] = {}
    for form, pgr in forms_pgr:
        if form.lower() not in pgr_by_form:
            pgr_by_form[form.lower()] = pgr
        else:
            existing = pgr_by_form[form.lower()]
            if pgr not in existing:
                pgr_by_form[form.lower()] = f"{existing}|{pgr}"

    result = []
    seen_forms: set[str] = set()
    for form in unique_forms:
        fl = form.lower()
        if fl in seen_forms:
            continue
        seen_forms.add(fl)

        info = analyses.get(form, {})
        fst_analyses = info.get("analyses", [])  # [(lemma, tags)]

        # Keep only analyses matching the entry lemma
        matching_tags: list[str] = []
        for lemma, tags in fst_analyses:
            if lemma.lower() == entry_lemma:
                matching_tags.append("+".join(tags))

        pgr = pgr_by_form.get(fl, "")

        result.append({
            "form": form,
            "tags": matching_tags,
            "pgr": pgr,
        })

    return result
