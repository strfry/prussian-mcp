"""FST-based tag analysis, tag matching, and form extraction.

Guarded import pattern (fsg_check.py): the server starts without FST, but
the FST tools then answer with an error (:func:`fst_error_message`) — there
is no dictionary-only fallback that would hide a missing FST.
"""

from __future__ import annotations

import re
from pathlib import Path
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
    from prussian_fst.cg3_pipeline import DEFAULT_FST, DEFAULT_FST_GEN, DEFAULT_LENIENT
except ImportError:
    _FST_DIR = Path(__file__).resolve().parents[2] / "prussian-fst" / "fst"
    DEFAULT_FST = _FST_DIR / "build/base.hfstol"
    DEFAULT_LENIENT = _FST_DIR / "build/lenient.hfstol"
    DEFAULT_FST_GEN = _FST_DIR / "build/base.gen.hfstol"



def _missing_artifacts() -> list[str]:
    return [str(p) for p in (DEFAULT_FST, DEFAULT_LENIENT, DEFAULT_FST_GEN)
            if not Path(p).exists()]


def fst_available() -> bool:
    """True if the FST pipeline is importable and analyzer/generator exist."""
    return fst_api is not None and not _missing_artifacts()


def fst_error_message() -> str:
    """Human-readable message when FST is not available."""
    if _IMPORT_ERROR:
        return _IMPORT_ERROR
    missing = _missing_artifacts()
    if missing:
        return (f"{', '.join(missing)} fehlt — im prussian-fst-Checkout "
                "`make all` ausführen")
    return ""


# ── Feature name mapping (human-readable → FST tag) ──────────────────────────

FEATURE_MAP: dict[str, str] = {
    # Partizipien: Typ (PrsPrc/PrfPrc) + Voice (Act/Pss).  "participle"
    # zielt auf das Perfekt-Partizip; Present/Passiv gezielt benennen.
    "participle": "PrfPrc",
    "present_participle": "PrsPrc",
    "conjunctive": "Subj",
    "subjunctive": "Subj",
    "optative": "Opt",
    "imperative": "Imprt",
    "indicative": "Ind",
    "present": "Prs",
    "preterite": "Prt",
    "past": "Prt",
    "infinitive": "Inf",
    "relative": "Rel",
    "passive": "Pss",
    "active": "Act",
    "reflexive": "Refl",
    "nominative": "Nom",
    "genitive": "Gen",
    "dative": "Dat",
    "accusative": "Acc",
    "acc": "Acc",
    "singular": "Sg",
    "plural": "Pl",
    "masculine": "Msc",
    "feminine": "Fem",
    "neuter": "Neu",
    "comparative": "Comp",
    "superlative": "Superl",
    "adverb": "Adv",
    "cardinal": "Card",
    "ordinal": "Ord",
}


def resolve_features(spec: str) -> list[str] | None:
    """Resolve a comma-separated feature spec to FST tags.

    Handles human-readable names (``"participle"``, ``"conjunctive"``),
    raw FST tags (``"PrfPrc"``, ``"Ind+Prs"``), and mixed input.
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
    "N", "A", "Pron", "Num", "V", "Adv", "Pr", "Adp", "Po",
    "CC", "CS", "Pcle", "Interj", "Prop",
    "Sg", "Pl", "Nom", "Gen", "Dat", "Acc",
    "Msc", "Fem", "Neu",
    "Prs", "Prt", "Inf", "Ind", "Opt", "Imprt", "Subj", "Rel",
    "Pss", "Act", "Refl", "Sg1", "Sg2", "SP3", "Pl1", "Pl2",
    "PrsPrc", "PrfPrc", "P1", "P2", "P3",
    "Comp", "Superl", "Card", "Ord", "Encl",
}
_KNOWN_FST_TAGS_LOWER = {t.lower(): t for t in _KNOWN_FST_TAGS}

# POS tags — to detect verbs for default filtering
_VERB_POS = {"V"}


# ── Tag matching ──────────────────────────────────────────────────────────────


def match_tags(tags: list[str], wanted: str) -> bool:
    """Check whether *tags* contain all wanted tags in at least one reading.

    ``wanted`` is split on ``+``, ``.``, ``,`` and whitespace; each piece
    is checked against at least one individual tag string (which may be
    ``+``-joined, e.g. ``"N+Sg+Acc+Fem"``).

    A form matches when at least one of its tag strings contains ALL
    wanted tags (case-insensitive).

    Examples::

        match_tags(["N+Sg+Acc+Fem"], "Acc+Sg")  → True
        match_tags(["N+Sg+Nom+Fem"], "Acc+Sg")  → False
        match_tags(["V+Ind+Prs+Sg1"], "Ind+Prs")  → True
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


def lemma_tags(word: str, *, macron: bool = False) -> dict[str, list[str]]:
    """Strict FST analyses of one word, grouped by lemma: ``{lemma: [tags]}``.

    Reflexive readings are grouped under the dictionary headword
    (``laikātun si``, see :func:`entry_lemma`).

    The dictionary lookup's form index: surface/lowercase/titlecase/short-prep
    stages of :func:`analyze_words`, never the lenient layer (the lookup's own
    prefix/ortho rules play that role).  ``macron=True`` analyzes with
    ``macron.hfstol`` instead (macron loss tolerated).
    """
    if fst_api is None or not word:
        return {}
    if macron:
        path = DEFAULT_FST.parent / "macron.hfstol"
        if not path.exists():
            return {}
        with fst_api._PIPELINE_LOCK:
            hits = flookup_batch(list(dict.fromkeys([word, word.lower()])), path)
        analyses = hits.get(word) or hits.get(word.lower()) or []
    else:
        info = analyze_words([word]).get(word, {})
        if info.get("via") in (None, "lenient"):
            return {}
        analyses = info["analyses"]
    out: dict[str, list[str]] = {}
    for lemma, tags in analyses:
        tag = "+".join(tags)
        group = out.setdefault(entry_lemma(lemma, tags), [])
        if tag not in group:
            group.append(tag)
    return out


# ── Paradigm generation: lemma → FST-generated forms ─────────────────────────


def split_refl(word: str) -> tuple[str, bool]:
    """``"laikātun si"`` → ``("laikātun", True)``.

    Reflexive entries carry the particle in the headword; the FST analyzes the
    bare verb with ``+Refl`` on every cell (``lāika`` → ``laikātun+…+Refl``).
    """
    if word.endswith(" si"):
        return word[:-3], True
    return word, False


def entry_lemma(lemma: str, tags) -> str:
    """Dictionary headword of an FST reading: ``+Refl`` → ``"<lemma> si"``."""
    return f"{lemma} si" if "Refl" in tags else lemma


def _own_readings(word: str) -> list[tuple[str, ...]]:
    """FST readings of *word* as its own lemma — for ``… si`` only the
    ``+Refl`` ones, for the bare verb only those without."""
    base, refl = split_refl(word)
    analyses = fst_api.tags([base]).get(base, {}).get("analyses", [])
    return [tuple(tags) for lemma, tags in analyses
            if lemma == base and ("Refl" in tags) == refl]


def generate_paradigm(entry: dict) -> list[dict]:
    """All paradigm cells of *entry*, generated by the FST.

    The lemma's own analyses give word class, gender and reflexivity;
    :func:`paradigm_cells` enumerates the plausible cells and the generator
    (``base.gen.hfstol``) keeps those that exist.  Lemmas outside FST
    coverage yield no forms.  A reflexive entry (``laikātun si``) shows
    every form with the particle, as in the dictionary (``lāika si``).

    Returns ``[{"form": str, "tags": [str]}, ...]`` in paradigm order, one
    item per (cell, surface form).
    """
    from prussian.engine.fst.paradigm_order import paradigm_cells

    word = entry.get("word", "")
    if fst_api is None or not word:
        return []
    base, refl = split_refl(word)
    cells = paradigm_cells(_own_readings(word))
    if not cells:
        return []
    queries = {c: f"{base}+{c}" for c in cells}
    generated = fst_api.generate(list(queries.values()), fst_path=DEFAULT_FST_GEN)
    particle = " si" if refl else ""
    return [{"form": surface + particle, "tags": [cell]}
            for cell, q in queries.items()
            for surface in generated.get(q, [])]


def paradigm_gap(word: str) -> str:
    """Why :func:`generate_paradigm` is empty although the FST knows *word*.

    ``""`` when the analyzer has no reading with *word* as its own lemma
    (a coverage gap, or *word* is an inflected form).  Otherwise the FST
    side that failed: no paradigm cells for the readings, or a generator
    that produces none of them (out of sync with the analyzer).
    """
    from prussian.engine.fst.paradigm_order import paradigm_cells

    if fst_api is None or not word:
        return ""
    readings = _own_readings(word)
    if not readings:
        return ""
    shown = ", ".join("+".join(r) for r in readings[:3])
    cells = paradigm_cells(readings)
    if not cells:
        return f"FST analyzes {word} as {shown}, but paradigm_order defines no cells for it"
    return (f"FST analyzes {word} as {shown}, but {DEFAULT_FST_GEN.name} generates "
            f"none of its {len(cells)} cells — generator out of sync with the analyzer?")
