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
    forms.  The ``features`` argument selects specific morphological
    categories.

    Args:
        engine: ``SearchEngine`` instance.
        lemma: Prussian base form (from ``lookup_prussian_word``).
        features: optional comma-separated feature filter.  Accepts
            human-readable names (``participle``, ``conjunctive``,
            ``optative``, ``imperative``, ``present``, ``preterite``,
            ``infinitive``) or raw FST tags (``PrfPrc+Pss``, ``Ind``,
            ``Gen+Pl``, ``Ind+Prs+SP3``).  Omit for the verb default
            (Ind+Prs) or full list (non-verbs).

    Returns:
        Compact text per entry: ``lemma (gender)  — desc``, one
        ``form: tags`` line per form, then the paradigm's feature groups
        (the tag combinations that actually occur, see
        :func:`_feature_groups`).  An empty filter result says which tag
        does not fit; a word without a paradigm of its own names the
        lemmas it is a form of.  Unknown lemma / feature → an ``error:``
        line (plus the valid features).
    """
    from prussian.engine.fst.tags import (
        fst_available,
        resolve_features,
        match_tags,
        generate_paradigm,
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
    form_of: str | None = None

    for entry in entries:
        fw = generate_paradigm(entry) if fst_available() else []
        cells = [f["tags"][0] for f in fw if f.get("tags")]

        # Detect POS from tag inventory
        is_verb = any(c.split("+")[0] == "V" for c in cells)

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
                if match_tags(f.get("tags", []), "Ind+Prs")
            ]
        else:
            filtered = fw

        inherent, groups = _feature_groups(fw)
        entry_out: dict[str, Any] = {
            "lemma": entry.get("word", ""),
            "desc": entry.get("desc", ""),
            "gender": entry.get("gender", ""),
            "forms": [{"form": f["form"], "tags": f["tags"]} for f in filtered],
            "inherent": inherent,
            "feature_groups": groups,
        }
        if not fw:
            if form_of is None:
                form_of = _form_of(entry.get("word", "")) if fst_available() else ""
            entry_out["note"] = form_of or "no paradigm (outside FST coverage)"
        elif wanted_tags is not None and not filtered:
            entry_out["note"] = _diagnose(cells, wanted_tags, inherent)
        results.append(entry_out)

    return _format(results)


# ── Feature groups ───────────────────────────────────────────────────────────

# Slot dimensions, in display order; everything else in a cell is its frame
# (POS, mood, participle type, voice, degree, person of pronouns, …).
_DIMS: tuple[tuple[str, ...], ...] = (
    ("Prs", "Prt"),
    ("Sg1", "Sg2", "SP3", "Pl1", "Pl2"),
    ("Msc", "Fem", "Neu"),
    ("Sg", "Pl"),
    ("Nom", "Gen", "Dat", "Acc"),
)
_DIM_OF = {t: i for i, dim in enumerate(_DIMS) for t in dim}
_MOODS = {"Ind", "Subj", "Opt", "Imprt", "Rel"}
_PARTICIPLES = {"PrsPrc", "PrfPrc"}
_POS = {"N", "A", "V", "Pron", "Num"}
# Groups shown without their POS tag (the filter does not need it).
_STRIP_POS = {"declension", "finite", "participle", "infinitive"}


def _label(frame: tuple[str, ...], lemma_pos: str) -> str:
    tags = set(frame)
    if tags & _PARTICIPLES:
        return "participle"
    if tags & _MOODS:
        return "finite"
    if "Inf" in tags:
        return "infinitive"
    if frame[:1] == ("Adv",) and lemma_pos == "A":
        return "derived adverb"
    if frame[:1] and frame[0] in _POS:
        return "declension"
    return "invariable"


def _feature_groups(fw: list[dict]) -> tuple[list[str], list[tuple[str, str]]]:
    """The tag combinations that occur in a paradigm, grouped.

    Returns ``(inherent, groups)``: ``inherent`` are the tags every cell
    carries (a noun's gender, a reflexive verb's ``Refl``, a pronoun's
    person); ``groups`` are ``(label, text)`` pairs such as
    ``("finite", "Ind × Prs|Prt × Sg1|Sg2|SP3|Pl1|Pl2; Opt × SP3")``.
    Within a group, frames with the same slot values are merged and
    ``×`` joins the slot values that combine freely; ``(gaps)`` marks a
    frame where not every combination exists.
    """
    cells = list(dict.fromkeys(f["tags"][0] for f in fw if f.get("tags")))
    if not cells:
        return [], []
    split = [c.split("+") for c in cells]
    lemma_pos = split[0][0]
    common = set(split[0][1:]).intersection(*(set(t[1:]) for t in split))
    inherent = [t for t in split[0][1:] if t in common]

    # (frame, dimensions it uses) → per-dimension values and cell count;
    # number-less cells (stas+Pron+Neu+Nom) thus form their own grid.
    frames: dict[tuple, dict] = {}
    for f, tags in zip(cells, split):
        tags = [t for t in tags if t not in common]
        frame = tuple(t for t in tags if t not in _DIM_OF)
        key = (frame, frozenset(_DIM_OF[t] for t in tags if t in _DIM_OF))
        info = frames.setdefault(key, {"dims": [set() for _ in _DIMS], "n": 0})
        info["n"] += 1
        for t in tags:
            if t in _DIM_OF:
                info["dims"][_DIM_OF[t]].add(t)
    examples = {}
    for f in fw:
        tags = f["tags"][0].split("+")
        frame = tuple(t for t in tags if t not in common and t not in _DIM_OF)
        examples.setdefault(frame, f["form"])

    # label → grid → frames (insertion order = paradigm order)
    by_label: dict[str, dict[tuple, list]] = {}
    for (frame, _), info in frames.items():
        grid = tuple(tuple(v for v in dim if v in vals)
                     for dim, vals in zip(_DIMS, info["dims"]))
        size = 1
        for vals in grid:
            size *= len(vals) or 1
        label = _label(frame, lemma_pos)
        by_label.setdefault(label, {}).setdefault(grid, []).append(
            (frame, info["n"] < size))

    groups = []
    for label, grids in by_label.items():
        parts = []
        for grid, members in grids.items():
            strip = label in _STRIP_POS and all(fr[:1] == (lemma_pos,) for fr, _ in members)
            head = _alternatives(["+".join(fr[1:] if strip else fr) for fr, _ in members])
            text = " × ".join(([head] if head else [])
                              + ["|".join(vals) for vals in grid if vals])
            if not text:  # nothing left to choose: the inherent tags say it all
                continue
            if any(gap for _, gap in members):
                text += " (gaps)"
            parts.append(text)
        if not parts:
            continue
        text = "; ".join(parts)
        if label == "derived adverb":
            text += f" ({examples[next(iter(grids.values()))[0][0]]})"
        groups.append((label, text))
    return inherent, groups


def _alternatives(frames: list[str], sep: str = "") -> str:
    """``["", "Comp", "Superl"]`` → ``[Comp|Superl]``; ``["Adv", "Adv+Comp"]``
    → ``Adv[+Comp]``; otherwise ``a | b``."""
    if len(frames) == 1:
        return frames[0]
    if "" in frames:
        return "[" + "|".join(sep + f for f in frames if f) + "]"
    first = frames[0].split("+")[0]
    rest = [f[len(first) + 1:] for f in frames]
    if "" in rest and all(f == first or f.startswith(first + "+") for f in frames):
        return first + _alternatives(rest, "+")
    return " | ".join(frames)


def _diagnose(cells: list[str], wanted: list[str], inherent: list[str]) -> str:
    """Why ``wanted`` matches no cell: an absent tag, or the one that clashes."""
    present = {t.lower() for c in cells for t in c.split("+")}
    missing = [t for t in wanted if t.lower() not in present]
    if missing:
        msg = f"{', '.join(missing)} not in this paradigm"
        if inherent:
            msg += f" (inherent: {'+'.join(inherent)})"
        return msg

    def hits(tags):
        return [c for c in cells
                if {t.lower() for t in tags} <= {p.lower() for p in c.split("+")}]

    clashes = [(t, hits([w for w in wanted if w != t])) for t in wanted]
    clashes = [(t, h) for t, h in clashes if h]
    if not clashes:
        return f"no form combines {'+'.join(wanted)}"
    out = []
    for t, h in clashes:
        rest = "+".join(w for w in wanted if w != t)
        pos = h[0].split("+")[0]
        label = _label(tuple(x for x in h[0].split("+") if x not in _DIM_OF), pos)
        out.append(f"{t} does not combine with {rest} ({label}, {len(h)} forms)")
    return "; ".join(out)


def _form_of(word: str) -> str:
    """``stāi is a form of stas (Pron+Fem+Sg+Nom, …)`` — or ``""``."""
    from prussian.engine.fst.tags import lemma_tags

    others = {lemma: tags for lemma, tags in lemma_tags(word).items()
              if lemma.lower() != word.lower()}
    if not others:
        return ""
    parts = [f"{lemma} ({', '.join(tags[:3])}{', …' if len(tags) > 3 else ''})"
             for lemma, tags in others.items()]
    return (f"{word} is a form of {'; '.join(parts)} — "
            "call get_word_forms on the lemma")


def _format(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no forms found."
    out: list[str] = []
    for entry in results:
        desc = f"— {entry['desc']}" if entry.get("desc") else ""
        out.append(head(entry.get("lemma", "?"), entry.get("gender", ""), desc))
        for f in entry["forms"]:
            tags = " | ".join(f["tags"])
            out.append(f"  {f['form']}: {tags}" if tags else f"  {f['form']}")
        if not entry["forms"]:
            out.append("  (no forms)")
        if entry.get("note"):
            out.append(f"  note: {entry['note']}")
        if entry.get("inherent"):
            out.append(f"  inherent: {'+'.join(entry['inherent'])}")
        for label, text in entry.get("feature_groups", []):
            out.append(f"  {label}: {text}")
    return "\n".join(out)


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
                    help="Feature filter: 'participle', 'Ind+Prs', etc.")
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
