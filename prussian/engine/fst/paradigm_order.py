"""Paradigm cells for the FST generation direction.

The generator matches ``lemma+Tag1+Tag2+...`` as a literal string, so the
tag order must be exactly the one the analyzer emits (prussian-fst
inventory, gender-first for nominals: ``N+Fem+Sg+Acc``,
``V+PrfPrc+Act+Msc+Sg+Nom``).  Instead of reading forms from the
dictionary JSON, we enumerate every plausible cell for the lemma's word
class and let the FST decide which exist (empty results are dropped by
the caller).

Word class, gender and reflexivity come from the FST analyses of the
lemma itself (its citation form), so no dictionary classification is
needed.  Pronouns are out of scope (their slot order differs and their
paradigms are hand-written).
"""

from __future__ import annotations

GENDERS = ("Msc", "Fem", "Neu")
NUMBERS = ("Sg", "Pl")
CASES = ("Nom", "Gen", "Dat", "Acc")
PERSONS = ("Sg1", "Sg2", "SP3", "Pl1", "Pl2")

_SLOTS = set(GENDERS) | set(NUMBERS) | set(CASES)
# Word classes whose gender agrees (all three genders inflect).
_AGREEING = {"A", "Num"}
DEGREES = ("", "Comp", "Superl")


def _decl(prefix: list[str], genders, numbers=NUMBERS) -> list[str]:
    return ["+".join([*prefix, *([g] if g else []), *([n] if n else []), c])
            for g in genders for n in numbers for c in CASES]


def _verb_cells() -> list[str]:
    cells = ["V+Inf"]
    for tense in ("Prs", "Prt"):
        cells += [f"V+Ind+{tense}+{p}" for p in PERSONS]
    cells += [f"V+Subj+{p}" for p in PERSONS]
    cells += ["V+Opt+SP3", "V+Imprt+Sg2", "V+Imprt+Pl2", "V+Rel+Prs+SP3"]
    for prc in ("PrsPrc+Act", "PrfPrc+Act", "PrfPrc+Pss"):
        cells += _decl(["V", *prc.split("+")], GENDERS)
    return cells


def paradigm_cells(lemma_readings: list[tuple[str, ...]]) -> list[str]:
    """Tag strings (without lemma) of all plausible cells, in paradigm order.

    *lemma_readings* are the tag tuples of the lemma's own FST analyses,
    e.g. ``[("N", "Fem", "Sg", "Nom")]`` for ``berzi`` or
    ``[("V", "Inf"), ("V", "Inf", "Refl")]`` for ``laikātun``.
    """
    cells: dict[str, None] = {}
    for tags in sorted(lemma_readings, key=len):
        if not tags:
            continue
        pos = tags[0]
        if pos == "Pron":
            # Every gender/number the lemma might have, number optional;
            # the generator keeps only the cells listed in the lexc.
            genders = GENDERS if any(t in GENDERS for t in tags) else ("",)
            prefix = [t for t in tags if t not in _SLOTS]
            cells.update((c, None) for c in _decl(prefix, genders, (*NUMBERS, "")))
            continue
        if pos == "V":
            refl = "+Refl" if "Refl" in tags else ""
            cells.update((c + refl, None) for c in _verb_cells())
            continue
        if pos == "Adv":
            cells.update((f"Adv{d and '+' + d}", None) for d in DEGREES)
            continue
        prefix = [t for t in tags if t not in _SLOTS]
        if len(prefix) == len(tags):  # invariable (Pr, CC, Pcle, bare N …)
            cells["+".join(tags)] = None
            continue
        own = [t for t in tags if t in GENDERS]
        genders = GENDERS if pos in _AGREEING else (own or [""])
        degrees = DEGREES if pos == "A" else ("",)
        for d in degrees:
            cells.update((c, None) for c in _decl(prefix + ([d] if d else []), genders))
        if pos == "A":
            cells.update((f"Adv{d and '+' + d}", None) for d in DEGREES)
    return list(cells)
