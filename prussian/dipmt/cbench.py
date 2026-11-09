"""Benchmark for the construction check on held-out videos.

- **False alarms**: share of held-out corpus sentences (correct as far as
  the corpus goes) for which ``analyse`` reports anything.
- **Case errors**: per sentence one noun / adjective / pronoun is regenerated
  in another case with the FST (``pāds+N+Msc+Sg+Dat`` → ``…+Nom``); the
  corrupted text is re-parsed like a draft.  *Detected* = a finding covers
  the changed word; *corrected* = its first suggested variant is the
  fragment of the original sentence.

The index holds the training videos and the article prose only, so neither
the sentence nor its lesson is in it.

``uv run python -m prussian.dipmt.cbench``
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

from prussian.dipmt.align import SPLIT_PATH
from prussian.dipmt.constructions import (TREEBANK_PATH, Constructions,
                                          fragments, parse_conllu)

CASES = ("Nom", "Gen", "Dat", "Acc")
CORRUPTIBLE = {"NOUN": "N", "ADJ": "A", "PRON": "Pron"}


def heldout_rows():
    held = set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["heldout"])
    with open(TREEBANK_PATH, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["sent_id"] in held:
                yield row


def corrupt(toks, rng):
    """(token, new_form, new_case) or None — an unambiguous cased word
    regenerated in another case whose form is not one of its own forms."""
    from prussian_fst import api
    cands = [t for t in toks if t.upos in CORRUPTIBLE and t.xpos not in ("", "_")
             and t.xpos.split("+")[-1] in CASES]
    rng.shuffle(cands)
    for t in cands:
        parts = t.xpos.split("+")
        own = parts[-1]
        others = [c for c in CASES if c != own]
        rng.shuffle(others)
        queries = {c: f"{t.lemma}+{'+'.join(parts[:-1])}+{c}" for c in CASES}
        gen = api.generate(list(queries.values()))
        own_forms = set(gen[queries[own]]) | {t.form.lower()}
        for c in others:
            forms = [f for f in gen[queries[c]] if f not in own_forms]
            if forms:
                f = forms[0]
                return t, (f[0].upper() + f[1:] if t.form[:1].isupper() else f), c
    return None


def main(argv: list[str] | None = None) -> int:
    from prussian_fst import api
    rng = random.Random(7)
    idx = Constructions()
    rows = list(heldout_rows())
    alarms = n = det = corr = tried = 0
    by_pos: Counter = Counter()
    by_pos_det: Counter = Counter()
    examples = []
    for row in rows:
        toks = parse_conllu(row["conllu"])
        if not toks:
            continue
        n += 1
        alarms += bool(idx.analyse(toks))
        c = corrupt(toks, rng)
        if c is None:
            continue
        t, new_form, new_case = c
        words = [x.form for x in toks]
        words[t.id - 1] = new_form
        text = " ".join(words).replace(" ,", ",").replace(" .", ".") \
            .replace(" ?", "?").replace(" !", "!")
        ctoks = parse_conllu(api.conllu(text, trace=False).split("\n\n")[0])
        if len(ctoks) != len(toks):
            continue
        tried += 1
        by_pos[t.upos] += 1
        orig = {f for f, _ in fragments(toks)}
        hits = [f for f in idx.analyse(ctoks) if t.id in f["ids"]]
        if hits:
            det += 1
            by_pos_det[t.upos] += 1
            if any(h["alternatives"][0][0] in orig for h in hits):
                corr += 1
            if len(examples) < 6:
                examples.append((text, t.form, new_form, hits[0]["fragment"],
                                 hits[0]["alternatives"][0]))
    print(f"{n} held-out sentences, index {len(idx.sents)} sentences")
    print(f"false alarms on correct sentences: {alarms}/{n} = {alarms / max(n, 1):.1%}")
    print(f"case errors: {tried}  detected {det / max(tried, 1):.1%}  "
          f"corrected@1 {corr / max(tried, 1):.1%}")
    for pos in by_pos:
        print(f"  {pos:5} {by_pos_det[pos]}/{by_pos[pos]} detected")
    for e in examples:
        print(f"  {e[1]} → {e[2]}: {e[3]}  ⇒ {e[4][0]} ({e[4][1]}×)\n    {e[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
