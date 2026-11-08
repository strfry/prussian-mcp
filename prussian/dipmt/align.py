"""Bilingual lexicon induction: eflomal over English lemmas ↔ Prussian lemmas.

DiPMT++ ran GIZA++ on word forms; here the Prussian side is already
FST-lemmatized and the English side is lemmatized with simplemma, so the
aligner sees ``die ↔ aulaūtwei`` instead of a dozen inflected variants.
Dictionary glosses go in as lexical priors (single-word English glosses →
headword), which keeps the small corpus (~2k unique pairs) from aligning
on co-occurrence noise alone.

Extraction from the symmetrized links (grow-diag-final-and):

* one English lemma ↔ one Prussian lemma → a lexicon entry;
* one English lemma ↔ a contiguous Prussian span → a multiword entry keyed
  by the surface forms (``maybe ↔ mazzi būtwei`` — frozen, not lemmatized);
* two adjacent English lemmas ↔ the same single Prussian token → an
  English bigram entry.

Each entry also counts the aligned surface form and its neighbours.  A
neighbour that (nearly) always comes along makes a frozen phrase: "maybe"
aligns to ``mazzi`` (FST lemma ``mazītwei``), but ``mazzi`` is followed by
``būtwei`` every time, so the entry offers ``mazzi būtwei``.

``uv run prussian-dipmt-build`` writes ``data/dipmt/lexicon.json`` and
``data/dipmt/split.json`` (held-out sent_ids).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from prussian.config import DATA_DIR, DICTIONARY_PATH
from prussian.dipmt.corpus import Pair, en_lemma, en_tokenize, load_pairs, split

OUT_DIR = DATA_DIR / "dipmt"
LEXICON_PATH = OUT_DIR / "lexicon.json"
SPLIT_PATH = OUT_DIR / "split.json"

_PRIOR_ALPHA = 2.0
_MAX_EXAMPLES = 5
_STOP_GLOSS = {"to", "a", "an", "the", "be", "of", "oneself", "sth", "smb",
               "something", "somebody"}


# ── priors from the dictionary ────────────────────────────────────────────────

def gloss_words(gloss: str) -> list[str]:
    """Content words of an English gloss, lemmatized (``to sink (go down)`` → [sink])."""
    gloss = re.sub(r"\([^)]*\)", " ", gloss)
    return [en_lemma(w) for w in en_tokenize(gloss) if w not in _STOP_GLOSS]


def dictionary_priors() -> list[tuple[str, str]]:
    """(english lemma, prussian headword) for every single-word English gloss."""
    from prussian_fst.nvh import read_entries
    out = set()
    for e in read_entries(DICTIONARY_PATH):
        word = (e.get("word") or "").lower()
        if not word or " " in word:
            continue
        for g in (e.get("translations") or {}).get("en", []) or []:
            for part in re.split(r"[,;](?![^()]*\))", g):
                ws = gloss_words(part)
                if len(ws) == 1:
                    out.add((ws[0], word))
    return sorted(out)


# ── alignment ─────────────────────────────────────────────────────────────────

def _read_links(path: str) -> list[set[tuple[int, int]]]:
    out = []
    with open(path) as fh:
        for line in fh:
            out.append({tuple(map(int, x.split("-"))) for x in line.split()})
    return out


def grow_diag_final_and(fwd: set, rev: set) -> set:
    """Koehn's symmetrization over (src, trg) link sets."""
    union, links = fwd | rev, set(fwd & rev)
    neigh = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    src_done = lambda i: any(a == i for a, _ in links)
    trg_done = lambda j: any(b == j for _, b in links)
    added = True
    while added:
        added = False
        for i, j in sorted(links):
            for di, dj in neigh:
                n = (i + di, j + dj)
                if n in union and n not in links and (not src_done(n[0]) or not trg_done(n[1])):
                    links.add(n)
                    added = True
    for i, j in sorted(union):
        if (i, j) not in links and not src_done(i) and not trg_done(j):
            links.add((i, j))
    return links


def align(pairs: list[Pair], priors: list[tuple[str, str]], quiet: bool = True
          ) -> list[set[tuple[int, int]]]:
    import eflomal
    src = [" ".join(p.en_lemmas) for p in pairs]
    trg = [" ".join(p.pru_lemmas) for p in pairs]
    prior_lines = [f"LEX\t{e}\t{w}\t{_PRIOR_ALPHA}" for e, w in priors]
    with tempfile.TemporaryDirectory() as tmp:
        fwd, rev = f"{tmp}/fwd", f"{tmp}/rev"
        eflomal.Aligner().align(src, trg, links_filename_fwd=fwd,
                                links_filename_rev=rev,
                                priors_input=prior_lines, quiet=quiet)
        return [grow_diag_final_and(f, r)
                for f, r in zip(_read_links(fwd), _read_links(rev))]


# ── lexicon extraction ────────────────────────────────────────────────────────

def extract(pairs: list[Pair], links: list[set[tuple[int, int]]]) -> dict:
    """``{"entries": {en: {pru: [count, [sent_ids]]}}, "en_count": {en: n}}``."""
    entries: dict[str, dict[str, list]] = defaultdict(dict)
    en_count: Counter = Counter()

    def add(en: str, pru: str, sid: str, p: Pair | None = None,
            j: int | None = None, free=lambda j: False) -> None:
        c = entries[en].setdefault(pru, [0, [], Counter()])
        c[0] += 1
        if len(c[1]) < _MAX_EXAMPLES and sid not in c[1]:
            c[1].append(sid)
        if p is not None:
            f = p.pru_forms
            c[2]["=" + f[j]] += 1
            # neighbours that belong to another English word are syntax
            # ("everyone has the right" ↛ "turri tikrōmiskan"), not a phrase
            if j + 1 < len(f) and free(j + 1):
                c[2][f"{f[j]} {f[j + 1]}"] += 1
            if j > 0 and free(j - 1):
                c[2][f"{f[j - 1]} {f[j]}"] += 1

    for p, ls in zip(pairs, links):
        by_src: dict[int, list[int]] = defaultdict(list)
        by_trg: dict[int, list[int]] = defaultdict(list)
        for i, j in ls:
            by_src[i].append(j)
            by_trg[j].append(i)
        for i, lemma in enumerate(p.en_lemmas):
            en_count[lemma] += 1
            js = sorted(by_src.get(i, []))
            if not js:
                continue
            # only links where the Prussian token is not shared with other
            # English words, else function words soak up content links
            if len(js) == 1 and len(by_trg[js[0]]) == 1:
                add(lemma, p.pru_lemmas[js[0]], p.sent_id, p, js[0],
                    free=lambda k, i=i: set(by_trg.get(k, [])) <= {i})
            elif len(js) > 1 and js == list(range(js[0], js[-1] + 1)) \
                    and all(by_trg[j] == [i] for j in js):
                add(lemma, " ".join(p.pru_forms[js[0]:js[-1] + 1]), p.sent_id)
        for j, is_ in by_trg.items():
            is_ = sorted(is_)
            if len(is_) == 2 and is_[1] == is_[0] + 1 and by_src[is_[0]] == [j] \
                    and by_src[is_[1]] == [j]:
                bigram = f"{p.en_lemmas[is_[0]]} {p.en_lemmas[is_[1]]}"
                en_count[bigram] += 1
                add(bigram, p.pru_lemmas[j], p.sent_id, p, j,
                    free=lambda k, s=set(is_): set(by_trg.get(k, [])) <= s)
    return {"entries": {en: {pru: _finish(c) for pru, c in d.items()}
                        for en, d in entries.items()},
            "en_count": dict(en_count)}


def _finish(c: list) -> dict:
    """``{"n", "examples", "form", "phrase"?}`` — the dominant surface form,
    and a frozen phrase when one neighbour bigram covers ≥80 % (n ≥ 3)."""
    n, examples, ctx = c
    out = {"n": n, "examples": examples}
    forms = [(k[1:], v) for k, v in ctx.items() if k.startswith("=")]
    if forms:
        out["form"] = max(forms, key=lambda kv: kv[1])[0]
    bigrams = [(k, v) for k, v in ctx.items() if not k.startswith("=")]
    if bigrams and n >= 3:
        k, v = max(bigrams, key=lambda kv: kv[1])
        if v >= 0.8 * n:
            out["phrase"] = k
    return out


def build(heldout_frac: float = 0.1, seed: int = 13, quiet: bool = True) -> dict:
    pairs = load_pairs()
    train, held = split(pairs, heldout_frac, seed)
    priors = dictionary_priors()
    print(f"{len(pairs)} unique pairs: {len(train)} train, {len(held)} held out; "
          f"{len(priors)} dictionary priors", file=sys.stderr)
    lex = extract(train, align(train, priors, quiet=quiet))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEXICON_PATH.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    SPLIT_PATH.write_text(json.dumps({
        "heldout_frac": heldout_frac, "seed": seed,
        "train": [p.sent_id for p in train],
        "heldout": [p.sent_id for p in held]}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {LEXICON_PATH} ({len(lex['entries'])} English keys)", file=sys.stderr)
    return lex


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="prussian-dipmt-build",
                                 description="Induce the English→Prussian lexicon.")
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    build(args.heldout_frac, args.seed, quiet=not args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
