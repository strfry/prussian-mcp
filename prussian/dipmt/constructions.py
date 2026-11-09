"""Construction search: is the grammar of a Prussian draft attested?

The corpus (training videos only) is parsed with the current FST + CG3
pipeline.  Every sentence is reduced to *delexicalized fragments*: function
words stay lexical (they are the construction — ``ēn``, ``ni``, ``kāi``,
``būtwei``), content words become their category plus the feature that
governs the construction (``NOUN.Dat``, ``VERB.Inf``, ``VERB.PrfPrc``,
``VERB.Subj``).  Fragments are

- adjacent pairs/triples opened by a function word
  (``ni būtwei.Pres NOUN.Gen``, ``ēn NOUN.Dat``),
- dependency edges with a label the CG3 grammar attaches reliably
  (``VERB.PrfPrc >xcomp> VERB.Inf``, ``NOUN.Acc >amod< ADJ.Acc``);
  ``>`` / ``<`` marks whether the dependent follows or precedes its head.

``Constructions().check(draft)`` parses a draft the same way and reports
fragments the corpus never shows next to single-feature variants it does
show (“``ni būtwei.Prs NOUN.Nom`` 0× — corpus: ``… NOUN.Gen`` 14×”), each
with an example sentence from another video that shares as few content
words with the draft as possible: the example teaches the construction,
not the vocabulary.

``python -m prussian.dipmt.constructions build`` parses the corpus into
``data/dipmt/treebank.jsonl``; ``… check "<draft>"`` prints a report.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

from prussian.config import PROJECT_ROOT
from prussian.dipmt.align import OUT_DIR, SPLIT_PATH
from prussian.dipmt.corpus import load_pairs, video

TREEBANK_PATH = OUT_DIR / "treebank.jsonl"
# Prussian prose from another source (no English side) — more attestations,
# no overlap with the subtitle lessons the eval is drawn from.
ARTICLE_DIRS = (PROJECT_ROOT.parent / "corpus" / "parsed" / "awizi_articles",)

FUNCTION_UPOS = frozenset({"ADP", "SCONJ", "CCONJ", "PART"})
# verbs that work as auxiliaries / copulas: lexical, with their tense/mood
FUNCTION_VERBS = frozenset({"būtwei", "wīrstwei", "pastātwei"})
FUNCTION_PRON = frozenset({"kas"})          # interrogative / relative
CASED = frozenset({"NOUN", "PROPN", "ADJ", "PRON", "NUM", "DET"})
EDGE_LABELS = frozenset({"case", "mark", "cop", "aux", "xcomp", "amod",
                         "det", "nmod", "obj", "iobj", "obl", "ccomp", "advcl"})
SKIP_UPOS = frozenset({"PUNCT", "_", "INTJ"})
_PARTICIPLE = re.compile(r"\b(PrsPrc|PrfPrc)\b")


@dataclass
class Tok:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: dict
    head: int
    deprel: str


def parse_conllu(block: str) -> list[Tok]:
    toks = []
    for line in block.splitlines():
        p = line.split("\t")
        if len(p) != 10 or not p[0].isdigit():
            continue
        feats = dict(f.split("=", 1) for f in p[5].split("|") if "=" in f)
        toks.append(Tok(int(p[0]), p[1], (p[2] if p[2] != "_" else p[1]).lower(),
                        p[3], p[4], feats, int(p[6]) if p[6].isdigit() else 0, p[7]))
    return toks


def is_function(t: Tok) -> bool:
    return (t.upos in FUNCTION_UPOS or t.lemma in FUNCTION_VERBS
            or t.lemma in FUNCTION_PRON)


def _verb_feat(t: Tok) -> str:
    m = _PARTICIPLE.search(t.xpos)
    if m:
        return m.group(1)
    if t.feats.get("VerbForm") == "Inf":
        return "Inf"
    if t.feats.get("VerbForm") == "Part":
        return "Part"
    mood = t.feats.get("Mood", "")
    if mood == "Ind":
        return t.feats.get("Tense", "Ind")
    return mood or "?"


def abstract(t: Tok) -> str:
    """The construction-relevant view of a token."""
    if t.upos == "VERB":
        head = t.lemma if t.lemma in FUNCTION_VERBS else "VERB"
        return f"{head}.{_verb_feat(t)}"
    if t.lemma in FUNCTION_PRON:
        return f"{t.lemma}.{t.feats.get('Case', '?')}"
    if t.upos in FUNCTION_UPOS:
        return t.lemma
    if t.upos in CASED:
        return f"{t.upos}.{t.feats.get('Case', '?')}"
    return t.upos


def fragments(toks: list[Tok]) -> list[tuple[str, tuple[int, ...]]]:
    """(fragment, token ids) — ids let a report point at the words."""
    words = [t for t in toks if t.upos not in SKIP_UPOS]
    ab = {t.id: abstract(t) for t in words}
    out: list[tuple[str, tuple[int, ...]]] = []
    for i, t in enumerate(words):
        for n in (2, 3):
            win = words[i:i + n]
            if len(win) < n or not is_function(win[0]):
                continue
            out.append((" ".join(ab[w.id] for w in win), tuple(w.id for w in win)))
    by_id = {t.id: t for t in words}
    for t in words:
        h = by_id.get(t.head)
        if h is None or t.deprel not in EDGE_LABELS:
            continue
        arrow = f">{t.deprel}>" if t.id > h.id else f"<{t.deprel}<"
        out.append((f"{ab[h.id]} {arrow} {ab[t.id]}", (h.id, t.id)))
    return out


def _slots(frag: str) -> list[str]:
    return frag.split(" ")


def _variants(frag: str, values: dict[str, set[str]]) -> list[str]:
    """Same fragment with one ``CAT.value`` slot changed to another value."""
    parts = _slots(frag)
    out = []
    for i, p in enumerate(parts):
        if "." not in p or p.startswith(("<", ">")):
            continue
        cat, val = p.split(".", 1)
        if val == "?":
            continue
        for other in values.get(cat, ()):
            if other not in (val, "?"):
                out.append(" ".join(parts[:i] + [f"{cat}.{other}"] + parts[i + 1:]))
    return out


# ── corpus ───────────────────────────────────────────────────────────────────

def build(path=TREEBANK_PATH) -> int:
    """Parse every corpus pair with the current pipeline (train and held-out;
    the index filters by split at load time)."""
    from prussian_fst import api
    from prussian_fst.cg3_pipeline import load_markdown_sentences
    rows = [{"sent_id": p.sent_id, "en": p.en, "pru": p.pru, "source": "youtube"}
            for p in load_pairs()]
    for d in ARTICLE_DIRS:
        rows += [{"sent_id": f"{d.name}/{s['sent_id']}", "en": "", "pru": s["text"],
                  "source": d.name} for s in load_markdown_sentences(d)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            r["conllu"] = api.conllu(r["pru"], trace=False)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} sentences)", file=sys.stderr)
    return len(rows)


@dataclass
class Sent:
    sent_id: str
    en: str
    pru: str
    lemmas: frozenset
    frags: frozenset


class Constructions:
    def __init__(self, path=TREEBANK_PATH, train_only: bool = True):
        train = (set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["train"])
                 if train_only else None)
        self.sents: list[Sent] = []
        self.count: Counter[str] = Counter()
        self.where: dict[str, list[int]] = defaultdict(list)
        self.values: dict[str, set[str]] = defaultdict(set)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if (train is not None and row.get("source", "youtube") == "youtube"
                        and row["sent_id"] not in train):
                    continue
                toks = parse_conllu(row["conllu"])
                frags = {f for f, _ in fragments(toks)}
                idx = len(self.sents)
                self.sents.append(Sent(row["sent_id"], row["en"], row["pru"],
                                       frozenset(t.lemma for t in toks if not is_function(t)),
                                       frozenset(frags)))
                for f in frags:
                    self.count[f] += 1
                    self.where[f].append(idx)
                for t in toks:
                    a = abstract(t)
                    if "." in a:
                        cat, val = a.split(".", 1)
                        self.values[cat].add(val)

    def example(self, frag: str, avoid: set[str], exclude_video: str | None = None
                ) -> Sent | None:
        """Shortest attestation sharing the fewest content lemmas with ``avoid``."""
        cands = [self.sents[i] for i in self.where.get(frag, ())
                 if exclude_video is None or video(self.sents[i].sent_id) != exclude_video]
        if not cands:
            return None
        return min(cands, key=lambda s: (len(s.lemmas & avoid), len(s.pru), s.sent_id))

    def analyse(self, toks: list[Tok], min_context: int = 20,
                max_share: float = 0.02) -> list[dict]:
        """Draft fragments whose feature value is (almost) never seen in a
        well-attested context: over all single-feature variants of the
        fragment the corpus has ``min_context``+ hits, of which the draft's
        own value takes at most ``max_share``."""
        found = []
        seen: set[str] = set()
        for frag, ids in fragments(toks):
            if "?" in frag or frag in seen:
                continue
            seen.add(frag)
            own = self.count[frag]
            alts = sorted(((self.count[v], v) for v in _variants(frag, self.values)
                           if self.count[v]), reverse=True)
            total = own + sum(n for n, _ in alts)
            if not alts or total < min_context or own > max_share * total:
                continue
            found.append({"fragment": frag, "ids": ids, "count": own,
                          "total": total,
                          "alternatives": [(v, n) for n, v in alts[:2]]})
        return found

    def check(self, draft: str, exclude_video: str | None = None) -> str:
        from prussian_fst import api
        lines = []
        for block in api.conllu(draft, trace=False).split("\n\n"):
            toks = parse_conllu(block)
            if not toks:
                continue
            avoid = {t.lemma for t in toks if not is_function(t)}
            forms = {t.id: t.form for t in toks}
            for f in self.analyse(toks):
                words = " ".join(forms[i] for i in sorted(set(f["ids"])))
                alt = "; ".join(f"{v} ({n}×)" for v, n in f["alternatives"])
                lines.append(f"- “{words}”: {f['fragment']} — {f['count']}× in the "
                             f"corpus; attested instead: {alt}")
                ex = self.example(f["alternatives"][0][0], avoid, exclude_video)
                if ex:
                    lines.append(f"    e.g. {ex.pru}" + (f"  ({ex.en})" if ex.en else ""))
        if not lines:
            return "CONSTRUCTIONS: every construction in the draft is attested in the corpus."
        return ("CONSTRUCTIONS rare or unattested in the corpus "
                "(with the attested variants):\n" + "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["build"]:
        build()
        return 0
    if argv[:1] == ["check"]:
        print(Constructions().check(" ".join(argv[1:])))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
