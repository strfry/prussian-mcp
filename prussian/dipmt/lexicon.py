"""Per-word Prussian candidates for an English sentence (DiPMT++ lexicon).

Each English token gets candidates from deterministic sources, tried in
order; every candidate records where it came from:

* ``dict``    — a dictionary gloss literally equals the word or its lemma
  (``SearchEngine._exact_lemmas``; ``to <verb>`` is tried too);
* ``corpus``  — the induced lexicon (:mod:`.align`); English bigrams
  (``burial site``) take precedence over their parts;
* ``wordnet`` — dictionary/corpus hits for WordNet synonyms of the lemma,
  only when the word itself found nothing;
* ``fuzzy``   — derivational stripping (``-ly``, ``-ness``) and the parts of
  hyphenated words, only when nothing else matched;
* ``search``  — optional fallback to the fused embedding search (top hit).

Nothing here is generated: every candidate is a dictionary headword or a
lemma/phrase observed in the aligned corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from prussian.dipmt.align import LEXICON_PATH
from prussian.dipmt.corpus import en_lemma, en_tokenize

STOPWORDS = frozenset("""
a an the and or but if so as of to in on at by for from with into onto over
under about than then there here this that these those it its i me my you your
he him his she her we us our they them their who whom whose what which is am
are was were be been being do does did have has had will would shall should can
could may might must not no yes very just also too all any some such s t don
""".split())

SOURCES = ("dict", "corpus", "wordnet", "fuzzy", "search")


@dataclass
class Candidate:
    lemma: str
    source: str
    n: int = 0                       # corpus alignment count (corpus source)
    form: str = ""                   # dominant surface form in the corpus
    phrase: str = ""                 # frozen phrase (``mazzi būtwei``)
    examples: list[str] = field(default_factory=list)


@dataclass
class WordCandidates:
    token: str
    lemma: str
    span: tuple[int, int]            # token index range [start, end)
    candidates: list[Candidate]


class Lexicon:
    def __init__(self, engine, lexicon_path=LEXICON_PATH, min_count: int = 1,
                 min_share: float = 0.0):
        self.engine = engine
        data = json.loads(lexicon_path.read_text(encoding="utf-8"))
        self.entries: dict[str, dict[str, dict]] = data["entries"]
        self.en_count: dict[str, int] = data["en_count"]
        self.min_count = min_count
        self.min_share = min_share

    # ── single sources ────────────────────────────────────────────────────────

    def from_dict(self, word: str) -> list[Candidate]:
        ids = []
        for q in dict.fromkeys([word, en_lemma(word), f"to {en_lemma(word)}"]):
            ids += [i for i in self.engine._exact_lemmas(q) if i not in ids]
        out = []
        for i in ids:
            e = self.engine.id_to_entry.get(i)
            if e and e["word"].lower() not in {c.lemma for c in out}:
                out.append(Candidate(e["word"].lower(), "dict"))
        return out

    def from_corpus(self, lemma: str) -> list[Candidate]:
        total = self.en_count.get(lemma, 0) or 1
        rows = sorted(self.entries.get(lemma, {}).items(),
                      key=lambda kv: -kv[1]["n"])
        return [Candidate(pru, "corpus", v["n"], v.get("form", ""),
                          v.get("phrase", ""), v.get("examples", []))
                for pru, v in rows
                if v["n"] >= self.min_count and v["n"] / total >= self.min_share]

    def from_wordnet(self, lemma: str) -> list[Candidate]:
        out: list[Candidate] = []
        for syn in _synonyms(lemma):
            for c in self.from_dict(syn) + self.from_corpus(syn):
                if c.lemma not in {x.lemma for x in out}:
                    out.append(Candidate(c.lemma, "wordnet", c.n, c.form,
                                         c.phrase, c.examples))
        return out

    def from_fuzzy(self, word: str) -> list[Candidate]:
        variants = []
        for suf, rep in (("ily", "y"), ("ly", ""), ("ness", ""), ("ful", "")):
            if word.endswith(suf) and len(word) - len(suf) >= 3:
                variants.append(word[: -len(suf)] + rep)
        out: list[Candidate] = []
        for v in variants:
            for c in self.from_dict(v) + self.from_corpus(en_lemma(v)):
                if c.lemma not in {x.lemma for x in out}:
                    out.append(Candidate(c.lemma, "fuzzy", c.n, c.form, c.phrase))
        return out

    def from_search(self, word: str) -> list[Candidate]:
        hits = self.engine.query(word, 1)
        return [Candidate(e["word"].lower(), "search")
                for h in hits for e in h.get("entries", [])[:1]]

    # ── per sentence ──────────────────────────────────────────────────────────

    def candidates(self, sentence: str, sources=SOURCES[:4], k: int = 2,
                   content_only: bool = True) -> list[WordCandidates]:
        """Up to ``k`` candidates per English word (bigrams merged)."""
        tokens = en_tokenize(sentence)
        lemmas = [en_lemma(t) for t in tokens]
        out: list[WordCandidates] = []
        i = 0
        while i < len(tokens):
            if "corpus" in sources and i + 1 < len(tokens):
                bigram = f"{lemmas[i]} {lemmas[i + 1]}"
                cands = self.from_corpus(bigram)
                if cands and not ({tokens[i], tokens[i + 1]} <= STOPWORDS):
                    out.append(WordCandidates(f"{tokens[i]} {tokens[i + 1]}",
                                              bigram, (i, i + 2), cands[:k]))
                    i += 2
                    continue
            tok, lem = tokens[i], lemmas[i]
            if content_only and (tok in STOPWORDS or lem in STOPWORDS):
                i += 1
                continue
            cands: list[Candidate] = []
            if "dict" in sources:
                cands += self.from_dict(tok)
            if "corpus" in sources:
                cands += self.from_corpus(lem)
            for src, fn, arg in (("wordnet", self.from_wordnet, lem),
                                 ("fuzzy", self.from_fuzzy, tok),
                                 ("search", self.from_search, tok)):
                if not cands and src in sources:
                    cands += fn(arg)
            total = self.en_count.get(lem, 0)
            out.append(WordCandidates(tok, lem, (i, i + 1), _rank(cands, total)[:k]))
            i += 1
        return out


def _rank(cands: list[Candidate], total: int = 0) -> list[Candidate]:
    """Merge duplicates and order by evidence.

    Score: the corpus share (how often the English word aligned to this
    lemma) plus a bonus for a literal dictionary gloss; a lemma seen in both
    leads.  Without corpus evidence the source order decides.
    """
    merged: dict[str, Candidate] = {}
    srcs: dict[str, set[str]] = {}
    for c in cands:
        if c.lemma in merged:
            m = merged[c.lemma]
            if c.source == "corpus":      # keep counts / phrase / examples
                c.source = m.source
                merged[c.lemma] = c
        else:
            merged[c.lemma] = c
        srcs.setdefault(c.lemma, set()).add("corpus" if c.n else c.source)
    order = {s: i for i, s in enumerate(("dict", "corpus", "wordnet", "fuzzy", "search"))}

    def score(c: Candidate) -> tuple:
        share = c.n / total if total else 0.0
        return (-(share + (0.5 if "dict" in srcs[c.lemma] else 0.0)),
                order.get(c.source, 9))
    return sorted(merged.values(), key=score)


@lru_cache(maxsize=8192)
def _synonyms(lemma: str) -> tuple[str, ...]:
    """WordNet synonyms (all senses, single words), most frequent senses first."""
    try:
        from nltk.corpus import wordnet as wn
        synsets = wn.synsets(lemma)
    except LookupError:
        return ()
    out: list[str] = []
    for s in synsets[:6]:
        for l in s.lemmas():
            name = l.name().lower()
            if "_" not in name and name != lemma and name not in out:
                out.append(name)
    return tuple(out[:12])
