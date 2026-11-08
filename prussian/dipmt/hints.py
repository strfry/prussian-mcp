"""The DiPMT++ prompt block: word hints plus retrieved exemplars.

``Hints(engine).render(sentence)`` returns plain text for the user message::

    WORD HINTS (candidate Prussian lemmas from the dictionary and an aligned
    corpus — pick by meaning, then fetch the exact form with get_word_forms):
    - maybe → mazzi būtwei (fixed phrase, corpus 69×)
    - die → aulaūtwei "to die" (dictionary + corpus 4×)
    ...
    SIMILAR SENTENCES FROM THE CORPUS:
    English: …
    Prussian: …

Exemplars are BM25 over the English lemmas of the *training* pairs only
(``data/dipmt/split.json``), so held-out and eval sentences never show up
as their own example.
"""

from __future__ import annotations

import json

from prussian.dipmt.align import SPLIT_PATH
from prussian.dipmt.corpus import en_lemma, en_tokenize, load_pairs
from prussian.dipmt.lexicon import STOPWORDS, Lexicon

_HEADER = ("WORD HINTS (candidate Prussian lemmas from the dictionary and an "
           "aligned corpus — pick by meaning, then fetch the exact form with "
           "get_word_forms; words without a hint: search_dictionary):")


def _content_lemmas(text: str) -> str:
    return " ".join(l for l in (en_lemma(t) for t in en_tokenize(text))
                    if l not in STOPWORDS)


class Hints:
    def __init__(self, engine, k: int = 2, n_exemplars: int = 3):
        from prussian_embeddings import BM25Index
        self.engine = engine
        self.lexicon = Lexicon(engine)
        self.k = k
        self.n_exemplars = n_exemplars
        train = set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["train"])
        self.pool = [p for p in load_pairs() if p.sent_id in train]
        self.bm25 = BM25Index([_content_lemmas(p.en) for p in self.pool],
                              tokenizer=str.split)

    def exemplars(self, sentence: str):
        hits = self.bm25.query(_content_lemmas(sentence), self.n_exemplars)
        return [self.pool[i] for i, _ in hits]

    def _gloss(self, lemma: str, en: str = "") -> str:
        """English gloss of the homograph that mentions ``en``, else the first
        (mēniks: "moon", not "gingerbread")."""
        glosses = [(e.get("translations") or {}).get("en") or []
                   for e in self.engine.word_to_entry.get(lemma, [])]
        glosses = [g for g in glosses if g]
        for g in glosses:
            if en and any(en in x.lower() for x in g):
                return ", ".join(g[:2])
        return ", ".join(glosses[0][:2]) if glosses else ""

    def _describe(self, c, en: str = "") -> str:
        head = c.phrase or c.lemma
        gloss = "" if c.phrase else self._gloss(c.lemma, en)
        bits = []
        if c.phrase:
            bits.append("fixed phrase")
        if c.source == "dict":            # also set when merged with corpus
            bits.append("dictionary")
        if c.n:
            bits.append(f"corpus {c.n}×")
        if c.source in ("wordnet", "fuzzy", "search"):
            bits.append({"wordnet": "via synonym", "fuzzy": "via word stem",
                         "search": "search hit"}[c.source])
        g = f' "{gloss}"' if gloss else ""
        return f"{head}{g} ({', '.join(bits)})" if bits else f"{head}{g}"

    def render(self, sentence: str) -> str:
        lines = [_HEADER]
        done: set[str] = set()
        for wc in self.lexicon.candidates(sentence, k=self.k):
            if wc.candidates and wc.lemma not in done:
                done.add(wc.lemma)
                lines.append(f"- {wc.token} → " + " | ".join(
                    self._describe(c, wc.lemma) for c in wc.candidates))
        if len(lines) == 1:
            lines.append("- (none)")
        ex = self.exemplars(sentence)
        if ex:
            lines.append("")
            lines.append("SIMILAR SENTENCES FROM THE CORPUS:")
            for p in ex:
                lines.append(f"English: {p.en}")
                lines.append(f"Prussian: {p.pru}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import sys
    from prussian.tools.runtime import get_engine
    print(Hints(get_engine()).render(" ".join(argv or sys.argv[1:])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
