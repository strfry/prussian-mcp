"""The parallel corpus: English subtitles ↔ FST-lemmatized Prussian.

Source is ``sentences_scored.jsonl`` from the eval (YouTube subtitles with
an FST/CG3 token analysis per sentence).  Subtitles repeat, so pairs are
deduplicated on the normalized Prussian text.  The held-out split always
contains the eval set's sentences (matched on text, not id) so neither the
induced lexicon nor the exemplar pool can leak eval targets.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from prussian.config import PROJECT_ROOT

EVAL_DATA = PROJECT_ROOT.parent / "eval" / "data"
CORPUS_PATH = Path(os.getenv("DIPMT_CORPUS", str(EVAL_DATA / "sentences_scored.jsonl")))
EVAL_SET = Path(os.getenv("DIPMT_EVAL_SET", str(EVAL_DATA / "eval_v2.jsonl")))

CONTENT_UPOS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})

_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


@dataclass
class Pair:
    sent_id: str
    en: str
    pru: str
    en_tokens: list[str]          # lowercased surface words
    en_lemmas: list[str]          # simplemma lemmas, aligned to en_tokens
    pru_forms: list[str]          # lowercased surface tokens (no punctuation)
    pru_lemmas: list[str]         # FST lemma, else the form
    pru_upos: list[str] = field(default_factory=list)

    def content_lemmas(self) -> set[str]:
        return {l for l, u in zip(self.pru_lemmas, self.pru_upos)
                if u in CONTENT_UPOS}


def norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


@lru_cache(maxsize=65536)
def en_lemma(word: str) -> str:
    import simplemma
    return simplemma.lemmatize(word, lang="en").lower()


def en_tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def _pair(row: dict) -> Pair | None:
    toks = [t for t in row.get("tokens", []) if re.search(r"\w", t.get("form", ""))]
    en_tokens = en_tokenize(row.get("translation_en") or "")
    if not toks or not en_tokens:
        return None
    lemma = lambda t: (t.get("lemma") if t.get("lemma") not in (None, "", "_")
                       else t["form"]).lower()
    return Pair(
        sent_id=row["sent_id"],
        en=row["translation_en"],
        pru=row.get("text_norm") or row["text_clean"],
        en_tokens=en_tokens,
        en_lemmas=[en_lemma(w) for w in en_tokens],
        pru_forms=[t["form"].lower() for t in toks],
        pru_lemmas=[lemma(t) for t in toks],
        pru_upos=[t.get("upos", "") for t in toks],
    )


def load_pairs(path: Path = CORPUS_PATH) -> list[Pair]:
    """All usable pairs, deduplicated on the normalized Prussian text."""
    seen: set[str] = set()
    out: list[Pair] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = norm_text(row.get("text_norm") or row["text_clean"])
            if key in seen:
                continue
            p = _pair(row)
            if p:
                seen.add(key)
                out.append(p)
    return out


def eval_texts(path: Path = EVAL_SET) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.add(norm_text(json.loads(line)["target"]))
    return out


def split(pairs: list[Pair], heldout_frac: float = 0.1, seed: int = 13
          ) -> tuple[list[Pair], list[Pair]]:
    """(train, heldout).  Heldout = the eval set + a seeded random sample."""
    evals = eval_texts()
    rng = random.Random(seed)
    train, held = [], []
    for p in pairs:
        if norm_text(p.pru) in evals or rng.random() < heldout_frac:
            held.append(p)
        else:
            train.append(p)
    return train, held
