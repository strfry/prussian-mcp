"""Lemma-recall benchmark for the search task on held-out sentences.

For every held-out pair (eval set + random sample, never seen by the
aligner) each system proposes Prussian lemmas for the English sentence; the
score is the share of the sentence's gold content lemmas (NOUN/VERB/ADJ/ADV
from the FST analysis, ``būtwei`` excluded) among the proposals, and the
cost is the number of proposals.  Alignment-free: the gold side needs no
English↔Prussian word links.

``uv run python -m prussian.dipmt.bench`` — results per system, overall and
for the eval-set sentences alone.
"""

from __future__ import annotations

import json
import sys
import time

from prussian.dipmt.align import SPLIT_PATH
from prussian.dipmt.corpus import (en_lemma, en_tokenize, eval_texts,
                                   load_pairs, norm_text)
from prussian.dipmt.lexicon import STOPWORDS, Lexicon

_SKIP_GOLD = {"būtwei"}


def base_vocab_words() -> set[str]:
    """Every word mentioned in prompts/base_vocab.md (lemmas and forms) —
    the agent gets these in its prompt, so they need no search."""
    import re
    from prussian.config import PROMPTS_DIR
    text = (PROMPTS_DIR / "base_vocab.md").read_text(encoding="utf-8").lower()
    return set(re.findall(r"[^\W\d_]+", text))


def search_system(engine, k: int):
    """Baseline: the fused search per English content word, top-k lemmas."""
    def run(sentence: str) -> set[str]:
        out: set[str] = set()
        for tok in en_tokenize(sentence):
            if tok in STOPWORDS or en_lemma(tok) in STOPWORDS:
                continue
            words: list[str] = []
            for chunk in engine.query(tok, k):
                for e in chunk.get("entries", []):
                    w = e["word"].lower()
                    if w not in words:
                        words.append(w)
            out.update(words[:k])
        return out
    return run


def dipmt_system(lex: Lexicon, sources, k: int):
    def run(sentence: str) -> set[str]:
        return {c.lemma for wc in lex.candidates(sentence, sources=sources, k=k)
                for c in wc.candidates}
    return run


def evaluate(systems: dict, pairs, evals: set[str], base: set[str]) -> list[dict]:
    rows = []
    for name, run in systems.items():
        t0 = time.time()
        hit = gold_n = cand_n = 0
        e_hit = e_gold = b_hit = b_gold = 0
        for p in pairs:
            gold = p.content_lemmas() - _SKIP_GOLD
            cands = run(p.en)
            h = len(gold & cands)
            hit += h
            gold_n += len(gold)
            cand_n += len(cands)
            b_hit += len((gold - base) & cands)
            b_gold += len(gold - base)
            if norm_text(p.pru) in evals:
                e_hit += h
                e_gold += len(gold)
        rows.append({"system": name, "recall": hit / max(gold_n, 1),
                     "eval_recall": e_hit / max(e_gold, 1),
                     "nobase_recall": b_hit / max(b_gold, 1),
                     "cands_per_sent": cand_n / max(len(pairs), 1),
                     "sec": time.time() - t0})
    return rows


def main(argv: list[str] | None = None) -> int:
    from prussian.tools.runtime import get_engine

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    held_ids = set(split["heldout"])
    pairs = [p for p in load_pairs() if p.sent_id in held_ids]
    evals = eval_texts()
    engine = get_engine()
    lex = Lexicon(engine)
    systems = {
        "search@2": search_system(engine, 2),
        "search@5": search_system(engine, 5),
        "dict@2": dipmt_system(lex, ("dict",), 2),
        "corpus@2": dipmt_system(lex, ("corpus",), 2),
        "dict+corpus@2": dipmt_system(lex, ("dict", "corpus"), 2),
        "+wordnet@2": dipmt_system(lex, ("dict", "corpus", "wordnet"), 2),
        "+fuzzy@2": dipmt_system(lex, ("dict", "corpus", "wordnet", "fuzzy"), 2),
        "+search@2": dipmt_system(lex, ("dict", "corpus", "wordnet", "fuzzy", "search"), 2),
        "+fuzzy@5": dipmt_system(lex, ("dict", "corpus", "wordnet", "fuzzy"), 5),
    }
    gold_total = sum(len(p.content_lemmas() - _SKIP_GOLD) for p in pairs)
    n_eval = sum(norm_text(p.pru) in evals for p in pairs)
    print(f"{len(pairs)} held-out sentences ({n_eval} from the eval set), "
          f"{gold_total} gold content lemmas", file=sys.stderr)
    base = base_vocab_words()
    print(f"{'system':16} {'recall':>7} {'no-base':>8} {'eval30':>7} {'cands/sent':>11} {'sec':>6}")
    for r in evaluate(systems, pairs, evals, base):
        print(f"{r['system']:16} {r['recall']:7.1%} {r['nobase_recall']:8.1%} {r['eval_recall']:7.1%} "
              f"{r['cands_per_sent']:11.1f} {r['sec']:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
