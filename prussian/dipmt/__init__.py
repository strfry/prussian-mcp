"""DiPMT++-style lexical support for English → Prussian.

After Zhang et al. 2024, "Teaching Large Language Models an Unseen Language
on the Fly" (Findings ACL 2024): instead of an agent searching word by word,
every word of the source sentence gets dictionary candidates from cheap,
deterministic resources, plus a few retrieved parallel exemplars.

* :mod:`.corpus`   — the lemmatized parallel corpus and the held-out split;
* :mod:`.align`    — bilingual lexicon induction (eflomal over English
  lemmas ↔ Prussian lemmas, dictionary glosses as priors);
* :mod:`.lexicon`  — per-word candidates: dictionary gloss match, induced
  lexicon, WordNet synonyms, compound/fuzzy matching;
* :mod:`.bench`    — lemma-recall benchmark on the held-out sentences.

Needs the ``dipmt`` extra (``eflomal`` builds with ``CC=clang`` here).
"""
