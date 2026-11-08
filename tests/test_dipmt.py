"""DiPMT++ lexicon: symmetrization, extraction, candidate ranking (offline)."""

import unittest

from prussian.dipmt.align import extract, grow_diag_final_and
from prussian.dipmt.corpus import Pair
from prussian.dipmt.lexicon import Candidate, _rank


def _pair(sid, en, forms, lemmas):
    return Pair(sid, " ".join(en), " ".join(forms), list(en), list(en),
                list(forms), list(lemmas), ["X"] * len(forms))


class TestAlign(unittest.TestCase):
    def test_grow_diag_adds_neighbour_of_intersection(self):
        fwd = {(0, 0), (1, 1)}
        rev = {(0, 0), (1, 2)}
        self.assertEqual(grow_diag_final_and(fwd, rev), {(0, 0), (1, 1), (1, 2)})

    def test_frozen_phrase_from_free_neighbour(self):
        # "maybe" ↔ mazzi; būtwei follows unaligned every time
        pairs = [_pair(f"s{i}", ["maybe", "you"], ["mazzi", "būtwei", "tū"],
                       ["mazītwei", "būtwei", "tū"]) for i in range(3)]
        links = [{(0, 0), (1, 2)}] * 3
        e = extract(pairs, links)["entries"]
        self.assertEqual(e["maybe"]["mazītwei"]["n"], 3)
        self.assertEqual(e["maybe"]["mazītwei"]["form"], "mazzi")
        self.assertEqual(e["maybe"]["mazītwei"]["phrase"], "mazzi būtwei")

    def test_no_phrase_when_neighbour_belongs_to_another_word(self):
        pairs = [_pair(f"s{i}", ["has", "right"], ["turri", "tikrōmiskan"],
                       ["turītun", "tikrōmiskan"]) for i in range(3)]
        links = [{(0, 0), (1, 1)}] * 3
        e = extract(pairs, links)["entries"]
        self.assertNotIn("phrase", e["right"]["tikrōmiskan"])

    def test_bigram_entry(self):
        pairs = [_pair("s1", ["burial", "site"], ["kapurna"], ["kapurna"])]
        e = extract(pairs, [{(0, 0), (1, 0)}])["entries"]
        self.assertEqual(e["burial site"]["kapurna"]["n"], 1)


class TestRank(unittest.TestCase):
    def test_dict_and_corpus_agreement_leads(self):
        cands = [Candidate("būda", "dict"),
                 Candidate("telepōns", "corpus", n=3),
                 Candidate("būda", "corpus", n=2, form="būdan")]
        ranked = _rank(cands, total=5)
        self.assertEqual([c.lemma for c in ranked], ["būda", "telepōns"])
        self.assertEqual(ranked[0].form, "būdan")   # corpus details kept


if __name__ == "__main__":
    unittest.main()
