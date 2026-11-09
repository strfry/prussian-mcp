"""DiPMT++ lexicon: symmetrization, extraction, candidate ranking (offline)."""

import unittest
from collections import Counter

from prussian.dipmt.align import extract, grow_diag_final_and
from prussian.dipmt.corpus import Pair
from prussian.dipmt.constructions import (Constructions, _variants,
                                          fragments, parse_conllu)
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


_PADU = """\
1\ten\tēn\tADP\tPr\t_\t3\tcase\t_\t_
2\tšismu\tšis\tPRON\tPron+Msc+Sg+Dat\tCase=Dat\t3\tdet\t_\t_
3\tpādu\tpāds\tNOUN\tN+Msc+Sg+Dat\tCase=Dat\t5\tobl\t_\t_
4\tni\tni\tPART\tPcle\t_\t5\tadvmod\t_\t_
5\tast\tbūtwei\tVERB\tV+Ind+Prs+SP3\tMood=Ind|Tense=Pres\t0\troot\t_\t_
6\t.\t.\tPUNCT\t_\t_\t5\tpunct\t_\t_
"""


class TestConstructions(unittest.TestCase):
    def test_fragments_keep_function_words_lexical(self):
        frags = {f for f, _ in fragments(parse_conllu(_PADU))}
        self.assertIn("ēn PRON.Dat NOUN.Dat", frags)
        self.assertIn("ni būtwei.Pres", frags)
        self.assertIn("NOUN.Dat <det< PRON.Dat", frags)
        # windows must open with a function word: no "NOUN.Dat ni"
        self.assertNotIn("NOUN.Dat ni", frags)

    def test_variants_change_one_slot(self):
        vs = _variants("ēn NOUN.Dat", {"NOUN": {"Dat", "Acc", "?"}})
        self.assertEqual(vs, ["ēn NOUN.Acc"])

    def test_analyse_flags_rare_value_in_attested_context(self):
        idx = Constructions.__new__(Constructions)
        idx.count = Counter({"ēn PRON.Dat NOUN.Dat": 30, "ēn NOUN.Dat": 40})
        idx.values = {"NOUN": {"Dat", "Nom"}, "PRON": {"Dat", "Nom"}}
        toks = parse_conllu(_PADU.replace("NOUN\tN+Msc+Sg+Dat\tCase=Dat",
                                          "NOUN\tN+Msc+Sg+Nom\tCase=Nom"))
        found = {f["fragment"]: f for f in idx.analyse(toks)}
        self.assertIn("ēn PRON.Dat NOUN.Nom", found)
        self.assertEqual(found["ēn PRON.Dat NOUN.Nom"]["alternatives"][0],
                         ("ēn PRON.Dat NOUN.Dat", 30))
        self.assertFalse(idx.analyse(parse_conllu(_PADU)))


if __name__ == "__main__":
    unittest.main()
