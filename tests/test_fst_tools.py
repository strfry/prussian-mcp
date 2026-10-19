"""Tests for FST-based tools: fst_tags, wordforms, lookup, search.

FST-dependent tests skip when prussian-fst artifacts are not built.
The pure-logic tests (match_tags, resolve_features) always run.
"""

import unittest

from prussian.engine.fst.tags import (
    match_tags,
    resolve_features,
    fst_available,
    FEATURE_MAP,
)

# Check FST availability for integration tests
FST_READY = fst_available()


class TestMatchTags(unittest.TestCase):
    """Pure-logic tests for match_tags — no FST needed."""

    def test_subset_match(self):
        self.assertTrue(match_tags(["Adj+Sg+Akk+Masc"], "Akk+Sg"))

    def test_exact_match(self):
        self.assertTrue(match_tags(["V+Ind+Pres+P1+Sg"],
                                   "V+Ind+Pres+P1+Sg"))

    def test_no_match(self):
        self.assertFalse(match_tags(["N+Sg+Nom"], "Akk"))

    def test_empty_wanted(self):
        self.assertTrue(match_tags(["N+Sg"], ""))

    def test_case_insensitive(self):
        self.assertTrue(match_tags(["adj+sg+akk"], "AKK+SG"))

    def test_comma_separated(self):
        self.assertTrue(match_tags(["V+Ind+Pres"], "Ind,Pres"))

    def test_space_separated(self):
        self.assertTrue(match_tags(["V+Ind+Pres"], "Ind Pres"))

    def test_dot_separated(self):
        self.assertTrue(match_tags(["N+Gen+Pl"], "Gen.Pl"))

    def test_partial_not_match(self):
        self.assertFalse(match_tags(["V+Ind+Pres"], "Opt"))

    def test_multiple_tags_in_list(self):
        self.assertTrue(match_tags(
            ["V+Ind+Pres+P1+Sg"], "P1+Sg"))

    def test_cross_reading_not_match(self):
        """Gen and Pl in different readings should NOT match Gen+Pl."""
        self.assertFalse(match_tags(
            ["N+Sg+Gen+Fem", "N+Pl+Nom+Fem"], "Gen+Pl"))

    def test_same_reading_match(self):
        """Gen and Pl in the same reading SHOULD match Gen+Pl."""
        self.assertTrue(match_tags(
            ["N+Pl+Gen+Fem"], "Gen+Pl"))


class TestResolveFeatures(unittest.TestCase):
    """Pure-logic tests for resolve_features — no FST needed."""

    def test_human_readable(self):
        result = resolve_features("participle")
        self.assertEqual(result, ["PrfPrc"])

    def test_multiple_human_readable(self):
        result = resolve_features("indicative,present")
        self.assertEqual(result, ["Ind", "Prs"])

    def test_raw_fst_tag(self):
        result = resolve_features("Ind+Prs")
        self.assertEqual(result, ["Ind", "Prs"])

    def test_mixed(self):
        result = resolve_features("participle,Gen+Pl")
        self.assertEqual(result, ["PrfPrc", "Gen", "Pl"])

    def test_empty(self):
        result = resolve_features("")
        self.assertIsNone(result)

    def test_none(self):
        result = resolve_features(None)
        self.assertIsNone(result)

    def test_unknown_feature(self):
        result = resolve_features("foobar")
        self.assertIsNone(result)

    def test_case_insensitive_name(self):
        result = resolve_features("Participle")
        self.assertEqual(result, ["PrfPrc"])

    def test_conjunctive_maps_to_subj(self):
        result = resolve_features("conjunctive")
        self.assertEqual(result, ["Subj"])

    def test_past_maps_to_pret(self):
        result = resolve_features("past")
        self.assertEqual(result, ["Prt"])


class TestParadigmCells(unittest.TestCase):
    """Cell enumeration from the lemma's own readings (pure logic)."""

    def setUp(self):
        from prussian.engine.fst.paradigm_order import paradigm_cells
        self.cells = paradigm_cells

    def test_noun_keeps_own_gender(self):
        cells = self.cells([("N", "Fem", "Sg", "Nom")])
        self.assertEqual(len(cells), 8)
        self.assertIn("N+Fem+Pl+Acc", cells)
        self.assertNotIn("N+Msc+Sg+Nom", cells)

    def test_adjective_all_genders_degrees_adverb(self):
        cells = self.cells([("A", "Msc", "Sg", "Nom")])
        self.assertIn("A+Fem+Pl+Dat", cells)
        self.assertIn("A+Superl+Neu+Sg+Gen", cells)
        self.assertIn("Adv+Comp", cells)
        self.assertEqual(len(cells), 3 * 24 + 3)

    def test_verb_cells_and_reflexive(self):
        cells = self.cells([("V", "Inf", "Refl"), ("V", "Inf")])
        self.assertEqual(cells[0], "V+Inf")
        self.assertIn("V+Ind+Prs+SP3", cells)
        self.assertIn("V+PrfPrc+Act+Msc+Sg+Nom", cells)
        self.assertIn("V+Ind+Prs+SP3+Refl", cells)

    def test_invariable(self):
        self.assertEqual(self.cells([("Pr",)]), ["Pr"])

    def test_pronoun_gender_first_number_optional(self):
        cells = self.cells([("Pron", "Msc", "Sg", "Nom")])
        self.assertIn("Pron+Fem+Pl+Dat", cells)
        self.assertIn("Pron+Neu+Nom", cells)
        self.assertEqual(len(cells), 3 * 3 * 4)

    def test_personal_pronoun_keeps_person_no_gender(self):
        cells = self.cells([("Pron", "P1", "Sg", "Nom")])
        self.assertIn("Pron+P1+Sg+Dat", cells)
        self.assertFalse(any("Msc" in c for c in cells))


def _fw(*cells):
    return [{"form": f"f{i}", "tags": [c]} for i, c in enumerate(cells)]


class TestFeatureGroups(unittest.TestCase):
    """Grouped feature summary and empty-match diagnosis (pure logic)."""

    def setUp(self):
        from prussian.tools import wordforms
        self.wf = wordforms

    def test_noun_gender_is_inherent(self):
        fw = _fw(*(f"N+Msc+{n}+{c}" for n in ("Sg", "Pl")
                   for c in ("Nom", "Gen", "Dat", "Acc")))
        inherent, groups = self.wf._feature_groups(fw)
        self.assertEqual(inherent, ["Msc"])
        self.assertEqual(groups, [("declension", "Sg|Pl × Nom|Gen|Dat|Acc")])

    def test_verb_finite_and_participle_separate(self):
        fw = _fw("V+Inf",
                 *(f"V+Ind+Prs+{p}" for p in ("Sg1", "Sg2", "SP3", "Pl1", "Pl2")),
                 "V+Imprt+Sg2", "V+Imprt+Pl2",
                 *(f"V+PrfPrc+Pss+{g}+{n}+Nom" for g in ("Msc", "Fem")
                   for n in ("Sg", "Pl")))
        groups = dict(self.wf._feature_groups(fw)[1])
        self.assertEqual(groups["finite"],
                         "Ind × Prs × Sg1|Sg2|SP3|Pl1|Pl2; Imprt × Sg2|Pl2")
        self.assertEqual(groups["participle"], "PrfPrc+Pss × Msc|Fem × Sg|Pl × Nom")
        self.assertEqual(groups["infinitive"], "Inf")

    def test_adjective_degrees_and_derived_adverb(self):
        fw = _fw("A+Msc+Sg+Nom", "A+Comp+Msc+Sg+Nom", "Adv", "Adv+Comp")
        groups = dict(self.wf._feature_groups(fw)[1])
        self.assertEqual(groups["declension"], "[Comp] × Msc × Sg × Nom")
        self.assertEqual(groups["derived adverb"], "Adv[+Comp] (f2)")

    def test_gaps_marked(self):
        fw = _fw("Pron+Msc+Sg+Nom", "Pron+Fem+Sg+Nom", "Pron+Msc+Pl+Nom",
                 "Pron+Msc+Sg+Gen")
        groups = dict(self.wf._feature_groups(fw)[1])
        self.assertEqual(groups["declension"], "Msc|Fem × Sg|Pl × Nom|Gen (gaps)")

    def test_diagnose_absent_tag_names_inherent(self):
        cells = ["N+Msc+Sg+Nom", "N+Msc+Pl+Nom"]
        self.assertEqual(self.wf._diagnose(cells, ["Fem"], ["Msc"]),
                         "Fem not in this paradigm (inherent: Msc)")

    def test_diagnose_clashing_tag(self):
        cells = ["V+Ind+Prs+Sg1", "V+Ind+Prs+SP3", "V+PrfPrc+Pss+Msc+Sg+Nom"]
        self.assertEqual(self.wf._diagnose(cells, ["Ind", "Prs", "Sg"], []),
                         "Sg does not combine with Ind+Prs (finite, 2 forms)")


@unittest.skipIf(not FST_READY, "prussian-fst-Artefakte nicht gebaut")
class TestGenerateParadigm(unittest.TestCase):
    def test_noun_and_verb_forms(self):
        from prussian.engine.fst.tags import generate_paradigm
        forms = {(f["form"], f["tags"][0]) for f in generate_paradigm({"word": "berzi"})}
        self.assertIn(("berzin", "N+Fem+Sg+Acc"), forms)
        forms = {(f["form"], f["tags"][0]) for f in generate_paradigm({"word": "būtwei"})}
        self.assertIn(("ast", "V+Ind+Prs+SP3"), forms)

    def test_pronoun_forms(self):
        from prussian.engine.fst.tags import generate_paradigm
        forms = {(f["form"], f["tags"][0]) for f in generate_paradigm({"word": "stas"})}
        self.assertIn(("stēimans", "Pron+Msc+Pl+Dat"), forms)
        self.assertIn(("sta", "Pron+Neu+Nom"), forms)

    def test_unknown_lemma_yields_nothing(self):
        from prussian.engine.fst.tags import generate_paradigm
        self.assertEqual(generate_paradigm({"word": "xyzzy"}), [])

    def test_reflexive_entry_gets_refl_cells_with_particle(self):
        from prussian.engine.fst.tags import generate_paradigm
        forms = {(f["form"], f["tags"][0])
                 for f in generate_paradigm({"word": "laikātun si"})}
        self.assertIn(("lāika si", "V+Ind+Prs+SP3+Refl"), forms)
        self.assertIn(("laikāsei si", "V+Opt+SP3+Refl"), forms)
        self.assertTrue(all("Refl" in t and f.endswith(" si") for f, t in forms))

    def test_bare_verb_has_no_refl_cells(self):
        from prussian.engine.fst.tags import generate_paradigm
        forms = {(f["form"], f["tags"][0])
                 for f in generate_paradigm({"word": "laikātun"})}
        self.assertIn(("lāika", "V+Ind+Prs+SP3"), forms)
        self.assertFalse(any("Refl" in t for _, t in forms))


class TestReflexiveHeadword(unittest.TestCase):
    """Headword ↔ (FST lemma, Refl) mapping (pure logic)."""

    def test_split_refl(self):
        from prussian.engine.fst.tags import split_refl
        self.assertEqual(split_refl("laikātun si"), ("laikātun", True))
        self.assertEqual(split_refl("laikātun"), ("laikātun", False))

    def test_entry_lemma(self):
        from prussian.engine.fst.tags import entry_lemma
        self.assertEqual(entry_lemma("laikātun", ["V", "Ind", "Refl"]), "laikātun si")
        self.assertEqual(entry_lemma("laikātun", ["V", "Ind"]), "laikātun")


@unittest.skipIf(not FST_READY, "prussian-fst-Artefakte nicht gebaut")
class TestLookupReflexive(unittest.TestCase):
    def test_refl_readings_go_to_the_si_entry(self):
        from types import SimpleNamespace
        from prussian.tools.lookup import lookup_tool
        engine = SimpleNamespace(word_to_entry={
            "laikātun": [{"word": "laikātun", "translations": {"de": ["halten"]}}],
            "laikātun si": [{"word": "laikātun si",
                             "translations": {"de": ["sich benehmen"]}}],
        })
        lines = lookup_tool(engine, "lāika si").splitlines()
        heads = {tuple(l.split()[:2]): l for l in lines if l.startswith("  laikātun")}
        refl = heads[("laikātun", "si")]
        bare = next(l for k, l in heads.items() if k[1] != "si")
        self.assertIn("V+Ind+Prs+SP3+Refl", refl)
        self.assertNotIn("Refl", bare)


@unittest.skipIf(not FST_READY, "prussian-fst-Artefakte nicht gebaut")
class TestValidateBattery(unittest.TestCase):
    """Regressionsbatterie (Arbeitsauftrag 2026-07, 0 false verifies):
    pinnt den MCP-sichtbaren Drei-Wert-Kontrakt von validate_prussian
    unabhängig von den prussian-fst-Interna.  Spiegel der Status-Tests
    in prussian-fst/tests/test_validate.py."""

    CASES = [
        ("As asma prūsiskan wīran.", "violations_found"),
        ("As asma prūsiskas wīrs.", "violations_found"),
        ("As asma prūsisks wīrs.", "verified_in_coverage"),
        ("Tū turri stas wīrs.", "violations_found"),
        ("Mes waida stan wīran.", "violations_found"),
        ("Tū turri stan wīran.", "verified_in_coverage"),
        ("As asma stan autōmatikin rekōnstruiwuns be sen grammatikin "
         "perbāndan plattinuns.", "out_of_coverage"),
    ]

    def test_three_valued_statuses(self):
        import json
        from prussian.engine.fst.validate import run_validate
        for text, expected in self.CASES:
            with self.subTest(text=text):
                result = json.loads(run_validate(text))
                self.assertEqual(result["overall"]["status"], expected,
                                 result)

    def test_unlicensed_case_reason(self):
        import json
        from prussian.engine.fst.validate import run_validate
        result = json.loads(run_validate(
            "As asma stan autōmatikin rekōnstruiwuns be sen grammatikin "
            "perbāndan plattinuns."))
        reasons = result["sentences"][0]["coverage"]["reasons"]
        self.assertIn("unlicensed_case", reasons)


if __name__ == "__main__":
    unittest.main()
