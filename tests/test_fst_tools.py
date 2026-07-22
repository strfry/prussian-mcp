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
        self.assertEqual(result, ["Part"])

    def test_multiple_human_readable(self):
        result = resolve_features("indicative,present")
        self.assertEqual(result, ["Ind", "Pres"])

    def test_raw_fst_tag(self):
        result = resolve_features("Ind+Pres")
        self.assertEqual(result, ["Ind", "Pres"])

    def test_mixed(self):
        result = resolve_features("participle,Gen+Pl")
        self.assertEqual(result, ["Part", "Gen", "Pl"])

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
        self.assertEqual(result, ["Part"])

    def test_conjunctive_maps_to_subj(self):
        result = resolve_features("conjunctive")
        self.assertEqual(result, ["Subj"])

    def test_past_maps_to_pret(self):
        result = resolve_features("past")
        self.assertEqual(result, ["Pret"])


class TestWordformsTool(unittest.TestCase):
    """Tests for wordforms_tool — uses mocked engine, no FST for unit tests."""

    def _make_engine(self):
        from unittest.mock import MagicMock
        engine = MagicMock()
        engine.word_to_entry = {
            "berzi": [{
                "word": "berzi",
                "gender": "fem",
                "translations": {"de": ["Birke"]},
                "desc": "n [berzi]",
                "forms": {
                    "declension": [{
                        "gender": "fem",
                        "cases": [
                            {"case": "Nominative", "singular": "berzi",
                             "plural": "berzans"},
                            {"case": "Genitive", "singular": "berzes",
                             "plural": "berzu"},
                            {"case": "Dative", "singular": "berzei",
                             "plural": "berzam"},
                            {"case": "Accusative", "singular": "berzin",
                             "plural": "berzans"},
                        ],
                    }],
                },
            }],
            "wīdātun": [{
                "word": "wīdātun",
                "gender": "",
                "translations": {"de": ["sehen"]},
                "desc": "v [widdai 113]",
                "forms": {
                    "indicative": [{
                        "tense": "Present",
                        "forms": [
                            {"pronoun": "as", "form": "wīda"},
                            {"pronoun": "tū", "form": "wīdi"},
                            {"pronoun": "mes", "form": "wīdamai"},
                            {"pronoun": "jūs", "form": "wīdatei"},
                        ],
                    }],
                },
            }],
        }
        return engine

    def test_not_found(self):
        from prussian.tools.wordforms import wordforms_tool
        engine = self._make_engine()
        result = wordforms_tool(engine, "nonexistent")
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_unknown_feature(self):
        from prussian.tools.wordforms import wordforms_tool
        engine = self._make_engine()
        result = wordforms_tool(engine, "berzi", features="foobar")
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("valid_features", result)

    def test_output_format(self):
        from prussian.tools.wordforms import wordforms_tool
        engine = self._make_engine()
        result = wordforms_tool(engine, "berzi")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        entry = result[0]
        self.assertEqual(entry["lemma"], "berzi")
        self.assertIn("forms", entry)
        self.assertIn("available_features", entry)
        self.assertNotIn("translations", entry)
        for form in entry["forms"]:
            self.assertIn("form", form)
            self.assertIn("tags", form)


@unittest.skipIf(not FST_READY, "prussian-fst-Artefakte nicht gebaut")
class TestWordformsFST(unittest.TestCase):
    """FST-dependent wordforms tests."""

    def _make_engine(self):
        from prussian.engine.search import SearchEngine
        engine = SearchEngine()
        return engine

    def test_verb_default_ind_pres(self):
        from prussian.tools.wordforms import wordforms_tool
        engine = self._make_engine()
        result = wordforms_tool(engine, "wīdātun")
        self.assertIsInstance(result, list)
        entry = result[0]
        # Default for verbs should only include Ind+Pres forms
        for form in entry.get("forms", []):
            tags = form.get("tags", [])
            if tags:
                has_ind_pres = any(
                    "ind" in t.lower() and "pres" in t.lower()
                    for t in tags
                )
                self.assertTrue(has_ind_pres,
                                f"Expected Ind+Pres, got {tags}")

    def test_participle_feature(self):
        from prussian.tools.wordforms import wordforms_tool
        engine = self._make_engine()
        result = wordforms_tool(engine, "wīdātun", features="participle")
        self.assertIsInstance(result, list)
        entry = result[0]
        for form in entry.get("forms", []):
            tags = form.get("tags", [])
            if tags:
                has_part = any("part" in t.lower() for t in tags)
                self.assertTrue(has_part,
                                f"Expected Part, got {tags}")

    def test_noun_full_list(self):
        from prussian.tools.wordforms import wordforms_tool
        engine = self._make_engine()
        result = wordforms_tool(engine, "berzi")
        self.assertIsInstance(result, list)
        entry = result[0]
        # Non-verb should have full list
        self.assertTrue(len(entry.get("forms", [])) > 4)


@unittest.skipIf(not FST_READY, "prussian-fst-Artefakte nicht gebaut")
class TestLookupFST(unittest.TestCase):
    """FST-dependent lookup tests."""

    def _make_engine(self):
        from prussian.engine.search import SearchEngine
        return SearchEngine()

    def test_sentence_lookup(self):
        from prussian.tools.lookup import lookup_tool
        engine = self._make_engine()
        results = lookup_tool(engine, "As wīda gaīlan berzin.")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        # Each result should have form and method
        for r in results:
            self.assertIn("form", r)
            self.assertIn("method", r)

    def test_fst_method_analyses(self):
        from prussian.tools.lookup import lookup_tool
        engine = self._make_engine()
        results = lookup_tool(engine, "As")
        self.assertTrue(len(results) > 0)
        r = results[0]
        self.assertEqual(r["method"], "fst")
        self.assertIn("analyses", r)
        for a in r["analyses"]:
            self.assertIn("lemma", a)
            self.assertIn("tags", a)

    def test_oov_fallback(self):
        from prussian.tools.lookup import lookup_tool
        engine = self._make_engine()
        results = lookup_tool(engine, "Ein Xyzzymorg")
        # Xyzzymorg should trigger dictionary_fallback
        fallback = [r for r in results if r["method"] == "dictionary_fallback"]
        self.assertTrue(len(fallback) > 0)

    def test_adjustment_field(self):
        from prussian.tools.lookup import lookup_tool
        engine = self._make_engine()
        # Titlecase: "dēiwan" is lowercase, FST knows "Dēiwan"
        results = lookup_tool(engine, "dēiwan")
        # Check if any token has adjustment
        adjusted = [r for r in results if r.get("adjustment")]
        # dēiwan might match via titlecase or lowercase
        if adjusted:
            adj = adjusted[0]["adjustment"]
            self.assertIn("via", adj)


@unittest.skipIf(not FST_READY, "prussian-fst-Artefakte nicht gebaut")
class TestSearchFST(unittest.TestCase):
    """FST-dependent search tests."""

    def _make_engine(self):
        from prussian.engine.search import SearchEngine
        return SearchEngine()

    def test_search_with_filter_tags(self):
        from prussian.tools.search import search_tool
        engine = self._make_engine()
        results = search_tool(engine, "Birke", top_k=5,
                              filter_tags="Akk+Sg")
        self.assertIsInstance(results, list)
        # Should find berzi with accusative singular forms
        for r in results:
            if r.get("forms"):
                for f in r["forms"]:
                    tags = f.get("tags", "")
                    self.assertTrue(
                        "akk" in tags.lower() or "sg" in tags.lower(),
                        f"Expected Akk+Sg in tags, got {tags}"
                    )

    def test_search_without_filter(self):
        from prussian.tools.search import search_tool
        engine = self._make_engine()
        results = search_tool(engine, "Birke", top_k=3)
        self.assertTrue(len(results) > 0)
        # Without filter, no forms field
        for r in results:
            self.assertNotIn("forms", r)


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
