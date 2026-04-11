"""Integration tests for search.py with PGR."""

import unittest
from unittest.mock import patch, MagicMock
from prussian_engine.pgr import extract_pgr_from_entry


class TestSearchPGRIntegration(unittest.TestCase):
    """Test PGR integration in search engine."""

    def test_form_to_pgr_lookup(self):
        """Test that form_to_pgr index is built correctly."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            engine.entries = []
            engine.embeddings = None
            engine.word_to_entry = {}
            engine.form_to_lemma = {}
            engine.form_to_pgr = {}

            entry = {
                "word": "Dēiws",
                "gender": "masc",
                "forms": {
                    "declension": [
                        {
                            "gender": "masc",
                            "cases": [
                                {
                                    "case": "Nominative",
                                    "singular": "Dēiws",
                                    "plural": "Deiwāi",
                                },
                                {
                                    "case": "Genitive",
                                    "singular": "Dēiwas",
                                    "plural": "Dēiwan",
                                },
                                {
                                    "case": "Accusative",
                                    "singular": "Dēiwan",
                                    "plural": "Dēiwans",
                                },
                            ],
                        }
                    ]
                },
            }
            engine.entries.append(entry)
            engine.word_to_entry["dēiws"] = [entry]

            forms_pgr = extract_pgr_from_entry(entry)
            for form, pgr in forms_pgr:
                form_lower = form.lower()
                if form_lower not in engine.form_to_pgr:
                    engine.form_to_pgr[form_lower] = []
                if pgr not in engine.form_to_pgr[form_lower]:
                    engine.form_to_pgr[form_lower].append(pgr)

            self.assertIn("dēiwan", engine.form_to_pgr)
            self.assertEqual(len(engine.form_to_pgr["dēiwan"]), 2)
            self.assertIn("GEN.PL.MASC", engine.form_to_pgr["dēiwan"])
            self.assertIn("ACC.SG.MASC", engine.form_to_pgr["dēiwan"])


class TestGetWordFormsFilter(unittest.TestCase):
    """Test get_word_forms with filter parameter."""

    def test_filter_gen_pl(self):
        """Test filtering for genitive plural."""
        entry = {
            "word": "Dēiws",
            "gender": "masc",
            "translations": {"miks": ["Gott"], "engl": ["god"]},
            "forms": {
                "declension": [
                    {
                        "gender": "masc",
                        "cases": [
                            {
                                "case": "Nominative",
                                "singular": "Dēiws",
                                "plural": "Deiwāi",
                            },
                            {
                                "case": "Genitive",
                                "singular": "Dēiwas",
                                "plural": "Dēiwan",
                            },
                        ],
                    }
                ]
            },
        }

        from prussian_engine.pgr import match_pgr

        forms_pgr = extract_pgr_from_entry(entry)
        filtered = [(f, p) for f, p in forms_pgr if match_pgr(p, "GEN.PL")]

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0], "Dēiwan")
        self.assertEqual(filtered[0][1], "GEN.PL.MASC")


class TestSimplifyPGR(unittest.TestCase):
    """Test _simplify_pgr function."""

    def test_no_change_single_pgr(self):
        """Single PGR should remain unchanged."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            result = engine._simplify_pgr("NOM.SG.MASC")
            self.assertEqual(result, "NOM.SG.MASC")

    def test_no_change_no_common(self):
        """PGRs with no common features shouldn't change."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            result = engine._simplify_pgr("NOM.SG.MASC|GEN.PL.FEM")
            self.assertEqual(result, "NOM.SG.MASC|GEN.PL.FEM")

    def test_simplify_common_features(self):
        """GEN.PL.MASC|GEN.PL.FEM|GEN.PL.NEUT → GEN.PL"""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            result = engine._simplify_pgr("GEN.PL.MASC|GEN.PL.FEM|GEN.PL.NEUT")
            self.assertEqual(result, "GEN.PL")


class TestExtractReferenceTarget(unittest.TestCase):
    """Test _extract_reference_target function."""

    def test_valid_reference(self):
        """Extract target from valid reference."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            result = engine._extract_reference_target("↑ Abbai dat")
            self.assertEqual(result, "Abbai")

    def test_no_reference(self):
        """Return None for non-reference."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            result = engine._extract_reference_target("normal description")
            self.assertIsNone(result)


class TestResolveReference(unittest.TestCase):
    """Test _resolve_reference function."""

    def test_resolve_existing_lemma(self):
        """Resolve reference to existing lemma."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()

            abbai_entry = {
                "word": "abbai",
                "gender": "masc",
                "translations": {"miks": ["beides"], "engl": ["both"]},
                "forms": {
                    "declension": [
                        {
                            "gender": "masc",
                            "cases": [
                                {
                                    "case": "Genitive",
                                    "singular": "abbejan",
                                    "plural": "",
                                },
                            ],
                        }
                    ]
                },
            }
            engine.word_to_entry = {"abbai": [abbai_entry]}
            engine.form_to_pgr = {"abbejan": ["GEN.SG.MASC"]}

            results = engine._resolve_reference("abbai", "abbejan")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["word"], "abbai")
            self.assertEqual(results[0]["matched_form"], "abbejan")
            self.assertEqual(results[0]["pgr"], "GEN.SG.MASC")

    def test_resolve_missing_lemma(self):
        """Return empty list for non-existent lemma."""
        from prussian_engine.search import SearchEngine

        with patch.object(SearchEngine, "__init__", lambda self: None):
            engine = SearchEngine()
            engine.word_to_entry = {}

            results = engine._resolve_reference("nonexistent", "form")
            self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
