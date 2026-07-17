"""Tests for prussian/pgr.py - Prussian Glossing Rules."""

import unittest
from prussian.engine.morphology import (
    parse_pgr,
    build_pgr,
    match_pgr,
    parse_reference_desc,
    extract_pgr_from_entry,
    normalize_feature,
    CASE_MAP,
    GENDER_MAP,
    NUMBER_MAP,
)


class TestParsePGR(unittest.TestCase):
    """Tests for parse_pgr function."""

    def test_simple_nom_sg_masc(self):
        """Parse simple nominal form."""
        result = parse_pgr("NOM.SG.MASC")
        self.assertEqual(result, {"CASE": "NOM", "NUMBER": "SG", "GENDER": "MASC"})

    def test_genitive_plural(self):
        """Parse genitive plural."""
        result = parse_pgr("GEN.PL")
        self.assertEqual(result, {"CASE": "GEN", "NUMBER": "PL"})

    def test_case_only(self):
        """Parse case only."""
        result = parse_pgr("ACC")
        self.assertEqual(result, {"CASE": "ACC"})

    def test_verb_pres_1_sg(self):
        """Parse verb present 1st singular indicative."""
        result = parse_pgr("PRES.1.SG.IND")
        self.assertEqual(
            result, {"TENSE": "PRS", "PERSON": "1", "NUMBER": "SG", "MOOD": "IND"}
        )

    def test_participle(self):
        """Parse participle."""
        result = parse_pgr("PC.PT.ACT.MASC.NOM.SG")
        self.assertEqual(
            result,
            {
                "TYPE": "PC",
                "PC_TYPE": "PT",
                "VOICE": "ACT",
                "CASE": "NOM",
                "NUMBER": "SG",
                "GENDER": "MASC",
            },
        )

    def test_verb_pres_1_sg(self):
        """Parse verb present 1st singular indicative."""
        result = parse_pgr("PRES.1.SG.IND")
        self.assertEqual(
            result, {"TENSE": "PRS", "PERSON": "1", "NUMBER": "SG", "MOOD": "IND"}
        )

    def test_lowercase_input(self):
        """Parse lowercase input."""
        result = parse_pgr("nom.sg.fem")
        self.assertEqual(result, {"CASE": "NOM", "NUMBER": "SG", "GENDER": "FEM"})

    def test_empty_string(self):
        """Handle empty string."""
        result = parse_pgr("")
        self.assertEqual(result, {})

    def test_none_input(self):
        """Handle None input."""
        result = parse_pgr(None)
        self.assertEqual(result, {})


class TestBuildPGR(unittest.TestCase):
    """Tests for build_pgr function."""

    def test_nom_sg_masc(self):
        """Build nominal form."""
        features = {"CASE": "NOM", "NUMBER": "SG", "GENDER": "MASC"}
        result = build_pgr(features)
        self.assertEqual(result, "NOM.SG.MASC")

    def test_gen_pl(self):
        """Build genitive plural."""
        features = {"CASE": "GEN", "NUMBER": "PL"}
        result = build_pgr(features)
        self.assertEqual(result, "GEN.PL")

    def test_verb_form(self):
        """Build verb form."""
        features = {"TENSE": "PRS", "PERSON": "1", "NUMBER": "SG", "MOOD": "IND"}
        result = build_pgr(features)
        self.assertEqual(result, "PRS.SG.1.IND")

    def test_partial_features(self):
        """Build with partial features."""
        features = {"CASE": "ACC"}
        result = build_pgr(features)
        self.assertEqual(result, "ACC")

    def test_empty_features(self):
        """Handle empty features."""
        result = build_pgr({})
        self.assertEqual(result, "")


class TestMatchPGR(unittest.TestCase):
    """Tests for match_pgr function."""

    def test_exact_match(self):
        """Exact match should return True."""
        self.assertTrue(match_pgr("NOM.SG.MASC", "NOM.SG.MASC"))

    def test_partial_filter(self):
        """Partial filter should match full form."""
        self.assertTrue(match_pgr("NOM.SG.MASC", "NOM"))
        self.assertTrue(match_pgr("NOM.SG.MASC", "SG"))
        self.assertTrue(match_pgr("NOM.SG.MASC", "MASC"))
        self.assertTrue(match_pgr("NOM.SG.MASC", "NOM.SG"))

    def test_no_match(self):
        """Non-matching filter should return False."""
        self.assertFalse(match_pgr("NOM.SG.MASC", "GEN"))
        self.assertFalse(match_pgr("NOM.SG.MASC", "PL"))
        self.assertFalse(match_pgr("NOM.SG.MASC", "FEM"))

    def test_ambiguous_form(self):
        """Ambiguous forms should match if any variant matches."""
        self.assertTrue(match_pgr("ACC.SG|GEN.PL", "ACC"))
        self.assertTrue(match_pgr("ACC.SG|GEN.PL", "GEN"))
        self.assertTrue(match_pgr("ACC.SG|GEN.PL", "GEN.PL"))
        self.assertFalse(match_pgr("ACC.SG|GEN.PL", "NOM"))

    def test_empty_filter(self):
        """Empty filter should always match."""
        self.assertTrue(match_pgr("NOM.SG.MASC", ""))
        self.assertTrue(match_pgr("ACC.SG|GEN.PL", None))


class TestParseReferenceDesc(unittest.TestCase):
    """Tests for parse_reference_desc function."""

    def test_simple_case_reference(self):
        """Parse simple case reference."""
        result = parse_reference_desc("↑ Dēiws acc")
        self.assertIsNotNone(result)
        target, features = result
        self.assertEqual(target, "dēiws")
        self.assertEqual(features.get("CASE"), "ACC")
        self.assertEqual(features.get("NUMBER"), "SG")

    def test_full_nominal_reference(self):
        """Parse full nominal reference with gender."""
        result = parse_reference_desc("↑ Madla nom sg m")
        self.assertIsNotNone(result)
        target, features = result
        self.assertEqual(target, "madla")
        self.assertEqual(features.get("CASE"), "NOM")
        self.assertEqual(features.get("NUMBER"), "SG")
        self.assertEqual(features.get("GENDER"), "MASC")

    def test_participle_reference(self):
        """Parse participle reference."""
        result = parse_reference_desc("↑ Palaipīntwei pc pt ac nom sg m")
        self.assertIsNotNone(result)
        target, features = result
        self.assertEqual(target, "palaipīntwei")
        self.assertEqual(features.get("TYPE"), "PC")
        self.assertEqual(features.get("CASE"), "NOM")
        self.assertEqual(features.get("NUMBER"), "SG")
        self.assertEqual(features.get("GENDER"), "MASC")

    def test_infinitive_reference(self):
        """Parse infinitive reference (just target) - returns None as no features."""
        result = parse_reference_desc("↑ Bilītun")
        self.assertIsNone(result)

    def test_not_reference(self):
        """Non-reference strings should return None."""
        self.assertIsNone(parse_reference_desc("v [verb description]"))
        self.assertIsNone(parse_reference_desc("n noun"))

    def test_dat_reference(self):
        """Parse dative reference."""
        result = parse_reference_desc("↑ Abbai dat")
        self.assertIsNotNone(result)
        target, features = result
        self.assertEqual(target, "abbai")
        self.assertEqual(features.get("CASE"), "DAT")


class TestExtractPGRFromEntry(unittest.TestCase):
    """Tests for extract_pgr_from_entry function."""

    def test_noun_declension(self):
        """Extract forms from noun declension."""
        entry = {
            "word": "deiws",
            "gender": "masc",
            "forms": {
                "declension": [
                    {
                        "gender": "masc",
                        "cases": [
                            {
                                "case": "Nominative",
                                "singular": "deiws",
                                "plural": "dīwi",
                            },
                            {
                                "case": "Genitive",
                                "singular": "dīwisa",
                                "plural": "dīwi",
                            },
                            {
                                "case": "Accusative",
                                "singular": "dīwin",
                                "plural": "dīwins",
                            },
                        ],
                    }
                ]
            },
        }

        results = extract_pgr_from_entry(entry)
        forms = {form: pgr for form, pgr in results}

        self.assertIn("deiws", forms)
        self.assertIn("dīwisa", forms)
        self.assertIn("dīwin", forms)
        self.assertIn("dīwi", forms)
        self.assertIn("dīwins", forms)

        self.assertEqual(forms["deiws"], "NOM.SG.MASC")
        self.assertEqual(forms["dīwisa"], "GEN.SG.MASC")
        self.assertEqual(forms["dīwin"], "ACC.SG.MASC")

    def test_verb_forms(self):
        """Extract forms from verb conjugation."""
        entry = {
            "word": "bītun",
            "forms": {
                "indicative": [
                    {
                        "tense": "Present",
                        "forms": [
                            {"pronoun": "as", "form": "būi"},
                            {"pronoun": "tū", "form": "būi"},
                            {"pronoun": "mes", "form": "būimai"},
                            {"pronoun": "jūs", "form": "būitei"},
                        ],
                    },
                    {
                        "tense": "Past",
                        "forms": [
                            {"pronoun": "as", "form": "būi"},
                            {"pronoun": "tū", "form": "būi"},
                        ],
                    },
                ]
            },
        }

        results = extract_pgr_from_entry(entry)
        self.assertTrue(len(results) > 0)

        forms_dict = dict(results)
        self.assertIn("būi", forms_dict)
        self.assertIn("būimai", forms_dict)

        for form, pgr in results:
            self.assertTrue(pgr.startswith("PRS") or pgr.startswith("PST"))

    def test_entry_without_forms(self):
        """Handle entry without forms."""
        entry = {"word": "test", "gender": "masc"}
        results = extract_pgr_from_entry(entry)
        self.assertEqual(results, [])


class TestNormalizeFeature(unittest.TestCase):
    """Tests for normalize_feature function."""

    def test_case_normalization(self):
        """Normalize case values."""
        self.assertEqual(normalize_feature("nom", CASE_MAP), "NOM")
        self.assertEqual(normalize_feature("NOM", CASE_MAP), "NOM")
        self.assertEqual(normalize_feature("acc", CASE_MAP), "ACC")

    def test_gender_normalization(self):
        """Normalize gender values."""
        self.assertEqual(normalize_feature("m", GENDER_MAP), "MASC")
        self.assertEqual(normalize_feature("masc", GENDER_MAP), "MASC")
        self.assertEqual(normalize_feature("f", GENDER_MAP), "FEM")

    def test_number_normalization(self):
        """Normalize number values."""
        self.assertEqual(normalize_feature("sg", NUMBER_MAP), "SG")
        self.assertEqual(normalize_feature("pl", NUMBER_MAP), "PL")
        self.assertEqual(normalize_feature("du", NUMBER_MAP), "DU")

    def test_unknown_value(self):
        """Handle unknown values."""
        self.assertIsNone(normalize_feature("xyz", CASE_MAP))
        self.assertIsNone(normalize_feature("", CASE_MAP))
        self.assertIsNone(normalize_feature(None, CASE_MAP))


if __name__ == "__main__":
    unittest.main()
