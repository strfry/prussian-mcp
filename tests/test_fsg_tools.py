"""Tests for prussian/fsg_check.py — in-process grammar tool.

Skips when the prussian-fst artifacts are not built (check_artifacts
non-empty) — the input-validation tests run regardless.
"""

import json
import unittest

from prussian.engine.fst.validate import (MAX_TEXT_LEN, check_fsg_pipeline,
                                       fst_api, run_validate)

ARTIFACTS_MISSING = fst_api is None or bool(fst_api.check_artifacts())


class TestInputValidation(unittest.TestCase):
    """ValueError paths — no pipeline needed."""

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            run_validate("")
        with self.assertRaises(ValueError):
            run_validate("   ")

    def test_text_too_long(self):
        with self.assertRaises(ValueError):
            run_validate("a" * (MAX_TEXT_LEN + 1))


@unittest.skipIf(ARTIFACTS_MISSING, "prussian-fst-Artefakte nicht gebaut")
class TestValidatePrussian(unittest.TestCase):

    def test_violation_and_overall(self):
        r = json.loads(run_validate("As pūwa sen laīwu. Labban dēinan!"))
        self.assertEqual(r["overall"]["status"], "violations_found")
        self.assertEqual(r["overall"]["n_sentences"], 2)
        self.assertGreaterEqual(r["overall"]["n_violations"], 1)
        statuses = [s["status"] for s in r["sentences"]]
        self.assertIn("violations_found", statuses)
        v = r["sentences"][0]["violations"][0]
        self.assertEqual(v["rule"], "prep-akk-dat")
        self.assertEqual(v["severity"], "error")

    def test_out_of_coverage_is_not_verified(self):
        # Litauischer Satz: alles OOV → out_of_coverage, nie verified
        r = json.loads(run_validate("Vakar buvau namie."))
        self.assertEqual(r["overall"]["status"], "out_of_coverage")
        self.assertIn("oov", r["sentences"][0]["coverage"]["reasons"])

    def test_coverage_block_present(self):
        r = json.loads(run_validate("Labban dēinan!"))
        cov = r["sentences"][0]["coverage"]
        for key in ("word_tokens", "oov", "collapsed", "ambig",
                    "checks_relevant", "reasons"):
            self.assertIn(key, cov)

    def test_no_conllu_by_default(self):
        r = json.loads(run_validate("Labban dēinan!"))
        self.assertNotIn("conllu", r["sentences"][0])

    def test_include_conllu(self):
        r = json.loads(run_validate("Labban dēinan!", include_conllu=True))
        block = r["sentences"][0]["conllu"]
        self.assertTrue(block.startswith("# sent_id = "))
        token_lines = [l for l in block.splitlines() if l and l[0].isdigit()]
        self.assertTrue(token_lines)
        for l in token_lines:
            self.assertEqual(len(l.split("\t")), 10)
        self.assertIn("Rule=", block)


class TestHealthCheck(unittest.TestCase):

    def test_never_raises(self):
        ok, msg = check_fsg_pipeline()
        self.assertIsInstance(ok, bool)
        self.assertTrue(msg)


if __name__ == "__main__":
    unittest.main()
