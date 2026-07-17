"""validate_prussian — FST + CG3 grammar check (shared core + CLI)."""

from __future__ import annotations

import json
import sys
import traceback


def validate_tool(text: str, include_conllu: bool = False) -> str:
    """Grammar check of Prussian text (FST + CG3 pipeline, three-valued).

    The single grammar tool: validates sentences and can also return the
    full dependency analysis.  Automatically corrects unambiguous OOV
    words via two-stage lookup (ortho → fuzzy) and re-validates.

    Args:
        text: Prussian sentence(s) to check (one or more sentences).
        include_conllu: also include each sentence's CoNLL-U dependency
            analysis (field ``conllu``, with rule provenance ``Rule=/``
            / ``AgrParent=`` in MISC) — same pipeline run, no extra cost.

    Returns:
        JSON string: ``{"overall": {"status", "n_sentences",
        "n_violations"}, "sentences": [{sent_id, text, status,
        violations, coverage, conllu?}, ...],
        "spelling_corrections": [...]}``
    """
    from prussian.engine.fst.validate import validate_with_corrections

    return validate_with_corrections(text, include_conllu=include_conllu)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="validate",
        description="Grammar check of Prussian text (FST + CG3 pipeline).",
    )
    ap.add_argument("text", help="Prussian sentence(s) to check.")
    ap.add_argument("--json", action="store_true",
                    help="emit raw JSON instead of human-readable output.")
    ap.add_argument("--include-conllu", action="store_true",
                    help="include CoNLL-U dependency analysis per sentence.")
    ap.add_argument("--verbose", action="store_true",
                    help="print full traceback on errors.")
    args = ap.parse_args(argv)

    try:
        raw = validate_tool(args.text, include_conllu=args.include_conllu)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None

    if args.json:
        sys.stdout.write(raw if isinstance(raw, str) else json.dumps(raw))
        sys.stdout.write("\n")
    else:
        if parsed and "overall" in parsed:
            overall = parsed["overall"]
            corrections = parsed.get("spelling_corrections", [])
            suffix = " (with corrections)" if corrections else ""
            print(f"overall: {overall.get('status')}{suffix} "
                  f"({overall.get('n_sentences')} sentences, "
                  f"{overall.get('n_violations')} violations)")
            for s in parsed.get("sentences", []):
                status = s.get("status", "")
                text = s.get("text", "")
                violations = s.get("violations", [])
                line = f"  [{s.get('sent_id')}] {status}: {text}"
                if violations:
                    for v in violations:
                        rule = v.get("rule", "")
                        msg = v.get("message", "")
                        sev = v.get("severity", "")
                        line += f"\n         {sev}: {rule} — {msg}"
                print(line)
            for c in corrections:
                certain = "" if c.get("certain", True) else ", unsicher"
                corrected = c["corrected"]
                method = c.get("method", "?")
                print(f"    spelling: {c['original']} → {corrected}"
                      f" ({method}{certain})")
        else:
            print(raw)

    status = (parsed or {}).get("overall", {}).get("status", "")
    return {"verified_in_coverage": 0, "out_of_coverage": 2,
            "violations_found": 3}.get(status, 1 if parsed is None else 0)
