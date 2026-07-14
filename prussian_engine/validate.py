"""Grammatik-Validierung — dreiwertiger Prüfer über die prussian-fst-Pipeline.

Wraps ``fst/scripts/cg3_pipeline.py --text - --validate`` (dritter
vislcg3-Pass ``validator.cg3``, Divvun-Muster: ADD-only-Regeln setzen
&-Fehler-Tags).  Pro Satz kommt ein Objekt zurück:

  status      violations_found | verified_in_coverage | out_of_coverage
  violations  [{rule, tag, index, form, severity, reading, message}]
  coverage    {word_tokens, oov, collapsed, ambig, checks_relevant, reasons}

Kernprinzip: „kein Fehler-Tag" heißt NICHT „korrekt".  Nur wenn keine
unbekannten Wörter, kein Lesarten-Kollaps, geringe Restambiguität UND
mindestens eine anwendbare Prüfregel vorliegen, lautet das Urteil
``verified_in_coverage`` — sonst ``out_of_coverage`` (Abstention).
Consumer (LLM-Agenten, Trainingsdaten-Filter) dürfen out_of_coverage
niemals als „korrekt" werten.

severity: ``error`` = Rektion/Valenz/Person (hohe Präzision), ``warning``
= Kongruenz/PP-Nominativ (auf attestiertem Text überwiegend Paradigmen-
Lücken bei Lehnwörtern).

Requires ``vislcg3`` and ``hfst-flookup`` on PATH and a built
``fst/build/base.fst`` in the prussian-fst checkout.
"""

import json
import subprocess
import sys

from .config import PRUSSIAN_FST_DIR

PIPELINE = PRUSSIAN_FST_DIR / "fst" / "scripts" / "cg3_pipeline.py"

# Guard against runaway inputs — the tool is meant for a few sentences.
MAX_TEXT_LEN = 4000

_STATUS_ORDER = ["verified_in_coverage", "out_of_coverage", "violations_found"]


def run_validate(text: str, timeout: float = 60.0) -> dict:
    """Validate Prussian text; return {overall_status, sentences}.

    overall_status ist der schlechteste Satz-Status (violations_found >
    out_of_coverage > verified_in_coverage).  Raises ValueError for
    unusable input and RuntimeError when the pipeline is missing or
    fails — FastMCP turns those into an ``isError`` tool result.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Kein Text übergeben.")
    if len(text) > MAX_TEXT_LEN:
        raise ValueError(
            f"Text zu lang ({len(text)} Zeichen, max. {MAX_TEXT_LEN})."
        )
    if not PIPELINE.exists():
        raise RuntimeError(
            f"prussian-fst-Pipeline nicht gefunden: {PIPELINE} — "
            "PRUSSIAN_FST_DIR setzen oder Checkout danebenlegen."
        )

    proc = subprocess.run(
        [sys.executable, str(PIPELINE), "--text", "-", "--validate"],
        input=text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError(
            "Validator-Pipeline fehlgeschlagen: "
            + (" | ".join(tail) or "kein stderr")
        )
    try:
        sentences = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Validator lieferte kein JSON: {e}") from e
    if not sentences:
        raise RuntimeError("Validator lieferte keine Analyse.")

    overall = max(
        (s.get("status", "out_of_coverage") for s in sentences),
        key=_STATUS_ORDER.index,
    )
    return {"overall_status": overall, "sentences": sentences}
