"""FSG/CG check — run the prussian-fst FST→CG3 pipeline on Prussian text.

Wraps ``fst/scripts/cg3_pipeline.py --text - --conllu --trace`` from the
prussian-fst checkout (config.PRUSSIAN_FST_DIR / $PRUSSIAN_FST_DIR):
FST-Lookup → CG3-Disambiguierung → Dependenzbaum → CoNLL-U, ein Block
pro Satz, mit Regel-Provenienz in MISC (``Rule=<name,…>`` — benannte
Grammatikregeln laut vislcg3 --trace — und ``AgrParent=<id>``, das
Kongruenz-Ziel der agr-head-Regeln).  Das ist die Tool-Signatur, die
das Chat-Frontend als Abhängigkeitsbaum rendert (isConllu-Erkennung).

Requires ``vislcg3`` and ``hfst-flookup`` on PATH and a built
``fst/build/base.fst`` in the prussian-fst checkout.
"""

import subprocess

from .config import PRUSSIAN_FST_DIR

PIPELINE = PRUSSIAN_FST_DIR / "fst" / "scripts" / "cg3_pipeline.py"

# Guard against runaway inputs — the tool is meant for a few sentences.
MAX_TEXT_LEN = 4000


def _pipeline_cmd(*extra_args: str) -> list[str]:
    """Build argv that runs the pipeline inside prussian-fst's venv."""
    return [
        "uv", "run", "--directory", str(PRUSSIAN_FST_DIR),
        str(PIPELINE),
        *extra_args,
    ]


def run_fsg_check(text: str, timeout: float = 60.0) -> str:
    """Parse Prussian text, return CoNLL-U (one block per sentence).

    Raises ValueError for unusable input and RuntimeError when the
    pipeline is missing or fails — FastMCP turns those into an
    ``isError`` tool result whose text is the message.
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
        _pipeline_cmd("--text", "-", "--conllu", "--trace"),
        input=text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError(
            "FSG/CG-Pipeline fehlgeschlagen: " + (" | ".join(tail) or "kein stderr")
        )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("FSG/CG-Pipeline lieferte keine Analyse.")
    return out + "\n"
