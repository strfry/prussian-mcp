"""FSG/CG-Grammatikprüfung — In-Process-Anbindung der prussian-fst-Pipeline.

Importiert die stabile API (prussian_fst.api, editierbare uv-Path-
Dependency, siehe pyproject [tool.uv.sources]): FST-Lookup läuft via
pyhfst im Prozess (Transducer-Cache bleibt warm), nur cg-proc ist ein
Subprozess pro Pass.  Kein `uv run`-Umweg mehr.

Ein Payload (das einzige Grammatik-Tool, validate_prussian):
  run_validate(text, include_conllu=False)
    → dreiwertiges Validierungs-JSON (validator.cg3): violations_found /
      verified_in_coverage / out_of_coverage, mit overall-Aggregat.
      Mit include_conllu trägt jeder Satz zusätzlich seinen CoNLL-U-
      Block (Rule=/AgrParent=-Provenienz in MISC) — gleicher
      Pipeline-Lauf, kein zweiter Durchlauf.

  validate_with_corrections(text, include_conllu=False)
    → wie run_validate(), aber korrigiert eindeutige OOV-Wörter über
      Zwei-Stufen-Lookup (ortho → fuzzy) und fügt spelling_corrections
      zum Resultat hinzu.

Voraussetzungen (nicht auto-gebaut): cg-proc im PATH und die Artefakte
im prussian-fst-Checkout — check_fsg_pipeline() nennt die konkreten
make-Kommandos, wenn etwas fehlt.
"""

import json
import subprocess

# Guarded Import: fehlt das Paket (venv nicht gesynct oder Checkout
# nicht am Pfad aus pyproject [tool.uv.sources]), soll der Server
# trotzdem starten — die Wörterbuch-Tools funktionieren ohne FST;
# der Healthcheck und das Tool selbst melden dann die Abhilfe.
try:
    from prussian_fst import api as fst_api
    _IMPORT_ERROR = None
except ImportError as e:
    fst_api = None
    _IMPORT_ERROR = (
        f"prussian_fst nicht importierbar ({e}) — im prussian-mcp-"
        "Checkout `uv sync` ausführen; der prussian-fst-Checkout muss "
        "am Pfad aus pyproject [tool.uv.sources] liegen "
        "(Default: ../prussian-fst)."
    )

# Guard against runaway inputs — the tool is meant for a few sentences.
MAX_TEXT_LEN = 4000

# Pro cg-proc-Pass (der FST-Lookup ist in-process und braucht keinen).
PIPELINE_TIMEOUT = 60.0

# Reihenfolge fürs overall-Aggregat: das „schlechteste" Satz-Urteil zählt.
_STATUS_ORDER = ["verified_in_coverage", "out_of_coverage", "violations_found"]


def _check_text(text: str) -> str:
    """Input-Validierung; ValueError wird von FastMCP als isError gerendert."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Kein Text übergeben.")
    if len(text) > MAX_TEXT_LEN:
        raise ValueError(
            f"Text zu lang ({len(text)} Zeichen, max. {MAX_TEXT_LEN})."
        )
    return text


def _translate_errors(fn, *args, **kwargs):
    """Pipeline-Fehler → RuntimeError mit brauchbarer Diagnose."""
    try:
        return fn(*args, **kwargs)
    except subprocess.TimeoutExpired:
        raise RuntimeError("FSG/CG-Pipeline: cg-proc-Timeout "
                           f"({PIPELINE_TIMEOUT:.0f}s).")
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError("FSG/CG-Pipeline fehlgeschlagen: "
                           + (" | ".join(tail) or "kein stderr"))
    except FileNotFoundError as e:
        raise RuntimeError(f"FSG/CG-Pipeline: {e} — "
                           "cg-proc installiert und Artefakte gebaut? "
                           "(siehe check_fsg_pipeline)")


def run_validate(text: str, include_conllu: bool = False) -> str:
    """Grammatikprüfung, dreiwertig — JSON mit overall-Aggregat.

    overall.status = schlechtestes Satz-Urteil (violations_found >
    out_of_coverage > verified_in_coverage).  Mit include_conllu
    bekommt jeder Satz sein "conllu"-Feld (Dependenzanalyse)."""
    if fst_api is None:
        raise RuntimeError(_IMPORT_ERROR)
    text = _check_text(text)
    sentences = _translate_errors(fst_api.validate, text,
                                  conllu=include_conllu,
                                  timeout=PIPELINE_TIMEOUT)
    overall = max((s["status"] for s in sentences), key=_STATUS_ORDER.index)
    return json.dumps({
        "overall": {
            "status": overall,
            "n_sentences": len(sentences),
            "n_violations": sum(len(s["violations"]) for s in sentences),
        },
        "sentences": sentences,
    }, ensure_ascii=False, indent=2)


# ── OOV-Korrektur (Zwei-Stufen-Lookup) ───────────────────────────────────────

_engine = None


def _get_engine():
    """Lazy-loaded SearchEngine Singleton."""
    global _engine
    if _engine is None:
        from prussian.engine.search import SearchEngine
        _engine = SearchEngine()
    return _engine


def _find_correction(word: str) -> dict | None:
    """Suche eine eindeutige Korrektur für ein OOV-Wort.

    Zwei Stufen:
      1. Ortho-Layer (prefix + ortho-Regeln, inkl. Macron-Normalisierung)
      2. Fuzzy (Levenshtein)

    Rückgabe: {original, corrected, method, certain} oder None.
    Zeigt die Dictionary-Form (nicht Lemma) als Korrektur.
    """
    engine = _get_engine()

    def _pick(candidates):
        c = candidates[0]
        # Bevorzuge inflected form über Lemma
        corrected = c.get("form") or c["word"]
        return {"original": word, "corrected": corrected,
                "method": c.get("method", "ortho"), "certain": True}

    # Stufe 1: ortho layer (kein fuzzy)
    candidates = engine.lookup(word, fuzzy=False)
    if len(candidates) == 1:
        return _pick(candidates)
    if len(candidates) > 1:
        return None  # mehrdeutig

    # Stufe 2: fuzzy (Levenshtein)
    candidates = engine.lookup(word, fuzzy=True)
    if len(candidates) == 1:
        c = candidates[0]
        corrected = c.get("form") or c["word"]
        return {"original": word, "corrected": corrected,
                "method": "fuzzy", "certain": False}
    return None  # mehrdeutig oder kein Treffer


def validate_with_corrections(text: str, include_conllu: bool = False) -> str:
    """Grammatikprüfung mit automatischer OOV-Korrektur.

    Wie ``run_validate()``, aber korrigiert eindeutige OOV-Wörter über
    den Zwei-Stufen-Lookup (ortho → fuzzy) und validiert den
    korrigierten Satz erneut.  Korrekturen erscheinen als
    ``spelling_corrections`` im JSON-Output.
    """
    raw = run_validate(text, include_conllu=include_conllu)
    result = json.loads(raw)

    # OOV-Wörter aus allen Sätzen sammeln
    oov_words = []
    for s in result.get("sentences", []):
        for oov in s.get("coverage", {}).get("oov", []):
            form = oov.get("form", "")
            if form and form not in oov_words:
                oov_words.append(form)

    if not oov_words:
        return raw

    # Korrekturen suchen
    corrections = []
    corrected_text = text
    for word in oov_words:
        c = _find_correction(word)
        if c:
            corrections.append(c)
            corrected_text = corrected_text.replace(word, c["corrected"])

    if not corrections:
        return raw

    # Korrigierten Satz neu validieren
    raw2 = run_validate(corrected_text, include_conllu=include_conllu)
    result2 = json.loads(raw2)
    result2["spelling_corrections"] = corrections
    return json.dumps(result2, ensure_ascii=False, indent=2)


def check_fsg_pipeline() -> tuple[bool, str]:
    """Startup-Healthcheck: Artefakte prüfen + In-Process-Smoke-Test.

    Returns ``(ok, message)`` — never raises.  Wärmt bei Erfolg zugleich
    den pyhfst-Transducer-Cache vor."""
    if fst_api is None:
        return False, f"FST/CG3-Pipeline nicht bereit: {_IMPORT_ERROR}"
    problems = fst_api.check_artifacts()
    if problems:
        return False, ("FST/CG3-Pipeline nicht bereit:\n  "
                       + "\n  ".join(problems))
    try:
        fst_api.analyze("Sta", timeout=15.0)
    except Exception as e:  # noqa: BLE001 — Healthcheck darf nie raisen
        return False, f"FST/CG3-Healthcheck fehlgeschlagen: {e}"
    return True, "FST/CG3-Pipeline bereit (in-process)."
