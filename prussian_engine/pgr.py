"""Prussian Glossing Rules (PGR) - Parser and utilities for morphological feature notation."""

import re
from typing import Dict, List, Optional, Tuple

CASE_MAP = {
    "nom": "NOM",
    "nominative": "NOM",
    "gen": "GEN",
    "genitive": "GEN",
    "dat": "DAT",
    "dative": "DAT",
    "acc": "ACC",
    "accusative": "ACC",
    "voc": "VOC",
    "vocative": "VOC",
    "instr": "INSTR",
    "instrumental": "INSTR",
    "prp": "PRP",
    "prepositional": "PRP",
}

GENDER_MAP = {
    "m": "MASC",
    "f": "FEM",
    "n": "NEUT",
    "masc": "MASC",
    "fem": "FEM",
    "neut": "NEUT",
}

NUMBER_MAP = {
    "sg": "SG",
    "pl": "PL",
    "du": "DU",
}

TENSE_MAP = {
    "present": "PRS",
    "pres": "PRS",
    "prs": "PRS",
    "past": "PST",
    "pst": "PST",
    "perfect": "PFT",
    "pft": "PFT",
    "future": "FUT",
    "fut": "FUT",
    "pluperfect": "PLPFT",
    "plpf": "PLPFT",
    "pres": "PRS",
}

MOOD_MAP = {
    "ind": "IND",
    "indicative": "IND",
    "opt": "OPT",
    "optative": "OPT",
    "imp": "IMP",
    "imperative": "IMP",
    "cond": "COND",
    "subj": "SUBJ",
    "subjunctive": "SUBJ",
}

PERSON_MAP = {
    "1": "1",
    "2": "2",
    "3": "3",
}

DEGREE_MAP = {
    "pos": "POS",
    "positive": "POS",
    "comp": "COMP",
    "comparative": "COMP",
    "sup": "SUP",
    "superlative": "SUP",
}

PC_TYPE_MAP = {
    "ps": "PS",
    "present": "PS",
    "pt": "PT",
    "past": "PT",
}

VOICE_MAP = {
    "act": "ACT",
    "active": "ACT",
    "pass": "PASS",
    "passive": "PASS",
}

FEATURE_ORDER = [
    "TYPE",
    "PC_TYPE",
    "VOICE",
    "TENSE",
    "CASE",
    "NUMBER",
    "GENDER",
    "PERSON",
    "MOOD",
    "DEGREE",
]


def parse_pgr(pgr_string: str) -> Dict[str, str]:
    """Parse a PGR string into a feature dictionary.

    Args:
        pgr_string: PGR notation like "NOM.SG.MASC" or "PRES.1.SG.IND" or "PC.PT.ACT.MASC.NOM.SG"

    Returns:
        Dictionary with feature names as keys, e.g. {"CASE": "NOM", "NUMBER": "SG", "GENDER": "MASC"}
    """
    if not pgr_string:
        return {}

    features: Dict[str, str] = {}
    parts = pgr_string.upper().split(".")

    for part in parts:
        part_lower = part.lower()
        if part in CASE_MAP.values() or part_lower in CASE_MAP:
            features["CASE"] = CASE_MAP.get(part_lower, part)
        elif part in NUMBER_MAP.values() or part_lower in NUMBER_MAP:
            features["NUMBER"] = NUMBER_MAP.get(part_lower, part)
        elif part in GENDER_MAP.values() or part_lower in GENDER_MAP:
            features["GENDER"] = GENDER_MAP.get(part_lower, part)
        elif part in TENSE_MAP.values() or part_lower in TENSE_MAP:
            features["TENSE"] = TENSE_MAP.get(part_lower, part)
        elif part in MOOD_MAP.values() or part_lower in MOOD_MAP:
            features["MOOD"] = MOOD_MAP.get(part_lower, part)
        elif part in PERSON_MAP:
            features["PERSON"] = part
        elif part in DEGREE_MAP.values() or part_lower in DEGREE_MAP:
            features["DEGREE"] = DEGREE_MAP.get(part_lower, part)
        elif part in PC_TYPE_MAP.values() or part_lower in PC_TYPE_MAP:
            features["TYPE"] = "PC"
            features["PC_TYPE"] = PC_TYPE_MAP.get(part_lower, part)
        elif part == "PC":
            features["TYPE"] = "PC"
        elif part in VOICE_MAP.values() or part_lower in VOICE_MAP:
            features["VOICE"] = VOICE_MAP.get(part_lower, part)

    return features


def build_pgr(features: Dict[str, str]) -> str:
    """Build a PGR string from a feature dictionary.

    Args:
        features: Dictionary like {"CASE": "NOM", "NUMBER": "SG", "GENDER": "MASC"}

    Returns:
        PGR string like "NOM.SG.MASC"
    """
    parts = []
    for key in FEATURE_ORDER:
        if key in features and features[key]:
            parts.append(features[key])
    return ".".join(parts)


def match_pgr(form_pgr: str, filter_pgr: str) -> bool:
    """Check if a form's PGR matches a filter PGR (partial match).

    Args:
        form_pgr: The form's full PGR, e.g. "ACC.SG.MASC" or "ACC.SG|GEN.PL"
        filter_pgr: The filter PGR, e.g. "GEN.PL" or "ACC"

    Returns:
        True if the form matches the filter (all filter features are present in form)
    """
    if not filter_pgr:
        return True

    filter_features = parse_pgr(filter_pgr)
    if not filter_features:
        return True

    form_pgrs = form_pgr.split("|")
    for fp in form_pgrs:
        form_features = parse_pgr(fp)
        if all(form_features.get(k) == v for k, v in filter_features.items()):
            return True

    return False


def normalize_feature(value: str, mapping: Dict[str, str]) -> Optional[str]:
    """Normalize a feature value using a mapping (case-insensitive)."""
    if not value:
        return None
    value_lower = value.lower()
    return mapping.get(value_lower)


def parse_reference_desc(desc: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """Parse a reference description from the dictionary 'desc' field.

    Examples:
        "↑ Dēiws acc" → ("dēiws", {"CASE": "ACC", "NUMBER": "SG"})
        "↑ Bilītun ip 2 sg m" → ("bilītun", {"TYPE": "PC", "TENSE": "PRS", "PERSON": "2", "NUMBER": "SG", "GENDER": "MASC"})
        "↑ Madla nom sg m" → ("madla", {"CASE": "NOM", "NUMBER": "SG", "GENDER": "MASC"})

    Args:
        desc: The description field from a dictionary entry, starting with ↑

    Returns:
        Tuple of (target_lemma, features_dict) or None if parsing fails
    """
    if not desc or not desc.startswith("↑"):
        return None

    desc = desc[1:].strip()
    tokens = desc.split()

    if not tokens:
        return None

    target = tokens[0].lower()
    remaining = tokens[1:]

    features: Dict[str, str] = {}

    i = 0
    while i < len(remaining):
        token = remaining[i].lower()

        if token in CASE_MAP:
            features["CASE"] = CASE_MAP[token]
            i += 1
            if i < len(remaining):
                next_tok = remaining[i].lower()
                if next_tok in NUMBER_MAP:
                    features["NUMBER"] = NUMBER_MAP[next_tok]
                    i += 1
                    if i < len(remaining):
                        next_tok = remaining[i].lower()
                        if next_tok in GENDER_MAP:
                            features["GENDER"] = GENDER_MAP[next_tok]
                            i += 1
                elif next_tok in GENDER_MAP:
                    features["GENDER"] = GENDER_MAP[next_tok]
                    i += 1
            continue

        elif token in NUMBER_MAP:
            features["NUMBER"] = NUMBER_MAP[token]
            i += 1
            if i < len(remaining):
                next_tok = remaining[i].lower()
                if next_tok in GENDER_MAP:
                    features["GENDER"] = GENDER_MAP[next_tok]
                    i += 1
            continue

        elif token in GENDER_MAP:
            features["GENDER"] = GENDER_MAP[token]
            i += 1
            continue

        elif token in PC_TYPE_MAP:
            features["TYPE"] = "PC"
            features["SUBTYPE"] = PC_TYPE_MAP[token]
            i += 1
            continue

        elif token in ("ip", "ps", "pc", "pt"):
            features["TYPE"] = "PC"
            if token == "ip":
                features["SUBTYPE"] = "PS"
            elif token == "ps":
                features["SUBTYPE"] = "PS"
            elif token == "pt":
                features["SUBTYPE"] = "PT"
            elif token == "pc":
                features["SUBTYPE"] = "PT"
            i += 1
            continue

        elif token in ("pa", "act", "active"):
            features["SUBTYPE"] = "ACT"
            i += 1
            continue

        elif token in ("pp", "pass", "passive"):
            features["SUBTYPE"] = "PASS"
            i += 1
            continue

        elif token in ("sg", "pl", "du"):
            features["NUMBER"] = NUMBER_MAP[token]
            i += 1
            continue

        elif token in ("m", "f", "n"):
            features["GENDER"] = GENDER_MAP[token]
            i += 1
            continue

        elif token in ("nom", "gen", "dat", "acc", "voc"):
            features["CASE"] = CASE_MAP[token]
            i += 1
            continue

        elif token in ("1", "2", "3"):
            features["PERSON"] = token
            i += 1
            continue

        elif token in ("ind", "opt", "imp", "cond", "subj"):
            features["MOOD"] = MOOD_MAP[token]
            i += 1
            continue

        else:
            i += 1

    if not features:
        if features.get("NUMBER"):
            pass
        else:
            return None

    if not features.get("NUMBER"):
        if features.get("CASE"):
            features["NUMBER"] = "SG"
        elif features.get("PERSON"):
            features["NUMBER"] = "SG"

    return (target, features)


def form_to_pgr(entry: Dict, form_data: Dict, form_type: str) -> str:
    """Convert a form's data into PGR notation.

    Args:
        entry: The dictionary entry
        form_data: The form data dict (e.g., {"case": "Nominative", "singular": "deiws"})
        form_type: Type of form ("declension", "indicative", "participle", etc.)

    Returns:
        PGR string, potentially with | for ambiguous forms
    """
    features: Dict[str, str] = {}

    if form_type == "declension":
        case = form_data.get("case", "").lower()
        if case in CASE_MAP:
            features["CASE"] = CASE_MAP[case]

        gender = entry.get("gender", "").lower()
        if gender in GENDER_MAP:
            features["GENDER"] = GENDER_MAP[gender]

        decl_data = form_data
        if decl_data.get("singular") and decl_data.get("plural"):
            singular_form = decl_data["singular"]
            plural_form = decl_data["plural"]

            singular_features = dict(features)
            singular_features["NUMBER"] = "SG"
            singular_pgr = build_pgr(singular_features)

            plural_features = dict(features)
            plural_features["NUMBER"] = "PL"
            plural_pgr = build_pgr(plural_features)

            return f"{singular_pgr}|{plural_pgr}"
        elif decl_data.get("singular"):
            features["NUMBER"] = "SG"
        elif decl_data.get("plural"):
            features["NUMBER"] = "PL"

    return build_pgr(features)


def extract_pgr_from_entry(entry: Dict) -> List[Tuple[str, str]]:
    """Extract all forms with their PGR from a dictionary entry.

    Args:
        entry: Dictionary entry with 'forms' field

    Returns:
        List of (form_text, pgr_string) tuples
    """
    results = []
    forms = entry.get("forms", {})
    gender = entry.get("gender", "").lower()

    if forms.get("declension"):
        for decl in forms["declension"]:
            decl_gender = decl.get("gender", gender)
            for case_data in decl.get("cases", []):
                case = case_data.get("case", "").lower()
                case_pgr = CASE_MAP.get(case, "")

                if case_data.get("singular"):
                    singular_features = {
                        "CASE": case_pgr,
                        "NUMBER": "SG",
                    }
                    if decl_gender in GENDER_MAP:
                        singular_features["GENDER"] = GENDER_MAP[decl_gender]
                    elif gender in GENDER_MAP:
                        singular_features["GENDER"] = GENDER_MAP[gender]

                    results.append(
                        (case_data["singular"], build_pgr(singular_features))
                    )

                if case_data.get("plural"):
                    plural_features = {
                        "CASE": case_pgr,
                        "NUMBER": "PL",
                    }
                    if decl_gender in GENDER_MAP:
                        plural_features["GENDER"] = GENDER_MAP[decl_gender]
                    elif gender in GENDER_MAP:
                        plural_features["GENDER"] = GENDER_MAP[gender]

                    results.append((case_data["plural"], build_pgr(plural_features)))

    if forms.get("indicative"):
        for mood_data in forms["indicative"]:
            tense = mood_data.get("tense", "").lower()
            tense_pgr = TENSE_MAP.get(tense, "PRS")

            for form_item in mood_data.get("forms", []):
                pronoun = form_item.get("pronoun", "").lower()
                form_text = form_item.get("form", "")

                if not form_text:
                    continue

                person, number = _parse_pronoun(pronoun)

                features = {
                    "TENSE": tense_pgr,
                    "PERSON": person,
                    "NUMBER": number,
                    "MOOD": "IND",
                }

                results.append((form_text, build_pgr(features)))

    if forms.get("imperative"):
        for form_item in forms["imperative"]:
            pronoun = form_item.get("pronoun", "").lower()
            form_text = form_item.get("form", "")

            if not form_text:
                continue

            person, number = _parse_pronoun(pronoun)

            features = {
                "TENSE": "PRS",
                "PERSON": person,
                "NUMBER": number,
                "MOOD": "IMP",
            }

            results.append((form_text, build_pgr(features)))

    if forms.get("participles"):
        for pc_data in forms["participles"]:
            pc_type = pc_data.get("type", "").lower()
            form_text = pc_data.get("form", "")

            if not form_text:
                continue

            features = {
                "TYPE": "PC",
                "SUBTYPE": PC_TYPE_MAP.get(pc_type, "PS"),
            }

            results.append((form_text, build_pgr(features)))

    return results


def _parse_pronoun(pronoun: str) -> Tuple[str, str]:
    """Parse a pronoun string to extract person and number.

    Args:
        pronoun: String like "as", "tū", "mes", "jūs", "tāns/tenā/tennan"

    Returns:
        Tuple of (person, number), defaults to ("3", "SG") if unknown
    """
    pronoun = pronoun.lower()

    if "as" in pronoun or pronoun == "mes":
        return ("1", "SG" if "as" in pronoun else "PL")
    elif "tū" in pronoun:
        return ("2", "SG")
    elif "mes" in pronoun:
        return ("1", "PL")
    elif "jūs" in pronoun or "tei" in pronoun:
        return ("2", "PL")
    else:
        return ("3", "SG")


def format_pgr_for_output(pgr_string: str) -> str:
    """Format a PGR string for user-facing output.

    Ambiguous forms are pipe-separated.
    """
    return pgr_string.upper()
