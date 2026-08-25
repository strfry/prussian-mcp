"""Kartesische Paradigmen-Zellen für die FST-Generierungsrichtung.

Der FST matcht ``lemma+Tag1+Tag2+...`` als literalen String (nicht als
Merkmalsmenge wie ``match_tags``) — die Tag-Reihenfolge muss exakt der
beim Lexc-Bau verwendeten entsprechen.  Statt Formen aus dem Dictionary-
JSON zu extrahieren (die Bugquelle des ursprünglichen ``get_word_forms``-
Fehlers: unvollständige/inkonsistente Feld-Auslesung), probieren wir hier
alle plausiblen Tag-Kombinationen pro Wortart durch und lassen den FST
entscheiden, welche davon tatsächlich existieren (leere Treffer werden
vom Aufrufer verworfen).

POS-Klassifikation, Valenz-Parsing und Reflexiv-Splitting werden direkt
aus ``prussian_fst.gen_lexc`` importiert (reine, seiteneffektfreie
Funktionen — kein Refactoring dort nötig, siehe Plan) statt hier ein
zweites Mal implementiert zu werden.
"""

from __future__ import annotations

from prussian_fst.gen_lexc import (
    CASE_MAP,
    GENDER_MAP,
    PERSON_TAGS,
    POS_TAGS,
    classify,
    numeral_subtype,
    prep_gov_tags,
    refl_tag,
    verb_valence_tags,
)

# Reihenfolge exakt wie gen_lexc.py:453 (Partizip-Deklination) bzw.
# gen_lexc.py:270-286 (Nominal-Deklination) — beide iterieren
# Kasus innerhalb Numerus, Genus als letztes Element angehängt.
CASES = list(CASE_MAP.values())  # ["Nom", "Gen", "Dat", "Akk"]
NUMBERS = ["Sg", "Pl"]
GENDERS_TAG = [g for g in GENDER_MAP.values() if g]  # ["+Masc", "+Fem", "+Neut"]

# P3 ist in PERSON_TAGS zweimal enthalten (3sg==3pl, kein Numerus-Tag
# nötig, siehe gen_lexc.py:48-51) — für die Kombinatorik einmal reicht.
PERSON_TAGS_UNIQ = list(dict.fromkeys(PERSON_TAGS))

# Wortarten, die vollständig entry-getrieben (Case/Number/Gender) generiert werden.
_NOMINAL_POS = {"noun", "proper_noun", "adjective", "numeral"}
# Wortarten mit genau einer unveränderlichen Zelle (kein Cartesian Product).
_INVARIABLE_POS = {"adverb", "conjunction", "particle", "interjection"}


def build_paradigm_queries(entry: dict, prep_words: set[str]) -> dict[str, str]:
    """{tag_suffix: full_query_string} für alle plausiblen Paradigmen-Zellen.

    *tag_suffix* ist ohne führendes Lemma, mit führendem "+" (z.B.
    ``"+V+Part+Pass+Sg+Nom+Neut"``); *full_query_string* ist
    ``f"{base}{tag_suffix}"``, direkt an ``prussian_fst.api.generate``
    übergebbar.  Enumeriert nur — ob eine Zelle im Lexikon existiert,
    entscheidet der FST-Lookup beim Aufrufer.
    """
    pos = classify(entry)
    word = entry.get("word", "")
    if not word:
        return {}

    queries: dict[str, str] = {}

    if pos == "verb":
        base, refl = refl_tag(word)
        val = "".join(verb_valence_tags(entry.get("desc", ""), prep_words))
        suffix_extra = f"{val}{refl}"

        queries["+V+Inf"] = f"{base}+V+Inf{suffix_extra}"
        for tense in ("Pres", "Pret"):
            for p in PERSON_TAGS_UNIQ:
                tag = f"+V+Ind+{tense}+{p}"
                queries[tag] = f"{base}{tag}{suffix_extra}"
        queries["+V+Opt+P3"] = f"{base}+V+Opt+P3{suffix_extra}"
        for p in ("P2+Sg", "P2+Pl"):
            tag = f"+V+Imp+{p}"
            queries[tag] = f"{base}{tag}{suffix_extra}"
        for p in PERSON_TAGS_UNIQ:
            tag = f"+V+Subj+{p}"
            queries[tag] = f"{base}{tag}{suffix_extra}"
        for ptype in ("Pres", "Pret", "Pass"):
            for num in NUMBERS:
                for case in CASES:
                    for gend in GENDERS_TAG:
                        tag = f"+V+Part+{ptype}+{num}+{case}{gend}"
                        queries[tag] = f"{base}{tag}{suffix_extra}"
        return queries

    if pos in _NOMINAL_POS:
        pos_tag = POS_TAGS[pos]
        subtype_tag = numeral_subtype(entry.get("desc", "")) if pos == "numeral" else ""
        degrees = ["", "+Cmp", "+Sup"] if pos == "adjective" else [""]
        if pos == "adjective":
            genders = GENDERS_TAG
        else:
            genders = [GENDER_MAP.get(entry.get("gender", "").lower(), "")]
        for deg_tag in degrees:
            for num in NUMBERS:
                for case in CASES:
                    for gend in genders:
                        tag = f"{pos_tag}{subtype_tag}{deg_tag}+{num}+{case}{gend}"
                        queries[tag] = f"{word}{tag}"
        return queries

    if pos in _INVARIABLE_POS:
        tag = POS_TAGS[pos]
        queries[tag] = f"{word}{tag}"
        return queries

    if pos == "preposition":
        tag = POS_TAGS[pos]
        gov_tags = prep_gov_tags(entry.get("desc", "")) or [""]
        for gov in gov_tags:
            full_tag = f"{tag}{gov}"
            queries[full_tag] = f"{word}{full_tag}"
        return queries

    # pronoun (handgeschrieben in pronouns.lexc) / unknown: außerhalb des
    # Geltungsbereichs, wie schon extract_pgr_from_entry keine
    # forms.declension-Daten für Pronomen liefert.
    return {}
