#!/usr/bin/env python3
"""
Verschiedene Strategien zur Text-Repräsentation von Wörterbucheinträgen.
Ziel: Optimale Vektoren generieren, die linguistische Semantik erfassen,
aber NICHT die JSON-Struktur.
"""

from typing import Dict, List
import re


def should_include_entry(entry: Dict) -> bool:
    """
    Prüfe ob Eintrag Übersetzungen hat (nicht nur Verweis).

    Filtert Deklinationsformen ohne Bedeutung aus:
    - Einträge mit leeren translations: skip
    - Einträge nur mit Verweisen (siehe X): skip
    - Einträge mit tatsächlichen Übersetzungen: include

    Args:
        entry: Wörterbuch-Eintrag mit 'translations' Dict

    Returns:
        True wenn Eintrag Übersetzungen hat, False sonst
    """
    translations = entry.get('translations', {})

    # Prüfe ob mindestens eine Sprache Übersetzungen hat
    has_translations = any(
        isinstance(trans_list, list) and len(trans_list) > 0
        for trans_list in translations.values()
    )

    return has_translations


class TextStrategy:
    """Basis-Klasse für Text-Generierungs-Strategien."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, entry: Dict) -> str:
        """Generiere Text-Repräsentation aus Wörterbuch-Eintrag."""
        raise NotImplementedError


class SimpleConcat(TextStrategy):
    """
    Strategie 1: Einfache Konkatenation (baseline).
    Problem: Enthält Struktur-Artefakte.
    """

    def __init__(self):
        super().__init__("simple_concat")

    def generate(self, entry: Dict) -> str:
        texts = []

        if "word" in entry:
            texts.extend([entry['word']] * 3)

        if "translations" in entry:
            for lang, trans_list in entry['translations'].items():
                if isinstance(trans_list, list):
                    texts.extend(trans_list)

        return " ".join(str(t) for t in texts if t)


class NaturalSentences(TextStrategy):
    """
    Strategie 2: Natürliche Sätze.
    Erstellt flüssige Sätze in verschiedenen Sprachen.
    """

    def __init__(self):
        super().__init__("natural_sentences")

        # Satz-Templates pro Sprache
        self.templates = {
            "engl": "In English: {word}",
            "miks": "Auf Deutsch: {word}",
            "leit": "Lietuviškai: {word}",
            "latt": "Latviešu: {word}",
            "pols": "Po polsku: {word}",
            "mask": "По-русски: {word}",
        }

    def generate(self, entry: Dict) -> str:
        parts = []

        # Hauptwort - mehrfach gewichtet
        if "word" in entry:
            prussian = entry['word']
            parts.extend([
                f"Old Prussian word: {prussian}",
                prussian,  # Raw form
                prussian,  # Nochmal für Gewichtung
            ])

        # Übersetzungen als Sätze
        if "translations" in entry:
            for lang_code, trans_list in entry['translations'].items():
                if isinstance(trans_list, list) and trans_list:
                    template = self.templates.get(lang_code, "{word}")
                    for trans in trans_list:
                        if trans:
                            parts.append(template.format(word=trans))

        return ". ".join(parts)


class WeightedMultilingual(TextStrategy):
    """
    Strategie 3: Gewichtete mehrsprachige Repräsentation.
    Keine Satzstruktur, aber klare Gewichtung nach linguistischer Relevanz.
    """

    def __init__(self, prussian_weight: int = 5, translation_weight: int = 2):
        super().__init__("weighted_multilingual")
        self.prussian_weight = prussian_weight
        self.translation_weight = translation_weight

    def generate(self, entry: Dict) -> str:
        tokens = []

        # Altpreußisches Wort: höchste Priorität
        if "word" in entry:
            prussian = self._clean_word(entry['word'])
            tokens.extend([prussian] * self.prussian_weight)

        # Übersetzungen: nach Sprachfamilie gewichtet
        if "translations" in entry:
            # Baltische Sprachen (näher an Altpreußisch)
            for lang in ["leit", "latt"]:
                if lang in entry['translations']:
                    for trans in entry['translations'][lang]:
                        if trans:
                            tokens.extend([self._clean_word(trans)] * 3)

            # Andere Sprachen (Englisch, Deutsch besonders wichtig für Suche)
            for lang in ["engl", "miks"]:
                if lang in entry['translations']:
                    for trans in entry['translations'][lang]:
                        if trans:
                            tokens.extend([self._clean_word(trans)] * self.translation_weight)

            # Slawische Sprachen
            for lang in ["pols", "mask"]:
                if lang in entry['translations']:
                    for trans in entry['translations'][lang]:
                        if trans:
                            tokens.append(self._clean_word(trans))

        return " ".join(tokens)

    @staticmethod
    def _clean_word(word: str) -> str:
        """Entferne technische Markierungen und normalisiere."""
        # Entferne Klammern, Zahlen, etc.
        cleaned = re.sub(r'[\[\]\(\)0-9]', '', str(word))
        # Entferne extra Whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()


class SemanticClusters(TextStrategy):
    """
    Strategie 4: Semantische Cluster.
    Gruppiert Übersetzungen nach Bedeutungs-Kernen.
    """

    def __init__(self):
        super().__init__("semantic_clusters")

    def generate(self, entry: Dict) -> str:
        clusters = []

        # Cluster 1: Hauptbegriff (alle Varianten)
        main_cluster = []
        if "word" in entry:
            main_cluster.append(entry['word'])

        if "translations" in entry:
            for lang_code, trans_list in entry['translations'].items():
                if isinstance(trans_list, list):
                    main_cluster.extend([t for t in trans_list if t])

        # Haupt-Cluster mehrfach
        clusters.extend(main_cluster * 2)

        # Cluster 2: Baltische Kognaten (wenn vorhanden)
        baltic = []
        if "translations" in entry:
            for lang in ["leit", "latt"]:
                if lang in entry['translations']:
                    baltic.extend(entry['translations'][lang])

        if baltic:
            clusters.extend(baltic)

        # Cluster 3: Beschreibung (wenn vorhanden)
        if "desc" in entry and entry['desc']:
            desc = self._extract_semantic_desc(entry['desc'])
            if desc:
                clusters.append(desc)

        return " ".join(str(c) for c in clusters if c)

    @staticmethod
    def _extract_semantic_desc(desc: str) -> str:
        """Extrahiere semantischen Inhalt aus Beschreibung."""
        # Entferne technische Referenzen in eckigen Klammern
        cleaned = re.sub(r'\[.*?\]', '', desc)
        # Entferne Paradigma-Nummern
        cleaned = re.sub(r'\d+', '', cleaned)
        # Entferne extra Whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()


class MinimalStructure(TextStrategy):
    """
    Strategie 5: Minimale Struktur - nur das Wesentliche.
    Vermeidet jegliche strukturelle Hinweise.
    """

    def __init__(self):
        super().__init__("minimal_structure")

    def generate(self, entry: Dict) -> str:
        # Sammle alle einzigartigen Wörter
        words = set()

        # Altpreußisch
        if "word" in entry:
            word = self._normalize(entry['word'])
            if word:
                words.add(word)

        # Übersetzungen
        if "translations" in entry:
            for lang_code, trans_list in entry['translations'].items():
                if isinstance(trans_list, list):
                    for trans in trans_list:
                        normalized = self._normalize(trans)
                        if normalized:
                            words.add(normalized)

        # Erstelle gewichteten Text: wichtigste Wörter mehrfach
        text_parts = []

        # Hauptwort (falls identifizierbar)
        if "word" in entry:
            main = self._normalize(entry['word'])
            if main:
                text_parts.extend([main] * 4)

        # Alle anderen Wörter einmal
        for word in sorted(words):  # Sortierung für Konsistenz
            if word not in text_parts:
                text_parts.append(word)

        return " ".join(text_parts)

    @staticmethod
    def _normalize(word) -> str:
        """Aggressive Normalisierung."""
        if not word:
            return ""

        word = str(word)

        # Nur Buchstaben und Bindestriche behalten
        word = re.sub(r'[^a-zA-ZäöüßĄąĘęĮįŲųĖėŪūàáâãäåèéêëìíîïòóôõöùúûüýÿĀāĒēĪīŌōŪūČčĎďĚěŇňŘřŠšŤťŮůŽžĄąĆćĘęŁłŃńÓóŚśŹźŻżА-Яа-я -]', '', word)

        # Entferne extra Whitespace
        word = ' '.join(word.split())

        return word.strip().lower()


class TranslationsOnly(TextStrategy):
    """
    Strategie 6: Nur Übersetzungen.
    Das Modell versteht kein Altpreußisch - daher nur bekannte Sprachen embedden.
    Altpreußisches Wort wird minimal (1×) für exakte Treffer erwähnt.
    """

    def __init__(self, include_prussian_once: bool = True, use_e5_prefix: bool = False):
        super().__init__("translations_only")
        self.include_prussian_once = include_prussian_once
        self.use_e5_prefix = use_e5_prefix

    def generate(self, entry: Dict) -> str:
        parts = []

        # Übersetzungen stark gewichtet
        if "translations" in entry:
            for lang_code, trans_list in entry['translations'].items():
                if isinstance(trans_list, list) and trans_list:
                    for trans in trans_list:
                        if trans:
                            cleaned = self._clean(trans)
                            # Jede Übersetzung 3× für Gewichtung
                            parts.extend([cleaned] * 3)

        # Altpreußisches Wort nur 1× am Ende (für exakte Suchen)
        if self.include_prussian_once and "word" in entry:
            parts.append(entry['word'])

        text = " ".join(parts)

        # E5-Modelle: "passage: " Präfix für bessere Performance
        if self.use_e5_prefix:
            text = "passage: " + text

        return text

    @staticmethod
    def _clean(text: str) -> str:
        """Entferne Klammern und technische Markierungen."""
        cleaned = re.sub(r'[\[\]\(\)0-9]', '', str(text))
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()


# Factory
STRATEGIES = {
    "simple": SimpleConcat,
    "sentences": NaturalSentences,
    "weighted": WeightedMultilingual,
    "clusters": SemanticClusters,
    "minimal": MinimalStructure,
    "translations_only": TranslationsOnly,
}


def get_strategy(name: str, **kwargs) -> TextStrategy:
    """Factory-Funktion für Strategien."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[name](**kwargs)
