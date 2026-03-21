#!/usr/bin/env python3
"""
Prussian Dictionary Search Skill
Lightweight embedding-basierte Suche (OHNE OpenVINO-Dependency für Server).

OpenVINO wird nur für Embedding-Generierung gebraucht (einmalig, offline).
Für Suche reicht sentence-transformers auf CPU (schnell genug für einzelne Queries).

Usage:
    # CLI
    python prussian_search_skill.py "son"
    python prussian_search_skill.py --interactive

    # Python API
    from prussian_search_skill import PrussianSearch

    search = PrussianSearch()
    results = search.query("son", top_k=5)

Dependencies (Server):
    - sentence-transformers
    - numpy
    (KEIN OpenVINO nötig!)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class PrussianSearch:
    """Lightweight Skill für Embedding-basierte Wörterbuchsuche (CPU)."""

    def __init__(
        self,
        embeddings_path: str = "embeddings_production",
        model_name: str = None,  # Auto-detect from metadata
        auto_load: bool = True
    ):
        """
        Args:
            embeddings_path: Pfad zu vorberechneten Embeddings
            model_name: Sentence-Transformer Modell (None = auto-detect from metadata)
            auto_load: Automatisch Embeddings beim Init laden
        """
        self.embeddings_path = embeddings_path
        self.model_name = model_name

        # Daten
        self.embeddings = None  # Vorberechnete Vektoren (geladen)
        self.entries = None     # Wörterbuch-Einträge
        self.embedding_dim = None
        self.metadata = None    # Metadata from embeddings

        # Modell (nur für Query → Vektor)
        self.model = None

        if auto_load:
            self.load()

    def load(self) -> None:
        """Lade vorberechnete Embeddings und Modell."""
        if self.embeddings is not None:
            return  # Bereits geladen

        embeddings_file = f"{self.embeddings_path}.embeddings.npy"
        entries_file = f"{self.embeddings_path}.entries.json"
        meta_file = f"{self.embeddings_path}.meta.json"

        # Prüfe Dateien
        if not Path(embeddings_file).exists():
            raise FileNotFoundError(
                f"Embeddings nicht gefunden: {embeddings_file}\n"
                f"Bitte erst generieren mit: python generate_production_embeddings.py"
            )

        # Lade Metadata
        if Path(meta_file).exists():
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)

            # Auto-detect model from metadata
            if self.model_name is None and "model" in self.metadata:
                self.model_name = self.metadata["model"]

        # Fallback model
        if self.model_name is None:
            self.model_name = "intfloat/multilingual-e5-large"

        # Lade vorberechnete Embeddings (numpy)
        print(f"📂 Lade Embeddings: {embeddings_file}")
        self.embeddings = np.load(embeddings_file)
        self.embedding_dim = self.embeddings.shape[1]

        # Lade Wörterbuch-Einträge
        with open(entries_file, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        print(f"   ✓ {len(self.entries)} Einträge, {self.embedding_dim}-dim")
        if self.metadata:
            print(f"   ℹ️  Model: {self.metadata.get('model', 'unknown')}")
            print(f"   ℹ️  Strategy: {self.metadata.get('strategy', 'unknown')}")

        # Lade Modell (nur für Query-Encoding)
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers nicht installiert!\n"
                "Install: pip install sentence-transformers"
            )

        print(f"📦 Lade Modell: {self.model_name} (CPU)")
        self.model = SentenceTransformer(self.model_name)
        print(f"   ✓ Bereit für Suche")

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        include_translations: bool = True,
        lang_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Suche im Wörterbuch (CPU-basiert, kein OpenVINO).

        Args:
            query_text: Suchbegriff (beliebige Sprache)
            top_k: Anzahl Ergebnisse
            include_translations: Übersetzungen in Ergebnis einbeziehen
            lang_filter: Nur bestimmte Sprachen zurückgeben

        Returns:
            Liste von Ergebnissen mit Score und Übersetzungen
        """
        if self.embeddings is None:
            self.load()

        # E5-Modelle: "query: " Präfix für asymmetrische Suche
        if "e5" in self.model_name.lower():
            query_text_prefixed = f"query: {query_text}"
        else:
            query_text_prefixed = query_text

        # 1. Query → Embedding (CPU, sehr schnell für 1 Text)
        query_embedding = self.model.encode(query_text_prefixed, convert_to_numpy=True)

        # 2. Kosinus-Ähnlichkeit mit allen vorberechneten Embeddings
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # 3. Top-K Ergebnisse
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # 4. Formatiere Ergebnisse
        formatted = []

        for idx in top_indices:
            entry = self.entries[idx]
            score = float(similarities[idx])

            result = {
                "word": entry.get("word", "?"),
                "score": score,
            }

            if include_translations and "translations" in entry:
                translations = {}

                for lang_code, trans_list in entry["translations"].items():
                    # Filter nach Sprachen wenn gewünscht
                    if lang_filter and lang_code not in lang_filter:
                        continue

                    if isinstance(trans_list, list) and trans_list:
                        translations[lang_code] = trans_list

                result["translations"] = translations

            # Optionale Felder
            if "desc" in entry and entry["desc"]:
                result["desc"] = entry["desc"]

            if "paradigm" in entry and entry["paradigm"]:
                result["paradigm"] = entry["paradigm"]

            if "gender" in entry and entry["gender"]:
                result["gender"] = entry["gender"]

            formatted.append(result)

        return formatted

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Berechne Kosinus-Ähnlichkeit (pure NumPy, sehr schnell)."""
        # Normalisiere Query-Vektor
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-10:
            vec = vec / vec_norm

        # Normalisiere Matrix
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norms = np.clip(matrix_norms, a_min=1e-10, a_max=None)
        matrix = matrix / matrix_norms

        # Skalarprodukt = Kosinus bei normalisierten Vektoren
        return np.dot(matrix, vec)

    def get_translation_summary(self, result: Dict) -> str:
        """Erstelle eine lesbare Zusammenfassung der Übersetzungen."""
        if "translations" not in result:
            return "-"

        # Prioritäts-Reihenfolge
        priority_langs = ["engl", "miks", "leit", "latt", "pols", "mask"]

        parts = []
        for lang in priority_langs:
            if lang in result["translations"]:
                trans = result["translations"][lang][:2]  # Max 2 pro Sprache
                parts.extend(trans)

            if len(parts) >= 4:  # Max 4 insgesamt
                break

        return ", ".join(parts) if parts else "-"

    def format_result(self, result: Dict, show_score: bool = True) -> str:
        """Formatiere ein Ergebnis für CLI-Ausgabe."""
        word = result["word"]
        trans = self.get_translation_summary(result)

        if show_score:
            score = result["score"]
            return f"[{score:.3f}] {word:25s} = {trans}"
        else:
            return f"{word:25s} = {trans}"

    def batch_query(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """Mehrere Suchen auf einmal."""
        return {
            query: self.query(query, top_k=top_k)
            for query in queries
        }

    def get_stats(self) -> Dict:
        """Statistiken über geladene Embeddings."""
        if self.embeddings is None:
            self.load()

        stats = {
            "num_entries": len(self.entries),
            "embedding_dim": self.embedding_dim,
            "model": self.model_name,
            "device": "CPU (sentence-transformers)",
        }

        # Add metadata if available
        if self.metadata:
            stats["strategy"] = self.metadata.get("strategy", "unknown")
            stats["generation_device"] = self.metadata.get("device", "unknown")
        else:
            stats["strategy"] = "unknown"

        return stats


# ============================================================================
# CLI Interface
# ============================================================================

def cli_search(args):
    """CLI: Einzelne Suche."""
    search = PrussianSearch()

    query = " ".join(args.query)
    results = search.query(query, top_k=args.top_k)

    print(f"\n🔍 Suche: '{query}'")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"{i:2d}. {search.format_result(result)}")

    print()


def cli_interactive(args):
    """CLI: Interaktiver Modus."""
    print("🔍 Prussian Dictionary Search - Interactive Mode")
    print("=" * 70)

    search = PrussianSearch()

    stats = search.get_stats()
    print(f"Loaded: {stats['num_entries']} entries, {stats['embedding_dim']}-dim vectors")
    print(f"Device: {stats['device']}")
    print(f"Strategy: {stats['strategy']}")
    print("\nType your search query (Ctrl+C to exit)")
    print("=" * 70)
    print()

    try:
        while True:
            query = input("Search: ").strip()

            if not query:
                continue

            results = search.query(query, top_k=args.top_k)

            print()
            for i, result in enumerate(results, 1):
                print(f"{i:2d}. {search.format_result(result)}")
            print()

    except KeyboardInterrupt:
        print("\n\nAuf Wiedersehen!")


def cli_batch(args):
    """CLI: Batch-Suche."""
    search = PrussianSearch()

    # Lese Queries
    if args.file:
        with open(args.file) as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = args.queries

    results = search.batch_query(queries, top_k=args.top_k)

    for query, query_results in results.items():
        print(f"\n📌 '{query}'")
        print("-" * 70)

        for i, result in enumerate(query_results, 1):
            print(f"  {i}. {search.format_result(result)}")

    print()


def cli_stats(args):
    """CLI: Zeige Statistiken."""
    search = PrussianSearch()
    stats = search.get_stats()

    print("\n📊 Prussian Dictionary Embeddings - Statistics")
    print("=" * 70)
    print(f"  Entries:        {stats['num_entries']}")
    print(f"  Embedding Dim:  {stats['embedding_dim']}")
    print(f"  Strategy:       {stats['strategy']}")
    print(f"  Model:          {stats['model']}")
    print(f"  Device:         {stats['device']}")
    print("=" * 70)
    print()


def main():
    """CLI Entry Point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prussian Dictionary Embedding Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single search
  %(prog)s "son"
  %(prog)s "water" --top-k 5

  # Interactive mode
  %(prog)s --interactive

  # Batch search
  %(prog)s --batch "son" "daughter" "mother"
  %(prog)s --batch --file queries.txt

  # Statistics
  %(prog)s --stats
"""
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Search query (can be multiple words)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive search mode"
    )

    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Batch search mode"
    )

    parser.add_argument(
        "-f", "--file",
        help="Read queries from file (one per line)"
    )

    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of results (default: 10)"
    )

    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="Show statistics"
    )

    parser.add_argument(
        "--queries",
        nargs="+",
        help="Multiple queries for batch mode"
    )

    args = parser.parse_args()

    # Route zu richtiger Funktion
    if args.stats:
        cli_stats(args)
    elif args.interactive:
        cli_interactive(args)
    elif args.batch:
        cli_batch(args)
    elif args.query:
        cli_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
