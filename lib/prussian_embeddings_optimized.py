#!/usr/bin/env python3
"""
Optimierte Embeddings für prußisches Wörterbuch mit OpenVINO GPU-Beschleunigung.

Features:
- Direkte OpenVINO-Integration (Intel Arc GPU)
- Verschiedene Text-Strategien zur Optimierung
- Keine JSON-Struktur-Artefakte in Embeddings
- Batch-Processing für Performance
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

from embedding_strategies import get_strategy, TextStrategy

try:
    from optimum.intel import OVModelForFeatureExtraction
    from transformers import AutoTokenizer
    import torch
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("⚠️  optimum[openvino] nicht installiert!")
    print("   Install: pip install optimum[openvino]")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class PrussianEmbeddingsOptimized:
    """
    Optimierte Embedding-Klasse mit konfigurierbaren Strategien.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        strategy: str = "weighted",
        use_openvino: bool = True,
        device: str = "GPU.0",
        batch_size: int = 32,
        **strategy_kwargs
    ):
        """
        Args:
            model_name: HuggingFace Model ID
            strategy: Text-Strategie ("simple", "sentences", "weighted", "clusters", "minimal")
            use_openvino: Nutze OpenVINO GPU (Intel Arc)
            device: OpenVINO Device ("GPU.0", "CPU")
            batch_size: Batch-Größe für Embedding-Generierung
            **strategy_kwargs: Zusätzliche Args für Strategie
        """
        self.model_name = model_name
        self.strategy_name = strategy
        self.use_openvino = use_openvino and HAS_OPENVINO
        self.device = device
        self.batch_size = batch_size

        # Text-Strategie
        self.text_strategy = get_strategy(strategy, **strategy_kwargs)
        print(f"📝 Text-Strategie: {self.text_strategy.name}")

        # Embeddings
        self.embeddings = None
        self.entries = []
        self.embedding_dim = None

        # Modell
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Lade Embedding-Modell (OpenVINO bevorzugt)."""

        if self.use_openvino:
            print(f"🔧 Lade OpenVINO-Modell: {self.model_name}")
            print(f"   Device: {self.device}")

            try:
                # OpenVINO IR-Format mit GPU
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    self.model_name,
                    export=True,
                    device=self.device
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.embedding_dim = self.model.config.hidden_size

                print(f"   ✓ GPU-Modell geladen (dim={self.embedding_dim})")

            except Exception as e:
                print(f"   ⚠️  OpenVINO fehlgeschlagen: {e}")
                print(f"   → Fallback zu sentence-transformers")
                self.use_openvino = False
                self._load_fallback()

        else:
            self._load_fallback()

    def _load_fallback(self):
        """Fallback: sentence-transformers (CPU)."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Weder OpenVINO noch sentence-transformers verfügbar!")

        print(f"📦 Lade sentence-transformers: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"   ✓ CPU-Modell geladen (dim={self.embedding_dim})")

    def load_dictionary(self, json_path: str) -> None:
        """Lade Wörterbuch und generiere Embeddings."""
        print(f"\n📖 Lade Wörterbuch: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Konvertiere zu Liste
        if isinstance(data, dict):
            self.entries = list(data.values())
        elif isinstance(data, list):
            self.entries = data
        else:
            raise ValueError("Erwartetes Format: List oder Dict")

        print(f"   ✓ {len(self.entries)} Einträge geladen")

        # Filter: Nur Einträge mit Übersetzungen
        from embedding_strategies import should_include_entry

        original_count = len(self.entries)
        self.entries = [e for e in self.entries if should_include_entry(e)]
        filtered_count = len(self.entries)

        print(f"   ℹ️  Gefiltert: {original_count} → {filtered_count} Einträge")
        print(f"      (Entfernt: {original_count - filtered_count} Verweis-Einträge)")

        # Generiere Embeddings
        self._generate_embeddings()

    def _generate_embeddings(self) -> None:
        """Generiere Embeddings mit aktueller Strategie."""
        print(f"\n🔢 Generiere Embeddings ({self.text_strategy.name})...")

        # Text-Repräsentationen generieren
        print(f"   1/3 Generiere Texte...")
        texts = []
        for i, entry in enumerate(self.entries):
            text = self.text_strategy.generate(entry)
            texts.append(text)

            if i % 1000 == 0 and i > 0:
                print(f"        {i}/{len(self.entries)}", end="\r")

        print(f"   ✓ {len(texts)} Texte generiert")

        # Embeddings berechnen
        print(f"   2/3 Berechne Embeddings...")
        if self.use_openvino:
            self.embeddings = self._encode_openvino(texts)
        else:
            self.embeddings = self._encode_sentence_transformers(texts)

        print(f"   3/3 Fertig!")
        print(f"   Shape: {self.embeddings.shape}")

    def _encode_openvino(self, texts: List[str]) -> np.ndarray:
        """Kodiere mit OpenVINO GPU."""
        start = time.time()
        embeddings = []

        print(f"   🚀 OpenVINO GPU (batch={self.batch_size}, device={self.device})")

        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Tokenisiere
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Inference
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pooling
            token_embeddings = outputs[0]
            attention_mask = encoded['attention_mask']

            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = (token_embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)

            sentence_embeddings = sum_embeddings / sum_mask

            embeddings.append(sentence_embeddings.detach().cpu().numpy())

            # Progress
            batch_num = (i // self.batch_size) + 1
            pct = (batch_num / num_batches) * 100
            print(f"        [{pct:5.1f}%] Batch {batch_num}/{num_batches}", end="\r")

        print()  # Newline nach Progress

        embeddings = np.vstack(embeddings)
        elapsed = time.time() - start

        print(f"   ✓ {elapsed:.1f}s ({len(texts)/elapsed:.0f} entries/s)")

        return embeddings

    def _encode_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Kodiere mit sentence-transformers (CPU)."""
        start = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        elapsed = time.time() - start
        print(f"   ✓ {elapsed:.1f}s ({len(texts)/elapsed:.0f} entries/s)")

        return embeddings

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Semantische Suche."""
        if self.embeddings is None:
            raise RuntimeError("Keine Embeddings geladen!")

        # Query-Embedding
        if self.use_openvino:
            query_embedding = self._encode_openvino([query])[0]
        else:
            query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Kosinus-Ähnlichkeit
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.entries[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Kosinus-Ähnlichkeit."""
        # Normalisiere Vektor
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-10:
            vec = vec / vec_norm

        # Normalisiere Matrix
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norms = np.clip(matrix_norms, a_min=1e-10, a_max=None)
        matrix = matrix / matrix_norms

        return np.dot(matrix, vec)

    def save_embeddings(self, output_path: str) -> None:
        """Speichere Embeddings + Metadaten."""
        print(f"\n💾 Speichere Embeddings: {output_path}")

        # Embeddings
        np.save(f"{output_path}.embeddings.npy", self.embeddings)

        # Einträge
        with open(f"{output_path}.entries.json", "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

        # Metadaten
        metadata = {
            "model": self.model_name,
            "strategy": self.strategy_name,
            "num_entries": len(self.entries),
            "embedding_dim": self.embedding_dim,
            "device": self.device if self.use_openvino else "CPU",
        }

        with open(f"{output_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"   ✓ Gespeichert")
        print(f"      - {output_path}.embeddings.npy")
        print(f"      - {output_path}.entries.json")
        print(f"      - {output_path}.meta.json")

    def load_embeddings(self, input_path: str) -> None:
        """Lade vorberechnete Embeddings."""
        print(f"\n📂 Lade Embeddings: {input_path}")

        # Embeddings
        self.embeddings = np.load(f"{input_path}.embeddings.npy")

        # Einträge
        with open(f"{input_path}.entries.json", "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        # Metadaten (optional)
        meta_path = f"{input_path}.meta.json"
        if Path(meta_path).exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                print(f"   ℹ️  Metadata: {metadata}")

        if self.embeddings.size > 0:
            self.embedding_dim = self.embeddings.shape[1]

        print(f"   ✓ {len(self.entries)} Einträge, {self.embeddings.shape[0]} Embeddings")

    def benchmark(self, queries: List[str], iterations: int = 1) -> Dict:
        """Benchmark Suchgeschwindigkeit."""
        if not queries:
            raise ValueError("Queries required for benchmark")

        print(f"\n⏱️  Benchmark: {len(queries)} Queries × {iterations} Iterationen")

        start = time.time()
        total_searches = 0

        for _ in range(iterations):
            for query in queries:
                self.search(query, top_k=10)
                total_searches += 1

        elapsed = time.time() - start

        avg_ms = (elapsed / total_searches) * 1000
        qps = total_searches / elapsed

        results = {
            "total_searches": total_searches,
            "elapsed_sec": elapsed,
            "avg_ms_per_query": avg_ms,
            "queries_per_sec": qps,
        }

        print(f"   ✓ {total_searches} Suchen in {elapsed:.2f}s")
        print(f"      Durchschnitt: {avg_ms:.1f}ms/Query")
        print(f"      Durchsatz: {qps:.0f} queries/s")

        return results


if __name__ == "__main__":
    # Test
    print("🌲 Prußische Embeddings (Optimized)\n")

    embedder = PrussianEmbeddingsOptimized(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        strategy="weighted",
        use_openvino=True,
        device="GPU.0"
    )

    # Lade Wörterbuch
    dict_file = "prussian_dictionary.json"
    if Path(dict_file).exists():
        embedder.load_dictionary(dict_file)

        # Test-Suchen
        test_queries = ["son", "daughter", "family", "light", "water"]
        print("\n🔍 Test-Suchen:")
        for q in test_queries:
            results = embedder.search(q, top_k=3)
            print(f"\n  '{q}':")
            for rank, (entry, score) in enumerate(results, 1):
                word = entry.get("word", "?")
                print(f"    {rank}. [{score:.3f}] {word}")

        # Speichere
        embedder.save_embeddings("embeddings_optimized")
    else:
        print(f"⚠️  {dict_file} nicht gefunden!")
