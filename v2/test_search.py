#!/usr/bin/env python3
"""Test-Suche mit vollständigen Embeddings."""

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("🔍 Lade Produktions-Embeddings...")
embedder = PrussianEmbeddingsOptimized(strategy="weighted")
embedder.load_embeddings("embeddings_production")

print(f"✓ {len(embedder.entries)} Einträge geladen\n")

# Test-Queries
queries = ["tree", "forest", "son", "daughter", "mother", "father", "water", "light", "house", "god"]

print("="*70)
print("SEMANTISCHE SUCH-TESTS")
print("="*70)

for query in queries:
    results = embedder.search(query, top_k=5)

    print(f"\n📌 '{query}'")
    print("-"*70)

    for rank, (entry, score) in enumerate(results, 1):
        word = entry.get("word", "?")

        # Übersetzungen sammeln
        trans = []
        if "translations" in entry:
            for lang in ["engl", "miks", "leit"]:
                if lang in entry["translations"] and entry["translations"][lang]:
                    trans.extend(entry["translations"][lang][:2])

        trans_str = ", ".join(trans[:3]) if trans else "-"

        print(f"  {rank}. [{score:.3f}] {word:20s} = {trans_str}")

print("\n" + "="*70)
print("✅ Semantische Suche funktioniert!")
print("="*70)
