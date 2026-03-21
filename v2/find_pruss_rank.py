#!/usr/bin/env python3
"""Finde wo prūss in den Suchergebnissen für 'pruße' rangiert."""

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("Testing with weighted strategy...")
embedder = PrussianEmbeddingsOptimized(
    strategy='weighted',
    use_openvino=True,
    device='GPU.0'
)
embedder.load_dictionary('prussian_dictionary.json')

# Suche nach "pruße" mit top_k=100
query = 'pruße'
results = embedder.search(query, top_k=100)

print(f"\nSearch query: '{query}'")
print("="*60)

# Finde prūss in den Ergebnissen
found_rank = None
for rank, (entry, score) in enumerate(results, 1):
    word = entry.get('word', '')
    if word == 'prūss':
        found_rank = rank
        print(f"✓ Found 'prūss' at rank {rank} with score {score:.3f}")
        print(f"  Translations: {entry.get('translations', {}).get('miks', [])}")
        break

if not found_rank:
    print("✗ 'prūss' NOT found in top 100 results!")

# Zeige Top 10 für Kontext
print("\nTop 10 results for context:")
print("-"*60)
for rank, (entry, score) in enumerate(results[:10], 1):
    word = entry.get('word', '')
    de_trans = entry.get('translations', {}).get('miks', [])
    marker = " ← TARGET" if word == 'prūss' else ""
    print(f"{rank:3}. [{score:.3f}] {word:<20} → {de_trans}{marker}")
