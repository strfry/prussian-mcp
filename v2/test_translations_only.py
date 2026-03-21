#!/usr/bin/env python3
"""Test der translations_only Strategie."""

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("="*60)
print("Testing: translations_only strategy")
print("="*60)

embedder = PrussianEmbeddingsOptimized(
    strategy='translations_only',
    use_openvino=True,
    device='GPU.0'
)
embedder.load_dictionary('prussian_dictionary.json')

# Test-Queries
test_queries = [
    'pruße',
    'preußen',
    'ostpreußen',
    'altpreußisch',
    'prussian',
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print('='*60)

    results = embedder.search(query, top_k=10)

    # Check ob prūss dabei ist
    pruss_found = False
    for rank, (entry, score) in enumerate(results, 1):
        word = entry.get('word', '')
        de_trans = entry.get('translations', {}).get('miks', [])

        marker = ""
        if 'prūss' in word.lower() or 'prusija' in word.lower():
            marker = " ← TARGET"
            pruss_found = True

        print(f"{rank:2}. [{score:.3f}] {word:<25} → {', '.join(de_trans[:2]) if de_trans else '—'}{marker}")

    if pruss_found:
        print(f"\n✓ Found relevant Prussian terms!")
    else:
        print(f"\n✗ No relevant Prussian terms in top 10")
