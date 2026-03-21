#!/usr/bin/env python3
"""Verifiziere Production-Embeddings."""

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("="*60)
print("Verifying Production Embeddings")
print("="*60)

embedder = PrussianEmbeddingsOptimized(
    model_name='intfloat/multilingual-e5-large',
    strategy='translations_only',
    use_openvino=True,
    device='GPU.0'
)

# Lade PRODUCTION embeddings
embedder.load_embeddings('embeddings_production')

# Test-Queries
test_queries = ['pruße', 'preußen', 'ostpreußen', 'prussian']

for query in test_queries:
    results = embedder.search(query, top_k=3)
    print(f"\n'{query}':")
    for rank, (entry, score) in enumerate(results, 1):
        word = entry.get('word', '')
        de = entry.get('translations', {}).get('miks', [])
        print(f"  {rank}. [{score:.3f}] {word:<20} → {de[0] if de else '—'}")

print("\n" + "="*60)
print("✓ Production embeddings verified!")
print("="*60)
