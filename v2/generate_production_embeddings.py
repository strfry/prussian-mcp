#!/usr/bin/env python3
"""Generiere Production-Embeddings mit bestem Setup."""

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("="*60)
print("Generating Production Embeddings")
print("="*60)
print("Model: intfloat/multilingual-e5-large")
print("Strategy: translations_only")
print("="*60)

embedder = PrussianEmbeddingsOptimized(
    model_name='intfloat/multilingual-e5-large',
    strategy='translations_only',
    use_openvino=True,
    device='GPU.0'
)

embedder.load_dictionary('prussian_dictionary.json')
embedder.save_embeddings('embeddings_production')

print("\n" + "="*60)
print("✓ Production embeddings saved!")
print("="*60)
print("Files:")
print("  - embeddings_production.embeddings.npy")
print("  - embeddings_production.entries.json")
print("  - embeddings_production.meta.json")
print("="*60)
