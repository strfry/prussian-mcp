#!/usr/bin/env python3
"""Generiere Embeddings mit E5-Präfixen ("passage: ...")."""

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("="*60)
print("Generating E5-Prefixed Embeddings")
print("="*60)
print("Model: intfloat/multilingual-e5-large")
print("Strategy: translations_only + E5 prefix")
print("Prefix: 'passage: ' for all entries")
print("="*60)

embedder = PrussianEmbeddingsOptimized(
    model_name='intfloat/multilingual-e5-large',
    strategy='translations_only',
    use_openvino=True,
    device='GPU.0',
    use_e5_prefix=True  # Enable E5 "passage: " prefix
)

embedder.load_dictionary('prussian_dictionary.json')
embedder.save_embeddings('embeddings_e5_prefix')

print("\n" + "="*60)
print("✓ E5-prefixed embeddings saved!")
print("="*60)
print("Files:")
print("  - embeddings_e5_prefix.embeddings.npy")
print("  - embeddings_e5_prefix.entries.json")
print("  - embeddings_e5_prefix.meta.json")
print("="*60)
print("\nNext: Test with 'query: ' prefix in search")
print("="*60)
