#!/usr/bin/env python3
"""Test E5-prefixed embeddings vs. non-prefixed."""

from prussian_search_skill import PrussianSearch

print("="*60)
print("Comparing: E5 Prefix vs. No Prefix")
print("="*60)

# Test queries
queries = ['pruße', 'preußen', 'ostpreußen']

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")

    # Without prefix (old)
    print("\n  WITHOUT E5 prefix (old):")
    search_old = PrussianSearch(embeddings_path='embeddings_production')
    results_old = search_old.query(query, top_k=5)
    for i, r in enumerate(results_old, 1):
        word = r['word']
        score = r['score']
        de = r.get('translations', {}).get('miks', ['—'])[0]
        print(f"    {i}. [{score:.3f}] {word:<20} → {de}")

    # With prefix (new)
    print("\n  WITH E5 prefix (new):")
    search_new = PrussianSearch(embeddings_path='embeddings_e5_prefix')
    results_new = search_new.query(query, top_k=5)
    for i, r in enumerate(results_new, 1):
        word = r['word']
        score = r['score']
        de = r.get('translations', {}).get('miks', ['—'])[0]
        marker = ""
        # Highlight improvements
        if i <= 3 and ('prūss' in word.lower() or 'prusija' in word.lower()):
            marker = " ✓"
        print(f"    {i}. [{score:.3f}] {word:<20} → {de}{marker}")

    print()

print("="*60)
print("Summary: Check if 'puttera' (Brei) is gone from top results!")
print("="*60)
