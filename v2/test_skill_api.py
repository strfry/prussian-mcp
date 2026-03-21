#!/usr/bin/env python3
"""Test der Python API des Prussian Search Skills."""

from prussian_search_skill import PrussianSearch

print("="*70)
print("PYTHON API TEST - Prussian Search Skill")
print("="*70)
print()

# Initialisiere
print("1. Initialisiere Search Skill...")
search = PrussianSearch()
print("   ✓ Geladen\n")

# Statistiken
print("2. Statistiken:")
stats = search.get_stats()
for key, value in stats.items():
    print(f"   {key:20s}: {value}")
print()

# Einzelne Suche
print("3. Einzelne Suche: 'forest'")
results = search.query("forest", top_k=3)
for i, result in enumerate(results, 1):
    word = result['word']
    score = result['score']
    trans = search.get_translation_summary(result)
    print(f"   {i}. [{score:.3f}] {word:20s} = {trans}")
print()

# Batch-Suche
print("4. Batch-Suche: ['god', 'sky', 'earth']")
batch_results = search.batch_query(["god", "sky", "earth"], top_k=2)
for query, results in batch_results.items():
    print(f"   '{query}':")
    for r in results:
        print(f"      - {r['word']:15s} ({r['score']:.3f})")
print()

# Sprachfilter
print("5. Suche mit Sprachfilter (nur Englisch & Deutsch): 'water'")
results = search.query("water", top_k=3, lang_filter=["engl", "miks"])
for i, result in enumerate(results, 1):
    word = result['word']
    engl = result.get('translations', {}).get('engl', [])
    miks = result.get('translations', {}).get('miks', [])
    print(f"   {i}. {word:20s} EN: {engl}  DE: {miks}")
print()

print("="*70)
print("✅ API Test erfolgreich!")
print("="*70)
