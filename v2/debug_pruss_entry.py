#!/usr/bin/env python3
"""Debug: Warum findet die Suche prūss nicht?"""

import json
from embedding_strategies import get_strategy

# Lade Dictionary
with open('prussian_dictionary.json', 'r') as f:
    data = json.load(f)

# Finde prūss
pruss_entry = None
for entry in (data if isinstance(data, list) else data.values()):
    if entry.get('word') == 'prūss':
        pruss_entry = entry
        break

if not pruss_entry:
    print("ERROR: prūss entry not found!")
    exit(1)

print("="*60)
print("Entry: prūss")
print("="*60)
print(f"Translations: {pruss_entry.get('translations', {})}")
print()

print("="*60)
print("Text Representations per Strategy")
print("="*60)

for strategy_name in ['simple', 'sentences', 'weighted', 'clusters', 'minimal']:
    strategy = get_strategy(strategy_name)
    text = strategy.generate(pruss_entry)

    print(f"\n{strategy_name.upper()}:")
    print(f"  Length: {len(text)} chars")
    print(f"  Text: {text}")
    print()

print("\n" + "="*60)
print("Testing if 'pruße' appears in texts")
print("="*60)

for strategy_name in ['simple', 'sentences', 'weighted', 'clusters', 'minimal']:
    strategy = get_strategy(strategy_name)
    text = strategy.generate(pruss_entry)

    # Check verschiedene Varianten
    checks = {
        'pruße (lowercase)': 'pruße' in text.lower(),
        'pruss': 'pruss' in text.lower(),
        'prussian': 'prussian' in text.lower(),
        'prūss': 'prūss' in text.lower()
    }

    print(f"\n{strategy_name}:")
    for check_name, found in checks.items():
        print(f"  {check_name}: {'✓' if found else '✗'}")
