#!/usr/bin/env python3
"""
Schneller Test: Zeige wie verschiedene Strategien Texte generieren.
Demonstriert den Unterschied zwischen den Ansätzen.
"""

import json
from embedding_strategies import STRATEGIES

# Beispiel-Eintrag aus dem Wörterbuch
example_entry = {
    "word": "sūnulis",
    "paradigm": "32",
    "gender": "masc",
    "desc": "[Sunulis 119 drv]",
    "translations": {
        "engl": ["son", "little son"],
        "miks": ["Sohn", "Söhnchen"],
        "leit": ["sūnus", "sūnelis"],
        "latt": ["dēls"],
        "pols": ["syn"],
        "mask": ["сын"]
    }
}

print("╔════════════════════════════════════════════════════════════╗")
print("║  Strategie-Vergleich: Text-Generierung                    ║")
print("╚════════════════════════════════════════════════════════════╝")
print()
print("Beispiel-Eintrag:")
print(f"  Wort: {example_entry['word']}")
print(f"  Übersetzungen: {len([t for ts in example_entry['translations'].values() for t in ts])} in {len(example_entry['translations'])} Sprachen")
print()
print("─" * 70)
print()

for strategy_name, strategy_class in STRATEGIES.items():
    strategy = strategy_class()
    text = strategy.generate(example_entry)

    print(f"📝 {strategy_name.upper()}")
    print(f"   Länge: {len(text)} Zeichen, {len(text.split())} Tokens")
    print()

    # Zeige ersten Teil (max 200 Zeichen)
    if len(text) <= 200:
        print(f"   {text}")
    else:
        print(f"   {text[:200]}...")

    print()
    print("─" * 70)
    print()

print("\n💡 Analyse:")
print()
print("• simple:     Reine Konkatenation, kann Struktur-Artefakte enthalten")
print("• sentences:  Natürliche Sätze, lernt aber auch 'In English:', etc.")
print("• weighted:   ⭐ EMPFOHLEN - Gewichtete Wörter ohne Struktur")
print("• clusters:   Semantische Gruppen, könnte Cluster-Grenzen lernen")
print("• minimal:    Absolut minimal, normalisiert und dedupliziert")
print()
print("Für semantische Suche: 'weighted' ist meist optimal!")
print()
