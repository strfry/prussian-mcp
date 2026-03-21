#!/usr/bin/env python3
"""
Quick test to verify the embedding filter works correctly.
"""

import json
from embedding_strategies import should_include_entry

# Load dictionary
with open('prussian_dictionary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

entries = list(data.values()) if isinstance(data, dict) else data

print(f"📊 Analysiere {len(entries)} Einträge...\n")

# Categorize entries
with_translations = []
without_translations = []
examples = {"with": [], "without": []}

for entry in entries:
    if should_include_entry(entry):
        with_translations.append(entry)
        if len(examples["with"]) < 3:
            examples["with"].append(entry)
    else:
        without_translations.append(entry)
        if len(examples["without"]) < 3:
            examples["without"].append(entry)

# Report
total = len(entries)
keep_count = len(with_translations)
remove_count = len(without_translations)

print(f"✅ MIT Übersetzungen:     {keep_count:5d} ({keep_count/total*100:.1f}%) ← BEHALTEN")
print(f"❌ OHNE Übersetzungen:    {remove_count:5d} ({remove_count/total*100:.1f}%) ← ENTFERNEN")
print(f"📦 Gesamt:                {total:5d}\n")

print(f"💾 Einsparung: ~{remove_count} Einträge, ~{remove_count/total*100:.1f}% weniger Embeddings\n")

# Show examples
print("=" * 60)
print("Beispiele für BEHALTENE Einträge (mit Übersetzungen):")
print("=" * 60)
for i, entry in enumerate(examples["with"], 1):
    word = entry.get('word', '?')
    trans = entry.get('translations', {})
    print(f"\n{i}. {word}")
    for lang, words in trans.items():
        if words:
            print(f"   {lang}: {', '.join(str(w) for w in words[:3])}")

print("\n" + "=" * 60)
print("Beispiele für ENTFERNTE Einträge (ohne Übersetzungen):")
print("=" * 60)
for i, entry in enumerate(examples["without"], 1):
    word = entry.get('word', '?')
    desc = entry.get('desc', '')
    trans = entry.get('translations', {})
    print(f"\n{i}. {word}")
    print(f"   desc: {desc}")
    print(f"   translations: {trans}")
