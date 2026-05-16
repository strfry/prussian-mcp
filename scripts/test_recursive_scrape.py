#!/usr/bin/env python3
"""Test recursive prefix scraping."""

import sys
sys.path.insert(0, 'scripts')

from scrape import search_prefix_recursive, entry_key, ALPHABET
import json

# Load existing wordlist to avoid duplicates
with open('data/wordlist.json', 'r') as f:
    wordlist = json.load(f)

existing = {entry_key(e) for e in wordlist}
existing_count = len(existing)

print(f"Existing wordlist: {len(wordlist)} entries")
print(f"\nTesting recursive scrape for prefix 'kw'...\n")

# Test recursive scrape
new_entries = search_prefix_recursive("kw", existing, max_depth=6)

print(f"\n{'='*60}")
print(f"Results:")
print(f"  New entries found: {len(new_entries)}")
print(f"  Total existing before: {existing_count}")
print(f"  Total existing after: {len(existing)}")

# Check if kwaitītun was found
kwaitītun_found = any(e['word'] == 'kwaitītun' for e in new_entries)
print(f"  kwaitītun found: {'✓ YES' if kwaitītun_found else '✗ NO'}")

if new_entries:
    print(f"\nFirst 10 new entries:")
    for e in new_entries[:10]:
        print(f"  - {e['word']} (paradigm: {e.get('paradigm', '')})")

    if kwaitītun_found:
        kwaitītun_entries = [e for e in new_entries if e['word'] == 'kwaitītun']
        print(f"\nkwaitītun details:")
        for e in kwaitītun_entries:
            print(f"  Word: {e['word']}")
            print(f"  Paradigm: {e.get('paradigm', '')}")
            print(f"  Desc: {e.get('desc', '')}")
            print(f"  Translations: {e.get('translations_engl', [])}")
