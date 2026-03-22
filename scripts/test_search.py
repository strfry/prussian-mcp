#!/usr/bin/env python3
"""Test script for semantic search."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prussian_engine import SearchEngine


def main():
    """Run search tests."""
    print("Loading search engine...")
    engine = SearchEngine()
    print()

    # Example searches
    queries = [
        ("Haus", 5),
        ("Gott", 5),
        ("Gruß Begrüßung Hallo", 5),
        ("house", 3),
    ]

    for query, top_k in queries:
        print(f"Search: '{query}' (top {top_k})")
        print("-" * 60)
        results = engine.query(query, top_k)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['word']:20s} — DE: {result['de']}")
            if result['en']:
                print(f"   {'':20s}    EN: {result['en']}")
            if 'score' in result:
                print(f"   {'':20s}    Score: {result['score']:.3f}")

        print()

    # Test lookup
    print("\nLookup tests:")
    print("-" * 60)

    words = ["semmē", "bēiti", "deiwas"]
    for word in words:
        print(f"\nLookup: '{word}'")
        results = engine.lookup(word)
        if results:
            for result in results:
                print(f"  {result['word']} — DE: {result['de']}, EN: {result['en']}")
                if result.get('paradigm'):
                    print(f"  Paradigm: {result['paradigm']}, Gender: {result.get('gender', 'N/A')}")
        else:
            print("  Not found")


if __name__ == "__main__":
    main()
