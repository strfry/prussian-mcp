#!/usr/bin/env python3
"""Vergleiche alle Embedding-Strategien für deutsche Preußen-Suchen."""

import json
import numpy as np
from pathlib import Path
from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

# Test-Queries
GERMAN_QUERIES = [
    "preußen",
    "pruße",
    "ich bin ein preusse",
    "altpreußisch",
    "ostpreußen"
]

CONTROL_QUERIES = [
    "son",      # Englisch
    "water",    # Englisch
    "sūnus",    # Litauisch
]

# Erwartete Treffer (word oder relevante Teilstrings)
EXPECTED_HITS = {
    "preußen": ["prūss", "deināprusija", "prūsiskan"],
    "pruße": ["prūss", "prūsisks"],
    "ostpreußen": ["deināprusija"],
    "altpreußisch": ["prūsiskan", "prūss"],
}

STRATEGIES = ['simple', 'sentences', 'weighted', 'clusters', 'minimal']


def check_expected_hit(entry: dict, expected_terms: list) -> bool:
    """Check ob Eintrag einen erwarteten Term enthält."""
    word = entry.get('word', '').lower()

    # Check Hauptwort
    for term in expected_terms:
        if term.lower() in word:
            return True

    # Check Übersetzungen
    translations = entry.get('translations', {})
    for lang_trans in translations.values():
        if isinstance(lang_trans, list):
            for trans in lang_trans:
                trans_lower = str(trans).lower()
                for term in expected_terms:
                    if term.lower() in trans_lower:
                        return True

    return False


def evaluate_strategy(strategy_name: str) -> dict:
    """Generiere Embeddings und teste Queries."""
    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"{'='*60}")

    # Generiere Embeddings
    embedder = PrussianEmbeddingsOptimized(
        strategy=strategy_name,
        use_openvino=True,
        device='GPU.0'
    )
    embedder.load_dictionary('prussian_dictionary.json')

    results = {
        'strategy': strategy_name,
        'queries': {}
    }

    # Teste deutsche Queries
    print("\n  German Queries:")
    for query in GERMAN_QUERIES:
        search_results = embedder.search(query, top_k=5)

        # Check ob erwartete Treffer dabei
        expected = EXPECTED_HITS.get(query, [])
        found_ranks = []

        for rank, (entry, score) in enumerate(search_results, 1):
            if expected and check_expected_hit(entry, expected):
                found_ranks.append((rank, score))

        results['queries'][query] = {
            'top_result': search_results[0][0]['word'],
            'top_score': float(search_results[0][1]),
            'expected_found': len(found_ranks) > 0,
            'best_rank': found_ranks[0][0] if found_ranks else None,
            'best_score': float(found_ranks[0][1]) if found_ranks else None,
            'all_results': [
                {
                    'word': entry.get('word', '?'),
                    'score': float(score),
                    'translations': entry.get('translations', {})
                }
                for entry, score in search_results
            ]
        }

        print(f"\n  Query: '{query}'")
        print(f"    Top result: {search_results[0][0]['word']} ({search_results[0][1]:.3f})")
        if found_ranks:
            print(f"    ✓ Expected hit found at rank {found_ranks[0][0]} (score: {found_ranks[0][1]:.3f})")
        else:
            print(f"    ✗ Expected hit NOT in top 5")

    # Teste Control Queries
    print("\n  Control Queries:")
    for query in CONTROL_QUERIES:
        search_results = embedder.search(query, top_k=3)
        results['queries'][query] = {
            'top_result': search_results[0][0]['word'],
            'top_score': float(search_results[0][1]),
            'all_results': [
                {
                    'word': entry.get('word', '?'),
                    'score': float(score)
                }
                for entry, score in search_results
            ]
        }
        print(f"    '{query}' → {search_results[0][0]['word']} ({search_results[0][1]:.3f})")

    return results


def main():
    print("="*60)
    print("BENCHMARK: German Search Quality Comparison")
    print("="*60)

    all_results = []

    for strategy in STRATEGIES:
        results = evaluate_strategy(strategy)
        all_results.append(results)

    # Summary Report
    print(f"\n\n{'='*60}")
    print("SUMMARY: German Query Performance")
    print(f"{'='*60}\n")

    # Erstelle Vergleichstabelle
    print(f"{'Strategy':<15} {'Hits':<10} {'Avg Rank':<12} {'Avg Score':<12}")
    print("-" * 60)

    best_strategy = None
    best_score = 0

    for result in all_results:
        strategy = result['strategy']
        queries = result['queries']

        # Nur deutsche Queries für Score
        german_results = {k: v for k, v in queries.items() if k in GERMAN_QUERIES}

        hits = sum(1 for q in german_results.values() if q['expected_found'])
        ranks = [q['best_rank'] for q in german_results.values() if q['best_rank']]
        scores = [q['best_score'] for q in german_results.values() if q['best_score']]

        avg_rank = np.mean(ranks) if ranks else None
        avg_score = np.mean(scores) if scores else None

        print(f"{strategy:<15} {hits}/{len(GERMAN_QUERIES):<10} "
              f"{avg_rank if avg_rank else 'N/A':<12} "
              f"{avg_score if avg_score else 'N/A':<12}")

        # Track beste Strategie (basierend auf Hit-Rate und Score)
        if avg_score and avg_score > best_score:
            best_score = avg_score
            best_strategy = strategy

    # Empfehlung
    print(f"\n{'='*60}")
    if best_strategy:
        print(f"RECOMMENDATION: Use strategy '{best_strategy}'")
        print(f"  - Best average score: {best_score:.3f}")
    else:
        print("WARNING: No strategy found expected hits consistently!")
    print(f"{'='*60}")

    # Save results
    output_file = 'strategy_comparison.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
