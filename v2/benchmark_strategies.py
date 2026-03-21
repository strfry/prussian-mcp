#!/usr/bin/env python3
"""
Benchmark verschiedener Text-Strategien für Embedding-Qualität.

Vergleicht:
1. Embedding-Generierungs-Geschwindigkeit
2. Such-Qualität (relevante Ergebnisse)
3. GPU-Auslastung
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np

from prussian_embeddings_optimized import PrussianEmbeddingsOptimized


# Test-Queries mit erwarteten Ergebnissen
TEST_QUERIES = {
    # Format: "query": ["expected_word_in_results", ...]
    "son": ["sūnulis", "sūnus"],  # Sohn
    "daughter": ["dūkti", "dukra"],  # Tochter
    "mother": ["mūti", "motina"],  # Mutter
    "father": ["tāws", "tėvas"],  # Vater
    "water": ["undan", "vanduo"],  # Wasser
    "light": ["swīgstan", "šviesa"],  # Licht
    "house": ["stūbā", "namas"],  # Haus
    "tree": ["garrin", "medis"],  # Baum
}


def evaluate_results(query: str, results: List[tuple], expected_words: List[str]) -> Dict:
    """Bewerte Suchergebnisse."""
    top_words = [entry.get("word", "").lower() for entry, score in results[:5]]

    # Wie viele erwartete Wörter sind in Top-5?
    hits = sum(1 for exp in expected_words if any(exp.lower() in w for w in top_words))

    # Durchschnittlicher Score
    avg_score = np.mean([score for _, score in results[:5]]) if results else 0.0

    # Beste Position eines erwarteten Wortes
    best_rank = None
    for i, (entry, score) in enumerate(results, 1):
        word = entry.get("word", "").lower()
        if any(exp.lower() in word for exp in expected_words):
            best_rank = i
            break

    return {
        "hits": hits,
        "total_expected": len(expected_words),
        "hit_rate": hits / len(expected_words) if expected_words else 0.0,
        "avg_score": avg_score,
        "best_rank": best_rank,
    }


def benchmark_strategy(
    strategy_name: str,
    dict_path: str = "prussian_dictionary.json",
    use_openvino: bool = True,
    device: str = "GPU.0",
    small_test: bool = False
):
    """Benchmark eine einzelne Strategie."""
    print(f"\n{'='*70}")
    print(f"STRATEGIE: {strategy_name.upper()}")
    print(f"{'='*70}\n")

    # Initialisiere
    embedder = PrussianEmbeddingsOptimized(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        strategy=strategy_name,
        use_openvino=use_openvino,
        device=device,
        batch_size=32
    )

    # Lade Wörterbuch (ggf. nur Subset für schnellen Test)
    print(f"📖 Lade Wörterbuch: {dict_path}")
    with open(dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = list(data.values())
    else:
        entries = data

    if small_test:
        # Nur 1000 Einträge für schnellen Test
        entries = entries[:1000]
        print(f"   ⚡ Small-Test-Modus: nur {len(entries)} Einträge")

    # Temporär als Datei speichern (für load_dictionary)
    temp_dict = f"temp_{strategy_name}.json"
    with open(temp_dict, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    # Generiere Embeddings
    start_gen = time.time()
    embedder.load_dictionary(temp_dict)
    gen_time = time.time() - start_gen

    # Cleanup
    Path(temp_dict).unlink()

    print(f"\n⏱️  Embedding-Generierung: {gen_time:.1f}s")
    print(f"   ({len(entries)/gen_time:.0f} entries/s)")

    # Teste Suchen
    print(f"\n🔍 Evaluiere Such-Qualität...\n")

    results_data = []

    for query, expected in TEST_QUERIES.items():
        # Suche
        start_search = time.time()
        results = embedder.search(query, top_k=10)
        search_time = (time.time() - start_search) * 1000  # ms

        # Evaluiere
        eval_result = evaluate_results(query, results, expected)

        results_data.append({
            "query": query,
            "search_ms": search_time,
            **eval_result
        })

        # Zeige Top-3
        print(f"  '{query}' ({search_time:.1f}ms):")
        for rank, (entry, score) in enumerate(results[:3], 1):
            word = entry.get("word", "?")
            marker = "✓" if any(exp.lower() in word.lower() for exp in expected) else " "
            print(f"    {marker} {rank}. [{score:.3f}] {word}")

    # Zusammenfassung
    print(f"\n📊 ZUSAMMENFASSUNG ({strategy_name})")
    print(f"   {'─'*60}")

    avg_hit_rate = np.mean([r["hit_rate"] for r in results_data])
    avg_score = np.mean([r["avg_score"] for r in results_data])
    avg_search_ms = np.mean([r["search_ms"] for r in results_data])

    best_ranks = [r["best_rank"] for r in results_data if r["best_rank"] is not None]
    avg_best_rank = np.mean(best_ranks) if best_ranks else None

    print(f"   Durchschnittliche Hit-Rate:    {avg_hit_rate*100:5.1f}%")
    print(f"   Durchschnittlicher Score:      {avg_score:.3f}")
    print(f"   Ø Beste Rank (wenn gefunden):  {avg_best_rank:.1f}" if avg_best_rank else "")
    print(f"   Ø Such-Zeit:                   {avg_search_ms:.1f}ms")
    print(f"   Embedding-Zeit:                {gen_time:.1f}s")

    return {
        "strategy": strategy_name,
        "avg_hit_rate": avg_hit_rate,
        "avg_score": avg_score,
        "avg_search_ms": avg_search_ms,
        "avg_best_rank": avg_best_rank,
        "embedding_time_sec": gen_time,
        "entries": len(entries),
        "results": results_data,
    }


def compare_all_strategies(
    dict_path: str = "prussian_dictionary.json",
    use_openvino: bool = True,
    device: str = "GPU.0",
    small_test: bool = False
):
    """Vergleiche alle verfügbaren Strategien."""
    strategies = ["simple", "sentences", "weighted", "clusters", "minimal"]

    print(f"\n{'#'*70}")
    print(f"# STRATEGIE-VERGLEICH")
    print(f"# OpenVINO: {use_openvino}, Device: {device}, Small-Test: {small_test}")
    print(f"{'#'*70}\n")

    all_results = []

    for strategy in strategies:
        try:
            result = benchmark_strategy(
                strategy,
                dict_path=dict_path,
                use_openvino=use_openvino,
                device=device,
                small_test=small_test
            )
            all_results.append(result)

        except Exception as e:
            print(f"\n❌ Fehler bei Strategie '{strategy}': {e}")
            import traceback
            traceback.print_exc()

    # Finale Vergleichstabelle
    print(f"\n\n{'='*70}")
    print(f"FINALE ERGEBNISSE - STRATEGIE-RANKING")
    print(f"{'='*70}\n")

    # Sortiere nach Hit-Rate
    all_results.sort(key=lambda x: x["avg_hit_rate"], reverse=True)

    print(f"{'Rang':<6} {'Strategie':<12} {'Hit-Rate':<12} {'Ø Score':<10} {'Ø Zeit (ms)':<12}")
    print(f"{'-'*70}")

    for rank, result in enumerate(all_results, 1):
        print(f"{rank:<6} {result['strategy']:<12} "
              f"{result['avg_hit_rate']*100:>5.1f}%       "
              f"{result['avg_score']:>6.3f}    "
              f"{result['avg_search_ms']:>6.1f}ms")

    # Speichere detaillierte Ergebnisse
    output_file = "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detaillierte Ergebnisse: {output_file}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Embedding-Strategien")
    parser.add_argument(
        "--dict",
        default="prussian_dictionary.json",
        help="Pfad zum Wörterbuch-JSON"
    )
    parser.add_argument(
        "--no-openvino",
        action="store_true",
        help="Deaktiviere OpenVINO (nutze CPU)"
    )
    parser.add_argument(
        "--device",
        default="GPU.0",
        help="OpenVINO Device (GPU.0, CPU, ...)"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Small-Test (nur 1000 Einträge für schnellen Test)"
    )
    parser.add_argument(
        "--strategy",
        choices=["simple", "sentences", "weighted", "clusters", "minimal", "all"],
        default="all",
        help="Nur eine bestimmte Strategie testen"
    )

    args = parser.parse_args()

    # Prüfe ob Wörterbuch existiert
    if not Path(args.dict).exists():
        print(f"❌ Wörterbuch nicht gefunden: {args.dict}")
        exit(1)

    use_openvino = not args.no_openvino

    if args.strategy == "all":
        # Alle Strategien vergleichen
        compare_all_strategies(
            dict_path=args.dict,
            use_openvino=use_openvino,
            device=args.device,
            small_test=args.small
        )
    else:
        # Nur eine Strategie
        benchmark_strategy(
            args.strategy,
            dict_path=args.dict,
            use_openvino=use_openvino,
            device=args.device,
            small_test=args.small
        )
