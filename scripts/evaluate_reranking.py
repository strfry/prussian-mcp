#!/usr/bin/env python3
"""Evaluate reranking performance - comparing embedding+rerank vs pure rerank."""

import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prussian_engine.rerank_search import RerankedSearchEngine
from prussian_engine.search import SearchEngine
import httpx
import json
import re

RERANKER_URL = "http://localhost:8001/v3/rerank"


def get_word_type(entry: dict) -> str:
    desc = entry.get("desc", "")
    if desc:
        match = re.match(r"^\s*(\w+)", desc)
        if match:
            return match.group(1).lower()
    return ""


WORD_TYPE_WEIGHTS = {
    "pn": 5.0,
    "prp": 4.5,
    "crd": 4.0,
    "ord": 4.0,
    "av": 3.5,
    "aj": 3.0,
    "nom": 3.0,
    "n": 3.0,
    "subst": 3.0,
    "pc": 2.5,
    "IJ": 0.1,
    "ij": 0.1,
}


# Basic vocabulary test queries - WITHOUT meta-terms like "pronouns"
TEST_QUERIES = {
    "Pronomen": {
        "query": "ich du er sie wir ihr I you he she we you",
        "expected": ["aīns", "as", "tu", "ten", "mes", "jūs"],
    },
    "Familie": {
        "query": "family father mother brother sister child Familie Vater Mutter Bruder Schwester Kind šeima tēvs māte brālis māsa bērns rodzina ojciec matka brat siostra dziecko семья отец мать брат сестра ребёнок",
        "expected": ["sēimī", "tāws", "māti", "brāti", "sestrā"],
    },
    "Zahlen": {
        "query": "one two three four five six seven eight nine ten eins zwei drei vier fünf sechs sieben acht neun zehn viens du trī četri pieci seši septiņi astoņi deviņi desmit jeden dwa trzy cztery pięć sześć siedem osiem dziewięć dziesięć один два три четыре пять шесть семь восемь девять десять",
        "expected": [
            "aīns",
            "dwēita",
            "trīs",
            "ketturjai",
            "pēnkjai",
            "ussjai",
            "septinnjai",
            "astōnjai",
            "newīnjai",
            "dessimts",
        ],
    },
    "Körper": {
        "query": "body head hand foot eye ear nose mouth Körper Kopf Hand Fuß Auge Ohr Nase Mund kūnas galva ranka pēda akis auss deguns mute ciało głowa ręka stopa oko ucho nos usta тело голова рука нога глаз ухо нос рот",
        "expected": [
            "kīrs",
            "galwā",
            "rānkan",
            "pēdan",
            "aks",
            "auss",
            "nāsī",
            "amūtan",
        ],
    },
    "Essen": {
        "query": "bread water meat fish eat drink Brot Wasser Fleisch Fisch essen trinken duona vanduo mėsa žuvis valgyti gerti chleb woda mięso ryba jeść pić хлеб вода мясо рыба есть пить",
        "expected": ["geītka", "wundan", "mēnsa", "ēstun", "pūtun"],
    },
    "Haus": {
        "query": "house home door window table chair bed Haus Heim Tür Fenster Tisch Stuhl Bett namai durys langas stalas kėdė lova dom drzwi okno stół krzesło łóżko дом дверь окно стол стул кровать",
        "expected": ["buttan", "wārtā", "langstā", "minsa", "klūmpis"],
    },
    "Präpositionen": {
        "query": "in on with for under in auf mit für unter ķiņš ar priekš no uz w na z za od do pod в на с для под",
        "expected": ["ēn", "na", "sēn", "pēr", "pa", "nō", "sen"],
    },
}


async def pure_rerank(query: str, entries: list, top_k: int = 20) -> list:
    """Rerank without embedding - directly on all entries."""
    combined_scores = {}

    docs = []
    for e in entries:
        word = e.get("word", "")
        wt = get_word_type(e)
        de = (
            e.get("translations", {}).get("miks", [""])[0]
            if e.get("translations", {}).get("miks")
            else ""
        )
        en = (
            e.get("translations", {}).get("engl", [""])[0]
            if e.get("translations", {}).get("engl")
            else ""
        )
        lt = (
            e.get("translations", {}).get("leit", [""])[0]
            if e.get("translations", {}).get("leit")
            else ""
        )
        lv = (
            e.get("translations", {}).get("latt", [""])[0]
            if e.get("translations", {}).get("latt")
            else ""
        )
        docs.append(f"{word} [{wt}]: {de} | {en} | {lt} | {lv}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        for batch_idx in range(0, len(entries), 50):
            batch = entries[batch_idx : batch_idx + 50]
            batch_docs = docs[batch_idx : batch_idx + 50]

            response = await client.post(
                RERANKER_URL,
                json={
                    "model": "BAAI/bge-reranker-large",
                    "query": query,
                    "documents": batch_docs,
                },
            )

            if response.status_code != 200:
                continue

            results = response.json().get("results", [])
            for item in results:
                idx = item.get("index", 0) + batch_idx
                score = item.get("relevance_score", 0)
                bonus = WORD_TYPE_WEIGHTS.get(get_word_type(entries[idx]), 1.0)
                combined_scores[idx] = combined_scores.get(idx, 0) + score * bonus

    sorted_indices = sorted(
        combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True
    )
    return [
        {
            "word": entries[i]["word"],
            "de": entries[i].get("translations", {}).get("miks", [""])[0],
            "rerank_score": combined_scores[i],
        }
        for i in sorted_indices[:top_k]
    ]


def evaluate_results(results: list, expected: list) -> dict:
    """Check how many expected terms are in results."""
    found = []
    for exp in expected:
        for r in results:
            if (
                r["word"].lower().startswith(exp.lower())
                or exp.lower() in r["word"].lower()
            ):
                found.append(exp)
                break
    return {"found": found, "precision": len(found) / len(expected) if expected else 0}


async def main():
    print("Loading search engines...")
    base_engine = SearchEngine()
    reranked_engine = RerankedSearchEngine(use_reranker=True)

    # Load all entries for pure reranking
    entries_file = Path("embeddings/embeddings_qwen.entries.json")
    with open(entries_file, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    entries_with_trans = [
        e
        for e in all_entries
        if e.get("translations", {}).get("miks")
        or e.get("translations", {}).get("engl")
    ]

    print("\n" + "=" * 90)
    print("COMPARING: Embedding-only vs Embedding+Rerank vs Pure Rerank")
    print("=" * 90)

    results_table = []

    for category, test_data in TEST_QUERIES.items():
        print(f"\n{category}")
        print("-" * 70)

        query = test_data["query"]
        expected = test_data["expected"]

        # 1. Embedding only
        start = time.time()
        emb_results = base_engine.query(query, top_k=20)
        emb_time = time.time() - start
        emb_eval = evaluate_results(emb_results, expected)

        # 2. Embedding + Rerank
        start = time.time()
        emb_rerank_results = await reranked_engine.search(
            query, top_k=20, rerank_candidates=100
        )
        emb_rerank_time = time.time() - start
        emb_rerank_eval = evaluate_results(emb_rerank_results, expected)

        # 3. Pure Rerank (all 9700 entries)
        start = time.time()
        pure_rerank_results = await pure_rerank(query, entries_with_trans, top_k=20)
        pure_rerank_time = time.time() - start
        pure_rerank_eval = evaluate_results(pure_rerank_results, expected)

        print(f"{'Method':<20} {'Precision':>10} {'Time':>10} {'Top 3':<35}")
        print("-" * 70)

        for name, results, prec, tim in [
            ("Embedding", emb_results, emb_eval["precision"], emb_time),
            (
                "Embed+Rerank",
                emb_rerank_results,
                emb_rerank_eval["precision"],
                emb_rerank_time,
            ),
            (
                "Pure Rerank",
                pure_rerank_results,
                pure_rerank_eval["precision"],
                pure_rerank_time,
            ),
        ]:
            top3 = ", ".join(r["word"][:12] for r in results[:3])
            print(f"{name:<20} {prec:>9.0%} {tim:>9.2f}s {top3[:35]}")

        results_table.append(
            {
                "category": category,
                "emb_prec": emb_eval["precision"],
                "emb_time": emb_time,
                "emb_rerank_prec": emb_rerank_eval["precision"],
                "emb_rerank_time": emb_rerank_time,
                "pure_rerank_prec": pure_rerank_eval["precision"],
                "pure_rerank_time": pure_rerank_time,
            }
        )

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Category':<12} {'Emb Prec':>9} {'Emb+Rerank':>11} {'Pure Rerank':>12}")
    print("-" * 50)

    avg_emb = sum(r["emb_prec"] for r in results_table) / len(results_table)
    avg_emb_rerank = sum(r["emb_rerank_prec"] for r in results_table) / len(
        results_table
    )
    avg_pure = sum(r["pure_rerank_prec"] for r in results_table) / len(results_table)

    for r in results_table:
        print(
            f"{r['category']:<12} {r['emb_prec']:>8.0%} {r['emb_rerank_prec']:>10.0%} {r['pure_rerank_prec']:>11.0%}"
        )

    print("-" * 50)
    print(f"{'AVERAGE':<12} {avg_emb:>8.0%} {avg_emb_rerank:>10.0%} {avg_pure:>11.0%}")


if __name__ == "__main__":
    asyncio.run(main())
