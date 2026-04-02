#!/usr/bin/env python3
"""Rerank dictionary entries by query relevance."""

import httpx
import asyncio
import json
import re
import sys
import time
from pathlib import Path


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

DEFAULT_QUERY = (
    "Pronomen pronouns ich du er sie wir ihr personal basic vocabulary essential"
)


def get_word_type(entry: dict) -> str:
    desc = entry.get("desc", "")
    if desc:
        match = re.match(r"^\s*(\w+)", desc)
        if match:
            return match.group(1).lower()
    return ""


def format_entry(entry: dict) -> str:
    word = entry.get("word", "")
    word_type = get_word_type(entry)
    gender = entry.get("gender", "")
    de = (
        entry.get("translations", {}).get("miks", [""])[0]
        if entry.get("translations", {}).get("miks")
        else ""
    )
    en = (
        entry.get("translations", {}).get("engl", [""])[0]
        if entry.get("translations", {}).get("engl")
        else ""
    )

    parts = [word]
    if word_type:
        parts.append(f"[{word_type}]")
    if gender:
        parts.append(f"({gender})")
    parts.append(f": {de}")
    if en:
        parts.append(f"| {en}")

    return " ".join(parts)


async def rerank(
    client: httpx.AsyncClient, query: str, entries: list[dict], batch_size: int = 32
) -> list[dict]:
    combined_scores: dict[int, float] = {}
    total_batches = (len(entries) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(entries), batch_size):
        batch_num = batch_idx // batch_size + 1
        batch = entries[batch_idx : batch_idx + batch_size]
        documents = [format_entry(e) for e in batch]

        start = time.time()
        response = await client.post(
            "http://localhost:8001/v3/rerank",
            json={
                "model": "BAAI/bge-reranker-large",
                "query": query,
                "documents": documents,
            },
            timeout=60.0,
        )
        elapsed = time.time() - start

        if response.status_code != 200:
            print(
                f"Batch {batch_num}/{total_batches}: ERROR {response.status_code}",
                file=sys.stderr,
            )
            continue

        results = response.json().get("results", [])
        for item in results:
            idx = item.get("index", 0) + batch_idx
            score = item.get("relevance_score", 0)
            combined_scores[idx] = combined_scores.get(idx, 0) + score

        print(
            f"Batch {batch_num}/{total_batches}: {len(batch)} docs, {elapsed:.2f}s",
            file=sys.stderr,
        )

    for i, entry in enumerate(entries):
        word_type = get_word_type(entry)
        bonus = WORD_TYPE_WEIGHTS.get(word_type, 1.0)
        if i in combined_scores:
            combined_scores[i] *= bonus

    sorted_indices = sorted(
        combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True
    )

    result = []
    for idx in sorted_indices:
        entry = entries[idx].copy()
        entry["rerank_score"] = combined_scores[idx]
        entry["word_type"] = get_word_type(entries[idx])
        result.append(entry)

    return result


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rerank dictionary entries by query relevance"
    )
    parser.add_argument(
        "query", nargs="?", default=DEFAULT_QUERY, help="Search query for reranking"
    )
    parser.add_argument(
        "--top-n", "-n", type=int, default=30, help="Number of top results to show"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/reranked_dictionary.json",
        help="Output file",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for reranking"
    )
    args = parser.parse_args()

    entries_file = Path("embeddings/embeddings_qwen.entries.json")
    print(f"Loading entries from {entries_file}...", file=sys.stderr)

    with open(entries_file, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    entries_with_translations = [
        e
        for e in all_entries
        if e.get("translations", {}).get("miks")
        or e.get("translations", {}).get("engl")
    ]

    print(
        f"Processing {len(entries_with_translations)} entries with query: {args.query}",
        file=sys.stderr,
    )

    start_total = time.time()

    async with httpx.AsyncClient() as client:
        reranked = await rerank(
            client, args.query, entries_with_translations, batch_size=args.batch_size
        )

    total_time = time.time() - start_total
    print(
        f"\nTotal: {total_time:.2f}s ({len(entries_with_translations)} entries)",
        file=sys.stderr,
    )

    print(f"\n=== Top {args.top_n} for: {args.query} ===")
    for i, entry in enumerate(reranked[: args.top_n], 1):
        word = entry.get("word", "")
        word_type = entry.get("word_type", "")
        gender = entry.get("gender", "")
        de = (
            entry.get("translations", {}).get("miks", [""])[0]
            if entry.get("translations", {}).get("miks")
            else ""
        )
        score = entry.get("rerank_score", 0)
        meta = f"{word_type} {gender}".strip()
        print(f"{i:2d}. {word:20s} [{meta:10s}] {de[:40]:40s} | {score:.4f}")

    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reranked, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(reranked)} entries to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
