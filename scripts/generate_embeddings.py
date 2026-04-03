#!/usr/bin/env python3
"""Generate embeddings via embedding API."""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prussian_engine.config import (
    RERANK_API_KEY,
    RERANK_EMBEDDING_MODEL,
    RERANK_EMBEDDING_DIM,
    RERANK_BASE_URL,
    DICTIONARY_PATH,
    EMBEDDINGS_PATH,
)
from prussian_engine.embedding_client import EmbeddingClient

BATCH_SIZE = 32

LANGUAGE_NAMES = {
    "engl": "English",
    "miks": "German",
    "leit": "Lithuanian",
    "latt": "Latvian",
    "pols": "Polish",
    "mask": "Russian",
}

LANGUAGE_ORDER = ["engl", "miks", "leit", "latt", "pols", "mask"]


def should_include_entry(entry: dict) -> bool:
    """Check if entry has translations (not just a reference)."""
    translations = entry.get("translations", {})
    return any(
        isinstance(trans_list, list) and len(trans_list) > 0
        for trans_list in translations.values()
    )


def make_passage(entry: dict) -> str:
    """Create embedding passage from translations."""
    parts = []

    translations = entry.get("translations", {})

    for lang_code in LANGUAGE_ORDER:
        if lang_code in translations:
            trans_list = translations[lang_code]
            if isinstance(trans_list, list) and trans_list:
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                words = ", ".join(str(t) for t in trans_list if t)
                if words:
                    parts.append(f"{lang_name}: {words}")

    return "; ".join(parts)


def main():
    if not RERANK_API_KEY:
        print("ERROR: RERANK_API_KEY environment variable is required")
        sys.exit(1)

    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60)
    print(f"Model:    {RERANK_EMBEDDING_MODEL}")
    print(f"API:      {RERANK_BASE_URL}")
    print(f"Dim:      {RERANK_EMBEDDING_DIM}")
    print(f"Strategy: translations_only")
    print(f"Batch:    {BATCH_SIZE}")
    print("=" * 60)

    # Load dictionary
    print(f"\nLoading dictionary: {DICTIONARY_PATH}")
    with open(DICTIONARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = list(data.values())
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Expected list or dict")

    original_count = len(entries)
    entries = [e for e in entries if should_include_entry(e)]
    print(f"  {original_count} -> {len(entries)} entries (filtered references)")

    # Generate text representations
    texts = []
    for entry in entries:
        text = make_passage(entry)
        texts.append(text)

    print(f"  {len(texts)} texts prepared")
    print(f"  Example: {texts[0][:120]}...")

    # Connect to embedding API
    client = EmbeddingClient()

    # Generate embeddings in batches
    print(f"\nGenerating embeddings...")
    start = time.time()
    all_embeddings = []
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        try:
            batch_embeddings = client.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"\n  Error on batch {i // BATCH_SIZE + 1}: {e}")
            all_embeddings.extend([np.zeros(RERANK_EMBEDDING_DIM)] * len(batch))

        batch_num = (i // BATCH_SIZE) + 1
        pct = (batch_num / num_batches) * 100
        elapsed = time.time() - start
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        print(
            f"  [{pct:5.1f}%] Batch {batch_num}/{num_batches} ({len(all_embeddings)} embeddings, {rate:.0f}/s)",
            end="\r",
        )

    print()
    elapsed = time.time() - start
    print(f"  Done: {elapsed:.1f}s ({len(texts) / elapsed:.0f} entries/s)")

    # Convert to numpy
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"  Shape: {embeddings.shape}")

    if embeddings.shape[1] != RERANK_EMBEDDING_DIM:
        print(
            f"  WARNING: Expected dim={RERANK_EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )

    # Save
    output_path = str(EMBEDDINGS_PATH)
    print(f"\nSaving to: {output_path}")

    np.save(f"{output_path}.embeddings.npy", embeddings)

    with open(f"{output_path}.entries.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    metadata = {
        "model": RERANK_EMBEDDING_MODEL,
        "provider": RERANK_BASE_URL,
        "strategy": "translations_only",
        "num_entries": len(entries),
        "embedding_dim": int(embeddings.shape[1]),
    }

    with open(f"{output_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(entries)} embeddings ({embeddings.shape[1]}d)")
    print(f"  - {output_path}.embeddings.npy")
    print(f"  - {output_path}.entries.json")
    print(f"  - {output_path}.meta.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
