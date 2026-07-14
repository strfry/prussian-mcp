#!/usr/bin/env python3
"""Generate embeddings with optimal format: Prussian word + 4 translations."""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prussian_engine.config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    API_BASE_URL,
    DICTIONARY_PATH,
    EMBEDDINGS_DIR,
    EMBEDDINGS_PATH,
    PASSAGE_PREFIX,
)
from prussian_engine.embedder import get_embedder

BATCH_SIZE = 256

LANGUAGE_ORDER = ["engl", "miks", "leit", "latt", "pols", "mask"]


def should_include_entry(entry: dict) -> bool:
    """Check if entry has translations."""
    translations = entry.get("translations", {})
    return any(
        isinstance(trans_list, list) and len(trans_list) > 0
        for trans_list in translations.values()
    )


def make_passage_with_prussian(entry: dict) -> str:
    """
    Generate embedding passage with Prussian word + translations.

    Format: "Document: buttan: Haus | house | namas namai | nms"
    """
    word = entry.get("word", "")
    translations = entry.get("translations", {})

    trans_parts = []
    for lang_code in LANGUAGE_ORDER[:4]:  # First 4 languages: EN, DE, LT, LV
        if lang_code in translations:
            trans_list = translations[lang_code]
            if isinstance(trans_list, list) and trans_list:
                trans_parts.append(trans_list[0])

    if not trans_parts:
        return ""

    return f"{PASSAGE_PREFIX}{word}: " + " | ".join(trans_parts)


def main():
    print("=" * 60)
    print("Generating Embeddings")
    print("=" * 60)
    print(f"Backend:  {EMBEDDING_BACKEND}")
    if EMBEDDING_BACKEND == "model2vec":
        print(f"Model:    {EMBEDDING_MODEL}")
    else:
        print(f"Model:    {EMBEDDING_MODEL}")
        print(f"API:      {API_BASE_URL}")
    print(f"Strategy: translations_only")
    print(f"Batch:    {BATCH_SIZE}")
    print("=" * 60)

    # Load the configured embedder (model2vec = local/CPU, api = remote)
    embedder = get_embedder()
    embedding_dim = embedder.dim
    print(f"Dim:      {embedding_dim}")

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
        text = make_passage_with_prussian(entry)
        if text:
            texts.append(text)

    print(f"  {len(texts)} texts prepared")
    print(f"  Example: {texts[0]}")
    print(f"  Example: {texts[100]}")
    print(f"  Example: {texts[1000]}")

    # Generate embeddings in batches
    print(f"\nGenerating embeddings...")
    start = time.time()
    all_embeddings = []
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embeddings = embedder.get_embeddings(batch)
        all_embeddings.extend(batch_embeddings)
        batch_num = (i // BATCH_SIZE) + 1
        pct = (batch_num / num_batches) * 100
        print(
            f"  [{pct:5.1f}%] Batch {batch_num}/{num_batches} ({len(all_embeddings)} embeddings)",
            end="\r",
        )

    print()
    elapsed = time.time() - start
    print(f"  Done: {elapsed:.1f}s ({len(texts) / elapsed:.0f} entries/s)")

    # Convert to numpy
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"  Shape: {embeddings.shape}")

    if embeddings.shape[1] != embedding_dim:
        print(f"  WARNING: Expected dim={embedding_dim}, got {embeddings.shape[1]}")

    # Filter entries to match texts
    filtered_entries = [e for e in entries if make_passage_with_prussian(e)]

    # Save
    output_path = str(EMBEDDINGS_PATH)
    print(f"\nSaving to: {output_path}")

    np.save(f"{output_path}.embeddings.npy", embeddings)

    with open(f"{output_path}.entries.json", "w", encoding="utf-8") as f:
        json.dump(filtered_entries, f, ensure_ascii=False, indent=2)

    metadata = {
        "backend": EMBEDDING_BACKEND,
        "model": EMBEDDING_MODEL,
        "provider": "local" if EMBEDDING_BACKEND == "model2vec" else API_BASE_URL,
        "strategy": "translations_only",
        "num_entries": len(filtered_entries),
        "embedding_dim": int(embeddings.shape[1]),
        "passage_prefix": PASSAGE_PREFIX,
    }

    with open(f"{output_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(filtered_entries)} embeddings ({embeddings.shape[1]}d)")
    print(f"  - {output_path}.embeddings.npy")
    print(f"  - {output_path}.entries.json")
    print(f"  - {output_path}.meta.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
