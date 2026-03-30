#!/usr/bin/env python3
"""Generiere Embeddings über den lokalen Embedding-Server (/embeddings API)."""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prussian_engine.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    PASSAGE_PREFIX,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    DICTIONARY_PATH,
    EMBEDDINGS_PATH,
)
from openai import OpenAI

BATCH_SIZE = 64

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
    """Prüfe ob Eintrag Übersetzungen hat (nicht nur Verweis)."""
    translations = entry.get("translations", {})
    return any(
        isinstance(trans_list, list) and len(trans_list) > 0
        for trans_list in translations.values()
    )


def make_passage(entry: dict) -> str:
    """
    Erzeuge Embedding-Passage als Liste der Quellsprachen.

    Format: "English: House, Home; German: Haus; Lithuanian: Namas Namai; ..."
    """
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


print("=" * 60)
print("Generating Embeddings via Model Server")
print("=" * 60)
print(f"Model:    {EMBEDDING_MODEL}")
print(f"Server:   {OPENAI_BASE_URL}")
print(f"Strategy: translations_only")
print(f"Prefix:   '{PASSAGE_PREFIX}'")
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
    text = PASSAGE_PREFIX + make_passage(entry)
    texts.append(text)

print(f"  {len(texts)} texts prepared")
print(f"  Example: {texts[0][:120]}...")

# Connect to embedding server
client = OpenAI(
    api_key=OPENAI_API_KEY or "dummy",
    base_url=OPENAI_BASE_URL,
)

# Generate embeddings in batches
print(f"\nGenerating embeddings...")
start = time.time()
all_embeddings = []
num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=batch,
    )
    batch_embeddings = [item.embedding for item in response.data]
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

if embeddings.shape[1] != EMBEDDING_DIM:
    print(f"  WARNING: Expected dim={EMBEDDING_DIM}, got {embeddings.shape[1]}")
    print(f"  Update EMBEDDING_DIM in config.py!")

# Save
output_path = str(EMBEDDINGS_PATH)
print(f"\nSaving to: {output_path}")

np.save(f"{output_path}.embeddings.npy", embeddings)

with open(f"{output_path}.entries.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

metadata = {
    "model": EMBEDDING_MODEL,
    "strategy": "translations_only",
    "num_entries": len(entries),
    "embedding_dim": int(embeddings.shape[1]),
    "server": OPENAI_BASE_URL,
}

with open(f"{output_path}.meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Saved {len(entries)} embeddings ({embeddings.shape[1]}d)")
print(f"  - {output_path}.embeddings.npy")
print(f"  - {output_path}.entries.json")
print(f"  - {output_path}.meta.json")
print("=" * 60)
