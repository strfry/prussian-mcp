#!/usr/bin/env python3
"""Generate embeddings with optimal format: Prussian word + 4 translations."""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prussian_engine.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    DICTIONARY_PATH,
    EMBEDDINGS_DIR,
    EMBEDDINGS_PATH,
    PASSAGE_PREFIX,
    RERANK_BASE_URL,
    RERANK_API_KEY,
)
from openai import OpenAI

BATCH_SIZE = 16

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

    Format: "buttan: Haus | house | namas namai | nms"
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


print("=" * 60)
print("Generating Embeddings WITH PRUSSIAN WORD")
print("=" * 60)
print(f"Model:    {EMBEDDING_MODEL}")
print(f"Server:   {RERANK_BASE_URL}")
print(f"Format:   'prefix + word: de | en | lt | lv'")
print(f"Prefix:   {PASSAGE_PREFIX!r}")
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
    text = make_passage_with_prussian(entry)
    if text:
        texts.append(text)

print(f"  {len(texts)} texts prepared")
print(f"  Example: {texts[0]}")
print(f"  Example: {texts[100]}")
print(f"  Example: {texts[1000]}")

# Connect to embedding server
client = OpenAI(
    api_key=RERANK_API_KEY or "dummy",
    base_url=RERANK_BASE_URL,
)

# Generate embeddings in batches (resumable)
CHECKPOINT_FILE = str(EMBEDDINGS_DIR / "embeddings_checkpoint.npy")
CHECKPOINT_META = str(EMBEDDINGS_DIR / "embeddings_checkpoint_meta.json")

# Resume from checkpoint if available
start_idx = 0
all_embeddings = []
if os.path.exists(CHECKPOINT_FILE) and os.path.exists(CHECKPOINT_META):
    with open(CHECKPOINT_META, "r") as f:
        cp_meta = json.load(f)
    if cp_meta.get("num_texts") == len(texts) and cp_meta.get("model") == EMBEDDING_MODEL:
        start_idx = cp_meta["completed"]
        all_embeddings = np.load(CHECKPOINT_FILE).tolist()
        print(f"\nResuming from checkpoint: {start_idx}/{len(texts)} already done")
    else:
        print(f"\nCheckpoint stale (different texts/model), starting fresh")

print(f"\nGenerating embeddings...")
start = time.time()
num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
MAX_CONSECUTIVE_ERRORS = 3
consecutive_errors = 0
last_good = len(all_embeddings)  # track position before errors started

for i in range(start_idx, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        consecutive_errors = 0
        last_good = len(all_embeddings)
    except Exception as e:
        consecutive_errors += 1
        print(f"\n  ERROR at batch {i // BATCH_SIZE + 1} (texts[{i}:{i+len(batch)}]):", file=sys.stderr)
        for j, t in enumerate(batch):
            print(f"    [{i+j}] {t!r}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)

        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            # Roll back to last successful position and save checkpoint
            all_embeddings = all_embeddings[:last_good]
            cp = np.array(all_embeddings, dtype=np.float32)
            np.save(CHECKPOINT_FILE, cp)
            with open(CHECKPOINT_META, "w") as f:
                json.dump({"completed": last_good, "num_texts": len(texts), "model": EMBEDDING_MODEL}, f)
            print(f"\n  {MAX_CONSECUTIVE_ERRORS} consecutive errors — saved checkpoint at {last_good}/{len(texts)}.", file=sys.stderr)
            print(f"  Restart server, then re-run this script to resume.", file=sys.stderr)
            sys.exit(1)

        # Pad with zeros for this batch, try to continue
        all_embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))

    batch_num = (i // BATCH_SIZE) + 1
    pct = (batch_num / num_batches) * 100
    print(
        f"  [{pct:5.1f}%] Batch {batch_num}/{num_batches} ({len(all_embeddings)} embeddings)",
        end="\r",
    )

# Clean up checkpoint on success
for f in (CHECKPOINT_FILE, CHECKPOINT_META):
    if os.path.exists(f):
        os.remove(f)

print()
elapsed = time.time() - start
print(f"  Done: {elapsed:.1f}s ({len(texts) / elapsed:.0f} entries/s)")

# Convert to numpy
embeddings = np.array(all_embeddings, dtype=np.float32)
print(f"  Shape: {embeddings.shape}")

if embeddings.shape[1] != EMBEDDING_DIM:
    print(f"  WARNING: Expected dim={EMBEDDING_DIM}, got {embeddings.shape[1]}")
    print(f"  Update EMBEDDING_DIM in config.py!")

# Filter entries to match texts
filtered_entries = [e for e in entries if make_passage_with_prussian(e)]

# Save
output_path = str(EMBEDDINGS_PATH)
print(f"\nSaving to: {output_path}")

np.save(f"{output_path}.embeddings.npy", embeddings)

with open(f"{output_path}.entries.json", "w", encoding="utf-8") as f:
    json.dump(filtered_entries, f, ensure_ascii=False, indent=2)

metadata = {
    "model": EMBEDDING_MODEL,
    "strategy": "with_prussian_word",
    "format": "word: de | en | lt | lv",
    "passage_prefix": PASSAGE_PREFIX,
    "num_entries": len(filtered_entries),
    "embedding_dim": int(embeddings.shape[1]),
    "server": RERANK_BASE_URL,
}

with open(f"{output_path}.meta.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'=' * 60}")
print(f"Saved {len(filtered_entries)} embeddings ({embeddings.shape[1]}d)")
print(f"  - {output_path}.embeddings.npy")
print(f"  - {output_path}.entries.json")
print(f"  - {output_path}.meta.json")
print("=" * 60)
