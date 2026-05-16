#!/usr/bin/env python3
"""Re-embed only homonym entries that were re-scraped."""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from prussian_engine.config import (
    EMBEDDING_MODEL,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    DICTIONARY_PATH,
    EMBEDDINGS_DIR,
)
from openai import OpenAI

BATCH_SIZE = 16
LANGUAGE_ORDER = ["engl", "miks", "leit", "latt", "pols", "mask"]
EMB_BASE = str(EMBEDDINGS_DIR / "embeddings_with_prussian")


def make_passage(entry):
    word = entry.get("word", "")
    translations = entry.get("translations", {})
    parts = []
    for lang in LANGUAGE_ORDER[:4]:
        tl = translations.get(lang, [])
        if isinstance(tl, list) and tl:
            parts.append(tl[0])
    return f"{word}: " + " | ".join(parts) if parts else ""


def entry_key(e):
    return (e["word"], e.get("paradigm", ""), e.get("desc", ""))


# Load current embeddings + entries
print("Loading embeddings...")
emb = np.load(f"{EMB_BASE}.embeddings.npy")
entries = json.load(open(f"{EMB_BASE}.entries.json"))
print(f"  {emb.shape[0]} entries, {emb.shape[1]}d")

# Load fresh dictionary and find homonyms
dict_entries = json.load(open(DICTIONARY_PATH))
counts = Counter((e["word"], e.get("paradigm", "")) for e in dict_entries)
dupes = {k for k, v in counts.items() if v > 1}

# Build lookup from fresh dictionary
fresh_by_key = {}
for e in dict_entries:
    k = entry_key(e)
    fresh_by_key[k] = e

# Find indices to update and collect texts
to_update = []
for i, old in enumerate(entries):
    if (old["word"], old.get("paradigm", "")) not in dupes:
        continue
    k = entry_key(old)
    fresh = fresh_by_key.get(k)
    if fresh:
        new_text = make_passage(fresh)
        old_text = make_passage(old)
        to_update.append((i, fresh, new_text, old_text))

print(f"\n{len(to_update)} homonym entries to re-embed")

changed = [(i, e, nt, ot) for i, e, nt, ot in to_update if nt != ot]
unchanged = [(i, e, nt, ot) for i, e, nt, ot in to_update if nt == ot]
print(f"  {len(changed)} with changed text, {len(unchanged)} unchanged")

if changed:
    print("\nChanged entries:")
    for i, e, nt, ot in changed[:20]:
        print(f"  [{i}] {ot!r}")
        print(f"    → {nt!r}")
    if len(changed) > 20:
        print(f"  ... and {len(changed) - 20} more")

if not changed:
    print("Nothing to do.")
    sys.exit(0)

# Generate new embeddings
client = OpenAI(api_key=OPENAI_API_KEY or "dummy", base_url=OPENAI_BASE_URL)

texts = [nt for _, _, nt, _ in changed]
indices = [i for i, _, _, _ in changed]

print(f"\nGenerating {len(texts)} embeddings...")
new_embeddings = []
for b in range(0, len(texts), BATCH_SIZE):
    batch = texts[b : b + BATCH_SIZE]
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    new_embeddings.extend([item.embedding for item in resp.data])
    print(f"  {min(b + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
print()

# Update in-place
for idx, new_emb in zip(indices, new_embeddings):
    emb[idx] = new_emb

# Update entries from fresh dictionary
for i, fresh, _, _ in to_update:
    entries[i] = fresh

# Save
print("Saving...")
np.save(f"{EMB_BASE}.embeddings.npy", emb)
with open(f"{EMB_BASE}.entries.json", "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

print(f"Done. Updated {len(changed)} embeddings + {len(to_update)} entries.")
