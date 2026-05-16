#!/usr/bin/env python3
"""
Convert prussian_dictionary.json to wordlist.json-compatible format.

prussian_dictionary.json has:
  - "translations": {"engl": [...], "miks": [...], "leit": [...], ...}
  - "forms": {"declension": [...]}

wordlist.json format:
  - "translations_engl": [...]
  - no "forms" field

This writes a NEW file, never modifying originals.
"""

import json
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).parent.parent
    src_path = base_dir / "prussian_dictionary.json"
    out_path = base_dir / "data" / "prussian_dictionary_flat.json"

    if not src_path.exists():
        print(f"Error: {src_path} not found")
        sys.exit(1)

    print(f"Loading {src_path}...")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  {len(data)} entries")

    result = []
    for entry in data:
        flat = {
            "word": entry["word"],
            "paradigm": entry.get("paradigm", ""),
            "gender": entry.get("gender", ""),
            "desc": entry.get("desc", ""),
            "audio": entry.get("audio", ""),
            "translations_engl": entry.get("translations", {}).get("engl", []),
            "description": entry.get("description", ""),
        }
        result.append(flat)

    print(f"Writing {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done! {len(result)} entries written.")


if __name__ == "__main__":
    main()
