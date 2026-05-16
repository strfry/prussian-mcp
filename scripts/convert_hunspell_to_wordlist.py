#!/usr/bin/env python3
"""
Convert Hunspell prg.dic/prg.aff to a wordlist.json-like format.

Hunspell .dic format:
  - Line 1: word count
  - Remaining lines: word[/affix_flags...]
  - Flags like /S, /aAun, /bun, /VK, /Pn, /pn, /W, /w, /un, /K, /n, /u etc.

This script:
  1. Extracts all entries from the .dic file
  2. Identifies lemmas (entries with no flags, or the base form of flagged entries)
  3. Outputs a JSON file compatible with wordlist.json structure
  4. Minimizes errors by preserving original data and writing to a NEW file
"""

import json
import re
import sys
from pathlib import Path


def parse_dic(dic_path: str) -> list[dict]:
    """Parse Hunspell .dic file and return list of word records."""
    entries = []
    with open(dic_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip first line (word count)
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split word from affix flags
        if "/" in line:
            word, flags = line.split("/", 1)
        else:
            word = line
            flags = ""

        entries.append({"word": word, "flags": flags})

    return entries


def identify_lemmas(entries: list[dict]) -> list[dict]:
    """
    Identify lemmas from the entries.

    Strategy:
    - Entries with no flags are likely lemmas or standalone words
    - Entries with flags are inflected forms derived from a lemma
    - We collect all unique words, marking which are likely lemmas (no flags)
    - For entries with flags, we keep them but mark them as inflected forms
    """
    # Track all words and their flag status
    word_info: dict[str, dict] = {}

    for entry in entries:
        word = entry["word"]
        flags = entry["flags"]

        if word not in word_info:
            word_info[word] = {
                "word": word,
                "paradigm": "",
                "gender": "",
                "desc": f"[hunspell flags: {flags}]" if flags else "",
                "audio": "",
                "translations_engl": [],
                "description": "hunspell-derived",
                "is_lemma": flags == "",
                "flags": flags,
            }
        else:
            # If we see the same word with no flags, it's definitely a lemma
            if flags == "":
                word_info[word]["is_lemma"] = True
                word_info[word]["desc"] = ""
            # Merge flags info
            existing_flags = word_info[word]["flags"]
            if flags and flags not in existing_flags:
                if existing_flags:
                    word_info[word]["flags"] = existing_flags + "," + flags
                else:
                    word_info[word]["flags"] = flags
                word_info[word]["desc"] = (
                    f"[hunspell flags: {word_info[word]['flags']}]"
                )

    return list(word_info.values())


def guess_paradigm_from_flags(flags: str) -> str:
    """Try to map Hunspell flags to paradigm numbers based on .aff comments."""
    # Mapping from .aff file comments:
    # A -> adjective declension (25,26,27,31)
    # a -> adjective with -is ending
    # b -> special declension
    # S -> noun declension (32,32y,33,35,52,54,49,45,52,53,54,58,60)
    # W/w/v/V/k/K -> verb conjugation (71,75,85,111,113,118,131,132,134,134a,136,138,139,142,143,144)
    # P/p -> participles
    # n -> negative prefix
    # u -> diminutive prefix

    flag_map = {
        "A": "25/26/27",
        "a": "adj-is",
        "b": "b-decl",
        "S": "32/33/35",
        "W": "85",
        "w": "71/75",
        "v": "134/134a",
        "V": "144/143",
        "k": "136/142",
        "K": "sen-wōkalin",
        "P": "act-pres-part",
        "p": "pass-part",
        "c": "act-pres-part-alt",
        "n": "neg-prefix",
        "u": "diminutive",
    }

    if not flags:
        return ""

    paradigms = []
    for flag in flags.replace(",", ""):
        if flag in flag_map:
            paradigms.append(flag_map[flag])

    return ",".join(paradigms) if paradigms else flags


def convert_to_wordlist_format(entries: list[dict]) -> list[dict]:
    """Convert parsed entries to wordlist.json-compatible format."""
    result = []
    for entry in entries:
        record = {
            "word": entry["word"],
            "paradigm": guess_paradigm_from_flags(entry.get("flags", "")),
            "gender": "",
            "desc": entry.get("desc", ""),
            "audio": "",
            "translations_engl": entry.get("translations_engl", []),
            "description": entry.get("description", "hunspell-derived"),
        }
        result.append(record)

    return result


def main():
    base_dir = Path(__file__).parent.parent
    dic_path = base_dir / "corpus" / "prg.dic"
    aff_path = base_dir / "corpus" / "prg.aff"

    if not dic_path.exists():
        print(f"Error: {dic_path} not found")
        sys.exit(1)

    print(f"Parsing {dic_path}...")
    entries = parse_dic(str(dic_path))
    print(f"  Found {len(entries)} entries")

    # Identify lemmas and deduplicate
    print("Identifying lemmas...")
    lemmas = identify_lemmas(entries)
    lemma_count = sum(1 for e in lemmas if e["is_lemma"])
    print(f"  {lemma_count} lemmas, {len(lemmas) - lemma_count} inflected forms")

    # Convert to wordlist format
    print("Converting to wordlist.json format...")
    wordlist = convert_to_wordlist_format(lemmas)

    # Write output to data/ directory (NOT overwriting original)
    output_path = base_dir / "data" / "wordlist_hunspell.json"
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(wordlist, f, ensure_ascii=False, indent=2)

    print(f"Done! {len(wordlist)} entries written to {output_path}")

    # Also create a lemmas-only version for comparison
    lemmas_only = [
        e
        for e in wordlist
        if e["desc"] == "" or "hunspell flags" not in e.get("desc", "")
    ]
    if lemmas_only:
        lemmas_path = base_dir / "data" / "wordlist_hunspell_lemmas.json"
        print(f"Writing lemmas-only to {lemmas_path}...")
        with open(lemmas_path, "w", encoding="utf-8") as f:
            json.dump(lemmas_only, f, ensure_ascii=False, indent=2)
        print(f"  {len(lemmas_only)} lemmas written")


if __name__ == "__main__":
    main()
