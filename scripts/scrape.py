#!/usr/bin/env python3
"""
Resumable scraper for the Prussian dictionary at wirdeins.twanksta.org

Two phases:
  1. Enumerate: discover all unique words via prefix search → wordlist.json
  2. Complete:  fetch all translations + forms per word → prussian_dictionary.json

Both phases are resumable. Ctrl+C safe.

Usage:
  python3 scrape.py                    # Run (resume where left off)
  python3 scrape.py --test             # Test with 5 words
  python3 scrape.py --delay=0.1        # Custom delay
  python3 scrape.py --status           # Show progress
"""

import json
import re
import html
import time
import sys
import os
import urllib.parse
import urllib.request

BASE = "https://wirdeins.twanksta.org"
DIALECT = "semba"
DELAY = 0.2
RESULT_CAP = 38

LANGUAGES = ["engl", "miks", "leit", "latt", "pols", "mask"]

ALPHABET = sorted(set(list("abdeghijklmnoprstuwz") + ["ā", "ē", "ī", "ō", "ū", "š", "ž"]))

WORDLIST_FILE = "wordlist.json"
OUTPUT_FILE = "prussian_dictionary.json"
STATE_FILE = "scrape_state.json"


# --- State ---

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"phase": "enumerate", "done_prefixes": [], "done_3letter": [], "completed_words": 0}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_wordlist():
    if os.path.exists(WORDLIST_FILE):
        with open(WORDLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_wordlist(words):
    with open(WORDLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)


def load_output():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_output(entries):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def entry_key(e):
    return (e["word"], e["paradigm"], e["desc"])


# --- HTTP ---

def fetch(url, params=None, post_data=None, retries=3):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    if post_data:
        data = urllib.parse.urlencode(post_data).encode()
        req = urllib.request.Request(url, data=data)
    else:
        req = urllib.request.Request(url)
    req.add_header("User-Agent", "PrussianDictionaryScraper/1.0 (linguistic research)")
    
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Try UTF-8 first, fall back to ISO-8859-1
                try:
                    return raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    return raw.decode("iso-8859-1", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                if attempt < retries - 1:
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    print(f"\n  [HTTP 503] Warte {wait}s...", file=sys.stderr, flush=True)
                    time.sleep(wait)
                else:
                    raise
            else:
                raise
        except Exception:
            raise


def clean(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    return " ".join(s.split()).strip()


# --- Parsing ---

def parse_search_results(html_text):
    entries = re.findall(r"<li>(.*?)</li>", html_text, re.DOTALL)
    results = []
    for entry_html in entries:
        word_m = re.search(r"class='word'>(.*?)</span>", entry_html)
        numb_m = re.search(r"class='numb'>(.*?)</span>", entry_html)
        gend_m = re.search(r'class="gend">(.*?)</span>', entry_html)
        desc_m = re.search(r"class='desc'>(.*?)</span>", entry_html)
        audio_m = re.search(r"src='(/upload/audio/[^']+)'", entry_html)

        translations = []
        for tm in re.finditer(
            r"class='translation-child'>.*?class='translation-number'>\d+</span>\s*(.*?)</span>",
            entry_html,
        ):
            t = clean(tm.group(1))
            if t:
                translations.append(t)

        desc_text = ""
        desc_div = re.search(
            r'<div class="descripcio">(.*?)</div>\s*</div>', entry_html, re.DOTALL
        )
        if desc_div:
            desc_text = clean(desc_div.group(1))

        word = clean(word_m.group(1)) if word_m else ""
        if not word:
            continue

        results.append({
            "word": word,
            "paradigm": numb_m.group(1).strip() if numb_m else "",
            "gender": gend_m.group(1).strip() if gend_m else "",
            "desc": clean(desc_m.group(1)) if desc_m else "",
            "audio": BASE + audio_m.group(1) if audio_m else "",
            "translations_engl": translations,
            "description": desc_text,
        })
    return results


def parse_forms(html_text):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse_forms import parse_forms as _parse
    return _parse(html_text)


# --- Phase 1: Enumerate ---

def search_prefix(prefix):
    """Search for a single prefix, return parsed results."""
    time.sleep(DELAY)
    params = {"s": prefix, "language": "engl", "dia": DIALECT}
    text = fetch(BASE + "/search/", params=params)
    return parse_search_results(text)


def phase_enumerate():
    """Phase 1: Build wordlist by searching all prefix combinations (2-letter, then 3-letter)."""
    state = load_state()
    wordlist = load_wordlist()
    existing = {entry_key(e) for e in wordlist}
    done_prefixes = set(state.get("done_prefixes", []))
    done_3letter = set(state.get("done_3letter", []))

    # Phase 1a: 2-letter prefixes
    all_2letter = [a + b for a in ALPHABET for b in ALPHABET]
    remaining_2letter = [p for p in all_2letter if p not in done_prefixes]

    if remaining_2letter:
        print(f"Phase 1a: 2-letter prefixes ({len(done_prefixes)}/{len(all_2letter)} done, {len(wordlist)} words)", file=sys.stderr)

        for i, prefix in enumerate(remaining_2letter):
            results = search_prefix(prefix)
            new = 0
            for r in results:
                k = entry_key(r)
                if k not in existing:
                    existing.add(k)
                    wordlist.append(r)
                    new += 1

            done_prefixes.add(prefix)

            # Save every 27 prefixes (= one "row" of the alphabet)
            if (i + 1) % 27 == 0 or (i + 1) == len(remaining_2letter):
                state["done_prefixes"] = sorted(done_prefixes)
                save_wordlist(wordlist)
                save_state(state)
                letter = prefix[0]
                print(f"  '{letter}*': {len(wordlist)} words total", file=sys.stderr, flush=True)

    # Phase 1b: 3-letter prefixes (for better coverage)
    print(f"\nPhase 1b: 3-letter prefixes (0/{len(ALPHABET)**3} done, {len(wordlist)} words)", file=sys.stderr)
    
    all_3letter = [a + b + c for a in ALPHABET for b in ALPHABET for c in ALPHABET]
    remaining_3letter = [p for p in all_3letter if p not in done_3letter]
    
    print(f"  Searching {len(remaining_3letter)} 3-letter prefixes for additional coverage...", file=sys.stderr)

    for i, prefix in enumerate(remaining_3letter):
        results = search_prefix(prefix)
        new = 0
        for r in results:
            k = entry_key(r)
            if k not in existing:
                existing.add(k)
                wordlist.append(r)
                new += 1

        done_3letter.add(prefix)

        # Save every 100 prefixes
        if (i + 1) % 100 == 0 or (i + 1) == len(remaining_3letter):
            state["done_3letter"] = sorted(done_3letter)
            state["done_prefixes"] = sorted(done_prefixes)
            save_wordlist(wordlist)
            save_state(state)
            progress = len(done_3letter)
            total = len(all_3letter)
            print(f"  [{progress}/{total}] {len(wordlist)} words total (+{new} last)", file=sys.stderr, flush=True)

    state["phase"] = "complete"
    state["done_3letter"] = sorted(done_3letter)
    save_state(state)
    print(f"\nEnumeration done: {len(wordlist)} unique entries", file=sys.stderr)


# --- Phase 2: Complete each entry ---

def phase_complete():
    """Phase 2: For each word, fetch all languages + forms."""
    wordlist = load_wordlist()
    entries = load_output()
    done_keys = {entry_key(e) for e in entries}
    total = len(wordlist)
    start_at = len(entries)

    print(f"Phase 2: Complete ({start_at}/{total} done)", file=sys.stderr)

    for i, stub in enumerate(wordlist):
        k = entry_key(stub)
        if k in done_keys:
            continue

        entry = {
            "word": stub["word"],
            "paradigm": stub["paradigm"],
            "gender": stub["gender"],
            "desc": stub["desc"],
            "audio": stub["audio"],
            "description": stub["description"],
            "translations": {"engl": stub["translations_engl"]},
        }

        # Fetch other languages
        for lang in LANGUAGES:
            if lang == "engl":
                continue
            time.sleep(DELAY)
            results = parse_search_results(
                fetch(BASE + "/search/", params={"s": entry["word"], "language": lang, "dia": DIALECT})
            )
            for r in results:
                if r["word"] == entry["word"] and r["paradigm"] == entry["paradigm"]:
                    if r["translations_engl"]:
                        entry["translations"][lang] = r["translations_engl"]
                    break

        # Fetch forms
        if entry["paradigm"]:
            time.sleep(DELAY)
            entry["forms"] = parse_forms(
                fetch(BASE + "/more/", post_data={
                    "word": entry["word"], "numb": entry["paradigm"],
                    "desc": entry["desc"], "dia": DIALECT,
                })
            )

        entries.append(entry)
        done_keys.add(k)

        idx = len(entries)
        if idx % 10 == 0 or idx == total:
            save_output(entries)
            print(f"  [{idx}/{total}] {entry['word']}", file=sys.stderr, flush=True)

    save_output(entries)
    print(f"\nDone: {len(entries)} entries in {OUTPUT_FILE}", file=sys.stderr)


# --- Test mode ---

def test_scrape():
    test_words = ["buttan", "ēitwei", "grazzus", "kwaitītun", "wundan"]
    entries = []

    for word in test_words:
        print(f"Searching: {word}", file=sys.stderr)
        results = parse_search_results(
            fetch(BASE + "/search/", params={"s": word, "language": "engl", "dia": DIALECT})
        )
        results = [r for r in results if r["word"] == word]

        for r in results:
            entry = {
                "word": r["word"], "paradigm": r["paradigm"], "gender": r["gender"],
                "desc": r["desc"], "audio": r["audio"], "description": r["description"],
                "translations": {"engl": r["translations_engl"]},
            }

            for lang in LANGUAGES:
                if lang == "engl":
                    continue
                time.sleep(DELAY)
                lang_results = parse_search_results(
                    fetch(BASE + "/search/", params={"s": word, "language": lang, "dia": DIALECT})
                )
                for lr in lang_results:
                    if lr["word"] == word and lr["paradigm"] == r["paradigm"]:
                        if lr["translations_engl"]:
                            entry["translations"][lang] = lr["translations_engl"]
                        break

            if entry["paradigm"]:
                time.sleep(DELAY)
                entry["forms"] = parse_forms(
                    fetch(BASE + "/more/", post_data={
                        "word": entry["word"], "numb": entry["paradigm"],
                        "desc": entry["desc"], "dia": DIALECT,
                    })
                )

            entries.append(entry)
            print(f"  ✓ {entry['word']} [{entry['paradigm']}]", file=sys.stderr)

    with open("prussian_test.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(entries)} entries to prussian_test.json", file=sys.stderr)


# --- Status ---

def show_status():
    state = load_state()
    wordlist = load_wordlist()
    entries = load_output()
    done = set(state.get("done_prefixes", []))
    all_prefixes = [a + b for a in ALPHABET for b in ALPHABET]

    print(f"Phase:       {state.get('phase', '?')}")
    print(f"Wordlist:    {len(wordlist)} entries")
    print(f"Completed:   {len(entries)} entries")
    print(f"Prefixes:    {len(done)}/{len(all_prefixes)} done")

    if entries:
        for lang in LANGUAGES:
            count = sum(1 for e in entries if lang in e.get("translations", {}))
            print(f"  {lang}: {count}/{len(entries)}")
        fc = sum(1 for e in entries if e.get("forms"))
        print(f"  forms: {fc}/{len(entries)}")


# --- Main ---

if __name__ == "__main__":
    for arg in sys.argv:
        if arg.startswith("--delay="):
            DELAY = float(arg.split("=", 1)[1])

    if "--status" in sys.argv:
        show_status()
    elif "--test" in sys.argv:
        test_scrape()
    else:
        state = load_state()
        if state["phase"] == "enumerate":
            phase_enumerate()
        if state["phase"] in ("complete", "enumerate"):
            state = load_state()
            if state["phase"] == "complete":
                phase_complete()
