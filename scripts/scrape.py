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
import asyncio
import time
import sys
import os
import httpx

BASE = "https://wirdeins.twanksta.org"
DIALECT = "semba"
DELAY = 0.2
RESULT_CAP = 30  # If results >= this, subdivide with longer prefix

LANGUAGES = ["engl", "miks", "leit", "latt", "pols", "mask"]

ALPHABET = sorted(set(list("abdeghijklmnoprstuwz") + ["ā", "ē", "ī", "ō", "ū", "š", "ž"]))

CONCURRENCY = 4

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


# --- HTTP with adaptive rate limiting ---

class AdaptiveThrottle:
    """Adjusts delay based on server response times.

    Tracks a baseline from the first few requests. When response times
    rise significantly above that baseline, the delay between requests
    is increased. When they drop back, the delay shrinks again.

    The semaphore limits how many requests are in-flight at once.
    """

    def __init__(self, min_delay=0.05, max_delay=2.0, concurrency=4, window=20):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.delay = min_delay
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.window = window
        self.times = []
        self.baseline = None
        self._lock = asyncio.Lock()

    async def record(self, elapsed):
        async with self._lock:
            self.times.append(elapsed)
            if len(self.times) > self.window:
                self.times.pop(0)

            if self.baseline is None:
                if len(self.times) >= self.window:
                    self.baseline = sorted(self.times)[len(self.times) // 2]
                    print(f"  [throttle] baseline: {self.baseline*1000:.0f}ms (concurrency={self.concurrency})", file=sys.stderr, flush=True)
                return

            current = sorted(self.times)[len(self.times) // 2]
            ratio = current / self.baseline

            if ratio > 2.0:
                self.delay = min(self.delay * 1.5, self.max_delay)
                print(f"  [throttle] slow ({current*1000:.0f}ms vs {self.baseline*1000:.0f}ms baseline), delay → {self.delay:.2f}s", file=sys.stderr, flush=True)
            elif ratio < 1.3 and self.delay > self.min_delay:
                self.delay = max(self.delay * 0.8, self.min_delay)

    def stats(self):
        if not self.times:
            return "no requests yet"
        med = sorted(self.times)[len(self.times) // 2]
        if self.baseline:
            return f"median={med*1000:.0f}ms delay={self.delay*1000:.0f}ms baseline={self.baseline*1000:.0f}ms"
        return f"warming up ({len(self.times)}/{self.window})"


_throttle = None

def get_throttle():
    global _throttle
    if _throttle is None:
        _throttle = AdaptiveThrottle(min_delay=DELAY, concurrency=CONCURRENCY)
    return _throttle


_client = None

def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            http2=True,
            timeout=30,
            headers={"User-Agent": "PrussianDictionaryScraper/1.0 (linguistic research)"},
            follow_redirects=True,
        )
    return _client


async def fetch(url, params=None, post_data=None, retries=3):
    client = get_client()
    throttle = get_throttle()
    async with throttle.semaphore:
        for attempt in range(retries):
            try:
                await asyncio.sleep(throttle.delay)
                t0 = time.monotonic()
                if post_data:
                    resp = await client.post(url, params=params, data=post_data)
                else:
                    resp = await client.get(url, params=params)
                elapsed = time.monotonic() - t0
                await throttle.record(elapsed)
                resp.raise_for_status()
                return resp.text
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600 and attempt < retries - 1:
                    wait = 5 * (2 ** attempt)
                    print(f"\n  [HTTP {e.response.status_code}] Warte {wait}s...", file=sys.stderr, flush=True)
                    await asyncio.sleep(wait)
                else:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                if attempt < retries - 1:
                    wait = 5 * (2 ** attempt)
                    print(f"\n  [Network Error: {e}] Warte {wait}s...", file=sys.stderr, flush=True)
                    await asyncio.sleep(wait)
                else:
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

async def search_prefix(prefix):
    """Search for a single prefix, return parsed results."""
    params = {"s": prefix, "language": "engl", "dia": DIALECT}
    text = await fetch(BASE + "/search/", params=params)
    return parse_search_results(text)


async def search_prefix_recursive(prefix, existing, max_depth=6):
    """Recursively search with prefix subdivision when results are capped."""
    results = await search_prefix(prefix)
    new_entries = []

    for r in results:
        k = entry_key(r)
        if k not in existing:
            existing.add(k)
            new_entries.append(r)

    if len(results) >= RESULT_CAP and len(prefix) < max_depth:
        print(f"    '{prefix}' has {len(results)} results (≥{RESULT_CAP}), subdividing...", file=sys.stderr, flush=True)
        # Subdivisions can run concurrently (throttle limits in-flight)
        tasks = [search_prefix_recursive(prefix + letter, existing, max_depth)
                 for letter in ALPHABET]
        for sub_entries in await asyncio.gather(*tasks):
            new_entries.extend(sub_entries)

    return new_entries


async def phase_enumerate():
    """Phase 1: Build wordlist by searching all prefix combinations."""
    state = load_state()
    wordlist = load_wordlist()
    existing = {entry_key(e) for e in wordlist}
    done_prefixes = set(state.get("done_prefixes", []))

    all_2letter = [a + b for a in ALPHABET for b in ALPHABET]
    remaining_2letter = [p for p in all_2letter if p not in done_prefixes]

    if remaining_2letter:
        print(f"Phase 1: 2-letter prefixes ({len(done_prefixes)}/{len(all_2letter)} done, {len(wordlist)} words)", file=sys.stderr)

        # Process in batches of 27 (one alphabet row)
        batch_size = 27
        for batch_start in range(0, len(remaining_2letter), batch_size):
            batch = remaining_2letter[batch_start:batch_start + batch_size]

            tasks = [search_prefix_recursive(p, existing, max_depth=6) for p in batch]
            results = await asyncio.gather(*tasks)

            for prefix, new_entries in zip(batch, results):
                wordlist.extend(new_entries)
                done_prefixes.add(prefix)

            state["done_prefixes"] = sorted(done_prefixes)
            save_wordlist(wordlist)
            save_state(state)
            letter = batch[0][0]
            print(f"  '{letter}*': {len(wordlist)} words total ({get_throttle().stats()})", file=sys.stderr, flush=True)

    state["phase"] = "complete"
    state["done_3letter"] = []
    save_state(state)
    print(f"\nEnumeration done: {len(wordlist)} unique entries ({get_throttle().stats()})", file=sys.stderr)


# --- Phase 2: Complete each entry ---

async def complete_entry(stub):
    """Fetch all languages + forms for a single word entry."""
    entry = {
        "word": stub["word"],
        "paradigm": stub["paradigm"],
        "gender": stub["gender"],
        "desc": stub["desc"],
        "audio": stub["audio"],
        "description": stub["description"],
        "translations": {"engl": stub["translations_engl"]},
    }

    # Fetch all languages concurrently
    async def fetch_lang(lang):
        results = parse_search_results(
            await fetch(BASE + "/search/", params={"s": entry["word"], "language": lang, "dia": DIALECT})
        )
        for r in results:
            if r["word"] == entry["word"] and r["paradigm"] == entry["paradigm"] and r["desc"] == entry["desc"]:
                if r["translations_engl"]:
                    return lang, r["translations_engl"]
                break
        return lang, None

    lang_tasks = [fetch_lang(lang) for lang in LANGUAGES if lang != "engl"]

    # Fetch forms concurrently alongside languages
    forms_task = None
    if entry["paradigm"]:
        async def fetch_forms():
            return parse_forms(
                await fetch(BASE + "/more/", post_data={
                    "word": entry["word"], "numb": entry["paradigm"],
                    "desc": entry["desc"], "dia": DIALECT,
                })
            )
        forms_task = fetch_forms()

    # Run all in parallel
    all_tasks = lang_tasks + ([forms_task] if forms_task else [])
    results = await asyncio.gather(*all_tasks)

    # Unpack language results
    for lang, translations in results[:len(lang_tasks)]:
        if translations:
            entry["translations"][lang] = translations

    # Unpack forms
    if forms_task:
        entry["forms"] = results[-1]

    return entry


async def phase_complete():
    """Phase 2: For each word, fetch all languages + forms."""
    wordlist = load_wordlist()
    entries = load_output()
    done_keys = {entry_key(e) for e in entries}
    total = len(wordlist)
    start_at = len(entries)

    remaining = [s for s in wordlist if entry_key(s) not in done_keys]
    print(f"Phase 2: Complete ({start_at}/{total} done)", file=sys.stderr)

    for i, stub in enumerate(remaining):
        entry = await complete_entry(stub)
        entries.append(entry)
        done_keys.add(entry_key(entry))

        idx = len(entries)
        if idx % 10 == 0 or idx == total:
            save_output(entries)
            print(f"  [{idx}/{total}] {entry['word']} ({get_throttle().stats()})", file=sys.stderr, flush=True)

    save_output(entries)
    print(f"\nDone: {len(entries)} entries in {OUTPUT_FILE}", file=sys.stderr)


# --- Test mode ---

async def test_scrape():
    test_words = ["buttan", "ēitwei", "grazzus", "kwaitītun", "wundan"]
    stubs = []

    for word in test_words:
        print(f"Searching: {word}", file=sys.stderr)
        results = parse_search_results(
            await fetch(BASE + "/search/", params={"s": word, "language": "engl", "dia": DIALECT})
        )
        stubs.extend(r for r in results if r["word"] == word)

    entries = []
    for stub in stubs:
        entry = await complete_entry(stub)
        entries.append(entry)
        print(f"  ✓ {entry['word']} [{entry['paradigm']}]", file=sys.stderr)

    with open("prussian_test.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(entries)} entries to prussian_test.json ({get_throttle().stats()})", file=sys.stderr)


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


# --- Rescrape homonyms ---

async def phase_rescrape_homonyms():
    """Re-fetch entries where (word, paradigm) is not unique.

    These are homonyms that the original scraper may have mixed up because
    it matched only on word+paradigm without checking desc.
    """
    from collections import Counter

    entries = load_output()
    counts = Counter((e["word"], e["paradigm"]) for e in entries)
    dupes = {k for k, v in counts.items() if v > 1}

    if not dupes:
        print("No homonyms found — nothing to rescrape.", file=sys.stderr)
        return

    to_rescrape = [(i, e) for i, e in enumerate(entries) if (e["word"], e["paradigm"]) in dupes]
    print(f"Rescraping {len(to_rescrape)} entries ({len(dupes)} homonym groups):", file=sys.stderr)
    for _, e in to_rescrape:
        print(f"  {e['word']} [{e['paradigm']}] {e['desc']}", file=sys.stderr)

    for idx, old_entry in to_rescrape:
        # Build stub from existing entry (no wordlist needed)
        translations = old_entry.get("translations", {})
        stub = {
            "word": old_entry["word"],
            "paradigm": old_entry["paradigm"],
            "gender": old_entry.get("gender", ""),
            "desc": old_entry["desc"],
            "audio": old_entry.get("audio", ""),
            "description": old_entry.get("description", ""),
            "translations_engl": translations.get("engl", []),
        }

        new_entry = await complete_entry(stub)
        # Preserve any existing translations that the rescrape didn't find
        for lang, trans in translations.items():
            if lang not in new_entry.get("translations", {}):
                new_entry["translations"][lang] = trans

        entries[idx] = new_entry
        print(f"  ✓ {new_entry['word']} [{new_entry['paradigm']}] {new_entry['desc']}", file=sys.stderr)

    save_output(entries)
    print(f"Done. Updated {len(to_rescrape)} entries.", file=sys.stderr)


# --- Main ---

async def run_and_close(coro):
    """Run a coroutine then close the shared HTTP client."""
    try:
        await coro
    finally:
        if _client:
            await _client.aclose()


async def main():
    state = load_state()
    if state["phase"] == "enumerate":
        await phase_enumerate()
    if state["phase"] in ("complete", "enumerate"):
        state = load_state()
        if state["phase"] == "complete":
            await phase_complete()


if __name__ == "__main__":
    for arg in sys.argv:
        if arg.startswith("--delay="):
            DELAY = float(arg.split("=", 1)[1])
        if arg.startswith("--concurrency="):
            CONCURRENCY = int(arg.split("=", 1)[1])

    if "--status" in sys.argv:
        show_status()
    elif "--test" in sys.argv:
        asyncio.run(run_and_close(test_scrape()))
    elif "--rescrape-homonyms" in sys.argv:
        asyncio.run(run_and_close(phase_rescrape_homonyms()))
    else:
        asyncio.run(run_and_close(main()))
