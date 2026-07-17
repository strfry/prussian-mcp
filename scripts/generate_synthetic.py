#!/usr/bin/env python3
"""
Generate synthetic Prussian sentences with LLM validation and dictionary lookup.

Usage:
    python scripts/generate_synthetic.py [num_sentences] [seed]

    num_sentences: Number of sentences to generate (default: 10)
    seed: Random seed for reproducibility (default: random)
"""

import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8001/v3")
MODEL = os.environ.get("OPENAI_MODEL", "eurollm-22b-instruct-int4")
OUTPUT_DIR = Path("corpus/synthetic")

# Try to use local search engine
_search_engine = None


def get_search_engine():
    """Get or create search engine instance."""
    global _search_engine
    if _search_engine is None:
        try:
            from prussian.engine.search import SearchEngine

            _search_engine = SearchEngine()
        except Exception as e:
            print(f"Could not load search engine: {e}", file=sys.stderr)
    return _search_engine


def search_dictionary(query: str, top_k: int = 5) -> list[dict]:
    """Search dictionary using local engine."""
    engine = get_search_engine()
    if engine is None:
        return []
    try:
        # Use query() instead of search()
        return engine.query(query, top_k=top_k)
    except Exception as e:
        print(f"Search error: {e}", file=sys.stderr)
        return []


def lookup_prussian_word(word: str, fuzzy: bool = True) -> list[dict]:
    """Lookup a Prussian word (lemma or inflected form)."""
    engine = get_search_engine()
    if engine is None:
        return []
    try:
        return engine.lookup(word, fuzzy=fuzzy)
    except Exception as e:
        print(f"Lookup error: {e}", file=sys.stderr)
        return []


def get_word_forms(lemma: str) -> dict:
    """Get all declension/conjugation forms for a lemma."""
    engine = get_search_engine()
    if engine is None:
        return {}
    try:
        return engine.get_word_forms(lemma)
    except Exception as e:
        print(f"Get forms error: {e}", file=sys.stderr)
        return {}


def format_leipzig(lookup_result: dict) -> str:
    """Format lookup result as Leipzig-style gloss."""
    word = lookup_result.get("word", "")
    de = lookup_result.get("de", "")
    en = lookup_result.get("en", "")
    gender = lookup_result.get("gender", "")
    forms = lookup_result.get("forms", {})
    matched = lookup_result.get("matched_form", "")

    lines = []
    lines.append(f"# {word}")
    lines.append(f"# [{gender}]" if gender else f"# [-]")
    lines.append(f"{de}; {en}")

    if matched and matched != word:
        lines.append(f"  → matched form: {matched}")

    if forms:
        for category, form_list in forms.items():
            lines.append(f"  {category}: {form_list}")

    return "\n".join(lines)


def load_grammar():
    """Load grammar rules from prompts/grammar.txt"""
    return Path("prompts/grammar.txt").read_text()


def call_mcp(tool_name: str, arguments: dict) -> dict:
    """Call MCP tool via JSON-RPC"""
    import urllib.request
    import urllib.error

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{MCP_URL}/rpc",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("result", {})
    except urllib.error.HTTPError as e:
        print(f"MCP HTTP Error: {e.code} {e.reason}", file=sys.stderr)
        return {"error": str(e)}
    except Exception as e:
        print(f"MCP Error: {e}", file=sys.stderr)
        return {"error": str(e)}


def get_word_forms(lemma: str) -> dict:
    """Get word forms using local search engine."""
    engine = get_search_engine()
    if engine is None:
        return {}
    try:
        return engine.get_word_forms(lemma)
    except Exception as e:
        print(f"Get forms error: {e}", file=sys.stderr)
        return {}


def generate_prompt(grammar: str, seed: int, count: int = 10) -> list[dict]:
    """Build the generation prompt with seed for variety."""

    templates = [
        "Alltagshandlungen (Essen, Schlafen, Gehen)",
        "Natur und Tiere (Wald, Vögel, Fisch)",
        "Familie und Beziehungen (Vater, Mutter, Kind)",
        "Gefühle und Zustände (Freude, Trauer, Sein)",
        "Zeit und Raum (heute, gestern, hier, dort)",
    ]

    selected = templates[seed % len(templates)]

    system_prompt = (
        "Du bist ein Assistent für Altpreußisch (Neo-Prußisch, Palmaitis-System).\n\n"
        "## Grammatikregeln\n" + grammar + "\n\n"
        f"## Bereich\n{selected}\n\n"
        "## Aufgabe\n"
        f"Generiere {count} einfache Beispielsätze Deutsch → Prußisch.\n"
        "Verwende verschiedene Personen (1sg, 2sg, 3sg, 1pl, 2pl).\n\n"
        "## Format (PFlicht!)\n"
        "Antworte NUR mit dieser Struktur, keine Erklärungen:\n"
        "DE;PR\n"
        "Ich bin König.;Asasma kunnegs.\n"
        "Du gehst nach Hause.;Tū ēitwei en buttan.\n"
        "Er ist gut.;Tāns ast labs.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generiere {count} Sätze (Seed: {seed})"},
    ]


def validate_with_dictionary(sentence_pair: dict) -> dict:
    """Validate using local search engine dictionary lookups."""
    pr_word = sentence_pair.get("pr", "").strip()

    if not pr_word:
        return {"valid": False, "issues": ["Leeres Prußisch"], "details": []}

    words = pr_word.replace(".", " ").replace(",", " ").split()
    issues = []
    details = []

    for word in words:
        word = word.strip()
        if not word or len(word) < 2:
            continue

        # Try lookup first (reverse lookup - prussian word to entry)
        lookup_result = lookup_prussian_word(word, fuzzy=True)

        if not lookup_result:
            # Try search as fallback (German/English → Prussian)
            search_result = search_dictionary(word, top_k=3)
            if not search_result:
                issues.append(f"Wort '{word}' nicht im Wörterbuch gefunden")
            else:
                details.append(
                    {
                        "word": word,
                        "found": True,
                        "via": "search",
                        "entries": search_result,
                    }
                )
        else:
            detail = {"word": word, "found": True, "via": "lookup", "entries": []}

            for entry in lookup_result:
                detail["entries"].append(
                    {
                        "lemma": entry.get("word", ""),
                        "de": entry.get("de", ""),
                        "en": entry.get("en", ""),
                        "gender": entry.get("gender", ""),
                        "matched_form": entry.get("matched_form", ""),
                        "forms": entry.get("forms", {}),
                        "leipzig": format_leipzig(entry),
                    }
                )

            details.append(detail)

    if issues:
        return {"valid": False, "issues": issues, "details": details}

    return {"valid": True, "issues": [], "details": details}


def validate_prompt(sentence_pair: dict, grammar: str) -> list[dict]:
    """Build validation prompt for a sentence pair."""

    system_prompt = (
        "Du bist ein Linguist für Altpreußisch.\n\n"
        "## Grammatik\n" + grammar + "\n\n"
        "## Prüfe\n"
        f"DE: {sentence_pair['de']}\n"
        f"PR: {sentence_pair['pr']}\n\n"
        "## Format\n"
        "Antworte NUR mit:\n"
        "valid;issues;fixed_pr\n"
        "true;;\n"
        "false;Fehler1,Fehler2;Korrektur\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Prüfe: {sentence_pair['de']} → {sentence_pair['pr']}",
        },
    ]


def call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """Call LLM via OpenAI-compatible API."""
    import urllib.request
    import urllib.error

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1000,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}", file=sys.stderr)
        print(e.read().decode("utf-8"), file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error calling LLM: {e}", file=sys.stderr)
        raise


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON or line-based sentences from LLM response."""
    text = text.strip()

    # Try array first
    start = text.find("[")
    end = text.rfind("]") + 1

    if start != -1 and end > 0:
        json_str = text[start:end]
        json_str = json_str.replace("```json", "").replace("```", "")
        try:
            return json.loads(json_str)
        except:
            pass

    # Try line-based format: DE;PR
    objects = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if ";" in line and not line.startswith("{"):
            parts = line.split(";", 1)
            if len(parts) == 2:
                de = parts[0].strip()
                pr = parts[1].strip()
                # Filter out examples and invalid entries
                if de and pr and len(de) > 2 and len(pr) > 2 and de.lower() != "de":
                    objects.append({"de": de, "pr": pr})

    if objects:
        return objects

    # Try finding all {de:, pr:} objects
    import re

    matches = re.findall(r'\{[^{}]*"de"[^{}]*"pr"[^{}]*\}', text)
    for m in matches:
        try:
            obj = json.loads(m)
            if "de" in obj and "pr" in obj:
                objects.append(obj)
        except:
            pass

    if objects:
        return objects

    raise ValueError(f"No JSON found in response: {text[:200]}")

    # Try object(s)
    import re

    # Match objects with nested braces
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    objects = []
    for m in matches:
        try:
            obj = json.loads(m)
            objects.append(obj)
        except:
            pass

    if objects:
        return objects

    # Fallback: simple find
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response: {text[:200]}")

    json_str = text[start:end]
    json_str = json_str.replace("```json", "").replace("```", "")

    return json.loads(json_str)


def main():
    num_sentences = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.randint(0, 999999)

    print(f"Generating {num_sentences} sentences with seed {seed}")

    random.seed(seed)

    grammar = load_grammar()
    print(f"Loaded grammar ({len(grammar)} chars)")

    # Test search engine
    print("Testing search engine...")
    test_search = search_dictionary("Haus", top_k=2)
    print(f"Search test: found {len(test_search)} entries for 'Haus'")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating sentences...")
    messages = generate_prompt(grammar, seed, num_sentences)

    for i, msg in enumerate(messages):
        print(f"[{i}] {msg['role']}: {msg['content'][:80]}...")

    response = call_llm(messages, temperature=0.9)
    print(f"\nGeneration response ({len(response)} chars)")

    try:
        sentences = parse_json_response(response)
    except Exception as e:
        print(f"Failed to parse generation response: {e}", file=sys.stderr)
        print(f"Response: {response[:500]}", file=sys.stderr)
        return 1

    print(f"Parsed {len(sentences)} sentences")

    # Validate each sentence with dictionary first, then LLM
    validated = []
    print("\nValidating sentences...")

    # Initialize search engine
    engine = get_search_engine()
    if engine:
        print(f"Search engine loaded successfully")
    else:
        print(
            f"Warning: No search engine available, skipping dictionary validation",
            file=sys.stderr,
        )

    for i, sent in enumerate(sentences):
        print(f"  [{i + 1}/{len(sentences)}] {sent.get('de', 'N/A')[:40]}...")

        # Step 1: Dictionary validation (if engine available)
        if engine:
            dict_result = validate_with_dictionary(sent)
        else:
            dict_result = {"valid": True, "issues": []}

        if dict_result.get("valid", False):
            print(f"    ✓ Dictionary check passed")
            sent["dict_valid"] = True
            sent["dict_details"] = dict_result.get("details", [])

            # Print Leipzig format for each word
            for detail in dict_result.get("details", []):
                for entry in detail.get("entries", []):
                    if entry.get("leipzig"):
                        print(f"    {entry.get('leipzig')}")
        else:
            issues = dict_result.get("issues", [])
            print(f"    ✗ Dictionary: {issues[0] if issues else 'Unknown'}")
            sent["dict_valid"] = False
            sent["issues"] = issues
            sent["dict_details"] = dict_result.get("details", [])

        # Step 2: LLM grammar validation
        try:
            val_msgs = validate_prompt(sent, grammar)
            val_response = call_llm(val_msgs, temperature=0.3)

            # Parse line-based response: valid;issues;fixed_pr
            result = {"valid": False, "issues": [], "fixed_pr": ""}

            lines = val_response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if ";" in line and not line.startswith("{"):
                    parts = line.split(";", 2)
                    if len(parts) >= 1:
                        valid_str = parts[0].strip().lower()
                        result["valid"] = valid_str in ("true", "yes", "ja", "valid")
                        if (
                            len(parts) >= 2
                            and parts[1].strip()
                            and parts[1].strip() not in ("issues", "")
                        ):
                            result["issues"] = [
                                p.strip() for p in parts[1].split(",") if p.strip()
                            ]
                        if len(parts) >= 3 and parts[2].strip():
                            result["fixed_pr"] = parts[2].strip()
                    break
            else:
                # Fallback: try JSON
                parsed = parse_json_response(val_response)
                if parsed:
                    if isinstance(parsed, list):
                        parsed = parsed[0]
                    result = parsed

            if result.get("valid", False):
                sent["llm_valid"] = True
                if not sent.get("issues"):
                    sent["validated"] = True
                    sent["issues"] = []
                print(f"    ✓ LLM check passed")
            else:
                sent["llm_valid"] = False
                issues = result.get("issues", [])
                if not sent.get("issues"):
                    sent["issues"] = issues
                    sent["fixed_pr"] = result.get("fixed_pr", sent.get("pr"))
                print(f"    ✗ LLM: {issues[0] if issues else 'Unknown issue'}")

        except Exception as e:
            print(f"    ! LLM validation error: {e}", file=sys.stderr)
            sent["llm_valid"] = False

        # Final validation status
        if sent.get("dict_valid") and sent.get("llm_valid"):
            sent["validated"] = True
        else:
            sent["validated"] = False

        validated.append(sent)
        time.sleep(0.3)

    valid = [s for s in validated if s.get("validated", False)]
    invalid = [s for s in validated if not s.get("validated", False)]

    print(f"\nValid: {len(valid)}, Invalid: {len(invalid)}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_path = OUTPUT_DIR / f"synthetic_{seed}.json"
    all_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "timestamp": timestamp,
                "valid": valid,
                "invalid": invalid,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved to {all_path}")

    combined_path = OUTPUT_DIR / "combined.json"
    combined = []
    if combined_path.exists():
        combined = json.loads(combined_path.read_text())

    combined.extend(valid)
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"Appended {len(valid)} valid sentences to {combined_path}")

    print(f"\n=== Summary ===")
    print(f"Seed: {seed}")
    print(f"Generated: {len(sentences)}")
    print(f"Valid: {len(valid)}")
    print(f"Invalid: {len(invalid)}")

    if valid:
        print("\nValid sentences:")
        for s in valid:
            print(f"  {s['de']} → {s['pr']}")

    if invalid:
        print("\nInvalid sentences (for review):")
        for s in invalid:
            print(f"  {s['de']} → {s['pr']}")
            for issue in s.get("issues", []):
                print(f"    ! {issue}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
