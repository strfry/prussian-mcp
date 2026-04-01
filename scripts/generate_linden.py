#!/usr/bin/env python3
"""
Generate synthetic Prussian sentences using Linden framework with XML Tool Calling.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from linden import AgentRunner, Configuration
from linden.config.configuration import (
    OpenAIConfig,
    OllamaConfig,
    GroqConfig,
    AnthropicConfig,
    GoogleConfig,
)

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8001/v3")
MODEL = os.environ.get("OPENAI_MODEL", "eurollm-22b-instruct-int4")
OUTPUT_DIR = Path("corpus/synthetic")

_search_engine = None


def get_search_engine():
    global _search_engine
    if _search_engine is None:
        try:
            from prussian_engine.search import SearchEngine

            _search_engine = SearchEngine()
        except Exception as e:
            print(f"Could not load search engine: {e}", file=sys.stderr)
    return _search_engine


def parse_xml_tool_call(text: str) -> dict | None:
    """Parse XML-style tool calls from LLM output.

    Format: <tool_name param1="value1" />
    Example: <search query="König" top_k="3" />
    """
    pattern = r"<(\w+)\s+([^/>]*)/>"

    for match in re.finditer(pattern, text):
        tool_name = match.group(1)
        attrs = match.group(2)
        params = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        return {"tool": tool_name, "params": params}

    return None


def execute_tool(tool_call: dict) -> str:
    """Execute a tool call and return the result as JSON."""
    tool = tool_call["tool"]
    params = tool_call["params"]
    engine = get_search_engine()

    if tool == "search":
        query = params.get("query", "")
        top_k = int(params.get("top_k", "5"))
        if engine:
            results = engine.query(query, top_k=top_k)
            return json.dumps(results[:top_k], ensure_ascii=False)
        return "[]"

    elif tool == "lookup":
        word = params.get("word", "")
        fuzzy = params.get("fuzzy", "true").lower() == "true"
        if engine:
            results = engine.lookup(word, fuzzy=fuzzy)
            return json.dumps(results[:10], ensure_ascii=False)
        return "[]"

    elif tool == "get_forms":
        lemma = params.get("lemma", "")
        if engine:
            results = engine.get_word_forms(lemma)
            return json.dumps(results, ensure_ascii=False)
        return "{}"

    elif tool == "validate":
        pr = params.get("pr", "")
        if not engine:
            return json.dumps({"valid": True, "issues": []})

        words = pr.replace(".", " ").replace(",", " ").split()
        issues = []
        for word in words:
            word = word.strip()
            if not word or len(word) < 2:
                continue
            lookup = engine.lookup(word, fuzzy=True)
            if not lookup:
                search = engine.query(word, top_k=3)
                if not search:
                    issues.append(f"Wort '{word}' nicht gefunden")

        return json.dumps(
            {"valid": len(issues) == 0, "issues": issues}, ensure_ascii=False
        )

    return json.dumps({"error": f"Unknown tool: {tool}"})


def load_grammar() -> str:
    return Path("prompts/grammar.txt").read_text()


def generate_system_prompt(grammar: str, seed: int, num_sentences: int = 10) -> str:
    templates = [
        "Alltagshandlungen (Essen, Schlafen, Gehen)",
        "Natur und Tiere (Wald, Vögel, Fisch)",
        "Familie und Beziehungen (Vater, Mutter, Kind)",
        "Gefühle und Zustände (Freude, Trauer, Sein)",
        "Zeit und Raum (heute, gestern, hier, dort)",
    ]
    selected = templates[seed % len(templates)]

    return f"""Du bist ein Assistent für Altpreußisch (Neo-Prußisch, Palmaitis-System).

## Grammatikregeln
{grammar}

## Bereich
{selected}

## Verfügbare Tools (XML-Format)
Wenn du Wörter nachschlagen musst, nutze EXAKT dieses Format:

<search query="Suchbegriff" top_k="5" />
<lookup word="Wort" fuzzy="true" />
<get_forms lemma="Lemma" />
<validate de="Deutsche Übersetzung" pr="Prußische Übersetzung" />

Beispiele:
<search query="König" top_k="3" />
<lookup word="buttan" fuzzy="true" />
<get_forms lemma="būtwei" />
<validate de="Ich bin König" pr="Asasma kunnegs" />

## Aufgabe
Generiere {num_sentences} einfache Beispielsätze Deutsch → Prußisch.
Verwende verschiedene Personen (1sg, 2sg, 3sg, 1pl, 2pl).

Nach jedem Satz kannst du Tools aufrufen um die Wörter zu validieren.
Wenn alle Wörter validiert sind, antworte mit den finalen Sätzen im Format:

ERGEBNIS:
Deutsch1;Prußisch1
Deutsch2;Prußisch2
...
"""


def main():
    num_sentences = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.randint(0, 999999)

    print(f"Generating {num_sentences} sentences with seed {seed}")
    random.seed(seed)

    grammar = load_grammar()
    print(f"Loaded grammar ({len(grammar)} chars)")

    # Initialize search engine
    engine = get_search_engine()
    if engine:
        print(f"Search engine loaded: {len(engine.entries)} entries indexed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create Linden configuration
    config = Configuration(
        openai=OpenAIConfig(
            api_key="dummy",  # Not used with custom URL
        ),
        ollama=OllamaConfig(),
        groq=GroqConfig(),
        anthropic=AnthropicConfig(),
        google=GoogleConfig(),
    )

    # Create agent
    agent = AgentRunner(
        config=config,
        user_id="synthetic-generator",
        name="prussian-agent",
        model=MODEL,
        temperature=0.9,
        system_prompt=generate_system_prompt(grammar, seed, num_sentences),
        enable_memory=False,
    )

    # Override client for custom URL
    agent._client._base_url = LLM_URL

    print("Running agent...")
    response = agent.run(f"Generiere {num_sentences} Sätze. Seed: {seed}")

    print(f"Response ({len(response)} chars): {response[:300]}...")

    # ReAct loop: parse tool calls, execute, repeat
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        tool_call = parse_xml_tool_call(response)

        if tool_call:
            print(f"  [{iteration}] Executing tool: {tool_call['tool']}")
            result = execute_tool(tool_call)
            print(f"      Result: {result[:150]}...")

            response = agent.run(
                f"Tool-Ergebnis: {result}\n\nFortfahren oder finale Sätze ausgeben."
            )
        else:
            if "ERGEBNIS:" in response:
                break
            print(f"  [{iteration}] No tool call found")
            break

    # Parse final results
    sentences = []

    if "ERGEBNIS:" in response:
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if ";" in line and not line.startswith("ERGEBNIS"):
                parts = line.split(";", 1)
                if len(parts) == 2:
                    de = parts[0].strip()
                    pr = parts[1].strip()
                    if de and pr:
                        sentences.append({"de": de, "pr": pr})

    if not sentences:
        for line in response.split("\n"):
            line = line.strip()
            if ";" in line:
                parts = line.split(";", 1)
                if len(parts) == 2:
                    de = parts[0].strip()
                    pr = parts[1].strip()
                    if de and len(de) > 2 and len(pr) > 2:
                        sentences.append({"de": de, "pr": pr})

    print(f"\n=== Summary ===")
    print(f"Generated: {len(sentences)} sentences")

    for s in sentences:
        print(f"  {s['de']} → {s['pr']}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_path = OUTPUT_DIR / f"linden_{seed}.json"
    all_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "timestamp": timestamp,
                "sentences": sentences,
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

    combined.extend(sentences)
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"Appended to {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
