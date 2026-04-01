#!/usr/bin/env python3
"""
Generate synthetic Prussian sentences with ReAct-style XML Tool Calling.
Lightweight framework that works with any LLM (no native tool calling needed).
"""

import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def parse_xml_tool_call(text: str) -> list[dict]:
    """Parse all XML-style tool calls from LLM output.

    Format: <tool_name param1="value1" />
    Example: <search query="König" top_k="3" />
    """
    tool_calls = []
    pattern = r"<(\w+)\s+([^/>]*)/>"

    for match in re.finditer(pattern, text):
        tool_name = match.group(1)
        attrs = match.group(2)
        params = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        tool_calls.append({"tool": tool_name, "params": params})

    return tool_calls


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


class ReActLLM:
    """Lightweight ReAct agent that works with any LLM via OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        base_url: str = "http://localhost:8001/v3",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    def call(self, user_message: str, temperature: float = None) -> str:
        """Call the LLM with a user message."""
        self.conversation_history.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": self.conversation_history,
            "temperature": temperature or self.temperature,
            "max_tokens": 8000,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=300) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    message = result["choices"][0]["message"]
                    # Prefer reasoning_content (CoT), fallback to content
                    content = message.get("reasoning_content") or message.get(
                        "content", ""
                    )

                    # Add response to history
                    self.conversation_history.append(
                        {"role": "assistant", "content": content}
                    )
                    return content

            except urllib.error.HTTPError as e:
                if attempt < self.max_retries - 1:
                    print(f"  Retry {attempt + 1}: HTTP {e.code}")
                    time.sleep(2)
                else:
                    raise

        raise RuntimeError("Failed to call LLM")

    def run(self, user_message: str, max_iterations: int = 10) -> str:
        """Run ReAct loop: generate → parse tools → execute → repeat."""

        response = self.call(user_message)

        for iteration in range(max_iterations):
            # Parse tool calls from response
            tool_calls = parse_xml_tool_call(response)

            if not tool_calls:
                # No tools to call, return response
                return response

            print(f"  [Iteration {iteration + 1}] Found {len(tool_calls)} tool call(s)")

            # Execute all tool calls and collect results
            tool_results = []
            for tc in tool_calls:
                print(f"    Executing: <{tc['tool']} {tc['params']} />")
                result = execute_tool(tc)
                print(f"      → {result[:100]}...")
                tool_results.append((tc, result))

            # Feed results back to LLM
            results_text = "\n\n".join(
                f"Tool <{tc['tool']}> Ergebnis: {result}" for tc, result in tool_results
            )

            response = self.call(
                f"Tool-Ergebnisse:\n{results_text}\n\n"
                "Wenn alle Wörter validiert sind, gib die finalen Sätze aus. "
                "Format: ERGEBNIS:\nDeutsch;Prußisch\n..."
            )

        return response


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


def parse_results(response: str) -> list[dict]:
    """Parse final results from LLM response."""
    sentences = []

    # Look for ERGEBNIS: marker
    if "ERGEBNIS:" in response:
        lines = response.split("\n")
        in_results = False
        for line in lines:
            line = line.strip()
            if line == "ERGEBNIS:":
                in_results = True
                continue
            if in_results and ";" in line:
                parts = line.split(";", 1)
                if len(parts) == 2:
                    de = parts[0].strip()
                    pr = parts[1].strip()
                    if de and pr:
                        sentences.append({"de": de, "pr": pr})

    # Fallback: parse any DE;PR pairs
    if not sentences:
        for line in response.split("\n"):
            line = line.strip()
            if ";" in line and not line.startswith("ERGEBNIS"):
                parts = line.split(";", 1)
                if len(parts) == 2:
                    de = parts[0].strip()
                    pr = parts[1].strip()
                    # Filter out examples
                    if de and pr and len(de) > 2 and len(pr) > 2:
                        if not any(
                            w in de.lower()
                            for w in ["beispiel", "beispiel:", "example"]
                        ):
                            sentences.append({"de": de, "pr": pr})

    return sentences


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

    # Create ReAct agent
    agent = ReActLLM(
        model=MODEL,
        system_prompt=generate_system_prompt(grammar, seed, num_sentences),
        base_url=LLM_URL,
        temperature=0.9,
    )

    print("Running ReAct agent...")
    response = agent.run(f"Generiere {num_sentences} Sätze. Seed: {seed}")

    print(f"\n=== Raw Response ({len(response)} chars) ===")
    print(response[:500] + "..." if len(response) > 500 else response)

    # Parse results
    sentences = parse_results(response)

    print(f"\n=== Summary ===")
    print(f"Generated: {len(sentences)} sentences")

    for s in sentences:
        print(f"  {s['de']} → {s['pr']}")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_path = OUTPUT_DIR / f"react_{seed}.json"
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
