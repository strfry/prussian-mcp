#!/usr/bin/env python3
"""
Generate synthetic Prussian sentences using Haystack Pipeline with native Tool-Calling.
For models that support function calling (GPT-4, Claude, etc.)
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("corpus/synthetic")

# Set default API key
os.environ.setdefault("OPENAI_API_KEY", "not-needed")

# ============================================================
# Search Engine (Core)
# ============================================================

from prussian_engine.search import SearchEngine

_search_engine = SearchEngine()
print(f"Search engine loaded: {len(_search_engine.entries)} entries indexed")


# ============================================================
# Tool Functions (Core - direct, no wrappers)
# ============================================================


_seen_words = set()  # Track words already returned with full translations


def search_dictionary(query: str, top_k: int = 5, filter_pgr: str = None) -> str:
    """Search the Prussian dictionary by German/English query, then optionally filter forms by PGR.

    Args:
        query: Search query in German or English (e.g. "Sohn", "give", "dog")
        top_k: Number of results to return
        filter_pgr: Optional PGR filter for grammatical forms (e.g. "GEN.SG", "ACC.PL", "PRS.3.SG.IND")
    """
    results = _search_engine.query(query, top_k=top_k)
    output = []
    for r in results[:top_k]:
        word = r["word"]
        # Only include full translations on first encounter
        if word in _seen_words:
            entry = {"word": word, "seen_before": True}
        else:
            entry = {"word": word, "translations": r["translations"]}
            _seen_words.add(word)
        if filter_pgr:
            forms_data = _search_engine.get_word_forms(word, filter_pgr=filter_pgr)
            if isinstance(forms_data, list):
                for fd in forms_data:
                    if fd.get("forms"):
                        # Put the matched form front and center so the model uses it
                        entry["matched_form"] = fd["forms"][0]["form"]
                        entry["matched_pgr"] = fd["forms"][0]["pgr"]
                        entry["gender"] = fd.get("gender", "")
                        break
        output.append(entry)
    return json.dumps(output, ensure_ascii=False)


def get_word_forms(lemma: str, filter_pgr: str = None) -> str:
    """Get word forms for a Prussian lemma.

    Args:
        lemma: The base form (lemma) of the Prussian word
        filter_pgr: Optional filter for specific form (e.g., 'GEN.PL.MASC')
    """
    results = _search_engine.get_word_forms(lemma, filter_pgr=filter_pgr)
    return json.dumps(results, ensure_ascii=False)


def verify_translation(de_sentence: str, pr_sentence: str) -> tuple[bool, list[str]]:
    """Verify each word in the Prussian sentence exists."""
    pr_words = pr_sentence.replace(".", " ").replace(",", " ").split()
    issues = []

    for word in pr_words:
        word = word.strip().lower()
        if len(word) < 2:
            continue

        results = _search_engine.lookup(word, fuzzy=False)
        if not results:
            norm = (
                word.replace("ā", "a")
                .replace("ē", "e")
                .replace("ī", "i")
                .replace("ō", "o")
                .replace("ū", "u")
            )
            results = _search_engine.lookup(norm, fuzzy=False)

        if not results:
            issues.append(f"'{word}' not in dictionary")

    return len(issues) == 0, issues


# ============================================================
# Haystack Pipeline with Native Tool Calling
# ============================================================

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.components.routers import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack.tools import create_tool_from_function
from haystack.utils import Secret
from typing import List as TypingList


def create_tools():
    """Create Haystack Tool objects from functions."""
    search_tool = create_tool_from_function(
        function=search_dictionary,
        description="Search the Prussian dictionary by German or English query",
    )
    forms_tool = create_tool_from_function(
        function=get_word_forms,
        description="Get all declension/conjugation forms for a Prussian lemma",
    )
    return [search_tool, forms_tool]


def create_pipeline(tools: list) -> Pipeline:
    """Create Haystack pipeline with native tool calling."""

    # Load prompt template
    template_path = Path(__file__).parent.parent / "prompts" / "generate_system.jinja2"
    template_content = template_path.read_text()

    pipeline = Pipeline()

    # Prompt Builder
    prompt = ChatPromptBuilder(
        template=[
            ChatMessage.from_system(template_content),
            ChatMessage.from_user("{{user_input}}"),
        ],
        variables=["phase", "user_input"],
        required_variables=["phase", "user_input"],
    )

    # LLM Generator with native tool support
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    base_url = os.environ.get("LLM_URL", "http://localhost:8001/v3")
    model = os.environ.get("OPENAI_MODEL", "gpt-oss-20b-int4-ov")

    generator = OpenAIChatGenerator(
        model=model,
        api_key=Secret.from_env_var("OPENAI_API_KEY", strict=False),
        api_base_url=base_url,
        http_client_kwargs={"timeout": 120.0},
        tools=tools,
        generation_kwargs={
            "temperature": 0.1,
            "max_completion_tokens": 8192,
        },
    )

    # Tool Invoker
    tool_invoker = ToolInvoker(tools=tools)

    # Router for tool calls vs text
    router = ConditionalRouter(
        routes=[
            {
                "condition": "{{replies[0].tool_calls | length > 0}}",
                "output": "{{replies}}",
                "output_name": "tool_calls",
                "output_type": TypingList[ChatMessage],
            },
            {
                "condition": "{{replies[0].tool_calls | length == 0}}",
                "output": "{{replies}}",
                "output_name": "text",
                "output_type": TypingList[ChatMessage],
            },
        ],
        unsafe=True,
    )

    # Add components
    pipeline.add_component("prompt", prompt)
    pipeline.add_component("generator", generator)
    pipeline.add_component("router", router)
    pipeline.add_component("tool_invoker", tool_invoker)

    # Connect
    pipeline.connect("prompt", "generator")
    pipeline.connect("generator.replies", "router")
    pipeline.connect("router.tool_calls", "tool_invoker")

    return pipeline


def run_with_tools(
    generator: OpenAIChatGenerator,
    tool_invoker: ToolInvoker,
    tools: list,
    user_input: str,
    phase: str,
    max_loops: int = 10,
    extra_vars: dict = None,
) -> str:
    """Run tool calling loop externally (for proper feedback)."""

    from haystack.dataclasses import ChatMessage

    # Build system prompt
    template_path = Path(__file__).parent.parent / "prompts" / "generate_system.jinja2"
    template_content = template_path.read_text()
    from jinja2 import Template

    template_vars = {"phase": phase, "user_input": user_input}
    if extra_vars:
        template_vars.update(extra_vars)
    system_prompt = Template(template_content).render(**template_vars)

    _seen_words.clear()

    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(user_input),
    ]

    for turn in range(max_loops):
        reply = generator.run(messages=messages)
        replies = reply["replies"]
        msg = replies[0]

        # Newline after streaming output
        print()

        if not msg.tool_calls:
            return msg.text

        # Show tool calls
        for tc in msg.tool_calls:
            args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in tc.arguments.items())
            print(f"  🔧 {tc.tool_name}({args_str})")

        # Execute tools
        tc_result = tool_invoker.run(messages=replies)
        tool_messages = tc_result["tool_messages"]

        # Show tool results (compact)
        for tm in tool_messages:
            for content in tm._content:
                result_str = content.result
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                print(f"  → {result_str}")

        messages.extend(replies)
        messages.extend(tool_messages)

    return "Max loops reached (increase max_loops?)"


# ============================================================
# Main
# ============================================================


def main():
    num_sentences = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.randint(0, 999999)

    print(f"Generating {num_sentences} sentences with seed {seed}")
    random.seed(seed)

    # Create tools
    tools = create_tools()
    print(f"Registered {len(tools)} tools")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create generator directly
    base_url = os.environ.get("LLM_URL", "http://localhost:8001/v3")
    model = os.environ.get("OPENAI_MODEL", "gpt-oss-20b-int4-ov")

    # Patch Haystack's chunk builder to pass through reasoning_content from OVMS
    import haystack.components.generators.chat.openai as _oai_mod
    _orig_convert = _oai_mod._convert_chat_completion_chunk_to_streaming_chunk
    def _patched_convert(chunk, previous_chunks, component_info=None):
        sc = _orig_convert(chunk, previous_chunks, component_info)
        # Extract reasoning_content from OVMS/Qwen deltas
        if chunk.choices:
            rc = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if rc:
                from haystack.dataclasses.chat_message import ReasoningContent
                sc.reasoning = ReasoningContent(reasoning_text=rc)
        return sc
    _oai_mod._convert_chat_completion_chunk_to_streaming_chunk = _patched_convert

    def stream_callback(chunk):
        """Print tokens as they arrive."""
        if chunk.reasoning and chunk.reasoning.reasoning_text:
            print(f"\033[2m{chunk.reasoning.reasoning_text}\033[0m", end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)

    generator = OpenAIChatGenerator(
        model=model,
        api_key=Secret.from_env_var("OPENAI_API_KEY", strict=False),
        api_base_url=base_url,
        http_client_kwargs={"timeout": 120.0},
        tools=tools,
        streaming_callback=stream_callback,
        generation_kwargs={
            "temperature": 0.1,
            "max_completion_tokens": 8192,
        },
    )

    tool_invoker = ToolInvoker(tools=tools)

    # Test with one sentence
    test_sentence = "Der Sohn des Bauern gibt dem Hund den Pfefferkuchen"

    print(f"\n=== Testing: {test_sentence} ===")

    # Phase 1: Vocabulary (with tools)
    print("\n--- Phase 1: Vocabulary ---")
    vocab_result = run_with_tools(
        generator, tool_invoker, tools, test_sentence, "vocabulary", max_loops=15
    )
    print(f"Vocabulary:\n{vocab_result}\n")

    # Phase 2: Compose (no tools — just grammar + vocab)
    print("--- Phase 2: Compose ---")
    compose_result = run_with_tools(
        generator, tool_invoker, tools, test_sentence, "compose",
        max_loops=1, extra_vars={"vocabulary": vocab_result},
    )
    print(f"Result: {compose_result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
