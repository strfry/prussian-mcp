"""Haystack pipeline for German → Old Prussian translation with tool calling."""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Callable

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.tools import create_tool_from_function
from haystack.utils import Secret

from prussian_engine.search import SearchEngine
from prussian_engine.config import OPENAI_BASE_URL, OPENAI_MODEL, PROMPTS_DIR


def create_lookup_tool(engine: SearchEngine):
    """Create the combined lookup tool function bound to an engine instance."""

    def lookup(query: str, pgr: str = None) -> str:
        """Search the Prussian dictionary and optionally filter by grammatical form.

        Use this to find Prussian translations for German words.
        If you need a specific grammatical form (e.g. accusative singular),
        pass the pgr parameter.

        Args:
            query: German word or phrase to search for
            pgr: Optional PGR filter for grammatical form (e.g. "NOM.SG.MASC", "ACC.PL", "GEN.SG.FEM")
        """
        # Step 1: Semantic search
        results = engine.query(query, top_k=5)

        output = {"matches": results}

        # Step 2: If pgr filter requested, get matching forms for top results
        if pgr and results:
            form_results = []
            for match in results[:3]:
                entries = engine.get_word_forms(match["word"], filter_pgr=pgr)
                if isinstance(entries, dict) and "error" in entries:
                    continue
                for entry in entries:
                    if entry.get("forms"):
                        form_results.append({
                            "lemma": entry["lemma"],
                            "matching_forms": entry["forms"],
                            "translations": entry.get("translations", {}),
                        })
            if form_results:
                output["forms"] = form_results

        return json.dumps(output, ensure_ascii=False)

    return lookup


def create_tools(engine: SearchEngine) -> list:
    """Create Haystack Tool objects for the pipeline."""
    lookup_fn = create_lookup_tool(engine)
    lookup_tool = create_tool_from_function(
        function=lookup_fn,
        description=(
            "Search the Prussian dictionary by German/English query. "
            "Optionally filter by PGR (Prussian Glossing Rules) to get "
            "the exact grammatical form needed."
        ),
    )
    return [lookup_tool]


def build_pipeline(
    engine: SearchEngine,
    *,
    model: str = None,
    api_base_url: str = None,
    api_key: str = None,
    streaming_callback: Optional[Callable] = None,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> tuple[Pipeline, list]:
    """Build and return the translation pipeline.

    Returns:
        Tuple of (pipeline, tools) for use with run_translation().
    """
    model = model or os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
    api_base_url = api_base_url or os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

    tools = create_tools(engine)

    # Load translation prompt template
    template_path = PROMPTS_DIR / "translate.jinja2"
    template_content = template_path.read_text()

    pipeline = Pipeline()

    # Prompt Builder
    prompt = ChatPromptBuilder(
        template=[
            ChatMessage.from_system(template_content),
            ChatMessage.from_user("Übersetze: {{user_input}}"),
        ],
        variables=["user_input"],
        required_variables=["user_input"],
    )

    # LLM Generator
    generator_kwargs = dict(
        model=model,
        api_key=Secret.from_token(api_key),
        api_base_url=api_base_url,
        http_client_kwargs={"timeout": 120.0},
        tools=tools,
        generation_kwargs={
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        },
    )
    if streaming_callback:
        generator_kwargs["streaming_callback"] = streaming_callback

    generator = OpenAIChatGenerator(**generator_kwargs)

    # Tool Invoker
    tool_invoker = ToolInvoker(tools=tools)

    # Router: tool_calls vs final text
    router = ConditionalRouter(
        routes=[
            {
                "condition": "{{replies[0].tool_calls | length > 0}}",
                "output": "{{replies}}",
                "output_name": "tool_calls",
                "output_type": List[ChatMessage],
            },
            {
                "condition": "{{replies[0].tool_calls | length == 0}}",
                "output": "{{replies}}",
                "output_name": "text",
                "output_type": List[ChatMessage],
            },
        ],
        unsafe=True,
    )

    # Wire components
    pipeline.add_component("prompt", prompt)
    pipeline.add_component("generator", generator)
    pipeline.add_component("router", router)
    pipeline.add_component("tool_invoker", tool_invoker)

    pipeline.connect("prompt", "generator")
    pipeline.connect("generator.replies", "router")
    pipeline.connect("router.tool_calls", "tool_invoker")

    return pipeline, tools


def run_translation(
    engine: SearchEngine,
    sentence: str,
    *,
    model: str = None,
    api_base_url: str = None,
    api_key: str = None,
    streaming_callback: Optional[Callable] = None,
    max_loops: int = 10,
    temperature: float = 0.1,
) -> str:
    """Translate a German sentence to Old Prussian using tool-calling loop.

    Args:
        engine: Initialized SearchEngine instance
        sentence: German sentence to translate
        model: LLM model name (default from env/config)
        api_base_url: LLM API base URL (default from env/config)
        api_key: API key (default from env)
        streaming_callback: Optional callback for token streaming
        max_loops: Maximum tool-calling iterations
        temperature: LLM temperature

    Returns:
        The final translated text from the LLM.
    """
    model = model or os.environ.get("OPENAI_MODEL", OPENAI_MODEL)
    api_base_url = api_base_url or os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

    tools = create_tools(engine)

    # Load prompt
    template_path = PROMPTS_DIR / "translate.jinja2"
    template_content = template_path.read_text()

    from jinja2 import Template

    system_prompt = Template(template_content).render(user_input=sentence)

    # Create generator
    generator_kwargs = dict(
        model=model,
        api_key=Secret.from_token(api_key),
        api_base_url=api_base_url,
        http_client_kwargs={"timeout": 120.0},
        tools=tools,
        generation_kwargs={
            "temperature": temperature,
            "max_completion_tokens": 800,
        },
    )
    if streaming_callback:
        generator_kwargs["streaming_callback"] = streaming_callback

    generator = OpenAIChatGenerator(**generator_kwargs)
    tool_invoker = ToolInvoker(tools=tools)

    # Tool-calling loop
    messages = [
        ChatMessage.from_system(system_prompt),
        ChatMessage.from_user(f"Übersetze: {sentence}"),
    ]

    for turn in range(max_loops):
        reply = generator.run(messages=messages)
        replies = reply["replies"]

        if not replies[0].tool_calls:
            return replies[0].text

        # Execute tool calls and feed results back
        tc_result = tool_invoker.run(messages=replies)
        tool_messages = tc_result["tool_messages"]

        messages.extend(replies)
        messages.extend(tool_messages)

    return "Max tool-calling loops reached"


# CLI entry point
if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", "not-needed")

    engine = SearchEngine()
    print(f"Loaded {len(engine.entries)} dictionary entries")

    sentence = sys.argv[1] if len(sys.argv) > 1 else "Der Sohn des Bauern gibt dem Hund den Pfefferkuchen"
    print(f"\nTranslating: {sentence}")

    tokens = []

    def on_token(chunk):
        if chunk.content:
            tokens.append(chunk.content)
            print(chunk.content, end="", flush=True)

    result = run_translation(engine, sentence, streaming_callback=on_token)
    print(f"\n\nResult: {result}")
