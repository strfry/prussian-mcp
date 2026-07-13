"""FastMCP server for Prussian Dictionary - MCP tools with streaming LLM proxy."""

import argparse
import json
import os
from typing import Any, AsyncIterator
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI
from starlette.responses import Response, StreamingResponse

import prussian_engine
from prussian_engine.fsg_check import check_fsg_pipeline
from prussian_engine.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    PROMPTS_DIR,
    SYSTEM_PROMPT_PATH,
)
from prussian_engine.tools import TOOLS
from prussian_engine.rerank_search import RerankedSearchEngine

# Parse command-line arguments at module level (before FastMCP construction)
parser = argparse.ArgumentParser(
    description="Prussian Dictionary MCP Server",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Transport modes:
  stdio (default)  - For local CLI clients (Claude Code, Claude Desktop)
  sse              - For HTTP clients (Claude Web) via SSE protocol

Examples:
  python mcp_server.py                    # Local: stdio on stdin/stdout
  python mcp_server.py --web              # Web: SSE on http://localhost:8001
  python mcp_server.py --web --port 9000  # Web: SSE on custom port
    """,
)
parser.add_argument(
    "--web",
    action="store_true",
    default=os.getenv("MCP_TRANSPORT") == "sse",
    help="Use SSE transport for Claude Web (default: stdio for local CLI)",
)
parser.add_argument(
    "--host",
    default=os.getenv("MCP_HOST", "127.0.0.1"),
    help="Server host for SSE mode (default: 127.0.0.1)",
)
parser.add_argument(
    "--port",
    type=int,
    default=int(os.getenv("MCP_PORT", "8001")),
    help="Server port for SSE mode (default: 8001)",
)

args = parser.parse_args()

# Initialize FastMCP with security settings
# Allow strfry.org for remote access via SSH tunnel
security_settings = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1:*",
        "localhost:*",
        "strfry.org:*",
        "strfry.org",
    ],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "https://strfry.org", "https://strfry.org:*"],
)
mcp = FastMCP(
    "Prussian Dictionary",
    transport_security=security_settings,
    host=args.host,
    port=args.port,
    debug=True,
    log_level="DEBUG",
    #    mount_path="/prussian-mcp",  # Tells FastMCP it's running under this prefix
)

# Load search engine at startup (no chat_engine needed for MCP tools)
print("Loading Prussian Dictionary search engine...")
search_engine = prussian_engine.SearchEngine()
reranked_engine = None
print("Search engine loaded successfully!")

# Quick health check of the FST/CG3 grammar pipeline
fsg_ok, fsg_msg = check_fsg_pipeline()
print(fsg_msg)

# Initialize OpenAI client for streaming proxy
llm_client = OpenAI(api_key=OPENAI_API_KEY or "dummy", base_url=OPENAI_BASE_URL)
llm_model = OPENAI_MODEL


# Load prompts from disk (hot-reload on every call)
PLAN_PROMPT_PATH = PROMPTS_DIR / "plan_prompt.txt"
FINAL_PROMPT_PATH = PROMPTS_DIR / "final_prompt.txt"


def _load_prompt(path: Path) -> str:
    """Load a prompt from file."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "You are an assistant."


def _build_tool_descriptions() -> str:
    """Build human-readable tool descriptions from TOOLS definitions."""
    lines = []
    for t in TOOLS:
        fn = t["function"]
        params = ", ".join(
            f"{name}: {schema.get('type', 'any')}"
            + (f" = {json.dumps(schema['default'])}" if "default" in schema else "")
            for name, schema in fn["parameters"].get("properties", {}).items()
        )
        desc = " ".join(fn["description"].split())
        lines.append(f"- {fn['name']}({params}): {desc}")
    return "\n".join(lines)


def _render_prompt(path: Path, language: str = "de") -> str:
    """Load prompt, substitute {lang_code} and {tools}."""
    lang_code = "LT" if language == "lt" else "DE"
    content = _load_prompt(path).replace("{lang_code}", lang_code)
    return content.replace("{tools}", _build_tool_descriptions())


@mcp.prompt()
def chat(language: str = "de") -> str:
    """System prompt for Prussian chatbot — understands user input via tool calls."""
    return _render_prompt(SYSTEM_PROMPT_PATH, language)


@mcp.prompt()
def plan(language: str = "de") -> str:
    """Planning prompt — responds in German and searches translations for each word."""
    return _render_prompt(PLAN_PROMPT_PATH, language)


@mcp.prompt()
def final(language: str = "de") -> str:
    """Final prompt — formulates the Prussian response from compacted results."""
    return _render_prompt(FINAL_PROMPT_PATH, language)


# ── Grammar Resources ─────────────────────────────────────────────────────────

GRAMMAR_SYNTAX_PATH = PROMPTS_DIR / "syntax_rules.txt"


@mcp.resource("grammar://syntax")
def grammar_syntax() -> str:
    """Prußische Syntaxregeln – Kondensierte Regeln zu Syntax, Prosodie und Kasusrektion (Morphologie steht im Wörterbuch)."""
    return _load_prompt(GRAMMAR_SYNTAX_PATH)


# ── Grammar Injection ──────────────────────────────────────────────────────────


def _inject_grammar(grammar: bool | list[str] | str | None) -> str:
    """Load grammar sources for injection after system prompt.

    Args:
        grammar: True for all sources, list of source names, or single source name.
                 False/None for no grammar.

    Returns:
        Grammar text to append, or empty string.
    """
    if not grammar:
        return ""

    grammar_sources = {
        "syntax": PROMPTS_DIR / "syntax_rules.txt",
    }

    if isinstance(grammar, str):
        grammar = [grammar]

    if isinstance(grammar, bool) or grammar == ["all"]:
        grammar = list(grammar_sources.keys())

    parts = []
    for key in grammar:
        path = grammar_sources.get(key)
        if path and path.exists():
            content = _load_prompt(path)
            parts.append(f"<grammar source=\"{key}\">\n{content}\n</grammar>")

    if not parts:
        return ""

    return "\n\n\n".join(parts)


# ── Streaming LLM Proxy ──────────────────────────────────────────────────────


def _format_system_prompt(language: str = "de", grammar: bool | list[str] | str | None = None) -> str:
    """Format system prompt with language code and optional grammar injection."""
    lang_code = "LT" if language == "lt" else "DE"
    content = _load_prompt(SYSTEM_PROMPT_PATH).replace("{lang_code}", lang_code)
    content = content.replace("{tools}", _build_tool_descriptions())

    grammar_text = _inject_grammar(grammar)
    if grammar_text:
        content += "\n\n" + grammar_text

    return content


def _build_llm_kwargs(
    messages, tools, temperature, max_tokens, language, grammar=None, *, stream=True, tool_choice=None
):
    """Build kwargs for llm_client.chat.completions.create."""
    system_content = _format_system_prompt(language, grammar=grammar)
    full_messages = [{"role": "system", "content": system_content}]
    full_messages.extend(messages)
    return dict(
        model=llm_model,
        messages=full_messages,
        tools=tools,
        tool_choice=tool_choice or ("required" if tools else None),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


def _sse_event(event_type: str, data: Any) -> str:
    """Format data as SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _stream_completions(
    messages: list,
    tools: list | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    language: str = "de",
    grammar: bool | list[str] | str | None = None,
    tool_choice: str | None = None,
) -> AsyncIterator[bytes]:
    """Stream completions from LLM with tool call support."""
    try:
        stream = llm_client.chat.completions.create(
            **_build_llm_kwargs(messages, tools, temperature, max_tokens, language, grammar=grammar, tool_choice=tool_choice)
        )

        tool_calls_buffer: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Handle reasoning content (DeepSeek R1 style)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield _sse_event(
                    "reasoning_delta", {"content": delta.reasoning_content}
                ).encode()

            # Handle content
            if delta.content:
                yield _sse_event("content_delta", {"content": delta.content}).encode()

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    if tc.id:
                        tool_calls_buffer[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buffer[idx]["function"]["name"] = (
                                tc.function.name
                            )
                        if tc.function.arguments:
                            tool_calls_buffer[idx]["function"]["arguments"] += (
                                tc.function.arguments
                            )

                    yield _sse_event(
                        "tool_call_delta",
                        {"index": idx, "tool_call": tool_calls_buffer[idx]},
                    ).encode()

            # Handle done
            if chunk.choices[0].finish_reason:
                yield _sse_event(
                    "done", {"finish_reason": chunk.choices[0].finish_reason}
                ).encode()

    except Exception as e:
        yield _sse_event("error", {"error": str(e)}).encode()


@mcp.custom_route("/api/completions", methods=["POST"])
async def completions_endpoint(request):
    """
    Streaming LLM proxy endpoint with custom SSE format.

    Request JSON:
        - messages: Chat messages array (list)
        - tools: Tool definitions array (list, optional)
        - temperature: Sampling temperature (float, default 0.7)
        - max_tokens: Maximum tokens (int, default 2000)
        - language: Response language 'de' or 'lt' (str, default 'de')
        - grammar: Grammar sources to inject (bool, list, str, or null; default null)

    Response: SSE stream with events:
        - content_delta: {"content": string}
        - reasoning_delta: {"content": string}
        - tool_call_delta: {"index": int, "tool_call": {...}}
        - done: {"finish_reason": string}
        - error: {"error": string}
    """
    try:
        data = await request.json()
        messages = data.get("messages", [])
        tools = data.get("tools", TOOLS)
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens", 2000))
        language = data.get("language", "de")
        grammar = data.get("grammar", None)

        return StreamingResponse(
            _stream_completions(messages, tools, temperature, max_tokens, language, grammar=grammar),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        return Response(
            f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n",
            media_type="text/event-stream",
            status_code=500,
        )


@mcp.custom_route("/v1/chat/completions", methods=["POST"])
async def openai_completions_endpoint(request):
    """
    OpenAI-compatible chat completions endpoint (non-streaming only).
    For streaming, use /api/completions.
    """
    try:
        data = await request.json()
        messages = data.get("messages", [])
        tools = data.get("tools", TOOLS)
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens", 2000))
        language = data.get("language", "de")
        model = data.get("model", "prussian-chat")
        grammar = data.get("grammar", None)
        tool_choice = data.get("tool_choice", None)

        if data.get("stream", False):
            return StreamingResponse(
                _stream_completions(messages, tools, temperature, max_tokens, language, grammar=grammar, tool_choice=tool_choice),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = llm_client.chat.completions.create(
            **_build_llm_kwargs(
                messages, tools, temperature, max_tokens, language, grammar=grammar, stream=False, tool_choice=tool_choice
            )
        )

        return Response(
            json.dumps(
                {
                    "id": f"chatcmpl-{response.id or 'none'}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                                "tool_calls": response.choices[0].message.tool_calls
                                or [],
                            },
                            "finish_reason": response.choices[0].finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else {},
                }
            ),
            media_type="application/json",
        )

    except Exception as e:
        return Response(
            json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


# ── MCP Tools ────────────────────────────────────────────────────────────────

from tools import search_tool, lookup_tool, wordforms_tool, validate_tool


@mcp.tool()
def search_dictionary(
    query: str, top_k: int = 10, use_reranker: bool = True, filter_tags: str = None
) -> list[dict[str, Any]]:
    """
    Semantic search in the Prussian dictionary.
    Use this when you have a concept or modern language word and want to find 
    the Prussian equivalent. Do NOT use for looking up known Prussian forms –
    use lookup_prussian_word instead.

    Args:
        query: Search query in multiple languages (German, English, Lithuanian, Latvian, Polish, Russian)
        top_k: Number of results to return
        use_reranker: More accurate but slower. Use False for simple lookups.
        filter_tags: Optional FST tag filter, e.g. "Akk+Sg", "Part+Pass", "Opt".
            When set, each entry's forms are filtered to those matching the tags.

    Returns:
        List of dictionary entries with translations and optionally filtered forms
    """
    global reranked_engine

    if use_reranker and reranked_engine is None:
        reranked_engine = RerankedSearchEngine(use_reranker=True)

    return search_tool(
        search_engine,
        query,
        top_k=top_k,
        use_reranker=use_reranker,
        filter_tags=filter_tags,
        reranked_engine=reranked_engine if use_reranker else None,
    )


@mcp.tool()
def lookup_prussian_word(text: str, fuzzy: bool = False) -> list[dict[str, Any]]:
    """
    Look up a Prussian sentence: tokenize, FST-analyze, enrich from dictionary.
    Each token is analyzed via the FST cascade and enriched with translations.
    Tokens without FST analyses fall back to dictionary lookup.

    Args:
        text: Prussian text (one or more sentences). Whole sentences are the normal case.
        fuzzy: Set to True for Levenshtein fallback on OOV tokens.
    """
    return lookup_tool(search_engine, text, fuzzy=fuzzy)


@mcp.tool()
def get_word_forms(lemma: str, features: str = None) -> list[dict[str, Any]]:
    """
    Get all declension or conjugation forms for a Prussian lemma.
    Returns a flat list of forms with their FST tags.
    For verbs, default shows indicative present forms plus available features.
    Use features to request specific categories (e.g. "participle", "Gen+Pl").

    Args:
        lemma: Prussian base form (from lookup_prussian_word result)
        features: Optional feature filter, e.g. "participle", "Ind+Pres",
            "Gen+Pl". Accepts human-readable names or raw FST tags.
    """
    return wordforms_tool(search_engine, lemma, features=features)


@mcp.tool()
def validate_prussian(text: str, include_conllu: bool = False) -> str:
    """
    Grammar check of Prussian text (FST + CG3 pipeline, three-valued).
    The single grammar tool: validates sentences and can also return the
    full dependency analysis.

    Args:
        text: Prussian sentence(s) to check (one or more sentences)
        include_conllu: also include each sentence's CoNLL-U dependency
            analysis (field "conllu", with rule provenance Rule=/
            AgrParent= in MISC) — same pipeline run, no extra cost.
            Set true when you need the parse, not just the verdict.

    Returns:
        JSON: {"overall": {"status", "n_sentences", "n_violations"},
        "sentences": [{sent_id, text, status, violations, coverage,
        conllu?}, ...]}

    Interpretation guide (IMPORTANT):
    - "verified_in_coverage" is the ONLY positive evidence of
      well-formedness: all words analyzed, low ambiguity, and at least
      one check family applied to the sentence.
    - "out_of_coverage" does NOT mean correct — the checker simply
      cannot verify (unknown words, collapsed analyses, residual
      ambiguity, no applicable check, or a nominal whose case has no
      licensing context — reason "unlicensed_case"; see
      coverage.reasons).  Never treat it as approval.
    - "violations_found": each violations[] entry names the rule, the
      offending form (with index and surviving reading), a message, and
      a severity — "error" (case government / valency / person or
      number clash) is reliable; "warning" (adjective agreement,
      nominative in PP, duplicate subject) is often a loanword paradigm
      gap rather than a real error.

    CoNLL-U note: the fallback deprel "dep" means the token could not
    be attached with a licensed relation; sentences with free-floating
    nominals degrade to out_of_coverage (unlicensed_case), never to
    verified.
    """
    return validate_tool(text, include_conllu=include_conllu)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if args.web:
        print("Starting MCP server in web mode (SSE)")
        print(f"  Address: http://{args.host}:{args.port}")
        print(f"  SSE endpoint: http://{args.host}:{args.port}/sse")
        print("\nConfigure in Claude Web with:")
        print(f"  {{'type': 'sse', 'url': 'http://{args.host}:{args.port}/sse'}}")

        # Run with streamable-http transport (modern MCP protocol)
        mcp.run(transport="streamable-http")
    else:
        print("Starting MCP server in local mode (stdio)")
        print(
            "Configure in .mcp.json with: {'command': 'python', 'args': ['mcp_server.py']}"
        )
        mcp.run(transport="stdio")
