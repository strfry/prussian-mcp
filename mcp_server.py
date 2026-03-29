"""FastMCP server for Prussian Dictionary - MCP tools with streaming LLM proxy."""

import argparse
import json
import mimetypes
import os
from typing import Any, AsyncIterator
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI
from starlette.responses import Response, StreamingResponse, FileResponse

import prussian_engine
from prussian_engine.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SYSTEM_PROMPT_PATH,
)
from prussian_engine.tools import TOOLS

# Initialize FastMCP with security settings
# Allow strfry.org for remote access via SSH tunnel
security_settings = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1:*",
        "localhost:*",
        "strfry.org",
    ],  # No port in Host header
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "https://strfry.org"],
)
mcp = FastMCP(
    "Prussian Dictionary",
    transport_security=security_settings,
    debug=True,
    log_level="DEBUG",
    #    mount_path="/prussian-mcp",  # Tells FastMCP it's running under this prefix
)

# Load search engine at startup (no chat_engine needed for MCP tools)
print("Loading Prussian Dictionary search engine...")
search_engine = prussian_engine.SearchEngine()
print("Search engine loaded successfully!")

# Initialize OpenAI client for streaming proxy
llm_client = OpenAI(api_key=OPENAI_API_KEY or "dummy", base_url=OPENAI_BASE_URL)
llm_model = OPENAI_MODEL


# Load system prompt
def _load_system_prompt() -> str:
    """Load system prompt from file."""
    if SYSTEM_PROMPT_PATH.exists():
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "You are an assistant."


system_prompt_template = _load_system_prompt()


# ── Streaming LLM Proxy ──────────────────────────────────────────────────────


def _format_system_prompt(language: str = "de") -> str:
    """Format system prompt with language code."""
    lang_code = "LT" if language == "lt" else "DE"
    return system_prompt_template.replace("{lang_code}", lang_code)


def _build_llm_kwargs(
    messages, tools, temperature, max_tokens, language, *, stream=True
):
    """Build kwargs for llm_client.chat.completions.create."""
    system_content = _format_system_prompt(language)
    full_messages = [{"role": "system", "content": system_content}]
    full_messages.extend(messages)
    return dict(
        model=llm_model,
        messages=full_messages,
        tools=tools,
        tool_choice="required" if tools else None,
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
) -> AsyncIterator[bytes]:
    """Stream completions from LLM with tool call support."""
    try:
        stream = llm_client.chat.completions.create(
            **_build_llm_kwargs(messages, tools, temperature, max_tokens, language)
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

        return StreamingResponse(
            _stream_completions(messages, tools, temperature, max_tokens, language),
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

        if data.get("stream", False):
            return StreamingResponse(
                _stream_completions(messages, tools, temperature, max_tokens, language),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        response = llm_client.chat.completions.create(
            **_build_llm_kwargs(
                messages, tools, temperature, max_tokens, language, stream=False
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


@mcp.tool()
def search_dictionary(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Semantic search in the Prussian dictionary.

    Args:
        query: Search query in German or English
        top_k: Number of results to return

    Returns:
        List of dictionary entries with translations
    """
    results = search_engine.query(query, top_k)
    return [{"word": r["word"], "de": r["de"], "en": r["en"]} for r in results]


@mcp.tool()
def lookup_prussian_word(word: str, fuzzy: bool = True) -> list[dict[str, Any]]:
    """
    Look up a specific Prussian word (lemma or inflected form).

    Args:
        word: Prussian word to look up

    Returns:
        List of matching entries with translations
    """
    return search_engine.lookup(word, fuzzy=fuzzy)


@mcp.tool()
def get_word_forms(lemma: str) -> dict[str, Any]:
    """
    Get all declension or conjugation forms for a Prussian lemma.

    Args:
        lemma: Prussian lemma (base form)

    Returns:
        Dictionary with lemma, translations, and all forms
    """
    return search_engine.get_word_forms(lemma)


# ── Static Files ────────────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "ui"

if static_dir.exists():

    @mcp.custom_route("/chatbot.html", methods=["GET"])
    async def serve_chatbot_html(request):
        """Serve the chatbot HTML page."""
        return FileResponse(static_dir / "chatbot.html")

    @mcp.custom_route("/chatbot.js", methods=["GET"])
    async def serve_chatbot_js(request):
        """Serve the chatbot JavaScript."""
        return FileResponse(
            static_dir / "chatbot.js", media_type="application/javascript"
        )

    @mcp.custom_route("/mcp-client.js", methods=["GET"])
    async def serve_mcp_client_js(request):
        """Serve the MCP client JavaScript."""
        return FileResponse(
            static_dir / "mcp-client.js", media_type="application/javascript"
        )

    @mcp.custom_route("/chat-engine.js", methods=["GET"])
    async def serve_chat_engine_js(request):
        """Serve the chat engine JavaScript."""
        return FileResponse(
            static_dir / "chat-engine.js", media_type="application/javascript"
        )

    @mcp.custom_route("/images/{filename}", methods=["GET"])
    async def serve_images(request):
        """Serve image files."""
        filename = request.path_params.get("filename", "")
        if ".." in filename or "/" in filename:
            return Response("Invalid filename", status_code=400)

        image_path = static_dir / "images" / filename
        if image_path.exists() and image_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(image_path))
            return FileResponse(image_path, media_type=mime_type)
        return Response("Image not found", status_code=404)

    print(f"Serving static files from: {static_dir}")
else:
    print(f"Warning: Static directory not found: {static_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    if args.web:
        # For SSE mode, set environment variables for uvicorn
        os.environ["FASTMCP_HOST"] = args.host
        os.environ["FASTMCP_PORT"] = str(args.port)

        print(f"Starting MCP server in web mode (SSE)")
        print(f"  Address: http://{args.host}:{args.port}")
        print(f"  SSE endpoint: http://{args.host}:{args.port}/sse")
        print(f"\nConfigure in Claude Web with:")
        print(f"  {{'type': 'sse', 'url': 'http://{args.host}:{args.port}/sse'}}")

        # Run with SSE transport
        mcp.run(transport="sse")
    else:
        print("Starting MCP server in local mode (stdio)")
        print(
            "Configure in .mcp.json with: {'command': 'python', 'args': ['mcp_server.py']}"
        )
        mcp.run(transport="stdio")
