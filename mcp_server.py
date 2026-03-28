"""FastMCP server for Prussian Dictionary - Pure MCP tools via stdio or SSE transport."""

import argparse
import os
from typing import Any

from mcp.server.fastmcp import FastMCP, Context
import mcp.types as types

import prussian_engine

# Initialize FastMCP
mcp = FastMCP("Prussian Dictionary")

# Load search engine at startup (no chat_engine needed for MCP tools)
print("Loading Prussian Dictionary search engine...")
search_engine = prussian_engine.SearchEngine()
print("Search engine loaded successfully!")


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
    return [{
        "word": r["word"],
        "de": r["de"],
        "en": r["en"]
    } for r in results]


@mcp.tool()
def lookup_prussian_word(word: str) -> list[dict[str, Any]]:
    """
    Look up a specific Prussian word (lemma or inflected form).

    Args:
        word: Prussian word to look up

    Returns:
        List of matching entries with translations
    """
    return search_engine.lookup(word)


@mcp.tool()
def get_word_forms(lemma: str) -> dict[str, Any]:
    """
    Get all declension or conjugation forms for a Prussian lemma.

    Args:
        lemma: Prussian lemma (base form)

    Returns:
        Dictionary with lemma, translations, and all forms
    """
    results = search_engine.lookup(lemma)
    if not results:
        return {"error": f"Word not found: {lemma}"}

    result = results[0]
    return {
        "lemma": result["word"],
        "translations": {
            "de": result["de"],
            "en": result["en"]
        },
        "paradigm": result.get("paradigm", ""),
        "gender": result.get("gender", ""),
        "forms": result.get("forms", {})
    }


@mcp.tool()
async def chat_prussian(
    ctx: Context,
    user_message: str,
    language: str = "de"
) -> dict[str, Any]:
    """
    Chat about Old Prussian using Claude with access to dictionary tools.

    The MCP server uses sampling to request Claude to generate responses.
    Claude can autonomously use the three dictionary tools during generation.

    Args:
        user_message: User's question or message in German or English
        language: Response language ('de' for German, 'en' for English)

    Returns:
        Dictionary with:
        - response: Claude's conversational response
        - model: Model used for generation
        - stop_reason: Why generation stopped (e.g., "end_turn")
    """
    session = ctx.session

    # Build system prompt for Claude
    language_name = "German" if language == "de" else "English"
    system_prompt = f"""You are an expert on Old Prussian language and culture.
You have access to a comprehensive Old Prussian dictionary with the following tools:
- search_dictionary: Find words by meaning (German/English to Prussian)
- lookup_prussian_word: Look up specific Prussian words or inflected forms
- get_word_forms: Show declensions, conjugations, and word paradigms

Guidelines:
1. Use the dictionary tools to provide accurate information
2. Explain word meanings, etymology, and usage patterns
3. Show examples when helpful
4. Always respond in {language_name}
5. Be helpful, accurate, and educational

If you don't have information, say so clearly."""

    # Create user message
    messages = [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(type="text", text=user_message)
        )
    ]

    try:
        # Request Claude to generate response via sampling
        # include_context="thisServer" tells Claude to use our 3 tools
        result = await session.create_message(
            messages=messages,
            max_tokens=2000,
            system_prompt=system_prompt,
            temperature=0.7,
            include_context="thisServer"
        )

        # Extract response text
        response_text = ""
        if hasattr(result, "content"):
            if isinstance(result.content, types.TextContent):
                response_text = result.content.text
            elif isinstance(result.content, str):
                response_text = result.content
            else:
                response_text = str(result.content)
        else:
            response_text = str(result)

        return {
            "response": response_text,
            "model": result.model if hasattr(result, "model") else "unknown",
            "stop_reason": result.stopReason if hasattr(result, "stopReason") else None
        }

    except Exception as e:
        return {
            "error": f"Sampling request failed: {str(e)}",
            "details": type(e).__name__
        }


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
        """
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=os.getenv("MCP_TRANSPORT") == "sse",
        help="Use SSE transport for Claude Web (default: stdio for local CLI)"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Server host for SSE mode (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "8001")),
        help="Server port for SSE mode (default: 8001)"
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
        print("Configure in .mcp.json with: {'command': 'python', 'args': ['mcp_server.py']}")
        mcp.run(transport="stdio")
