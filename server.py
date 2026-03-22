"""FastMCP server for Prussian Dictionary."""

import json
from pathlib import Path
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

import prussian_engine

# Initialize FastMCP
mcp = FastMCP("Prussian Dictionary")

# Load engine at startup
print("Loading Prussian Dictionary engine...")
search_engine, chat_engine = prussian_engine.load()
print("Engine loaded successfully!")


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


# ── REST Endpoints ───────────────────────────────────────────────────────────


@mcp.custom_route("/prussian-api/chat", methods=["POST"])
async def chat_endpoint(request):
    """
    Chat endpoint for the Prussian Dictionary chatbot.

    Request JSON:
        - message: User message (str)
        - language: Output language 'de' or 'lt' (str)
        - history: Conversation history (list)

    Response JSON:
        - prussian: Response in Old Prussian (str)
        - german/lithuanian: Translation (str)
        - usedWords: List of dictionary words used (list)
        - debugInfo: Debug information (dict)
        - history: Updated conversation history (list)
    """
    try:
        data = await request.json()
        message = data.get("message", "")
        language = data.get("language", "de")
        history = data.get("history", [])

        if not message:
            return {"error": "No message provided"}, 400

        # Process message
        result = chat_engine.send_message(message, language, history)

        return result, 200

    except Exception as e:
        return {"error": str(e)}, 500


# ── Static Files ─────────────────────────────────────────────────────────────

# Serve static files from ui/ directory
static_dir = Path(__file__).parent / "ui"
if static_dir.exists():
    mcp.static_files("/", static_dir)
    print(f"Serving static files from: {static_dir}")
else:
    print(f"Warning: Static directory not found: {static_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run server
    mcp.run()
