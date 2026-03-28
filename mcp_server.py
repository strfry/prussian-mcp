"""FastMCP server for Prussian Dictionary - Pure MCP tools via stdio transport."""

from typing import Any

from mcp.server.fastmcp import FastMCP

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run server with stdio transport (default MCP protocol)
    mcp.run(transport="stdio")
