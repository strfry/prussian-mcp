"""FastMCP server for Prussian Dictionary."""

import json
from pathlib import Path
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

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
        - translation: Translation (str)
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
            return JSONResponse({"error": "No message provided"}, status_code=400)

        # Process message
        result = chat_engine.send_message(message, language, history)

        return JSONResponse(result, status_code=200)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Static Files ─────────────────────────────────────────────────────────────

# Serve static files from ui/ directory
static_dir = Path(__file__).parent / "ui"

if static_dir.exists():
    from starlette.responses import FileResponse, Response, JSONResponse
    import mimetypes

    @mcp.custom_route("/chatbot.html", methods=["GET"])
    async def serve_chatbot_html(request):
        """Serve the chatbot HTML page."""
        return FileResponse(static_dir / "chatbot.html")

    @mcp.custom_route("/chatbot.js", methods=["GET"])
    async def serve_chatbot_js(request):
        """Serve the chatbot JavaScript."""
        return FileResponse(static_dir / "chatbot.js", media_type="application/javascript")

    @mcp.custom_route("/images/{filename}", methods=["GET"])
    async def serve_images(request):
        """Serve image files."""
        filename = request.path_params.get("filename", "")
        # Prevent directory traversal
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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run server with SSE transport to enable HTTP endpoints
    mcp.run(transport="sse")
