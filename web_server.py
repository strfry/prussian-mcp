"""FastMCP server for Prussian Dictionary - Web server with REST API and static files."""

import json
import mimetypes
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse, FileResponse, Response

import prussian_engine

# Initialize FastMCP with SSE transport for HTTP support
mcp = FastMCP("Prussian Dictionary Web")

# Load both engines at startup
print("Loading Prussian Dictionary engines...")
search_engine, chat_engine = prussian_engine.load()
print("Engines loaded successfully!")


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


# ── Static Files ──────────────────────────────────────────────────────────────

# Serve static files from ui/ directory
static_dir = Path(__file__).parent / "ui"

if static_dir.exists():

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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run server with SSE transport to enable HTTP endpoints
    mcp.run(transport="sse")
