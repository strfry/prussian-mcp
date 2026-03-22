#!/usr/bin/env python3
"""Flask API for Prussian Dictionary RAG endpoints."""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import sys
import os
import requests

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib"))
from prussian_search_skill import PrussianSearch
from chat_engine import ChatEngine

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__, static_folder=os.path.join(project_root, "ui"), static_url_path=""
)
CORS(app)  # Enable CORS for browser requests

# Load embeddings once at startup
print("Loading embeddings...", file=sys.stderr)
search = PrussianSearch(
    embeddings_path=os.path.join(project_root, "embeddings", "embeddings_e5_prefix")
)
print("Embeddings loaded successfully!", file=sys.stderr)

# Load full dictionary for forms lookup
print("Loading dictionary...", file=sys.stderr)
with open(
    os.path.join(project_root, "data", "prussian_dictionary.json"),
    "r",
    encoding="utf-8",
) as f:
    dictionary = json.load(f)
    if isinstance(dictionary, dict):
        dictionary = list(dictionary.values())
print(f"Dictionary loaded: {len(dictionary)} entries", file=sys.stderr)

# Initialize chat engine
print("Initializing chat engine...", file=sys.stderr)
chat_engine = ChatEngine(
    search=search,
    dictionary=dictionary,
    hf_token=os.environ.get("HF_TOKEN", ""),  # Not needed for local model
    hf_model=os.environ.get("HF_MODEL", "localhost:8001/v3"),
)
print("Chat engine initialized!", file=sys.stderr)


@app.route("/api/search", methods=["POST"])
def semantic_search():
    """
    Semantic search endpoint.

    Request: {"query": "pruße", "top_k": 10}
    Response: {"results": [...]}
    """
    data = request.json
    query = data.get("query", "")
    top_k = data.get("top_k", 10)

    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        results = search.query(query, top_k=top_k)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forms", methods=["POST"])
def get_forms():
    """
    Forms lookup endpoint.

    Given a Prussian lemma, return all inflected forms.

    Request: {"lemma": "prūss"}
    Response: {
        "lemma": "prūss",
        "forms": {
            "declension": [...],
            "indicative": [...],
            ...
        }
    }
    """
    data = request.json
    lemma = data.get("lemma", "")

    if not lemma:
        return jsonify({"error": "Missing lemma"}), 400

    # Find entry by lemma (case-insensitive)
    entry = next(
        (e for e in dictionary if e.get("word", "").lower() == lemma.lower()), None
    )

    if not entry:
        return jsonify({"error": f"Lemma '{lemma}' not found"}), 404

    return jsonify(
        {
            "lemma": lemma,
            "forms": entry.get("forms", {}),
            "translations": entry.get("translations", {}),
            "paradigm": entry.get("paradigm"),
            "gender": entry.get("gender"),
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "embeddings_loaded": search.embeddings is not None,
            "dictionary_entries": len(dictionary),
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint - handles full conversation with tool calling.

    Request: {
        "message": "Hi",
        "language": "de",  # or "lt"
        "history": [],  # optional, previous conversation history
        "system_prompt": "..."  # optional, generate if missing
    }

    Response: {
        "prussian": "Kaīls!",
        "german": "Hallo!",
        "usedWords": ["kaīls"],
        "history": [...],  # updated history for client to maintain
        "debugInfo": {
            "query": "Hi",
            "toolCalls": [...],
            "results": [...],
            "usedWords": [...],
            "systemPrompt": "..."
        }
    }
    """
    data = request.json
    if not data or not data.get("message"):
        return jsonify({"error": "Missing message"}), 400

    message = data["message"]
    language = data.get("language", "de")
    history = data.get("history", [])  # Client sends conversation history

    # Generate system prompt if not provided
    system_prompt = data.get("system_prompt")
    if not system_prompt:
        system_prompt = generate_system_prompt(language)

    try:
        result = chat_engine.send_message(message, system_prompt, history)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_system_prompt(language="de"):
    """Generate system prompt from template file."""
    lang_code = "DE" if language == "de" else "LT"

    # Load template from file
    prompts_dir = os.path.join(project_root, "prompts")
    template_path = os.path.join(prompts_dir, "system_prompt.txt")

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read().strip()

    return template.format(lang_code=lang_code)


@app.route("/api/llm", methods=["POST"])
def llm_proxy():
    """
    LLM proxy endpoint - forwards requests to HuggingFace.

    Converts Anthropic format -> OpenAI format -> HuggingFace -> Anthropic format

    Request: {"system": "...", "messages": [...], "max_tokens": 1000, "tools": [...]}
    Response: {"content": [{"type": "text", "text": "..."}] or [..., {"type": "tool_use", ...}]}
    """
    data = request.json
    if not data:
        return jsonify({"error": "No input"}), 400

    # Convert Anthropic format to OpenAI format
    messages = []
    if data.get("system"):
        messages.append({"role": "system", "content": data["system"]})

    for msg in data.get("messages", []):
        # Handle tool results (Anthropic format)
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    # Convert to OpenAI tool result format
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        }
                    )
        # Handle assistant messages with tool calls
        elif msg["role"] == "assistant" and isinstance(msg.get("content"), list):
            # Extract text and tool calls
            text_parts = []
            tool_calls = []

            for block in msg["content"]:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            }
                        )

            assistant_msg = {
                "role": "assistant",
                "content": " ".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
        else:
            # Regular message
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Convert tools from Anthropic to OpenAI format
    tools = None
    if data.get("tools"):
        tools = []
        for tool in data["tools"]:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )

    # HuggingFace configuration
    HF_TOKEN = os.environ.get("HF_TOKEN", "HF_TOKEN_REDACTED")
    MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    # Build request body
    request_body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": data.get("max_tokens", 1000),
        "temperature": 0.7,
        "stream": False,
    }

    # Add tools if present
    if tools:
        request_body["tools"] = tools

    # Call HuggingFace
    try:
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {HF_TOKEN}",
            },
            json=request_body,
            timeout=30,
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), response.status_code

        # Convert OpenAI format back to Anthropic format
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]

            # Check if there are tool calls
            if message.get("tool_calls"):
                # Convert tool calls to Anthropic format
                content = []
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})

                for tool_call in message["tool_calls"]:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": json.loads(tool_call["function"]["arguments"]),
                        }
                    )

                return jsonify({"content": content})
            else:
                # Regular text response
                text = message.get("content", "")
                return jsonify({"content": [{"type": "text", "text": text}]})
        else:
            return jsonify({"error": "No response from LLM"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


# Serve UI files
@app.route("/")
def index():
    """Serve the main chatbot UI."""
    return send_from_directory(app.static_folder, "chatbot.html")


@app.route("/data/<path:filename>")
def serve_data(filename):
    """Serve data files (dictionary, wordlist, etc.)."""
    data_dir = os.path.join(project_root, "data")
    return send_from_directory(data_dir, filename)


@app.route("/ui/<path:filename>")
def serve_ui(filename):
    """Serve UI assets."""
    return send_from_directory(app.static_folder, filename)


if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=5000, debug=True)
