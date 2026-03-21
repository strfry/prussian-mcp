#!/usr/bin/env python3
"""Flask API for Prussian Dictionary RAG endpoints."""

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import json
import sys
import os
import requests

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))
from prussian_search_skill import PrussianSearch

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__,
            static_folder=os.path.join(project_root, 'ui'),
            static_url_path='')
CORS(app)  # Enable CORS for browser requests

# Load embeddings once at startup
print("Loading embeddings...", file=sys.stderr)
search = PrussianSearch(embeddings_path='../embeddings/embeddings_e5_prefix')
print("Embeddings loaded successfully!", file=sys.stderr)

# Load full dictionary for forms lookup
print("Loading dictionary...", file=sys.stderr)
with open('../data/prussian_dictionary.json', 'r', encoding='utf-8') as f:
    dictionary = json.load(f)
    if isinstance(dictionary, dict):
        dictionary = list(dictionary.values())
print(f"Dictionary loaded: {len(dictionary)} entries", file=sys.stderr)


@app.route('/api/search', methods=['POST'])
def semantic_search():
    """
    Semantic search endpoint.

    Request: {"query": "pruße", "top_k": 10}
    Response: {"results": [...]}
    """
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 10)

    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        results = search.query(query, top_k=top_k)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/forms', methods=['POST'])
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
    lemma = data.get('lemma', '')

    if not lemma:
        return jsonify({"error": "Missing lemma"}), 400

    # Find entry by lemma (case-insensitive)
    entry = next((e for e in dictionary if e.get('word', '').lower() == lemma.lower()), None)

    if not entry:
        return jsonify({"error": f"Lemma '{lemma}' not found"}), 404

    return jsonify({
        "lemma": lemma,
        "forms": entry.get('forms', {}),
        "translations": entry.get('translations', {}),
        "paradigm": entry.get('paradigm'),
        "gender": entry.get('gender')
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "embeddings_loaded": search.embeddings is not None,
        "dictionary_entries": len(dictionary)
    })


@app.route('/api/llm', methods=['POST'])
def llm_proxy():
    """
    LLM proxy endpoint - forwards requests to HuggingFace.

    Converts Anthropic format -> OpenAI format -> HuggingFace -> Anthropic format

    Request: {"system": "...", "messages": [...], "max_tokens": 1000}
    Response: {"content": [{"type": "text", "text": "..."}]}
    """
    data = request.json
    if not data:
        return jsonify({"error": "No input"}), 400

    # Convert Anthropic format to OpenAI format
    messages = []
    if data.get('system'):
        messages.append({"role": "system", "content": data['system']})

    for msg in data.get('messages', []):
        messages.append({"role": msg['role'], "content": msg['content']})

    # HuggingFace configuration
    HF_TOKEN = os.environ.get('HF_TOKEN', 'HF_TOKEN_REDACTED')
    MODEL = os.environ.get('HF_MODEL', 'Qwen/Qwen2.5-72B-Instruct')

    # Call HuggingFace
    try:
        response = requests.post(
            'https://router.huggingface.co/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {HF_TOKEN}'
            },
            json={
                'model': MODEL,
                'messages': messages,
                'max_tokens': data.get('max_tokens', 1000),
                'temperature': 0.7,
                'stream': False
            },
            timeout=30
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), response.status_code

        # Convert OpenAI format back to Anthropic format
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            text = result['choices'][0]['message']['content']
            return jsonify({
                "content": [{"type": "text", "text": text}]
            })
        else:
            return jsonify({"error": "No response from LLM"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


# Serve UI files
@app.route('/')
def index():
    """Serve the main chatbot UI."""
    return send_from_directory(app.static_folder, 'chatbot.html')


@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve data files (dictionary, wordlist, etc.)."""
    data_dir = os.path.join(project_root, 'data')
    return send_from_directory(data_dir, filename)


@app.route('/ui/<path:filename>')
def serve_ui(filename):
    """Serve UI assets."""
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
