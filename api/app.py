#!/usr/bin/env python3
"""Flask API for Prussian Dictionary RAG endpoints."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))
from prussian_search_skill import PrussianSearch

app = Flask(__name__)
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


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
