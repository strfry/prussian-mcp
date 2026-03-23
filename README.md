# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian chatbot and dictionary with semantic search using E5 multilingual embeddings and LLM-powered conversations.

## Project Structure

```
├── prussian_engine/    Python package (search, chat, tools)
├── server.py           FastMCP server (MCP tools + REST API)
├── data/               Dictionary data (~10,000 entries)
├── embeddings/         Pre-computed E5 embeddings
├── prompts/            System prompts for LLM
├── ui/                 Web interface (HTML/JS)
├── scripts/            CLI tools and utilities
└── venv/               Virtual environment
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure LLM (Optional)

Set environment variables for your LLM endpoint:

```bash
export OPENAI_BASE_URL="http://localhost:8001/v1"
export OPENAI_MODEL="gpt-oss-20b-int4-ov"
export OPENAI_API_KEY="dummy"  # or your API key
```

For local LLM servers, you can use any OpenAI-compatible endpoint.

### 3. Start Server

```bash
python server.py
```

- **Web UI**: http://localhost:8000/chatbot.html
- **Chat API**: POST http://localhost:8000/prussian-api/chat
- **MCP Tools**: Available for MCP clients (e.g., Claude Desktop)

## CLI Testing

Test the engine directly without the web server:

```bash
# Test semantic search
python scripts/test_search.py

# Test chat functionality
python scripts/test_chat.py
```

## API

### REST Endpoint

**POST** `/prussian-api/chat`

Request:
```json
{
  "message": "Hallo!",
  "language": "de",
  "history": []
}
```

Response:
```json
{
  "prussian": "Kails tū assei!",
  "translation": "Gegrüßt seist du!",
  "usedWords": ["kails", "tū", "assei"],
  "debugInfo": {...},
  "history": [...]
}
```

### MCP Tools

- `search_dictionary(query, top_k)` - Semantic search
- `lookup_prussian_word(word)` - Lookup Prussian word
- `get_word_forms(lemma)` - Get declension/conjugation

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

**Key Components:**
- **prussian_engine**: Modular Python package with search, chat, and tools
- **FastMCP**: Web server with MCP support and REST endpoints
- **E5 Embeddings**: Semantic search using multilingual-e5-large (1024-dim)
- **Tool Calling**: LLM uses tools to search dictionary and build responses

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Target architecture and data flow
- [Integration Complete](docs/INTEGRATION_COMPLETE.md) - E5 integration technical details
- [Data Provenance](docs/DATA_PROVENANCE.md) - Dictionary data sources

## Regenerating Embeddings

To regenerate the E5 embeddings from the dictionary:

```bash
source venv/bin/activate
python scripts/generate_embeddings.py
```

## Development

The `prussian_engine` package is designed to be importable for CLI tools:

```python
from prussian_engine import load

search_engine, chat_engine = load()
results = search_engine.query("Haus", top_k=5)
```

## License

[To be determined]
