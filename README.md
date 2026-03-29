# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian chatbot and dictionary with semantic search using E5 multilingual embeddings and LLM-powered conversations.

## Project Structure

```
├── prussian_engine/       Python package (search, chat, tools)
├── mcp_server.py          MCP tools via stdio (lightweight, for Claude Code/Desktop)
├── web_server.py          Web server (REST API + static files + LLM)
├── data/                  Dictionary data (~10,000 entries)
├── embeddings/            Pre-computed E5 embeddings
├── prompts/               System prompts for LLM
├── ui/                    Web interface (HTML/JS)
├── scripts/               CLI tools and utilities
├── .mcp.json              MCP client configuration
├── archive/               Archived files (old server.py)
└── venv/                  Virtual environment
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

### 3. Start Server(s)

Choose based on your use case:

**Option A: MCP Server - Local CLI (Claude Code/Desktop)**
```bash
source venv/bin/activate
python mcp_server.py
```
- **Transport**: stdio (pure MCP protocol)
- **No LLM needed** - just dictionary tools
- 3 MCP tools available:
  - `search_dictionary` - Semantic search
  - `lookup_prussian_word` - Word lookup
  - `get_word_forms` - Declensions/conjugations
- **Configure**: `.mcp.json` (already set up)
- **Best for**: Local development with Claude Code/Desktop

**Option B: MCP Server - Web Mode (Claude Web)**
```bash
source venv/bin/activate
python mcp_server.py --web
```
- **Transport**: SSE (Server-Sent Events over HTTP)
- **URL**: http://localhost:8000/sse
- **No LLM needed** - just dictionary tools
- **Configure in Claude Web**:
  ```json
  {
    "type": "sse",
    "url": "http://localhost:8000/sse"
  }
  ```
- **Best for**: Testing with Claude Web

**Option C: Web Server (for browser UI + REST API)**
```bash
source venv/bin/activate
python web_server.py
```
- **URL**: http://localhost:8000
- **Requires LLM endpoint** configuration (see step 2)
- **Web UI**: http://localhost:8000/chatbot.html
- **Chat API**: POST http://localhost:8000/prussian-api/chat
- **Best for**: Web UI and REST API integration

**Option D: Both Servers (different terminals)**
```bash
# Terminal 1: MCP Server (stdio for local Claude Code/Desktop)
python mcp_server.py

# Terminal 2: Web Server (HTTP for browser - REST API + static files)
python web_server.py
```

## CLI Testing

Test the engine directly without the web server:

```bash
# Test semantic search
python scripts/test_search.py
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

- `search_dictionary(query, top_k)` - Semantic search (German/English → Prussian)
- `lookup_prussian_word(word)` - Lookup Prussian word (Prussian → German/English)
- `get_word_forms(lemma)` - Get declensions/conjugations

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

**Key Components:**
- **prussian_engine**: Modular Python package with search, chat, and tools
- **mcp_server.py**: Lightweight MCP server (stdio transport) for Claude clients
- **web_server.py**: HTTP server with REST API and static file serving
- **E5 Embeddings**: Semantic search using multilingual-e5-large (1024-dim)
- **Tool Calling**: LLM uses tools to search dictionary and build responses

**Two Runtime Modes:**
1. **MCP Mode** (mcp_server.py): Pure MCP protocol via stdio, no HTTP overhead
2. **Web Mode** (web_server.py): Full-featured with REST API, static files, and LLM integration

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

search_engine = load()
results = search_engine.query("Haus", top_k=5)
```

## License

[To be determined]
