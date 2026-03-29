# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian chatbot and dictionary with semantic search using E5 multilingual embeddings and LLM-powered conversations.

## Project Structure

```
├── prussian_engine/       Python package (search, chat, tools)
├── mcp_server.py          MCP server (stdio + web modes, REST API, static files)
├── data/                  Dictionary data (~10,000 entries)
├── embeddings/            Pre-computed E5 embeddings
├── prompts/               System prompts for LLM
├── ui/                    Web interface (HTML/JS)
├── scripts/               CLI tools and utilities
├── .mcp.json              MCP client configuration
├── archive/               Archived files
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

### 3. Start Server

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

**Option B: MCP Server - Web Mode (Combined MCP + Web UI + OpenAI-compatible API)**
```bash
source venv/bin/activate
python mcp_server.py --web
```
- **Modes**:
  - **MCP Protocol** (SSE): http://localhost:8001/sse
  - **Web UI**: http://localhost:8001/chatbot.html
  - **OpenAI-compatible API**: POST http://localhost:8001/v1/chat/completions
- **Requires LLM endpoint** configuration (see step 2)
- **Configure MCP in Claude Web**:
  ```json
  {
    "type": "sse",
    "url": "http://localhost:8001/sse"
  }
  ```
- **Best for**: Everything - single server for MCP protocol, web UI, and REST API

## CLI Testing

Test the engine directly without the web server:

```bash
# Test semantic search
python scripts/test_search.py
```

## API

### REST Endpoints

**OpenAI-compatible Chat Completion API**

**POST** `/v1/chat/completions` (streaming)

Request:
```json
{
  "model": "prussian-chat",
  "messages": [
    {"role": "user", "content": "Was bedeutet 'lauxnos'?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_dictionary",
        "description": "Search Prussian dictionary",
        "parameters": {...}
      }
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2000,
  "stream": true,
  "language": "de"
}
```

Response (streaming SSE):
```
data: {"id": "chatcmpl-...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "Die"}}]}
data: {"id": "chatcmpl-...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " Inschrift..."}}]}
data: [DONE]
```

Features:
- OpenAI-compatible format
- Streaming and non-streaming modes (`stream: true/false`)
- Tool calling support
- Custom `language` parameter (`de` or `lt`)
- Reasoning content support (DeepSeek R1)

### MCP Tools

- `search_dictionary(query, top_k)` - Semantic search (German/English → Prussian)
- `lookup_prussian_word(word)` - Lookup Prussian word (Prussian → German/English)
- `get_word_forms(lemma)` - Get declensions/conjugations

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

**Key Components:**
- **prussian_engine**: Modular Python package with search, chat, and tools
- **mcp_server.py**: FastMCP server with stdio and web transports
  - MCP Protocol (SSE for remote clients)
  - OpenAI-compatible REST API (`/v1/chat/completions`)
  - Static file serving (Web UI)
- **E5 Embeddings**: Semantic search using multilingual-e5-large (1024-dim)
- **Tool Calling**: LLM uses tools to search dictionary and build responses

**Two Runtime Modes:**
1. **Local Mode** (`python mcp_server.py`): Pure MCP protocol via stdio for Claude Code/Desktop
2. **Web Mode** (`python mcp_server.py --web`): All-in-one server with MCP (SSE), OpenAI-compatible API, and Web UI

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
