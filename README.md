# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian chatbot and dictionary with semantic search using E5 multilingual embeddings and LLM-powered conversations.

## Project Structure

```
├── prussian_engine/       Python package (search, chat, tools)
├── mcp_server.py          MCP server (stdio + web modes, REST API)
├── data/                  Dictionary data and wordlists
├── embeddings/            Pre-computed E5 embeddings
├── prompts/               System prompts for LLM
├── scripts/               Data pipeline and development scripts
├── tests/                 MCP server tests
├── .mcp.json              MCP client configuration
└── venv/                  Virtual environment
```

## Quick Start

### 1. Setup Environment

Das Projekt ist ein uv-Projekt; `prussian-fst` wird als editierbare
Path-Dependency aus dem Geschwister-Checkout eingebunden (Pfad in
`pyproject.toml` unter `[tool.uv.sources]` anpassen, falls er woanders
liegt).

```bash
# Voraussetzung: prussian-fst-Checkout mit gebauten Artefakten
make -C ../prussian-fst/fst all cg3-check   # braucht hfst + cg-comp
# außerdem: cg-proc im PATH (cg3/Apertium)

uv sync
```

Das Grammatik-Tool (`validate_prussian`, dreiwertige Prüfung, optional
mit CoNLL-U-Dependenzanalyse) läuft in-process (pyhfst); fehlen
Artefakte, meldet der Server beim Start die konkreten make-Kommandos.

### Deployment (Server)

Beide Repos nebeneinander auschecken — die Build-Artefakte sind NICHT
im Git und müssen einmalig auf dem Zielsystem gebaut werden:

```bash
git clone <...>/prussian-fst
git clone <...>/prussian-mcp

# 1. Artefakte bauen (braucht hfst-lexc/hfst-xfst + cg-comp;
#    zur Laufzeit wird nur noch cg-proc gebraucht, Lookup ist pyhfst)
make -C prussian-fst/fst all cg3-sets cg3-check

# 2. venv verdrahten (installiert prussian_fst editierbar aus ../prussian-fst)
cd prussian-mcp
uv sync

# 3. starten
.venv/bin/python mcp_server.py        # stdio; --web für SSE/HTTP
```

Liegt der prussian-fst-Checkout woanders: Pfad in `pyproject.toml`
unter `[tool.uv.sources]` anpassen und `uv sync` erneut ausführen.
Fehlt etwas (Paket nicht gesynct, Artefakte nicht gebaut), startet der
Server trotzdem — die Wörterbuch-Tools laufen, und der Healthcheck
bzw. `validate_prussian` melden die konkrete Abhilfe.

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
- 4 MCP tools available:
  - `search_dictionary` - Semantic search
  - `lookup_prussian_word` - Word lookup
  - `get_word_forms` - Declensions/conjugations
  - `validate_prussian` - FSG/CG grammar check (three-valued, optional CoNLL-U)
- **Configure**: `.mcp.json` (already set up)
- **Best for**: Local development with Claude Code/Desktop

**Option B: MCP Server - Web Mode (Combined MCP + OpenAI-compatible API)**
```bash
source venv/bin/activate
python mcp_server.py --web
```
- **Modes**:
  - **MCP Protocol** (SSE): http://localhost:8001/sse
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

## Development & Testing

Test the engine directly:

```bash
# Test semantic search
python scripts/test_search.py

# Test word lookup and reranking
python scripts/test_reranked_search.py
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
- `validate_prussian(text, include_conllu)` - Grammar check with the FST/CG3
  pipeline from [`prussian-fst`](https://github.com/strfry/prussian-fst)
  (in-process). Returns three-valued JSON per sentence — `verified_in_coverage`
  (only positive evidence), `out_of_coverage` (cannot verify — NOT approval),
  `violations_found` (rule/severity/message per violation). With
  `include_conllu=true` each sentence also carries its CoNLL-U block
  (dependency analysis; MISC carries rule provenance `Rule=<name,…>` from
  named CG3 rules and `AgrParent=<id>` from the agreement `SETPARENT` layer).
  Requires a built prussian-fst checkout (`fst/build/base.fst`) plus
  `vislcg3`/`hfst-flookup` on PATH; location via `PRUSSIAN_FST_DIR`
  (default: sibling directory `../prussian-fst`).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

**Key Components:**
- **prussian_engine**: Modular Python package with search, chat, and tools
- **mcp_server.py**: FastMCP server with stdio and web transports
  - MCP Protocol (SSE for remote clients)
  - OpenAI-compatible REST API (`/v1/chat/completions`)
- **E5 Embeddings**: Semantic search using multilingual-e5-large (1024-dim)
- **Tool Calling**: LLM uses tools to search dictionary and build responses

**Two Runtime Modes:**
1. **Local Mode** (`python mcp_server.py`): Pure MCP protocol via stdio for Claude Code/Desktop
2. **Web Mode** (`python mcp_server.py --web`): All-in-one server with MCP (SSE) and OpenAI-compatible API (Web UI lives in the separate prussian-bot project)

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Target architecture and data flow
- [DATA_PROVENANCE.md](DATA_PROVENANCE.md) - Dictionary data sources and analysis

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
