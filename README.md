# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian chatbot and dictionary with semantic search using E5 multilingual embeddings and LLM-powered conversations.

## Project Structure

```
├── prussian/              The one package
│   ├── tools/             The four tools — single source of truth (+ CLI scripts)
│   ├── engine/            Dictionary + FST/CG3 engine (search, morphology, embeddings, fst)
│   ├── adapters/          Thin wrappers: mcp.py, inspect_tools.py, agent/ (CLI)
│   └── config.py          Env-driven configuration
├── evals/                 inspect-ai reconstruction eval harness
├── data/                  Dictionary data and wordlists
├── embeddings/            Pre-computed embeddings
├── prompts/               System prompts for LLM
├── scripts/               Data pipeline and development scripts
├── tests/                 Offline + engine tests
├── .mcp.json              MCP client configuration
└── .venv/                 Virtual environment
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
uv run prussian-mcp                   # stdio; --web für SSE/HTTP
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
uv run prussian-mcp                   # or: .venv/bin/python -m prussian.adapters.mcp
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
uv run prussian-mcp --web
```
- **Transport**: streamable-http (MCP protocol for remote clients)
- **Host/port**: `--host` / `--port` or `MCP_HOST` / `MCP_PORT` (default `127.0.0.1:8001`)
- **Configure MCP in Claude Web**:
  ```json
  {
    "type": "sse",
    "url": "http://localhost:8001/sse"
  }
  ```
- **Best for**: Remote MCP clients over HTTP

## Development & Testing

Test the engine directly:

```bash
# Test semantic search
python scripts/test_search.py

# Test word lookup and reranking
python scripts/test_reranked_search.py
```

## MCP Tools

The four tools live in `prussian/tools/` and are wrapped 1:1 by every
adapter (MCP, inspect-ai eval, CLI agent):

- `search_dictionary(query, top_k, use_reranker, filter_tags)` - Semantic search (German/English/… → Prussian)
- `lookup_prussian_word(text, fuzzy)` - Tokenize + FST-analyze a Prussian sentence, enrich from dictionary
- `get_word_forms(lemma, features)` - Declensions/conjugations, optional feature filter
- `validate_prussian(text, include_conllu)` - Grammar check with the FST/CG3
  pipeline from [`prussian-fst`](https://github.com/strfry/prussian-fst)
  (in-process). Returns three-valued JSON per sentence — `verified_in_coverage`
  (only positive evidence), `out_of_coverage` (cannot verify — NOT approval),
  `violations_found` (rule/severity/message per violation). With
  `include_conllu=true` each sentence also carries its CoNLL-U block
  (dependency analysis; MISC carries rule provenance `Rule=<name,…>` from
  named CG3 rules and `AgrParent=<id>` from the agreement `SETPARENT` layer).
  Requires a built prussian-fst checkout (`fst/build/base.fst`) plus
  `vislcg3`/`hfst-flookup` on PATH; the checkout is an editable uv
  path-dependency (`[tool.uv.sources]`, default `../prussian-fst`).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

**Key Components:**
- **`prussian.tools`**: the four tools — single source of truth wrapped by every adapter
- **`prussian.engine`**: search, morphology, embeddings, and the FST/CG3 validator
- **`prussian.adapters.mcp`**: FastMCP server (stdio / streamable-http) — the four tools + FST health, nothing else
- **`prussian.adapters.agent`** / **`prussian.adapters.inspect_tools`**: the CLI agent and the inspect-ai eval

**Two MCP transports:**
1. **stdio** (`uv run prussian-mcp`): pure MCP for Claude Code/Desktop
2. **streamable-http** (`uv run prussian-mcp --web`): MCP for remote clients

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

The `prussian` package is designed to be importable for CLI tools:

```python
from prussian import load

search_engine = load()
results = search_engine.query("Haus", top_k=5)
```

## License

[To be determined]
