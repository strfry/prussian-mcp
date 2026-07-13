# Prussian Dictionary – Architektur

## Überblick

RAG-System für ein Altpreußisch-Wörterbuch:

- Semantische Suche via Embeddings
- Tool-gestützte Übersetzung via LLM-Agent (smolagents)
- FastMCP als Webserver mit MCP-Tools, -Prompts und -Resources

```
User → prussian-agent CLI → smolagents ToolCallingAgent → LLM
                              ↕ (4 Tools)                  ↓
                        prussian_engine              PRUSSIAN: <Satz>
                        (SearchEngine + FST/CG3)

MCP-Clients → FastMCP Server → prussian_engine
              (4 Tools + Prompts + Resources)
```

## Verzeichnisstruktur

```
prussian-mcp/
├── agents/                  # prussian-agent CLI + Runner
│   ├── cli.py               # argparse entry point
│   ├── runner.py            # run_agent, RunResult, extract_candidate
│   └── tools.py             # in-process smolagents Tools (Single Source of Truth)
├── prussian_engine/         # Python-Paket (importierbar für CLI)
│   ├── __init__.py          # Hauptexport
│   ├── search.py            # Embedding-basierte Suche + get_word_forms
│   ├── fsg_check.py         # FST/CG3 Grammar Pipeline
│   └── config.py            # Env-Konfiguration
├── tools/                   # Shared CLI entry points (validate, search, lookup, wordforms)
├── mcp_server.py            # FastMCP-Server (stdio / streamable-http)
├── prompts/
│   ├── agent_system_en.md   # Kanonischer Agent-Prompt
│   ├── plan_prompt.txt      # MCP plan-Prompt
│   ├── final_prompt.txt     # MCP final-Prompt
│   ├── syntax_rules.txt     # MCP Resource: Syntaxregeln
│   └── base_vocab.md        # MCP Resource: Basisvokabular
├── pyproject.toml           # uv-Projekt; prussian-fst als editierbare Path-Dependency
└── ARCHITECTURE.md
```

## Daten

### Wörterbuch (data/twanksta_entries.json)
- ~10.737 Einträge (via Release-Asset von strfry/prussian-corpus)
- Felder: word, paradigm, gender, desc, translations, forms
- Übersetzungen: miks (DE), engl (EN), leit (LT), lett (Lettisch), pols (Polnisch), mask (Russisch)

### Embeddings (embeddings/)
- Semantische Suche mit konfigurierbarem Embedding-Modell
- Query/Passage Prefix via ENV konfigurierbar

## Komponenten

### 1. prussian_engine

Das Herzstück – unabhängig vom Webserver importierbar für CLI-Tools.

- `search.py`: `SearchEngine` – lädt Embeddings, Kosinus-Ähnlichkeit via NumPy
- `fsg_check.py`: FST/CG3 Grammar Pipeline
- `config.py`: Liest Env-Variablen (`OPENAI_*`, `EMBEDDING_*`, `RERANK_*`)

### 2. agents (prussian-agent CLI)

- `cli.py`: argparse entry point, `--validate-only`, `--mcp-url`, `--json`
- `runner.py`: `run_agent()` – smolagents ToolCallingAgent, `extract_candidate()`, `parse_last_validation()`
- `tools.py`: 4 in-process smolagents Tools (Single Source of Truth für Docstrings)

### 3. mcp_server.py (FastMCP)

MCP-Tools (4):
- `search_dictionary` – Semantische Suche
- `lookup_prussian_word` – Tokenize + FST-Analyse
- `get_word_forms` – Deklination/Konjugation
- `validate_prussian` – FST/CG3 Grammatik-Check

MCP-Prompts:
- `plan` – Planungs-Prompt
- `final` – Antwortformulierung

MCP-Resources:
- `grammar://syntax` – Kondensierte Syntaxregeln
- `vocabulary://base` – Basisvokabular (Pronomen, Präpositionen,高频 Verben)

Streaming LLM Proxy:
- `/api/completions` (SSE)
- `/v1/chat/completions` (OpenAI-kompatibel)

### 4. Tools (shared CLI entry points)

`tools/` – Direkte CLI-Aufrufe ohne Agent:
- `validate`, `search`, `lookup`, `wordforms`

## Tech-Stack

| Komponente   | Technologie              |
|--------------|--------------------------|
| Agent        | smolagents               |
| Web Server   | FastMCP                  |
| Embeddings   | numpy                    |
| LLM Client   | openai (compat)          |
| Grammar      | prussian-fst (FST/CG3)   |
