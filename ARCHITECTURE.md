# Prussian Dictionary – Architektur

## Überblick

RAG-System für ein Altpreußisch-Wörterbuch:

- Semantische Suche via Embeddings
- Tool-gestützte Übersetzung via LLM-Agent (smolagents)
- FastMCP-Server, der dieselben vier Tools über das MCP-Protokoll exponiert

Alles liegt in **einem** Paket, `prussian/`.  Die vier Tool-Funktionen in
`prussian/tools/` sind die *Single Source of Truth*; jeder Adapter (MCP,
inspect-ai, CLI) ist ein dünner 1:1-Wrapper darum.

```
                       ┌─────────────────────────────┐
                       │  prussian/tools/  (4 Tools)  │  ← Single Source of Truth
                       │  search · lookup · wordforms │
                       │  · validate                  │
                       └──────────────┬──────────────┘
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
 adapters/agent (CLI)          adapters/mcp                 adapters/inspect_tools
 smolagents Agent → LLM        FastMCP (stdio/http)         inspect-ai @tools
 → PRUSSIAN: <Satz>            MCP-Clients                  evals/reconstruction.py
        └─────────────────────────────┴─────────────────────────────┘
                                       ▼
                          prussian/engine  (SearchEngine + FST/CG3)
```

## Verzeichnisstruktur

```
prussian-mcp/
├── prussian/                     # das eine Paket
│   ├── __init__.py               # re-export: SearchEngine, load(), Version
│   ├── config.py                 # Env-Konfiguration (Import-Zeit)
│   ├── tools/                    # DIE vier Tools + CLI entry points
│   │   ├── __init__.py           # search_tool, lookup_tool, wordforms_tool, validate_tool
│   │   ├── search.py  lookup.py  wordforms.py  validate.py
│   ├── engine/                   # Wörterbuch- + FST/CG3-Engine
│   │   ├── search.py             # SearchEngine (Embedding-Suche, Lookup, Formen)
│   │   ├── morphology.py         # PGR (Prussian Glossing Rules)
│   │   ├── embeddings/           # backend.py (model2vec/API), client.py, rerank.py
│   │   └── fst/                  # tags.py (FST-Analyse), validate.py (CG3-Pipeline)
│   └── adapters/                 # dünne Framework-Wrapper
│       ├── mcp.py                # FastMCP (stdio / streamable-http) — 4 Tools + Health
│       ├── inspect_tools.py      # inspect-ai @tool-Wrapper
│       └── agent/                # prussian-agent CLI (cli.py, runner.py, tools.py)
├── evals/                        # inspect-ai Eval-Harness (reconstruction.py, corpus_dataset.py)
├── prompts/                      # agent_system_en.md, syntax_rules.txt, base_vocab.md
├── tests/
├── pyproject.toml                # uv-Projekt; prussian-fst als editierbare Path-Dependency
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

### 1. prussian.tools — die vier Tools (Single Source of Truth)

`search_tool`, `lookup_tool`, `wordforms_tool`, `validate_tool`.  Reine
Funktionen über einer `SearchEngine`; alle Adapter rufen genau diese auf.
Zusätzlich exponiert als CLI-Scripts `validate` / `search` / `lookup` /
`wordforms`.  `runtime.py` hält die gemeinsamen Lazy-Singletons
`get_engine()` / `get_reranker()`, die jeder Adapter teilt (statt je einer
eigenen Kopie).  `spec.py` ist die *Single Source of Truth* für die
Tool-Beschreibungen (Prosa + Argument-Dokumentation): jeder Adapter bezieht
seinen Beschreibungstext daraus, sodass MCP, inspect-ai und CLI identische
Tool-Beschreibungen zeigen.

### 2. prussian.engine — Wörterbuch- + FST/CG3-Engine

- `search.py`: `SearchEngine` – lädt den Chunk-Embedding-Store fürs Retrieval (BM25+dense RRF oder dense) und die Wort-/Formen-Indizes aus dem Wörterbuch (`DICTIONARY_PATH`); `query()` liefert Chunks `{lemma, members, pos, score, text, entries}`. Es gibt nur diesen einen Suchpfad (kein Entry-/Chunk-Umschalten mehr).
- `morphology.py`: PGR-Parsing und Feature-Utilities
- `embeddings/`: `backend.py` (model2vec lokal / API), `client.py` (HTTP), `rerank.py` (`RerankedSearchEngine`)
- `fst/tags.py`: FST-Morphologieanalyse, Tag-Matching, Formengenerierung
- `fst/validate.py`: dreiwertige CG3-Grammatikprüfung — `run_validate`, `validate_with_corrections`, `check_fsg_pipeline`
- `config.py`: liest Env-Variablen (`OPENAI_*`, `EMBEDDING_*`, `RERANK_*`)

### 3. prussian.adapters — dünne Wrapper

Alle Adapter teilen `SearchEngine` + Reranker über
`prussian.tools.runtime` (ein Lazy-Singleton-Paar), beziehen ihre
Tool-Beschreibungen aus `prussian.tools.spec` und wrappen dieselben vier
`prussian.tools`-Funktionen 1:1.  So bezieht jeder Adapter denselben
Beschreibungstext, obwohl die Frameworks ihn unterschiedlich lesen: FastMCP
und inspect-ai über `__doc__` (via `spec.docstring`), smolagents über
Attribut-Override nach dem Bau (`spec.apply_to_smolagents_tool`), weil es den
Quelltext parst.

- **`mcp.py`** (FastMCP): exponiert die vier Tools über stdio / streamable-http, plus FST-Health-Check beim Start. Kein LLM-Proxy, keine Prompts/Resources mehr. Entry point `prussian-mcp`.
- **`inspect_tools.py`**: inspect-ai `@tool`-Wrapper für die Reconstruction-Eval in `evals/`.
- **`agent/`** (prussian-agent CLI, *legacy*): `cli.py` (argparse, `--validate-only`, `--mcp-url`, `--json`), `runner.py` (`run_agent`, `RunResult`, `extract_candidate`, `parse_last_validation`), `tools.py` (smolagents-Wrapper). smolagents ist deprecated — der Pfad bleibt lauffähig, ist aber kein Designtreiber mehr.

Die vier MCP-Tools:
- `search_dictionary` – Semantische Suche
- `lookup_prussian_word` – Tokenize + FST-Analyse
- `get_word_forms` – Deklination/Konjugation
- `validate_prussian` – FST/CG3 Grammatik-Check

## Tech-Stack

| Komponente   | Technologie              |
|--------------|--------------------------|
| Agent        | smolagents               |
| MCP Server   | FastMCP                  |
| Embeddings   | numpy / model2vec        |
| LLM Client   | openai (compat)          |
| Grammar      | prussian-fst (FST/CG3)   |
| Eval         | inspect-ai               |
