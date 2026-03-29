# Prussian Dictionary – Architektur

## Status
Dieses Dokument beschreibt die Ziel-Architektur für den Rewrite. Die aktuelle Implementierung dient als Ausgangsbasis.

## Überblick

Das Projekt ist ein RAG-System für ein Altpreußisch-Wörterbuch:

- Semantische Suche via E5-Embeddings
- Tool-gestützte Chat-Konversation mit einem LLM (Frontend-seitig)
- FastMCP als Webserver mit statischem Frontend und REST-Endpoint

```
Browser → FastMCP Server → prussian_engine → Search/Chat → LLM
                                       ↓
                                  Data Layer
                             (Dictionary + Embeddings)
```

## Verzeichnisstruktur

```
prussian-dictionary/
├── prussian_engine/        # Python-Paket (importierbar für CLI)
│   ├── __init__.py         # Hauptexport
│   ├── search.py           # Embedding-basierte Suche + get_word_forms
│   ├── tools.py            # Tool-Definitionen
│   └── config.py           # Env-Konfiguration
├── mcp_server.py           # FastMCP-Server
│   ├── Static Files (ui/)
│   ├── /api/completions Endpoint
│   └── MCP-Tools registrieren
├── prompts/
│   └── system_prompt.txt   # Unverändert
├── scripts/                # CLI-Tools zum Testen
│   └── test_search.py
├── ui/                     # Frontend (JS)
├── data/                   # Unverändert
├── embeddings/             # Unverändert
├── requirements.txt
└── ARCHITECTURE.md
```

## Daten

### Wörterbuch (data/prussian_dictionary.json)
- ~10.172 Einträge
- Felder: word, paradigm, gender, desc, translations, forms
- Übersetzungen: miks (DE), engl (EN), leit (LT), lett (Lettisch), pols (Polnisch), mask (Russisch)
- Wichtig: Einträge ohne translations sind Formen/Verweise auf Hauptlemmas

### Embeddings (embeddings/embeddings_e5_prefix.*)
- Modell: intfloat/multilingual-e5-large (1024-dim)
- Strategie: translations_only
- Wichtig: Query strings erhalten `query:` Prefix, Embeddings nutzen `passage:` Prefix

### Prompts (prompts/system_prompt.txt)
Wird unverändert übernommen.

## Komponenten

### 1. prussian_engine Modul

Das Herzstück – unabhängig vom Webserver importierbar für CLI-Tools.

#### `__init__.py`
- Exports: `SearchEngine`, `load()`

#### `config.py`
- Liest Env-Variablen: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`
- Datenpfade: `data/prussian_dictionary.json`, `embeddings/`

#### `search.py`
- Lädt vorberechnete E5-Embeddings beim Start
- Kosinus-Ähnlichkeit via NumPy
- Methoden:
  - `query(query: str, top_k: int)` – Semantische Suche, gibt nur Lemmas mit Übersetzung zurück
  - `lookup(prussian_word: str)` – Reverse Lookup (Lemma + alle Formen)
  - `get_word_forms(lemma: str)` – Deklination/Konjugation für ein Lemma

#### `tools.py`
Tool-Definitionen im OpenAI-Format (JSON Schema):
1. `search_dictionary` – Semantische Suche, DE/EN → Preußisch
   - Returns: `[{word, de, en}, ...]` (knapp: nur Lemma + Übersetzung)
2. `lookup_prussian_word` – Reverse Lookup, Preußisch → DE/EN (sucht auch flektierte Formen)
   - Returns: `[{word, de, en, forms?}, ...]`
3. `get_word_forms` – Deklination/Konjugation für ein Lemma
   - Returns: `{lemma, forms, translations, paradigm, gender}`

### 2. mcp_server.py (FastMCP)

Webserver-Funktionen:

- Statische Files aus ui/ serven
- `/api/completions` Endpoint (SSE Streaming)
  - Request (POST JSON):
    - `messages`: list – Chat messages
    - `tools`: list – Tool definitions (optional, default aus tools.py)
    - `temperature`: float – Sampling temperature (default 0.7)
    - `max_tokens`: int – Maximum tokens (default 2000)
    - `language`: str – Response language 'de' oder 'lt' (default 'de')
  - Response (SSE stream):
    - `content_delta`: {"content": string}
    - `reasoning_delta`: {"content": string}
    - `tool_call_delta`: {"index": int, "tool_call": {...}}
    - `done`: {"finish_reason": string}
    - `error`: {"error": string}

- MCP-Tools:
  - Registriert die 3 Dictionary-Tools für das LLM
  - Ermöglicht MCP-Clients (z.B. Claude Desktop) direkt zu nutzen

Startup:
- Lädt prussian_engine einmalig beim Start
- Alle Daten (Dictionary + Embeddings) in RAM

### 3. CLI-Tools (scripts/)

Test-Skripte ohne Webserver:

- `scripts/test_search.py` – Suchanfragen direkt an SearchEngine

Keine großen Frameworks – Plain Python scripts.

### 4. Frontend (ui/)

- Modernes JavaScript mit Chat-Interface
- Kommunikation mit `/api/completions` Endpoint
- Chat-Logik client-seitig (ChatEngine in JavaScript)

## Tech-Stack

| Komponente   | Technologie              |
|--------------|--------------------------|
| Web Server   | FastMCP                  |
| Embeddings   | numpy                    |
| LLM Client   | openai (compat)          |

## Konfiguration

Via Environment Variables:

```
OPENAI_API_KEY=...      # API Key (local: kann leer sein)
OPENAI_BASE_URL=...     # Endpoint (z.B. http://localhost:8001/v1)
OPENAI_MODEL=...        # Modellname (z.B. gpt-oss-20b-int4-ov)
```

Datenpfade hardcoded:
- `data/prussian_dictionary.json`
- `embeddings/`

## Offene Punkte

1. ~~Tool-Definitionen: Weiterhin als Python-Datei oder als JSON extrahieren?~~
2. Logging: Gewünscht?
3. ~~Frontend: Client-side Dictionary Loading (ui/chatbot.js) ist veraltet und sollte entfernt werden~~

## Datenfluss (Chat)

```
POST /api/completions (FastMCP, SSE Streaming)
    Request: {messages, tools?, temperature, max_tokens, language}
    → Stream LLM responses
        → content_delta, reasoning_delta, tool_call_delta
    → Client executes tools via MCP
    → Client sends results back as assistant message
```
