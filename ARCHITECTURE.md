Prussian Dictionary – Architektur
Status
Dieses Dokument beschreibt die Ziel-Architektur für den Rewrite. Die aktuelle Implementierung dient als Ausgangsbasis.
Überblick
Das Projekt ist ein RAG-System für ein Altpreußisch-Wörterbuch:
- Semantische Suche via E5-Embeddings
- Tool-gestützte Chat-Konversation mit einem LLM
- FastMCP als Webserver mit statischem Frontend und REST-Endpoint
Browser → FastMCP Server → prussian_engine → Search/Chat → LLM
                                    ↓
                               Data Layer
                          (Dictionary + Embeddings)
Verzeichnisstruktur
prussian-dictionary/
├── prussian_engine/        # Python-Paket (importierbar für CLI)
│   ├── __init__.py         # Hauptexport
│   ├── chat.py             # ChatEngine + Prompts
│   ├── search.py           # Embedding-basierte Suche
│   ├── tools.py            # Tool-Definitionen
│   └── config.py           # Env-Konfiguration
├── server.py               # FastMCP-Server
│   ├── Static Files (ui/)
│   ├── /prussian-api/chat Endpoint
│   └── MCP-Tools registrieren
├── prompts/
│   └── system_prompt.txt   # Unverändert
├── scripts/                # CLI-Tools zum Testen
│   ├── test_search.py
│   └── test_chat.py
├── ui/                     # Frontend (JS modernisieren)
├── data/                   # Unverändert
├── embeddings/             # Unverändert
├── api/                    # Alte Flask-Impl. (wird ersetzt)
├── requirements.txt
└── ARCHITECTURE.md
Daten
Wörterbuch (data/prussian_dictionary.json)
- ~10.172 Einträge
- Felder: word, paradigm, gender, desc, translations, forms
- Übersetzungen: miks (DE), engl (EN), leit (LT), latt (Lettisch), pols (Polnisch), mask (Russisch)
- Wichtig: Einträge ohne translations sind Formen/Verweise auf Hauptlemmas
Embeddings (embeddings/embeddings_e5_prefix.*)
- Modell: intfloat/multilingual-e5-large (1024-dim)
- Strategie: translations_only
- Wichtig: Query strings erhalten query: Prefix, Embeddings nutzen passage: Prefix
Prompts (prompts/system_prompt.txt)
Wird unverändert übernommen.
Komponenten
1. prussian_engine Modul
Das Herzstück – unabhängig vom Webserver importierbar für CLI-Tools.
__init__.py
- Exports: SearchEngine, ChatEngine, load()
config.py
- Liest Env-Variablen: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
- Datenpfade: data/prussian_dictionary.json, embeddings/
search.py
- Lädt vorberechnete E5-Embeddings beim Start
- Kosinus-Ähnlichkeit via NumPy
- Methoden:
  - query(query: str, top_k: int) – Semantische Suche, gibt nur Lemmas mit Übersetzung zurück
  - lookup(prussian_word: str) – Reverse Lookup (Lemma + alle Formen)
tools.py
Tool-Definitionen im OpenAI-Format (JSON Schema):
1. search_dictionary – Semantische Suche, DE/EN → Preußisch
   - Returns: [{word, de, en}, ...] (knapp: nur Lemma + Übersetzung)
2. lookup_prussian_word – Reverse Lookup, Preußisch → DE/EN (sucht auch flektierte Formen)
   - Returns: [{word, de, en, forms?}, ...]
3. get_word_forms – Deklination/Konjugation für ein Lemma
   - Returns: {lemma, forms, translations, paradigm, gender}
chat.py
- Nutzt tools.py + System Prompt (prompts/system_prompt.txt)
- LLM-Call via OpenAI (compat) Client
- Tool-Calling Loop: LLM → Tool → Ergebnis → LLM
- Parameter: message (str), language (str), history (list)
- Return: {prussian, german, usedWords, debugInfo, history}
2. server.py (FastMCP)
Webserver-Funktionen:
- Statische Files aus ui/ serveen
- /prussian-api/chat Endpoint (REST, kein MCP-Call)
  Request (POST JSON):
    - message: str – Benutzernachricht
    - language: str – Ausgabesprache ('de' oder 'lt')
    - history: list – Conversation History (vom Frontend gehalten)
  Response (JSON):
    - prussian: str – Antwort auf Altpreußisch
    - german: str – Deutsche Übersetzung
    - usedWords: list – Verwendete Wörterbucheinträge
    - debugInfo: object – Debug-Informationen (optional)
    - history: list – Aktualisierte Conversation History
MCP-Tools:
- Registriert die 3 Dictionary-Tools für das LLM
- Ermöglicht MCP-Clients (z.B. Claude Desktop) direkt zu nutzen
Startup:
- Lädt prussian_engine einmalig beim Start
- Alle Daten (Dictionary + Embeddings) in RAM
3. CLI-Tools (scripts/)
Test-Skripte ohne Webserver:
- scripts/test_search.py – Suchanfragen direkt an SearchEngine
- scripts/test_chat.py – Vollständige Prompts mit ChatEngine
Keine großen Frameworks – Plain Python scripts.
4. Frontend (ui/)
- Bleibt im Wesentlichen wie aktuell
- Modernisierung des JavaScript (keine größeren Feature-Änderungen)
- Kommunikation mit /chat Endpoint
Tech-Stack
Komponente	Technologie
Web Server	FastMCP
Embeddings	numpy
LLM Client	openai (compat)
Konfiguration
Via Environment Variables:
OPENAI_API_KEY=...      # API Key (local: kann leer sein)
OPENAI_BASE_URL=...     # Endpoint (z.B. http://localhost:8001/v1)
OPENAI_MODEL=...        # Modellname (z.B. gpt-oss-20b-int4-ov)
Datenpfade hardcoded:
- data/prussian_dictionary.json
- embeddings/
Offene Punkte
1. Tool-Definitionen: Weiterhin als Python-Datei oder als JSON extrahieren?
2. Logging: Gewünscht?
3. Frontend: Client-side Dictionary Loading (ui/chatbot.js:406-422) ist veraltet und sollte entfernt werden
Datenfluss (Chat)
POST /prussian-api/chat (FastMCP, REST)
    Request: {message, language, history}
    → ChatEngine.send_message(message, language, history)
        → Load system_prompt.txt
        → LLM Call (mit Tools + History)
            → search_dictionary → SearchEngine.query()
            → lookup_prussian_word → SearchEngine.lookup()
            → get_word_forms → SearchEngine.lookup()
        → LLM Call (mit Tool-Ergebnissen)
        → Parse [DE: …] / [LT: …]
        → Update History
    → Response: {prussian, german, usedWords, debugInfo, history}
