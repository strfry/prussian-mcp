# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian dictionary with semantic search capabilities using E5 multilingual embeddings.

## Project Structure

```
├── api/          Flask backend for semantic search
├── lib/          Core Python modules
├── data/         Dictionary data
├── embeddings/   Pre-computed E5 embeddings
├── ui/           Web interface
├── scripts/      Utility scripts
├── docs/         Documentation
└── v2/           Development workspace
```

## Quick Start

### Run the Backend

```bash
cd api
source ../v2/venv/bin/activate
python app.py
```

The API will start on `http://localhost:5000`

### Open the UI

Open `ui/chatbot.html` in your browser, or serve it with a local web server:

```bash
python -m http.server 8000
# Then visit http://localhost:8000/ui/chatbot.html
```

## API Endpoints

- `POST /api/search` - Semantic search for Prussian words
- `POST /api/forms` - Get inflected forms for a lemma
- `GET /api/health` - Health check

## Documentation

- [Integration Complete](docs/INTEGRATION_COMPLETE.md) - E5 integration technical details
- [Data Provenance](docs/DATA_PROVENANCE.md) - Dictionary data sources

## Development

The `v2/` directory is a development workspace with symlinks to production files in `lib/`, `data/`, and `embeddings/`. This avoids file duplication while allowing development work.

## License

[To be determined]
