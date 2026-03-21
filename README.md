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

### One-Command Start

```bash
./start.sh
```

This starts the Flask server which serves both the API and the web UI.

- **Web UI**: http://localhost:5000/
- **API**: http://localhost:5000/api/

### Manual Start

```bash
cd api
source ../v2/venv/bin/activate
python app.py
```

Then open your browser to http://localhost:5000/

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
