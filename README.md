# Prussian Dictionary with E5 Semantic Search

AI-powered Old Prussian dictionary with semantic search capabilities using E5 multilingual embeddings.

## Project Structure

```
├── api/          Flask backend (app.py)
├── lib/          Core Python modules (search, strategies)
├── data/         Dictionary data
├── embeddings/   Pre-computed E5 embeddings
├── ui/           Web interface
├── scripts/      Utilities (scraper, embedding generator)
├── docs/         Documentation
├── venv/         Virtual environment
└── start.sh      Quick start script
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
source venv/bin/activate
cd api
python app.py
```

Then open your browser to http://localhost:5000/

### First Time Setup

If you don't have a virtual environment yet:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## API Endpoints

- `POST /api/search` - Semantic search for Prussian words
- `POST /api/forms` - Get inflected forms for a lemma
- `GET /api/health` - Health check

## Documentation

- [Integration Complete](docs/INTEGRATION_COMPLETE.md) - E5 integration technical details
- [Data Provenance](docs/DATA_PROVENANCE.md) - Dictionary data sources

## Regenerating Embeddings

To regenerate the E5 embeddings from the dictionary:

```bash
source venv/bin/activate
python scripts/generate_embeddings.py
```

## License

[To be determined]
