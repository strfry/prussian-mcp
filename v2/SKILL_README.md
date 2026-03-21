# Prussian Dictionary Search Skill

Embedding-basierte semantische Suche im altpreußischen Wörterbuch.

## Features

- ⚡ **Schnell**: OpenVINO GPU-beschleunigt (Intel Arc)
- 🧠 **Semantisch**: Findet konzeptuell verwandte Begriffe
- 🌍 **Mehrsprachig**: Suche in 6+ Sprachen
- 🔌 **Einfach**: CLI & Python API

## Quick Start

### 1. CLI - Einzelne Suche

```bash
python prussian_search_skill.py "son"
python prussian_search_skill.py "water" --top-k 5
```

Output:
```
🔍 Suche: 'son'
======================================================================
 1. [0.764] dēlina                = -
 2. [0.760] dēlan                 = -
 3. [0.759] dēlas                 = -
 4. [0.748] dēinans               = -
 5. [0.737] sūnus                 = -
```

### 2. CLI - Interaktiver Modus

```bash
python prussian_search_skill.py --interactive
```

```
🔍 Prussian Dictionary Search - Interactive Mode
======================================================================
Loaded: 10172 entries, 384-dim vectors
Device: GPU.0
Strategy: weighted

Search: tree
 1. [0.768] krāusī               = pear-tree, Birnbaum, kriaušė (medis)
 2. [0.669] meddjan              = forest, Wald, miškas
...
```

### 3. CLI - Batch-Suche

```bash
# Mehrere Queries
python prussian_search_skill.py --batch --queries "son" "daughter" "mother"

# Aus Datei
echo -e "tree\nwater\nlight" > queries.txt
python prussian_search_skill.py --batch --file queries.txt
```

### 4. CLI - Statistiken

```bash
python prussian_search_skill.py --stats
```

## Python API

### Basis-Nutzung

```python
from prussian_search_skill import PrussianSearch

# Initialisiere
search = PrussianSearch()

# Suche
results = search.query("son", top_k=5)

# Ergebnisse verarbeiten
for result in results:
    print(f"{result['word']}: {result['score']:.3f}")
    print(f"  Übersetzungen: {search.get_translation_summary(result)}")
```

### Erweiterte Nutzung

```python
# Spezifische Sprachen
results = search.query(
    "water",
    top_k=10,
    lang_filter=["engl", "miks"]  # Nur Englisch und Deutsch
)

# Batch-Suche
queries = ["tree", "water", "light"]
all_results = search.batch_query(queries, top_k=5)

for query, results in all_results.items():
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  - {r['word']}")

# Statistiken
stats = search.get_stats()
print(f"Geladen: {stats['num_entries']} Einträge")
```

### Integration in eigene Apps

```python
from prussian_search_skill import PrussianSearch

class MyApp:
    def __init__(self):
        # Lazy loading - wird erst bei erster Suche geladen
        self.search = PrussianSearch(auto_load=False)

    def search_prussian(self, user_query):
        # Auto-load beim ersten Aufruf
        results = self.search.query(user_query, top_k=5)

        # Konvertiere zu eigenem Format
        return [
            {
                "word": r["word"],
                "relevance": r["score"],
                "english": r["translations"].get("engl", []),
                "german": r["translations"].get("miks", []),
            }
            for r in results
        ]
```

## API-Referenz

### `PrussianSearch`

#### Constructor

```python
PrussianSearch(
    embeddings_path: str = "embeddings_production",
    auto_load: bool = True,
    use_openvino: bool = True,
    device: str = "GPU.0"
)
```

- `embeddings_path`: Pfad zu Embeddings (ohne Extension)
- `auto_load`: Embeddings sofort laden
- `use_openvino`: OpenVINO GPU nutzen
- `device`: GPU Device ("GPU.0", "CPU")

#### Methoden

**`query(query_text, top_k=10, include_translations=True, lang_filter=None)`**

Suche im Wörterbuch.

- `query_text`: Suchbegriff (beliebige Sprache)
- `top_k`: Anzahl Ergebnisse
- `include_translations`: Übersetzungen zurückgeben
- `lang_filter`: Liste von Sprachcodes (z.B. `["engl", "miks"]`)

Returns: Liste von Dicts mit `word`, `score`, `translations`, etc.

**`batch_query(queries, top_k=5)`**

Mehrere Suchen parallel.

- `queries`: Liste von Suchbegriffen

Returns: Dict mit `{query: [results]}`

**`get_translation_summary(result)`**

Lesbare Zusammenfassung der Übersetzungen.

Returns: String (z.B. "son, Sohn, sūnus")

**`format_result(result, show_score=True)`**

Formatiere Ergebnis für CLI-Ausgabe.

**`get_stats()`**

Statistiken über Embeddings.

Returns: Dict mit `num_entries`, `embedding_dim`, `strategy`, etc.

## Ergebnis-Format

```python
{
    "word": "māti",               # Altpreußisches Wort
    "score": 0.660,               # Similarity Score (0-1)
    "translations": {             # Übersetzungen
        "engl": ["mother"],
        "miks": ["Mutter"],
        "leit": ["motina"],
        "latt": ["māte"],
        "pols": ["matka"],
        "mask": ["мать"]
    },
    "desc": "...",                # Optionale Beschreibung
    "paradigm": "32",             # Optionales Paradigma
    "gender": "fem"               # Optionales Geschlecht
}
```

## Sprach-Codes

- `engl`: English
- `miks`: Deutsch (Mischsprache)
- `leit`: Lietuvių (Litauisch)
- `latt`: Latviešu (Lettisch)
- `pols`: Polski (Polnisch)
- `mask`: Русский (Russisch)

## Performance

- **Suche**: ~300-350 queries/s (nach Init)
- **Init**: ~2-3s (Laden der Embeddings)
- **Memory**: ~160MB (10.172 Einträge × 384 dim)

## Fehlerbehandlung

```python
from prussian_search_skill import PrussianSearch

try:
    search = PrussianSearch()
    results = search.query("test")
except FileNotFoundError as e:
    print("Embeddings nicht gefunden!")
    print("Bitte erst generieren: ./run_embedding_setup.sh")
```

## Tipps

### Beste Ergebnisse

- **Kurze Begriffe**: "son" statt "the son of the father"
- **Basis-Formen**: "walk" statt "walking"
- **Mehrsprachig**: Funktioniert in allen 6+ Sprachen

### Performance-Optimierung

```python
# GPU nutzen (Standard)
search = PrussianSearch(use_openvino=True, device="GPU.0")

# CPU (wenn keine GPU)
search = PrussianSearch(use_openvino=False)

# Lazy Loading für schnelleren Start
search = PrussianSearch(auto_load=False)
# ... später dann:
results = search.query("test")  # Lädt beim ersten Aufruf
```

## Beispiele

### Web-API Integration

```python
from flask import Flask, request, jsonify
from prussian_search_skill import PrussianSearch

app = Flask(__name__)
search = PrussianSearch()

@app.route('/search')
def api_search():
    query = request.args.get('q', '')
    top_k = int(request.args.get('k', 10))

    results = search.query(query, top_k=top_k)

    return jsonify({
        'query': query,
        'results': results
    })

if __name__ == '__main__':
    app.run(port=8000)
```

### CLI Tool

```python
#!/usr/bin/env python3
import sys
from prussian_search_skill import PrussianSearch

def main():
    search = PrussianSearch()

    for query in sys.argv[1:]:
        results = search.query(query, top_k=3)
        print(f"\n{query}:")
        for r in results:
            print(f"  - {r['word']} ({r['score']:.2f})")

if __name__ == '__main__':
    main()
```

## Troubleshooting

**Problem**: `FileNotFoundError: embeddings_production.embeddings.npy`

**Lösung**: Embeddings erst generieren
```bash
./run_embedding_setup.sh
# Wähle Option 3: "Produktions-Embeddings generieren"
```

**Problem**: OpenVINO GPU nicht gefunden

**Lösung**: CPU-Fallback nutzen
```python
search = PrussianSearch(use_openvino=False)
```

**Problem**: Import-Fehler

**Lösung**: Dependencies installieren
```bash
source venv/bin/activate
pip install openvino optimum[openvino] transformers torch numpy
```
