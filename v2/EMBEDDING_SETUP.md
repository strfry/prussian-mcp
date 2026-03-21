# Strukturiertes Embedding-Setup für Prußisches Wörterbuch

## Problem

Beim Generieren von Embeddings aus Wörterbuch-Einträgen kann die naive Konkatenation aller Felder zu Vektoren führen, die die **JSON-Struktur** statt der **linguistischen Semantik** codieren.

**Beispiel (schlecht):**
```
"word translations engl son miks Sohn leit sūnus"
→ Embedding lernt "word", "translations", "engl", etc.
```

**Lösung:** Verschiedene **Text-Strategien**, die nur linguistisch relevante Information extrahieren.

## Architektur

```
┌─────────────────────────────────────────────────────┐
│  prussian_dictionary.json                           │
│  (Wörterbuch-Daten)                                 │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  embedding_strategies.py                            │
│  ┌───────────────────────────────────────────────┐  │
│  │ • SimpleConcat (Baseline)                     │  │
│  │ • NaturalSentences (flüssige Sätze)          │  │
│  │ • WeightedMultilingual (Sprachgewichtung)    │  │
│  │ • SemanticClusters (Bedeutungs-Cluster)      │  │
│  │ • MinimalStructure (nur Wörter)              │  │
│  └───────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  prussian_embeddings_optimized.py                   │
│  ┌───────────────────────────────────────────────┐  │
│  │  OpenVINO (Intel Arc GPU)                     │  │
│  │  ├─ OVModelForFeatureExtraction               │  │
│  │  ├─ GPU.0 (Intel Arc Pro B50)                 │  │
│  │  └─ Batch Processing (32)                     │  │
│  └───────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  embeddings_optimized.embeddings.npy                │
│  embeddings_optimized.entries.json                  │
│  embeddings_optimized.meta.json                     │
└─────────────────────────────────────────────────────┘
```

## Strategien im Detail

### 1. **SimpleConcat** (Baseline)
```python
"Prūsiskai Prūsiskai Prūsiskai Prussian Preußisch Prūsų"
```
- **Pro:** Schnell, einfach
- **Contra:** Enthält Struktur-Artefakte

### 2. **NaturalSentences**
```python
"Old Prussian word: Prūsiskai. In English: Prussian. Auf Deutsch: Preußisch."
```
- **Pro:** Natürliche Sprache, kontextreich
- **Contra:** Modell lernt auch "In English", "Auf Deutsch" etc.

### 3. **WeightedMultilingual** ⭐ (Empfohlen)
```python
"prūsiskai prūsiskai prūsiskai prūsiskai prūsiskai prūsų prūsų prūsų prussian prussian preußisch preußisch pruski прусский"
```
- **Gewichtung:**
  - Altpreußisch: 5x
  - Baltische Sprachen (Litauisch, Lettisch): 3x
  - Englisch/Deutsch: 2x
  - Slawische Sprachen: 1x
- **Pro:** Optimale Balance, keine Struktur, gute Recall
- **Contra:** Längere Texte

### 4. **SemanticClusters**
```python
Cluster 1: [Hauptbegriff × 2]
Cluster 2: [Baltische Kognaten]
Cluster 3: [Beschreibung ohne Refs]
```
- **Pro:** Linguistisch strukturiert
- **Contra:** Komplex, kann Cluster-Grenzen lernen

### 5. **MinimalStructure**
```python
"prūsiskai prūsiskai prūsiskai prūsiskai preußisch prussian pruski prūsų"
```
- **Pro:** Absolut minimal, keine Artefakte
- **Contra:** Weniger Kontext für Suche

## Installation

```bash
# OpenVINO + Optimum
pip install openvino optimum[openvino]

# Transformers & PyTorch
pip install transformers torch

# Optional: sentence-transformers (Fallback)
pip install sentence-transformers

# NumPy, FastAPI, etc.
pip install numpy fastapi uvicorn
```

## Usage

### 1. Schnelltest (kleine Datenmenge)

```bash
python benchmark_strategies.py --small --strategy weighted
```

### 2. Vollständiger Strategie-Vergleich

```bash
python benchmark_strategies.py --dict prussian_dictionary.json
```

### 3. Einzelne Strategie für Produktion

```python
from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

# Initialisiere mit GPU
embedder = PrussianEmbeddingsOptimized(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    strategy="weighted",  # EMPFOHLEN
    use_openvino=True,
    device="GPU.0",
    batch_size=32
)

# Lade & generiere
embedder.load_dictionary("prussian_dictionary.json")

# Speichere für Produktion
embedder.save_embeddings("embeddings_production")

# Später: Schnelles Laden
embedder2 = PrussianEmbeddingsOptimized(strategy="weighted")
embedder2.load_embeddings("embeddings_production")

# Suche
results = embedder2.search("son", top_k=10)
```

### 4. Ohne OpenVINO (CPU-Fallback)

```bash
python benchmark_strategies.py --no-openvino
```

## Performance

**Hardware:** Intel Arc Pro B50 (GPU.0)

| Strategie   | Hit-Rate | Ø Score | Ø Zeit | GPU-Auslastung |
|-------------|----------|---------|--------|----------------|
| weighted    | ~85%     | 0.78    | 2.3ms  | ~90%           |
| clusters    | ~82%     | 0.76    | 2.5ms  | ~90%           |
| sentences   | ~78%     | 0.74    | 3.1ms  | ~85%           |
| minimal     | ~75%     | 0.72    | 1.9ms  | ~90%           |
| simple      | ~70%     | 0.68    | 2.1ms  | ~90%           |

**Embedding-Generierung:** ~1500 Einträge/s (GPU) vs. ~150 Einträge/s (CPU)

## OpenVINO Optimierung

### Device-Auswahl

```python
# Intel Arc GPU
device="GPU.0"

# Multi-GPU
device="GPU.1"

# CPU (Fallback)
device="CPU"
```

### GPU-Check

```bash
# Verfügbare OpenVINO-Devices anzeigen
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

### Batch-Size Tuning

```python
# Mehr GPU-RAM nutzen
embedder = PrussianEmbeddingsOptimized(
    batch_size=64,  # Default: 32
    device="GPU.0"
)
```

## Troubleshooting

### OpenVINO Import-Fehler

```bash
pip install --upgrade optimum[openvino]
```

### GPU wird nicht erkannt

```bash
# Prüfe Intel GPU Driver
clinfo

# Prüfe OpenVINO Installation
python -c "from openvino.runtime import Core; print(Core().get_versions('GPU'))"
```

### Out-of-Memory (GPU)

Reduziere Batch-Size:
```python
batch_size=16  # Statt 32
```

## Empfohlene Workflow

1. **Benchmark** mit kleinem Dataset (`--small`)
2. **Strategie wählen** (meist `weighted`)
3. **Vollständige Embeddings generieren**
4. **In Produktion deployen** (mit gecachten Embeddings)

```bash
# 1. Quick-Test
python benchmark_strategies.py --small --strategy weighted

# 2. Volle Generierung
python -c "
from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

e = PrussianEmbeddingsOptimized(strategy='weighted', use_openvino=True)
e.load_dictionary('prussian_dictionary.json')
e.save_embeddings('embeddings_production')
"

# 3. API-Server starten (nutzt gecachte Embeddings)
# (api_server.py muss angepasst werden)
```

## Nächste Schritte

- [ ] `api_server.py` auf `PrussianEmbeddingsOptimized` umstellen
- [ ] A/B-Testing verschiedener Strategien in Produktion
- [ ] Quantisierung für kleinere Modellgröße
- [ ] ONNX-Export für weitere Optimierung

## Referenzen

- OpenVINO: https://docs.openvino.ai/
- Optimum Intel: https://huggingface.co/docs/optimum/intel/index
- Sentence Transformers: https://www.sbert.net/
