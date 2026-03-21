#!/bin/bash
#
# Strukturiertes Embedding-Setup für Prußisches Wörterbuch
# Direkte OpenVINO GPU-Beschleunigung, keine Umwege
#

set -e  # Exit bei Fehler

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Prußisches Wörterbuch - Embedding Setup                  ║${NC}"
echo -e "${BLUE}║  OpenVINO GPU-Beschleunigung (Intel Arc)                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Aktiviere venv falls vorhanden
if [ -d "venv" ]; then
    echo -e "${GREEN}→${NC} Aktiviere venv..."
    source venv/bin/activate
fi

# Prüfe Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗${NC} Python3 nicht gefunden!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"

# Prüfe Dependencies
echo -e "\n${YELLOW}[1/4]${NC} Prüfe Dependencies..."

MISSING_DEPS=()

python3 -c "import openvino" 2>/dev/null || MISSING_DEPS+=("openvino")
python3 -c "import optimum" 2>/dev/null || MISSING_DEPS+=("optimum[openvino]")
python3 -c "import transformers" 2>/dev/null || MISSING_DEPS+=("transformers")
python3 -c "import torch" 2>/dev/null || MISSING_DEPS+=("torch")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC}  Fehlende Dependencies: ${MISSING_DEPS[@]}"
    echo -e "${YELLOW}→${NC} Installiere..."
    pip install -q openvino optimum[openvino] transformers torch numpy
fi

echo -e "${GREEN}✓${NC} Dependencies OK"

# Prüfe OpenVINO GPU
echo -e "\n${YELLOW}[2/4]${NC} Prüfe OpenVINO GPU..."

GPU_CHECK=$(python3 -c "
from openvino.runtime import Core
core = Core()
devices = core.available_devices
gpu_found = any('GPU' in d for d in devices)
if gpu_found:
    print('OK')
    for d in devices:
        if 'GPU' in d:
            print(f'  Device: {d}')
else:
    print('NO_GPU')
" 2>&1)

if [[ "$GPU_CHECK" == *"OK"* ]]; then
    echo -e "${GREEN}✓${NC} OpenVINO GPU verfügbar"
    echo "$GPU_CHECK" | grep "Device:"
    USE_GPU=true
else
    echo -e "${YELLOW}⚠${NC}  Keine GPU gefunden - nutze CPU-Fallback"
    USE_GPU=false
fi

# Prüfe Wörterbuch
echo -e "\n${YELLOW}[3/4]${NC} Prüfe Wörterbuch..."

DICT_FILE="prussian_dictionary.json"

if [ ! -f "$DICT_FILE" ]; then
    echo -e "${RED}✗${NC} $DICT_FILE nicht gefunden!"

    # Versuche alternative Namen
    if [ -f "prussian_dict.json" ]; then
        DICT_FILE="prussian_dict.json"
        echo -e "${YELLOW}→${NC} Nutze $DICT_FILE stattdessen"
    else
        echo -e "${RED}Fehler:${NC} Kein Wörterbuch gefunden!"
        echo "Bitte prussian_dictionary.json bereitstellen."
        exit 1
    fi
fi

DICT_SIZE=$(stat -f%z "$DICT_FILE" 2>/dev/null || stat -c%s "$DICT_FILE" 2>/dev/null)
DICT_SIZE_MB=$((DICT_SIZE / 1024 / 1024))

echo -e "${GREEN}✓${NC} Wörterbuch: $DICT_FILE (${DICT_SIZE_MB}MB)"

# Menü
echo -e "\n${YELLOW}[4/4]${NC} Wähle Aktion:\n"
echo "  1) Quick-Test (1000 Einträge, eine Strategie)"
echo "  2) Strategie-Benchmark (alle Strategien vergleichen)"
echo "  3) Produktions-Embeddings generieren (weighted)"
echo "  4) Interaktive Suche (mit existierenden Embeddings)"
echo "  5) GPU-Info anzeigen"
echo ""
read -p "Auswahl [1-5]: " CHOICE

case $CHOICE in
    1)
        echo -e "\n${BLUE}═══ Quick-Test ═══${NC}\n"

        if [ "$USE_GPU" = true ]; then
            python3 benchmark_strategies.py --small --strategy weighted --dict "$DICT_FILE"
        else
            python3 benchmark_strategies.py --small --strategy weighted --no-openvino --dict "$DICT_FILE"
        fi
        ;;

    2)
        echo -e "\n${BLUE}═══ Vollständiger Strategie-Benchmark ═══${NC}\n"
        echo -e "${YELLOW}Warnung:${NC} Dies kann 10-30 Minuten dauern!"
        read -p "Fortfahren? [y/N]: " CONFIRM

        if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
            if [ "$USE_GPU" = true ]; then
                python3 benchmark_strategies.py --dict "$DICT_FILE"
            else
                python3 benchmark_strategies.py --no-openvino --dict "$DICT_FILE"
            fi

            echo -e "\n${GREEN}✓${NC} Ergebnisse: benchmark_results.json"
        fi
        ;;

    3)
        echo -e "\n${BLUE}═══ Produktions-Embeddings (weighted) ═══${NC}\n"

        OUTPUT_PATH="embeddings_production"

        if [ "$USE_GPU" = true ]; then
            GPU_FLAG="True"
            DEVICE="GPU.0"
        else
            GPU_FLAG="False"
            DEVICE="CPU"
        fi

        python3 << EOF
from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("Initialisiere Embedder...")
embedder = PrussianEmbeddingsOptimized(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    strategy="weighted",
    use_openvino=$GPU_FLAG,
    device="$DEVICE",
    batch_size=32
)

print("Lade Wörterbuch & generiere Embeddings...")
embedder.load_dictionary("$DICT_FILE")

print("Speichere Embeddings...")
embedder.save_embeddings("$OUTPUT_PATH")

print("\n✓ Fertig!")
print(f"  - {OUTPUT_PATH}.embeddings.npy")
print(f"  - {OUTPUT_PATH}.entries.json")
print(f"  - {OUTPUT_PATH}.meta.json")
EOF

        echo -e "\n${GREEN}✓${NC} Produktions-Embeddings generiert: $OUTPUT_PATH.*"
        ;;

    4)
        echo -e "\n${BLUE}═══ Interaktive Suche ═══${NC}\n"

        # Prüfe ob Embeddings existieren
        EMBED_FILES=( embeddings_production embeddings_optimized embeddings_cache )
        FOUND=""

        for EMB in "${EMBED_FILES[@]}"; do
            if [ -f "${EMB}.embeddings.npy" ]; then
                FOUND="$EMB"
                break
            fi
        done

        if [ -z "$FOUND" ]; then
            echo -e "${RED}✗${NC} Keine Embeddings gefunden!"
            echo "Bitte erst Option 3 ausführen."
            exit 1
        fi

        echo -e "${GREEN}→${NC} Nutze Embeddings: $FOUND"
        echo ""

        python3 << EOF
from prussian_embeddings_optimized import PrussianEmbeddingsOptimized

print("Lade Embeddings...")
embedder = PrussianEmbeddingsOptimized(strategy="weighted")
embedder.load_embeddings("$FOUND")

print("\n" + "="*70)
print("Interaktive Suche (Ctrl+C zum Beenden)")
print("="*70 + "\n")

try:
    while True:
        query = input("Suchterm: ").strip()

        if not query:
            continue

        results = embedder.search(query, top_k=10)

        print(f"\n✓ {len(results)} Ergebnisse:\n")

        for rank, (entry, score) in enumerate(results, 1):
            word = entry.get("word", "?")

            # Übersetzungen
            trans = []
            if "translations" in entry:
                for lang in ["engl", "miks"]:
                    if lang in entry["translations"] and entry["translations"][lang]:
                        trans.extend(entry["translations"][lang][:2])

            trans_str = ", ".join(trans[:3]) if trans else "-"

            print(f"{rank:2d}. [{score:.3f}] {word:25s} ({trans_str})")

        print()

except KeyboardInterrupt:
    print("\n\nAuf Wiedersehen!")
EOF
        ;;

    5)
        echo -e "\n${BLUE}═══ GPU Info ═══${NC}\n"

        python3 << EOF
from openvino.runtime import Core

core = Core()

print("OpenVINO Version:", core.get_versions("CPU")["CPU"].description)
print("\nVerfügbare Devices:")
for device in core.available_devices:
    print(f"  - {device}")
    if "GPU" in device:
        props = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"    Name: {props}")

print("\nGPU Properties:")
try:
    gpu_props = core.get_property("GPU", "FULL_DEVICE_NAME")
    print(f"  Full Name: {gpu_props}")
except:
    print("  Keine GPU gefunden")
EOF
        ;;

    *)
        echo -e "${RED}Ungültige Auswahl!${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}═══ Fertig! ═══${NC}\n"
