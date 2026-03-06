#!/bin/bash
# Search the Prussian dictionary in local JSON file
# Usage: search.sh <word> [language] [max_results]
# language: engl (default), miks (German), leit (Lithuanian), latt (Latvian), pols (Polish), mask (Russian), prus (Prussian lemma/forms)
# max_results: maximum number of results to return (default: 20)

WORD="$1"
LANG="${2:-engl}"
MAX_RESULTS="${3:-20}"

if [ -z "$WORD" ]; then
  echo "Usage: search.sh <word> [language] [max_results]"
  echo "Languages: engl (English), miks (German), leit (Lithuanian), latt (Latvian), pols (Polish), mask (Russian), prus (Prussian)"
  echo "Searches in Prussian word forms and translations. Returns matching entries with full data."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DICT_FILE="$SCRIPT_DIR/../prussian_dictionary.json"

if [ ! -f "$DICT_FILE" ]; then
  echo "Error: Dictionary file not found: $DICT_FILE"
  exit 1
fi

# Call Python search script
exec python3 "$SCRIPT_DIR/search_dict.py" "$DICT_FILE" "$WORD" "$LANG" "$MAX_RESULTS"

