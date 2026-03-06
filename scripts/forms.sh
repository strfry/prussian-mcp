#!/bin/bash
# Get declension/conjugation forms for a Prussian word from local JSON
# Usage: forms.sh <word> [paradigm_number]

WORD="$1"
PARADIGM="${2:-}"

if [ -z "$WORD" ]; then
  echo "Usage: forms.sh <word> [paradigm_number]"
  echo "Returns full entry with all forms for the given word."
  echo "Paradigm number is optional for disambiguation."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DICT_FILE="$SCRIPT_DIR/../prussian_dictionary.json"

if [ ! -f "$DICT_FILE" ]; then
  echo "Error: Dictionary file not found: $DICT_FILE"
  exit 1
fi

# Call Python forms script
exec python3 "$SCRIPT_DIR/forms_dict.py" "$DICT_FILE" "$WORD" "$PARADIGM"

