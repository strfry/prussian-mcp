#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get forms for a specific Prussian word from the dictionary JSON
Usage: forms_dict.py <dict_file> <word> [paradigm]
"""

import json
import sys
import io

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize(text):
    """Remove diacritics for matching"""
    if not text:
        return ''
    replacements = {'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
                    'ã': 'a', 'ẽ': 'e', 'ĩ': 'i', 'õ': 'o', 'ũ': 'u'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()

def decode_arg(arg):
    """Fix UTF-8 decoding issues with command line arguments"""
    if isinstance(arg, str):
        try:
            return arg.encode('utf-8', 'surrogateescape').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return arg
    return arg

def main():
    if len(sys.argv) < 3:
        print("Usage: forms_dict.py <dict_file> <word> [paradigm]", file=sys.stderr)
        sys.exit(1)
    
    dict_file = decode_arg(sys.argv[1])
    word = decode_arg(sys.argv[2]).lower().strip()
    paradigm = decode_arg(sys.argv[3]).strip() if len(sys.argv) > 3 else None
    
    try:
        with open(dict_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except Exception as e:
        print(f"Error loading dictionary: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Find matching entries
    matches = []
    for entry in entries:
        entry_word = entry.get('word', '').lower()
        if word == entry_word or normalize(word) == normalize(entry_word):
            if paradigm:
                if str(entry.get('paradigm', '')) == paradigm:
                    matches.append(entry)
            else:
                matches.append(entry)
    
    if not matches:
        print('No matching entry found.')
        sys.exit(1)
    
    # Return all matches (or first if only one)
    if len(matches) == 1:
        print(json.dumps(matches[0], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(matches, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
