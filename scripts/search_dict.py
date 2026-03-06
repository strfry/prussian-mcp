#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search the Prussian dictionary JSON file
Usage: search_dict.py <dict_file> <query> <language> <max_results>
"""

import json
import sys
import re
import io

# Ensure UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize(text):
    """Remove diacritics for fuzzy matching"""
    if not text:
        return ''
    replacements = {'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u',
                    'ã': 'a', 'ẽ': 'e', 'ĩ': 'i', 'õ': 'o', 'ũ': 'u'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower()

def extract_all_forms(entry):
    """Extract all inflected forms from declension/conjugation data"""
    forms = set()
    
    # Declension forms
    if 'forms' in entry and 'declension' in entry['forms']:
        for decl in entry['forms']['declension']:
            for case in decl.get('cases', []):
                if case.get('singular'):
                    forms.add(case['singular'].lower())
                if case.get('plural'):
                    forms.add(case['plural'].lower())
    
    # Conjugation forms
    if 'forms' in entry:
        # Indicative has tense groups
        if 'indicative' in entry['forms'] and isinstance(entry['forms']['indicative'], list):
            for tense_group in entry['forms']['indicative']:
                if isinstance(tense_group, dict) and 'forms' in tense_group:
                    for form_item in tense_group['forms']:
                        if isinstance(form_item, dict) and 'form' in form_item:
                            forms.add(form_item['form'].lower())
        
        # Other moods are direct arrays of {pronoun, form}
        for mood in ['subjunctive', 'optative', 'imperative']:
            if mood in entry['forms'] and isinstance(entry['forms'][mood], list):
                for item in entry['forms'][mood]:
                    if isinstance(item, dict) and 'form' in item:
                        forms.add(item['form'].lower())
        
        # Participles and infinitives
        for part_type in ['participles', 'infinitives']:
            if part_type in entry['forms'] and isinstance(entry['forms'][part_type], list):
                for item in entry['forms'][part_type]:
                    if isinstance(item, dict) and 'form' in item:
                        forms.add(item['form'].lower())
    
    return forms

def matches(entry, query, lang):
    """Check if entry matches the search query"""
    q_norm = normalize(query)
    
    # Search in Prussian word (lemma)
    if lang in ['prus', 'engl', 'miks', 'leit', 'latt', 'pols', 'mask']:
        word = entry.get('word', '').lower()
        if query in word or q_norm in normalize(word):
            return True
        
        # Search in all inflected forms
        forms = extract_all_forms(entry)
        for form in forms:
            if query in form or q_norm in normalize(form):
                return True
    
    # Search in translations
    if lang != 'prus':
        translations = entry.get('translations', {}).get(lang, [])
        for trans in translations:
            trans_lower = trans.lower()
            trans_words = re.split(r'[\s,;]+', trans_lower)
            # Match whole words or substrings
            if query in trans_lower or q_norm in normalize(trans_lower):
                return True
            for word in trans_words:
                if query in word or word in query:
                    return True
    
    return False

def decode_arg(arg):
    """Fix UTF-8 decoding issues with command line arguments"""
    if isinstance(arg, str):
        try:
            # Fix surrogate pairs using surrogateescape
            return arg.encode('utf-8', 'surrogateescape').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return arg
    return arg

def main():
    if len(sys.argv) < 5:
        print("Usage: search_dict.py <dict_file> <query> <language> <max_results>", file=sys.stderr)
        sys.exit(1)
    
    dict_file = decode_arg(sys.argv[1])
    query = decode_arg(sys.argv[2]).lower().strip()
    lang = decode_arg(sys.argv[3])
    max_results = int(sys.argv[4])
    
    try:
        with open(dict_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except Exception as e:
        print(f"Error loading dictionary: {e}", file=sys.stderr)
        sys.exit(1)
    
    results = [e for e in entries if matches(e, query, lang)][:max_results]
    
    if not results:
        print('No results found.')
        sys.exit(0)
    
    # Output as JSON
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
