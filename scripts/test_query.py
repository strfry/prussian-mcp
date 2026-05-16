#!/usr/bin/env python3
"""Quick test for the chat engine."""

import prussian_engine

# Load engine
print("Loading engine...")
search_engine, chat_engine = prussian_engine.load()
print("Engine loaded!\n")

# Test query
query = "As kwai minintun prusan"
print(f"Testing query: {query}\n")
print("="*80)

result = chat_engine.send_message(query, language="de")

print("\n" + "="*80)
print("\n📋 RESULT:")
print(f"Prussian: {result['prussian']}")
print(f"Translation: {result['translation']}")
print(f"Used words: {result['usedWords']}")
