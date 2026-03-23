#!/usr/bin/env python3
"""Test script for chat functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prussian_engine import load


def main():
    """Run chat tests."""
    print("Loading Prussian Dictionary engine...")
    search_engine, chat_engine = load()
    print()

    # Test conversations
    conversations = [
        ("Hallo!", "de"),
        ("Wie geht es dir?", "de"),
        ("Gott sei mit dir", "de"),
        ("Labas!", "lt"),
    ]

    history = []

    for message, language in conversations:
        print(f"\n{'='*60}")
        print(f"User ({language}): {message}")
        print(f"{'='*60}")

        result = chat_engine.send_message(message, language, history)

        # Update history
        history = result.get("history", [])

        # Display response
        print(f"\nPrussian: {result['prussian']}")
        if "translation" in result:
            print(f"Translation: {result['translation']}")

        # Display used words
        if result.get("usedWords"):
            print(f"\nUsed words: {', '.join(result['usedWords'])}")

        # Display tool calls in debug mode
        if result.get("debugInfo", {}).get("toolCalls"):
            print(f"\nTool calls:")
            for call in result["debugInfo"]["toolCalls"]:
                print(f"  - {call['name']}: {call['input']}")
                if isinstance(call['result'], list):
                    print(f"    Results: {len(call['result'])} entries")


if __name__ == "__main__":
    main()
