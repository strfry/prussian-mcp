"""The system prompt — one text for every chat-style client.

Served by the MCP server as the ``system`` prompt (prussian-bot) and used
as the instructions of ``prussian-agent``.  Its tool-workflow core
(``prompts/contract.md``) and ``prompts/base_vocab.md`` are the same files
the reconstruction eval (``instruct=basevocab``) puts in front of every
sentence, so all three run the same surface.
"""

from prussian.config import PROMPTS_DIR


def _read(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def system_prompt() -> str:
    return _read("chat_intro.md") + "\n" + _read("contract.md") + "\n\n" + _read("base_vocab.md")
