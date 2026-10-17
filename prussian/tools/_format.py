"""Shared text helpers for the tools' compact output (search + lookup)."""

from __future__ import annotations

from typing import Any

# Dictionary language keys → ISO codes, in display order (two per line).
_LANGS = (("miks", "de"), ("engl", "en"), ("leit", "lt"),
          ("latt", "lv"), ("pols", "pl"), ("mask", "ru"))

_GENDER = {"fem": "f", "masc": "m", "neut": "n"}


def translations(tr: Any, indent: str) -> list[str]:
    """``de: Birke · en: birch`` / ``lt: … · lv: …`` / ``pl: … · ru: …``."""
    if not isinstance(tr, dict):
        return [f"{indent}{tr}"] if tr else []
    order = [k for k, _ in _LANGS] + [k for k in tr if k not in dict(_LANGS)]
    parts = []
    for key in order:
        vals = tr.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        if vals:
            parts.append(f"{dict(_LANGS).get(key, key)}: {', '.join(vals)}")
    return [f"{indent}{' · '.join(parts[i:i + 2])}" for i in range(0, len(parts), 2)]


def head(word: str, gender: str = "", extra: str = "") -> str:
    """``berzi (f)`` plus optional trailing info."""
    g = _GENDER.get(gender, gender)
    out = f"{word} ({g})" if g else word
    return f"{out}  {extra}" if extra else out
