from __future__ import annotations

import re
import sys
from typing import Optional


def get_input_text(cli_text: Optional[str]) -> str:
    if cli_text and cli_text.strip():
        return cli_text.strip()

    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped

    print("Paste your text below. Press ENTER twice to finish:\n")
    lines: list[str] = []
    while True:
        line = input()
        if not line and lines:
            break
        lines.append(line)

    return "\n".join(lines).strip()


def split_text_for_batch(text: str, max_chars: int = 140) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks
