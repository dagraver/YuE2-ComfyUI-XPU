"""Whisper's tokenizer, read with the ``tokenizers`` library ComfyUI already installs."""

from __future__ import annotations

import re

TIMESTAMP = re.compile(r"<\|(\d+\.\d+)\|>")
"""A timestamp token as text: the release keeps them as ordinary added tokens, so decoding writes them out."""


class Tokenizer:
    """Ids back to text, from the release's ``tokenizer.json``."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer as Backend

        self._backend = Backend.from_file(path)

    def decode(self, ids) -> str:
        """The words of ``ids``, special tokens and timestamps left out, as transformers' Whisper tokenizer gives them."""
        text = self._backend.decode([int(token) for token in ids], skip_special_tokens=True)
        return TIMESTAMP.sub("", text)
