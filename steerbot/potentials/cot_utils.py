"""Shared helpers for CoT potentials (decode generation bytes → text, first ``<thought>`` block)."""

from __future__ import annotations

import re

_THOUGHT_BLOCK = re.compile(r"<thought>(.*?)(?:</thought>|$)", flags=re.DOTALL)


def context_to_text(context) -> str:
    """Decode a sampler ``context`` (bytes, ints, Tokens, or list thereof) to UTF-8 text.

    ``Coerced(..., f=b"".join)`` passes a single :class:`bytes` object; iterating it
    yields ints, so we must treat ``bytes`` as a whole.
    """
    if isinstance(context, (bytes, bytearray)):
        raw = bytes(context)
    else:
        buf = bytearray()
        for t in context:
            if isinstance(t, int):
                buf.append(t & 0xFF)
            elif isinstance(t, (bytes, bytearray)):
                buf.extend(t)
            else:
                buf.extend(bytes(t))
        raw = bytes(buf)
    return raw.decode("utf-8", errors="replace")


def word_count_in_first_thought(text: str) -> int:
    """Words inside the first ``<thought>...</thought>`` block (or open ``<thought>`` to EOS)."""
    m = _THOUGHT_BLOCK.search(text)
    if not m:
        return 0
    return len(m.group(1).split())


def word_count_in_first_thought_from_context(context) -> int:
    return word_count_in_first_thought(context_to_text(context))


def _sentence_count_in_body(body: str) -> int:
    """Heuristic sentence count for a free-form thought body.

    We treat a sentence boundary as a run of ``.``/``!``/``?`` followed by either:
    - whitespace (space/newline/tab), OR
    - end-of-string, OR
    - an ASCII letter/digit (to avoid the easy hack ``Sentence1.Sentence2.``).

    This is still heuristic (e.g. ``e.g.`` / decimals) but makes the cap much harder
    to bypass via formatting quirks.
    """
    body = body.strip()
    if not body:
        return 0
    # Split *after* sentence-ending punctuation when it's followed by:
    # - whitespace, OR
    # - end-of-string, OR
    # - an alnum character (handles ".A" and ".2" without requiring a space).
    #
    # Counting segments (not just punctuation) ensures the final sentence is counted even if
    # it doesn't end with ".?!", which is a common "hack" under caps.
    # Python regex lookbehind must be fixed-width, so we split using a capturing group
    # and reconstruct sentence-like chunks.
    tokens = re.split(r"([.!?]+)(?=(?:\s|$|[A-Za-z0-9]))", body)
    # tokens alternates: [text, punct, text, punct, ...] (punct may be absent at end)
    parts: list[str] = []
    i = 0
    while i < len(tokens):
        text = tokens[i]
        punct = tokens[i + 1] if i + 1 < len(tokens) and re.fullmatch(r"[.!?]+", tokens[i + 1]) else ""
        chunk = (text + punct).strip()
        if chunk:
            parts.append(chunk)
        i += 2 if punct else 1

    return len(parts)


def sentence_count_in_first_thought(text: str) -> int:
    """Sentences (heuristic) inside the first ``<thought>...</thought>`` block."""
    m = _THOUGHT_BLOCK.search(text)
    if not m:
        return 0
    return _sentence_count_in_body(m.group(1))


def sentence_count_in_first_thought_from_context(context) -> int:
    return sentence_count_in_first_thought(context_to_text(context))
