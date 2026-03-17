"""Utilities for sanitizing chat-template rendered prompts.

Some HF tokenizers ship chat templates that prepend boilerplate system blocks
like "You are ChatGPT, a large language model..." plus metadata.

That boilerplate is not part of our constructed messages and can waste context
window and interfere with downstream behavior. This module provides a
conservative stripper that only activates when the canonical boilerplate is
detected.
"""

from __future__ import annotations

import re

__all__ = ["strip_leading_chatgpt_boilerplate"]


_CHATGPT_BOILERPLATE_RE = re.compile(
    r"\A\s*<\|start\|>system<\|message\|>.*?<\|end\|>\s*",
    flags=re.DOTALL,
)


def strip_leading_chatgpt_boilerplate(prompt_text: str) -> str:
    """Remove common chat-template boilerplate from a rendered prompt.

    Parameters
    ----------
    prompt_text:
        Prompt text after applying a tokenizer chat template.

    Returns
    -------
    str
        The prompt text with the leading boilerplate system message removed when
        detected. If no boilerplate is detected, returns the original text.
    """

    if not prompt_text:
        return prompt_text

    # Fast-path: only attempt stripping when the canonical phrase is present.
    if "You are ChatGPT, a large language model" not in prompt_text:
        return prompt_text

    # Tokenizer-template variant (e.g., <|start|>system<|message|>...<|end|>)
    if prompt_text.lstrip().startswith("<|start|>system<|message|>"):
        match = _CHATGPT_BOILERPLATE_RE.match(prompt_text)
        if match:
            stripped = prompt_text[match.end() :]
            return stripped.lstrip("\n")

    # Plain-text variant: conservatively drop only the first paragraph when it
    # starts with the canonical phrase.
    lstripped = prompt_text.lstrip()
    if lstripped.startswith("You are ChatGPT, a large language model"):
        parts = lstripped.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].lstrip("\n")
        return ""

    return prompt_text
