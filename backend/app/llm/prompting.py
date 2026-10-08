"""Prompt construction (Phase 8).

The prompt is a *structured dump of the pipeline's decisions* — intent,
confidence, entities, memory context, raw message. The LLM never decides
what the user meant; it only words the reply. That split is what keeps
Phase 3/4/5/6/7 behaviour stable when the model behind the prompt changes.

Design notes:
* system prompt carries persona + hard rules (no invented facts),
* user prompt is deterministic (same inputs -> same prompt) for testability,
* history is capped so prompt size stays bounded regardless of window size.
"""

from __future__ import annotations

from app.llm.base import LLMRequest
from app.memory.base import MemoryMessage

SYSTEM_PROMPT = (
    "You are NOVA, the assistant inside the NOVA chat application. "
    "Reply in one to three plain-text sentences in the user's language. "
    "Use ONLY the facts supplied in the message below: the conversation "
    "context, the detected intent, and the extracted entities. "
    "If the intent is 'unknown', ask the user to rephrase instead of guessing. "
    "Never invent names, times, dates, prices or document contents — "
    "if a fact is not in the supplied information, say you don't have it."
)

MAX_HISTORY_LINES = 8


def build_chat_request(
    *,
    message: str,
    intent_label: str,
    confidence: float,
    entity_texts: list[str],
    messages: list[MemoryMessage],
    known_name: str | None,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> LLMRequest:
    """Build the completion request for one chat turn."""
    lines: list[str] = []

    lines.append(f"Detected intent: {intent_label} (confidence {confidence:.2f})")
    lines.append(
        "Entities: " + (", ".join(entity_texts) if entity_texts else "none")
    )
    if known_name:
        lines.append(f"User's name (from memory): {known_name}")

    history = messages[-MAX_HISTORY_LINES:]
    if history:
        lines.append("Recent conversation:")
        for msg in history:
            speaker = "user" if msg.role == "user" else "NOVA"
            lines.append(f"  {speaker}: {msg.content}")

    lines.append(f"User message: {message}")
    lines.append("Reply as NOVA.")

    return LLMRequest(
        prompt="\n".join(lines),
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
    )
