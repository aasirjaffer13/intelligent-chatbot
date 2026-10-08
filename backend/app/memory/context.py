"""Building the session context the reply generator consumes (Phase 6).

The context window is *derived from stored history only* — nothing about a
user is hardcoded anywhere. "My name is Zephyr" followed by "What is my
name?" works because turn 1 was persisted and turn 2 reads it back.

Name handling is deliberately rule-based: a small set of statement/question
patterns plus a predicate blacklist ("I'm fine" must not record a name).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.memory.base import MemoryMessage

# "My name is X" / "I am X" / "I'm X" / "Call me X"
_NAME_STATEMENT_RE = re.compile(
    r"\b(?:my name is|i am|i'm|call me|this is)\s+([A-Za-z][A-Za-z'\-]{1,20})\b",
    re.IGNORECASE,
)

_NAME_QUESTION_RE = re.compile(
    r"\b(?:what(?:'s| is) my name|who am i|do you (?:know|remember) my name"
    r"|what do you call me)\b",
    re.IGNORECASE,
)

# Common completions of "I'm ..." / "I am ..." that are NOT names.
_NON_NAME_PREDICATES = {
    "fine", "good", "ok", "okay", "great", "tired", "happy", "sad", "angry",
    "here", "back", "not", "so", "a", "an", "the", "new", "just", "sure",
    "done", "ready", "looking", "trying", "going", "working", "curious",
    "bored", "well", "afraid", "sorry", "confused", "lost", "stuck",
    "your", "his", "her", "someone", "anyone", "everyone",
}


def extract_name(text: str) -> str | None:
    """The name this utterance declares, if any (validated against a blacklist)."""
    match = _NAME_STATEMENT_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip(".,!?;:'\"")
    if name.lower() in _NON_NAME_PREDICATES:
        return None
    # Preserve normal capitalization; capitalize a fully-lowercase name.
    return name if any(c.isupper() for c in name) else name.capitalize()


def is_name_question(text: str) -> bool:
    return bool(_NAME_QUESTION_RE.search(text))


@dataclass
class MemoryContext:
    """Everything the reply generator may use from history."""

    messages: list[MemoryMessage] = field(default_factory=list)
    turn_count: int = 0                     # user turns seen so far (incl. current)
    known_name: str | None = None           # latest name declared in history

    @property
    def has_history(self) -> bool:
        return bool(self.messages)


def build_context(recent: list[MemoryMessage], current_message: str) -> MemoryContext:
    """Assemble the context window from stored history + the incoming message.

    The *newest* name statement wins (a user changing their name is honored).
    The current message participates too, so a one-shot "My name is X" can be
    acknowledged with X in the same turn.
    """
    known_name: str | None = None
    user_turns = 0
    for message in recent:
        if message.role == "user":
            user_turns += 1
            declared = extract_name(message.content)
            if declared:
                known_name = declared

    declared_now = extract_name(current_message)
    if declared_now:
        known_name = declared_now
    if current_message.strip():
        user_turns += 1

    return MemoryContext(messages=list(recent), turn_count=user_turns, known_name=known_name)
