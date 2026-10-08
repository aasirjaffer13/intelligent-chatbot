"""Tool contract for the agent loop (Phase 9).

A tool is a plain class: name + description + parameter docs for the
prompt, and one ``run(args)`` method that returns a :class:`ToolResult`.
The agent loop never inspects tool internals — that's what makes tools
modular: register a class, it becomes available to the model.

Errors are *data*, not exceptions: a tool that fails returns
``ToolResult(ok=False, output="reason")`` and the loop feeds that reason
back to the model as an observation. Uncaught exceptions are also caught
by the loop, so one broken tool can never crash a conversation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolResult:
    """What a tool run produced — success or a readable failure."""

    ok: bool
    output: str
    data: dict = field(default_factory=dict)


class Tool(ABC):
    """Base class every tool implements."""

    name: str = ""
    description: str = ""
    parameters: dict[str, str] = {}  # param name -> one-line description

    @abstractmethod
    def run(self, args: dict) -> ToolResult:
        """Execute with the model-provided arguments. Never raise for
        *expected* problems (bad input, missing data) — return ok=False."""

    def signature(self) -> str:
        """``name(param: description, ...)`` — one line for the prompt."""
        if not self.parameters:
            return f"{self.name}()"
        args = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({args})"
