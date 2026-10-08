"""ReAct-style agent loop (Phase 9).

    user -> LLM -> tool call? -> tool -> observation -> LLM -> ... -> answer

Each iteration the model sees the original prompt, every prior tool call
and observation, and either (a) answers in plain text — loop ends — or
(b) emits ``{"tool": ..., "args": ...}`` — the loop executes it and feeds
the observation back. Bounded by ``max_steps`` so a confused model can
never spin forever; on exhaustion the loop returns an honest limit
message instead of a truncated answer.

Parsing is deliberately forgiving: *anything that is not a JSON object
with a ``tool`` key is treated as the final answer*, so a model that
ignores the format still produces a reply.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.llm.base import LLMProvider, LLMRequest
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

STEPS_EXCEEDED = (
    "I hit my step limit before finishing that — please rephrase, "
    "or ask something that needs fewer lookups."
)


@dataclass(frozen=True)
class ToolCall:
    """A parsed model decision to call one tool."""

    name: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AgentStep:
    """One entry of the loop's trace (for logs, tests, future UIs)."""

    kind: str  # "llm" | "tool" | "observation"
    text: str


@dataclass
class AgentResult:
    reply: str
    steps: list[AgentStep] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind == "tool")


def parse_decision(text: str) -> ToolCall | None:
    """ToolCall when the model emitted a tool JSON object, else None.

    Handles fenced blocks (```json ... ```), and treats every other
    successful parse (or parse failure) as a plain-text answer.
    """
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("tool"), str):
        args = payload.get("args", {})
        return ToolCall(name=payload["tool"], args=args if isinstance(args, dict) else {})
    return None


class AgentLoop:
    """Drives one conversation turn through the provider and tools."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.provider = provider
        self.registry = registry
        self.max_steps = max_steps

    def run(self, request: LLMRequest) -> AgentResult:
        block = self.registry.describe()
        prompt = f"{request.prompt}\n\n{block}" if block else request.prompt
        steps: list[AgentStep] = []
        used: list[str] = []

        for _ in range(self.max_steps):
            response = self.provider.complete(
                LLMRequest(
                    prompt=prompt,
                    system=request.system,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            steps.append(AgentStep("llm", response.text))

            call = parse_decision(response.text)
            if call is None:
                return AgentResult(reply=response.text.strip(), steps=steps, used_tools=used)

            call_json = json.dumps({"tool": call.name, "args": call.args})
            steps.append(AgentStep("tool", call_json))
            observation = self._execute(call)
            if self.registry.get(call.name) is not None:
                used.append(call.name)
            steps.append(AgentStep("observation", observation))
            logger.debug("agent step: %s -> %s", call_json, observation[:200])

            prompt = (
                f"{request.prompt}\n\n{block}\n\n" if block else f"{request.prompt}\n\n"
            )
            prompt += f"Tool call: {call_json}\nObservation: {observation}\nContinue."

        logger.warning("agent hit max_steps=%d without a final answer", self.max_steps)
        return AgentResult(reply=STEPS_EXCEEDED, steps=steps, used_tools=used)

    def _execute(self, call: ToolCall) -> str:
        """Run one tool; failures become observations, never exceptions."""
        tool = self.registry.get(call.name)
        if tool is None:
            available = ", ".join(self.registry.names()) or "none"
            return f"error: unknown tool {call.name!r} (available: {available})"
        try:
            result: ToolResult = tool.run(call.args)
        except Exception as exc:
            return f"error: tool {call.name!r} crashed: {exc}"
        return result.output
