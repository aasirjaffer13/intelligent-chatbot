"""Tool registry (Phase 9).

The registry is the single source of truth for *what the model can call*.
``describe()`` renders the registry into the prompt; the loop resolves
calls through :meth:`get`. Registering is one line:

    registry.register(MyTool())

A duplicate name raises — silently shadowing a tool is how agent bugs
start.
"""

from __future__ import annotations

from typing import Iterable

from app.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def describe(self, *, max_tools: int | None = None) -> str:
        """Render the 'available tools' block the prompt embeds."""
        lines = []
        for name in self.names():
            tool = self._tools[name]
            lines.append(f"  {tool.signature()} - {tool.description}")
            if max_tools is not None and len(lines) >= max_tools:
                break
        if not lines:
            return ""
        return (
            "Available tools (call one by replying with ONLY a JSON object: "
            '{"tool": "<name>", "args": {...}}):\n'
            + "\n".join(lines)
            + "\nTo answer directly, reply with plain text instead of JSON."
        )


def build_default_registry() -> ToolRegistry:
    """The shipped toolset. document_search wires itself to the shared
    RAG store lazily, so building a registry never touches the database."""
    from app.tools.builtin import (
        CalculatorTool,
        CurrentTimeTool,
        DocumentSearchTool,
        WebSearchTool,
        WeatherTool,
    )

    return ToolRegistry(
        [CalculatorTool(), CurrentTimeTool(), DocumentSearchTool(), WeatherTool(), WebSearchTool()]
    )
