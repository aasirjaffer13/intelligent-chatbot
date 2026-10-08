"""Tool package (Phase 9): contract, registry, built-in tools."""

from app.tools.base import Tool, ToolResult
from app.tools.builtin import (
    CalculatorTool,
    CurrentTimeTool,
    DocumentSearchTool,
    WebSearchTool,
    WeatherTool,
)
from app.tools.registry import ToolRegistry, build_default_registry

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "build_default_registry",
    "CalculatorTool",
    "CurrentTimeTool",
    "DocumentSearchTool",
    "WeatherTool",
    "WebSearchTool",
]
