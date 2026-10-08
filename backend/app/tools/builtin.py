"""Built-in tools (Phase 9).

Five shipped tools, each honest about failure:

* **calculator** — safe arithmetic via an AST whitelist (never ``eval``);
* **current_time** — the clock, same source as the deterministic reply;
* **document_search** — RAG retrieval over uploaded documents;
* **weather** — Open-Meteo (keyless), injectable base URLs for offline tests;
* **web_search** — DuckDuckGo instant answers, injectable for offline tests.

The two HTTP tools return ``ok=False`` with a readable reason when the
network misbehaves — the loop passes that back to the model as an
observation instead of crashing the conversation.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime

import httpx

from app.tools.base import Tool, ToolResult

# --- calculator --------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_INT = 10**15
_MAX_EXPR_CHARS = 200


class CalculatorTool(Tool):
    """Arithmetic through a whitelist of AST nodes — no eval, no names,
    no attribute access, no imports: only numbers, operators and parens."""

    name = "calculator"
    description = "evaluate arithmetic: + - * / // % ** and parentheses"
    parameters = {"expression": "expression, e.g. '(2 + 3) * 4'"}

    def run(self, args: dict) -> ToolResult:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return ToolResult(False, "error: 'expression' is required")
        if len(expr) > _MAX_EXPR_CHARS:
            return ToolResult(False, "error: expression too long")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return ToolResult(False, f"error: cannot parse expression: {expr!r}")
        try:
            value = self._eval(tree.body)
        except ZeroDivisionError:
            return ToolResult(False, "error: division by zero")
        except (ValueError, TypeError, OverflowError) as exc:
            return ToolResult(False, f"error: {exc}")
        return ToolResult(True, _format_number(value), {"value": value})

    def _eval(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"unsupported value: {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported operator: {type(node.op).__name__}")
            if isinstance(node.op, ast.Pow):
                base, exp = self._eval(node.left), self._eval(node.right)
                if abs(exp) > 4096 or abs(base) > 10**6:
                    raise ValueError("power out of range")
                return op(base, exp)
            left, right = self._eval(node.left), self._eval(node.right)
            result = op(left, right)
            if isinstance(result, float) and (
                result != result or result in (float("inf"), float("-inf"))
            ):
                raise ValueError("non-finite result")
            if isinstance(result, int) and abs(result) > _MAX_INT:
                raise ValueError("result out of range")
            return result
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported operator: {type(node.op).__name__}")
            return op(self._eval(node.operand))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


# --- current time ------------------------------------------------------------


class CurrentTimeTool(Tool):
    name = "current_time"
    description = "current local date and time with timezone"
    parameters = {}

    def run(self, args: dict) -> ToolResult:
        now = datetime.now().astimezone()
        text = (
            f"{now.strftime('%A, %B %d, %Y %H:%M:%S')} "
            f"(timezone {now.strftime('%Z')}, UTC offset {now.strftime('%z')})"
        )
        return ToolResult(True, text, {"iso": now.isoformat()})


# --- document search ---------------------------------------------------------


class DocumentSearchTool(Tool):
    """Top-k similarity search over uploaded documents (the RAG retriever)."""

    name = "document_search"
    description = "search the uploaded documents for a passage answering a question"
    parameters = {"query": "question or keywords to search for"}

    def run(self, args: dict) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(False, "error: 'query' is required")
        try:
            from app.rag import Retriever, get_rag_store
            from app.services.embedding_service import get_embedding_service

            store = get_rag_store()
            if not store.list_documents():
                return ToolResult(True, "no documents are uploaded yet")
            hits = Retriever(store, get_embedding_service()).search(query, top_k=3)
            if not hits:
                return ToolResult(True, "no matching passages found")
            lines = [
                f"[score {hit.score:.3f}] {hit.filename} (part {hit.chunk_index + 1}): "
                f"{hit.content}"
                for hit in hits
            ]
            return ToolResult(True, "\n".join(lines), {"hits": len(hits)})
        except Exception as exc:  # store/embedder problems are observations, not crashes
            return ToolResult(False, f"document search failed: {exc}")


# --- weather (Open-Meteo, keyless) -------------------------------------------

_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    80: "light showers", 81: "moderate showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}


class WeatherTool(Tool):
    """Two keyless HTTP calls: geocode the place, then read current weather.

    Base URLs and the transport are injectable so tests never touch the
    real network.
    """

    name = "weather"
    description = "current weather (temperature, conditions) for a city or place"
    parameters = {"location": "city or place name, e.g. 'Nairobi'"}

    def __init__(
        self,
        *,
        geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search",
        forecast_url: str = "https://api.open-meteo.com/v1/forecast",
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.geocode_url = geocode_url
        self.forecast_url = forecast_url
        self.timeout = timeout
        self._transport = transport

    def run(self, args: dict) -> ToolResult:
        location = str(args.get("location", "")).strip()
        if not location:
            return ToolResult(False, "error: 'location' is required")
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                geo = client.get(
                    self.geocode_url,
                    params={"name": location, "count": 1, "language": "en"},
                )
                geo.raise_for_status()
                results = geo.json().get("results") or []
                if not results:
                    return ToolResult(False, f"no place found named {location!r}")
                place = results[0]
                lat, lon = place["latitude"], place["longitude"]

                forecast = client.get(
                    self.forecast_url,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                    },
                )
                forecast.raise_for_status()
                current = forecast.json()["current"]
        except httpx.HTTPError as exc:
            return ToolResult(False, f"weather lookup failed: {exc}")
        except (ValueError, KeyError, TypeError) as exc:
            return ToolResult(False, f"unexpected weather response: {exc}")

        code = int(current.get("weather_code", -1))
        conditions = _WMO.get(code, f"weather code {code}")
        text = (
            f"{place.get('name', location)}: {current.get('temperature_2m', '?')}°C, "
            f"{conditions}, wind {current.get('wind_speed_10m', '?')} km/h"
        )
        return ToolResult(
            True, text, {"temperature_c": current.get("temperature_2m"), "code": code}
        )


# --- web search (DuckDuckGo instant answers) ---------------------------------


class WebSearchTool(Tool):
    name = "web_search"
    description = "quick web search for a fact or short answer"
    parameters = {"query": "search terms"}

    def __init__(
        self,
        *,
        base_url: str = "https://api.duckduckgo.com/",
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self._transport = transport

    def run(self, args: dict) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(False, "error: 'query' is required")
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                response = client.get(
                    self.base_url,
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            return ToolResult(False, f"web search failed: {exc}")
        except ValueError as exc:
            return ToolResult(False, f"unexpected search response: {exc}")

        parts: list[str] = []
        for key in ("Answer", "AbstractText"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        for topic in (body.get("RelatedTopics") or [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        if not parts:
            return ToolResult(True, f"no instant answer found for {query!r}")
        return ToolResult(True, "\n".join(parts[:4]))
