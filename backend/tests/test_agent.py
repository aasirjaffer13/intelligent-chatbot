"""Phase 9 tests: tools, registry, and the ReAct-style agent loop.

Fully offline: HTTP tools use ``httpx.MockTransport``; the loop runs on a
scripted provider; chat integration injects the script via ``llm=``.
"""

from __future__ import annotations

import httpx
import pytest

from app.agent import (
    STEPS_EXCEEDED,
    AgentLoop,
    AgentResult,
    ToolCall,
    parse_decision,
)
from app.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse
from app.memory import InMemoryStore
from app.services.chat_service import ChatService
from app.tools import (
    CalculatorTool,
    CurrentTimeTool,
    DocumentSearchTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
    WeatherTool,
    build_default_registry,
)


class ScriptedProvider(LLMProvider):
    """Replies from a script; records every request. Test double only."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self.replies:
            raise LLMError("script exhausted")
        return LLMResponse(text=self.replies.pop(0), provider=self.name, model="scripted")


def _loop(replies: list[str], registry: ToolRegistry | None = None, *, max_steps: int = 6):
    provider = ScriptedProvider(replies)
    return provider, AgentLoop(provider, registry or build_default_registry(), max_steps=max_steps)


# --- calculator --------------------------------------------------------------


class TestCalculatorTool:
    def setup_method(self) -> None:
        self.tool = CalculatorTool()

    @pytest.mark.parametrize(
        ("expr", "expected"),
        [
            ("2 + 2", "4"),
            ("(2 + 3) * 4", "20"),
            ("10 / 4", "2.5"),
            ("7 // 2", "3"),
            ("7 % 3", "1"),
            ("2 ** 10", "1024"),
            ("-5 + 3", "-2"),
            ("0.1 + 0.2", "0.3"),
        ],
    )
    def test_valid_expressions(self, expr: str, expected: str) -> None:
        result = self.tool.run({"expression": expr})
        assert result.ok, result.output
        assert result.output == expected

    def test_division_by_zero_is_an_error_result(self) -> None:
        result = self.tool.run({"expression": "1 / 0"})
        assert not result.ok
        assert "division by zero" in result.output

    @pytest.mark.parametrize("expr", ["__import__('os')", "open('/etc/passwd')", "2 +", "a + 1", "True + 1"])
    def test_non_arithmetic_is_rejected(self, expr: str) -> None:
        result = self.tool.run({"expression": expr})
        assert not result.ok
        assert result.output.startswith("error:")

    def test_power_limits(self) -> None:
        assert not self.tool.run({"expression": "9 ** 999999"}).ok
        assert not self.tool.run({"expression": "2 ** 5000"}).ok

    def test_missing_expression(self) -> None:
        assert not self.tool.run({}).ok

    def test_oversized_expression(self) -> None:
        assert not self.tool.run({"expression": "1+1" * 200}).ok


# --- registry ----------------------------------------------------------------


class TestToolRegistry:
    def test_register_get_names(self) -> None:
        registry = ToolRegistry([CalculatorTool(), CurrentTimeTool()])
        assert registry.names() == ["calculator", "current_time"]
        assert isinstance(registry.get("calculator"), CalculatorTool)
        assert registry.get("weather") is None
        assert len(registry) == 2

    def test_duplicate_name_rejected(self) -> None:
        registry = ToolRegistry([CalculatorTool()])
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CalculatorTool())

    def test_unregister(self) -> None:
        registry = ToolRegistry([CalculatorTool()])
        assert registry.unregister("calculator") is True
        assert registry.unregister("calculator") is False

    def test_describe_lists_signatures(self) -> None:
        registry = ToolRegistry([CalculatorTool(), CurrentTimeTool()])
        text = registry.describe()
        assert "calculator(expression: " in text
        assert "current_time()" in text
        assert '{"tool": "<name>", "args": {...}}' in text

    def test_describe_empty_registry(self) -> None:
        assert ToolRegistry().describe() == ""

    def test_default_registry_has_all_five(self) -> None:
        registry = build_default_registry()
        assert registry.names() == [
            "calculator",
            "current_time",
            "document_search",
            "weather",
            "web_search",
        ]


# --- other built-in tools ----------------------------------------------------


class TestBuiltinTools:
    def test_current_time(self) -> None:
        result = CurrentTimeTool().run({})
        assert result.ok
        assert "timezone" in result.output
        assert result.data["iso"]

    def test_document_search_without_documents(self, monkeypatch, tmp_path) -> None:
        from app.rag import RagStore

        store = RagStore(f"sqlite:///{tmp_path / 'rag.db'}")
        monkeypatch.setattr("app.rag.get_rag_store", lambda: store)
        result = DocumentSearchTool().run({"query": "vacation"})
        assert result.ok
        assert "no documents" in result.output

    def test_weather_with_mock_transport(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "geocoding" in str(request.url):
                return httpx.Response(
                    200, json={"results": [{"name": "Nairobi", "latitude": -1.3, "longitude": 36.8}]}
                )
            return httpx.Response(
                200,
                json={
                    "current": {
                        "temperature_2m": 24.5,
                        "weather_code": 2,
                        "wind_speed_10m": 12.0,
                    }
                },
            )

        tool = WeatherTool(transport=httpx.MockTransport(handler))
        result = tool.run({"location": "Nairobi"})
        assert result.ok, result.output
        assert "Nairobi: 24.5°C" in result.output
        assert "partly cloudy" in result.output

    def test_weather_unknown_place(self) -> None:
        tool = WeatherTool(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))
        )
        result = tool.run({"location": "Gotham City"})
        assert not result.ok
        assert "no place found" in result.output

    def test_weather_network_failure_is_error_result(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        result = WeatherTool(transport=httpx.MockTransport(handler)).run({"location": "x"})
        assert not result.ok
        assert "failed" in result.output

    def test_web_search_abstract(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"AbstractText": "Python is a programming language."}
            )

        result = WebSearchTool(transport=httpx.MockTransport(handler)).run(
            {"query": "python"}
        )
        assert result.ok
        assert "programming language" in result.output

    def test_web_search_no_results(self) -> None:
        tool = WebSearchTool(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        result = tool.run({"query": "zzz nothing"})
        assert result.ok
        assert "no instant answer" in result.output


# --- decision parsing --------------------------------------------------------


class TestParseDecision:
    def test_plain_text_is_final(self) -> None:
        assert parse_decision("Just a normal answer.") is None

    def test_json_tool_call(self) -> None:
        call = parse_decision('{"tool": "calculator", "args": {"expression": "1+1"}}')
        assert call == ToolCall(name="calculator", args={"expression": "1+1"})

    def test_fenced_json_tool_call(self) -> None:
        call = parse_decision('```json\n{"tool": "current_time", "args": {}}\n```')
        assert call is not None and call.name == "current_time"

    def test_dict_without_tool_key_is_final(self) -> None:
        assert parse_decision('{"answer": "42"}') is None

    def test_broken_json_is_final(self) -> None:
        assert parse_decision('{"tool": "calc') is None

    def test_args_must_be_object(self) -> None:
        call = parse_decision('{"tool": "x", "args": "nope"}')
        assert call is not None and call.args == {}


# --- agent loop --------------------------------------------------------------


class TestAgentLoop:
    def test_direct_answer_single_step(self) -> None:
        provider, loop = _loop(["Hello! How can I help?"])
        result = loop.run(LLMRequest(prompt="user: hi", system="sys"))
        assert isinstance(result, AgentResult)
        assert result.reply == "Hello! How can I help?"
        assert result.tool_calls == 0
        assert result.used_tools == []
        assert len(provider.calls) == 1
        assert "calculator" in provider.calls[0].prompt  # tools described

    def test_tool_then_final_answer(self) -> None:
        provider, loop = _loop(
            [
                '{"tool": "calculator", "args": {"expression": "6 * 7"}}',
                "The answer is 42.",
            ]
        )
        result = loop.run(LLMRequest(prompt="what is six times seven?", system="sys"))

        assert result.reply == "The answer is 42."
        assert result.used_tools == ["calculator"]
        assert [step.kind for step in result.steps] == ["llm", "tool", "observation", "llm"]
        assert "42" in result.steps[2].text  # observation carries the tool output
        # second LLM call sees the observation
        assert "Observation: 42" in provider.calls[1].prompt
        assert "Tool call:" in provider.calls[1].prompt

    def test_unknown_tool_becomes_observation(self) -> None:
        provider, loop = _loop(
            ['{"tool": "teleport", "args": {}}', "Sorry, I can't do that."]
        )
        result = loop.run(LLMRequest(prompt="go", system="sys"))
        assert result.reply == "Sorry, I can't do that."
        observation = result.steps[2].text
        assert "unknown tool" in observation
        assert "calculator" in observation  # available tools listed
        assert result.used_tools == []

    def test_crashing_tool_is_an_observation(self) -> None:
        from app.tools.base import Tool

        class Bomb(Tool):
            name = "bomb"
            description = "crashes"
            parameters: dict = {}

            def run(self, args: dict) -> ToolResult:
                raise RuntimeError("boom")

        registry = ToolRegistry([Bomb()])
        provider, loop = _loop(
            ['{"tool": "bomb", "args": {}}', "Recovered."], registry=registry
        )
        result = loop.run(LLMRequest(prompt="x", system="sys"))
        assert result.reply == "Recovered."
        assert "crashed" in result.steps[2].text

    def test_max_steps_rail(self) -> None:
        tool_call = '{"tool": "calculator", "args": {"expression": "1+1"}}'
        provider, loop = _loop([tool_call, tool_call], max_steps=2)
        result = loop.run(LLMRequest(prompt="loop forever", system="sys"))
        assert result.reply == STEPS_EXCEEDED
        assert len(provider.calls) == 2  # bounded — no runaway model
        assert result.tool_calls == 2

    def test_provider_failure_propagates(self) -> None:
        provider, loop = _loop([])  # script exhausted -> LLMError
        with pytest.raises(LLMError):
            loop.run(LLMRequest(prompt="x", system="sys"))

    def test_empty_reply_returns_empty(self) -> None:
        provider, loop = _loop(["   "])
        result = loop.run(LLMRequest(prompt="x", system="sys"))
        assert result.reply == ""  # chat layer treats this as fallback

    def test_invalid_max_steps(self) -> None:
        with pytest.raises(ValueError):
            AgentLoop(ScriptedProvider([]), ToolRegistry(), max_steps=0)


# --- chat service integration ------------------------------------------------


class TestChatWithAgent:
    def test_calculator_loop_through_chat(self) -> None:
        scripted = ScriptedProvider(
            [
                '{"tool": "calculator", "args": {"expression": "6 * 7"}}',
                "The answer is 42.",
            ]
        )
        service = ChatService(memory=InMemoryStore(), llm=scripted)

        from app.schemas import ChatRequest

        # NB: "six times seven" classifies as the *time* intent (0.60) and
        # time answers are deterministic — so use a neutral phrasing here.
        response = service.handle(ChatRequest(message="can you calculate something for me"))

        assert response.response == "The answer is 42."
        assert len(scripted.calls) == 2
        assert "calculator(expression" in scripted.calls[0].prompt
        assert "Observation: 42" in scripted.calls[1].prompt

    def test_direct_llm_reply_still_works(self) -> None:
        scripted = ScriptedProvider(["Nice weather we're having."])
        service = ChatService(memory=InMemoryStore(), llm=scripted)

        from app.schemas import ChatRequest

        response = service.handle(ChatRequest(message="tell me something"))
        assert response.response == "Nice weather we're having."
        assert len(scripted.calls) == 1

    def test_deterministic_answers_bypass_the_agent(self) -> None:
        scripted = ScriptedProvider(['{"tool": "current_time", "args": {}}'])
        service = ChatService(memory=InMemoryStore(), llm=scripted)

        from app.schemas import ChatRequest

        response = service.handle(ChatRequest(message="what time is it"))
        assert scripted.calls == []  # clock answers never need a model
        assert "timezone" in response.response

    def test_agent_failure_falls_back_to_templates(self) -> None:
        class ExplodingProvider(LLMProvider):
            name = "exploding"

            def complete(self, request: LLMRequest) -> LLMResponse:
                raise LLMError("everything is broken")

        broken = ChatService(memory=InMemoryStore(), llm=ExplodingProvider())
        control = ChatService(memory=InMemoryStore(), llm=None)

        from app.schemas import ChatRequest

        a = broken.handle(ChatRequest(message="hello there"))
        b = control.handle(ChatRequest(message="hello there"))
        assert a.response == b.response

    def test_injected_agent_is_used_directly(self) -> None:
        scripted = ScriptedProvider(["from the injected agent"])
        registry = ToolRegistry([CalculatorTool()])
        agent = AgentLoop(scripted, registry, max_steps=3)
        service = ChatService(memory=InMemoryStore(), llm=None, agent=agent)

        from app.schemas import ChatRequest

        response = service.handle(ChatRequest(message="hello there"))
        assert response.response == "from the injected agent"
        assert len(scripted.calls) == 1
