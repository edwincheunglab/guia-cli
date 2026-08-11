from __future__ import annotations

import asyncio

import pytest
from google.adk.models.lite_llm import LiteLlm

from guia_cli.agents.scientific_critic import (
    SCIENTIFIC_CRITIC_INSTRUCTION,
    SCIENTIFIC_CRITIC_TOOLS,
    ScientificCriticAgent,
    build_scientific_critic,
)


class FakeAgentRuntime:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        user_id: str = "local-user",
    ) -> str:
        self.calls.append((prompt, session_id))
        return self.response


def _tool_name(tool: object) -> str:
    return getattr(tool, "name", getattr(tool, "__name__", ""))


def test_scientific_critic_has_exact_basic_tool_allowlist() -> None:
    agent = build_scientific_critic(LiteLlm(model="openai/test-model"))

    assert {_tool_name(tool) for tool in agent.tools} == {
        tool.__name__ for tool in SCIENTIFIC_CRITIC_TOOLS
    }
    assert {_tool_name(tool) for tool in agent.tools} == {
        "rest_api_call",
        "list_session_files",
        "read_session_file",
        "write_markdown_result",
        "write_csv_result",
    }


def test_scientific_critic_has_no_subagents_or_code_executor() -> None:
    agent = build_scientific_critic(LiteLlm(model="openai/test-model"))

    assert agent.sub_agents == []
    assert agent.code_executor is None


def test_scientific_critic_has_rigorous_review_guidance() -> None:
    assert "/entrez/eutils/esearch.fcgi" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "/entrez/eutils/esummary.fcgi" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "association from causation" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "multiple" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "testing" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "abstract-only check" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "Issue" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "Correction" in SCIENTIFIC_CRITIC_INSTRUCTION
    assert "Do not delegate work to another agent" in (
        SCIENTIFIC_CRITIC_INSTRUCTION
    )
    assert "browser automation, MCP servers, remote agents" in (
        SCIENTIFIC_CRITIC_INSTRUCTION
    )


def test_scientific_critic_executes_through_runtime() -> None:
    runtime = FakeAgentRuntime("The central claim is only partly supported.")
    agent = ScientificCriticAgent(runtime=runtime)

    result = asyncio.run(
        agent.run(
            "Critique the evidence supporting this report.",
            session_id="session-critic",
        )
    )

    assert result == "The central claim is only partly supported."
    assert runtime.calls == [
        ("Critique the evidence supporting this report.", "session-critic")
    ]


def test_scientific_critic_rejects_empty_task() -> None:
    runtime = FakeAgentRuntime("unused")
    agent = ScientificCriticAgent(runtime=runtime)

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(agent.run("  "))

    assert runtime.calls == []


def test_scientific_critic_requires_one_runtime_source() -> None:
    with pytest.raises(ValueError):
        ScientificCriticAgent()

    with pytest.raises(ValueError):
        ScientificCriticAgent(
            model=LiteLlm(model="openai/test-model"),
            runtime=FakeAgentRuntime("unused"),
        )
