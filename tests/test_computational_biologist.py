from __future__ import annotations

import asyncio

import pytest
from google.adk.models.lite_llm import LiteLlm

from guia_cli.agents.computational_biologist import (
    COMPUTATIONAL_BIOLOGIST_INSTRUCTION,
    COMPUTATIONAL_BIOLOGIST_TOOLS,
    ComputationalBiologistAgent,
    build_computational_biologist,
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


def test_computational_biologist_has_exact_basic_tool_allowlist() -> None:
    agent = build_computational_biologist(
        LiteLlm(model="openai/test-model")
    )

    assert {_tool_name(tool) for tool in agent.tools} == {
        tool.__name__ for tool in COMPUTATIONAL_BIOLOGIST_TOOLS
    }
    assert {_tool_name(tool) for tool in agent.tools} == {
        "rest_api_call",
        "list_session_files",
        "read_session_file",
        "write_markdown_result",
        "write_csv_result",
    }


def test_computational_biologist_has_no_subagents_or_code_executor() -> None:
    agent = build_computational_biologist(
        LiteLlm(model="openai/test-model")
    )

    assert agent.sub_agents == []
    assert agent.code_executor is None


def test_computational_biologist_has_validated_api_and_scope_guidance() -> None:
    assert "/entrez/eutils/esearch.fcgi" in (
        COMPUTATIONAL_BIOLOGIST_INSTRUCTION
    )
    assert "/entrez/eutils/esummary.fcgi" in (
        COMPUTATIONAL_BIOLOGIST_INSTRUCTION
    )
    assert "https://api.platform.opentargets.org/api/v4/graphql" in (
        COMPUTATIONAL_BIOLOGIST_INSTRUCTION
    )
    assert "https://rest.uniprot.org/uniprotkb/P38398.json" in (
        COMPUTATIONAL_BIOLOGIST_INSTRUCTION
    )
    assert "Open Targets scores" in COMPUTATIONAL_BIOLOGIST_INSTRUCTION
    assert "Do not claim to perform differential expression" in (
        COMPUTATIONAL_BIOLOGIST_INSTRUCTION
    )


def test_computational_biologist_executes_through_runtime() -> None:
    runtime = FakeAgentRuntime("Integrated gene evidence.")
    agent = ComputationalBiologistAgent(runtime=runtime)

    result = asyncio.run(
        agent.run(
            "Summarize evidence for human BRCA1.",
            session_id="session-1",
        )
    )

    assert result == "Integrated gene evidence."
    assert runtime.calls == [
        ("Summarize evidence for human BRCA1.", "session-1")
    ]


def test_computational_biologist_rejects_empty_task() -> None:
    runtime = FakeAgentRuntime("unused")
    agent = ComputationalBiologistAgent(runtime=runtime)

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(agent.run("  "))

    assert runtime.calls == []


def test_computational_biologist_requires_one_runtime_source() -> None:
    with pytest.raises(ValueError):
        ComputationalBiologistAgent()

    with pytest.raises(ValueError):
        ComputationalBiologistAgent(
            model=LiteLlm(model="openai/test-model"),
            runtime=FakeAgentRuntime("unused"),
        )
