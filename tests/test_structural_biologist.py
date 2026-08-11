from __future__ import annotations

import asyncio

import pytest
from google.adk.models.lite_llm import LiteLlm

from guia_cli.agents.structural_biologist import (
    STRUCTURAL_BIOLOGIST_INSTRUCTION,
    STRUCTURAL_BIOLOGIST_TOOLS,
    StructuralBiologistAgent,
    build_structural_biologist,
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


def test_structural_biologist_has_exact_basic_tool_allowlist() -> None:
    agent = build_structural_biologist(
        LiteLlm(model="openai/test-model")
    )

    assert {_tool_name(tool) for tool in agent.tools} == {
        tool.__name__ for tool in STRUCTURAL_BIOLOGIST_TOOLS
    }
    assert {_tool_name(tool) for tool in agent.tools} == {
        "rest_api_call",
        "list_session_files",
        "read_session_file",
        "write_markdown_result",
        "write_csv_result",
    }


def test_structural_biologist_has_no_subagents_or_code_executor() -> None:
    agent = build_structural_biologist(
        LiteLlm(model="openai/test-model")
    )

    assert agent.sub_agents == []
    assert agent.code_executor is None


def test_structural_biologist_has_validated_api_and_quality_guidance() -> None:
    assert "https://search.rcsb.org/rcsbsearch/v2/query" in (
        STRUCTURAL_BIOLOGIST_INSTRUCTION
    )
    assert "https://data.rcsb.org/rest/v1/core/entry/1M17" in (
        STRUCTURAL_BIOLOGIST_INSTRUCTION
    )
    assert "https://rest.uniprot.org/uniprotkb/P00533.json" in (
        STRUCTURAL_BIOLOGIST_INSTRUCTION
    )
    assert '"return_type":"entry"' in STRUCTURAL_BIOLOGIST_INSTRUCTION
    assert "asymmetric unit from the biological assembly" in (
        STRUCTURAL_BIOLOGIST_INSTRUCTION
    )
    assert "Do not claim to perform structure prediction" in (
        STRUCTURAL_BIOLOGIST_INSTRUCTION
    )


def test_structural_biologist_executes_through_runtime() -> None:
    runtime = FakeAgentRuntime("RCSB PDB evidence.")
    agent = StructuralBiologistAgent(runtime=runtime)

    result = asyncio.run(
        agent.run(
            "Inspect the experimental method for PDB 1M17.",
            session_id="session-1",
        )
    )

    assert result == "RCSB PDB evidence."
    assert runtime.calls == [
        ("Inspect the experimental method for PDB 1M17.", "session-1")
    ]


def test_structural_biologist_rejects_empty_task() -> None:
    runtime = FakeAgentRuntime("unused")
    agent = StructuralBiologistAgent(runtime=runtime)

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(agent.run("  "))

    assert runtime.calls == []


def test_structural_biologist_requires_one_runtime_source() -> None:
    with pytest.raises(ValueError):
        StructuralBiologistAgent()

    with pytest.raises(ValueError):
        StructuralBiologistAgent(
            model=LiteLlm(model="openai/test-model"),
            runtime=FakeAgentRuntime("unused"),
        )
