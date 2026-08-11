from __future__ import annotations

import asyncio

import pytest
from google.adk.models.lite_llm import LiteLlm

from guia_cli.agents.medicinal_chemist import (
    MEDICINAL_CHEMIST_INSTRUCTION,
    MEDICINAL_CHEMIST_TOOLS,
    MedicinalChemistAgent,
    build_medicinal_chemist,
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


def test_medicinal_chemist_has_exact_basic_tool_allowlist() -> None:
    agent = build_medicinal_chemist(
        LiteLlm(model="openai/test-model")
    )

    assert {_tool_name(tool) for tool in agent.tools} == {
        tool.__name__ for tool in MEDICINAL_CHEMIST_TOOLS
    }
    assert {_tool_name(tool) for tool in agent.tools} == {
        "rest_api_call",
        "list_session_files",
        "read_session_file",
        "write_markdown_result",
        "write_csv_result",
    }


def test_medicinal_chemist_has_no_subagents_or_code_executor() -> None:
    agent = build_medicinal_chemist(
        LiteLlm(model="openai/test-model")
    )

    assert agent.sub_agents == []
    assert agent.code_executor is None


def test_medicinal_chemist_has_validated_api_recovery_guidance() -> None:
    assert "/chembl/api/data/target/search.json" in MEDICINAL_CHEMIST_INSTRUCTION
    assert "/chembl/api/data/activity.json" in MEDICINAL_CHEMIST_INSTRUCTION
    assert "/rest/pug/compound/name/aspirin/cids/JSON" in (
        MEDICINAL_CHEMIST_INSTRUCTION
    )
    assert '"ok":false' in MEDICINAL_CHEMIST_INSTRUCTION
    assert "Never repeat an identical failed request" in (
        MEDICINAL_CHEMIST_INSTRUCTION
    )


def test_medicinal_chemist_executes_through_runtime() -> None:
    runtime = FakeAgentRuntime("Reported ChEMBL evidence.")
    agent = MedicinalChemistAgent(runtime=runtime)

    result = asyncio.run(
        agent.run(
            "Find reported EGFR inhibitors.",
            session_id="session-1",
        )
    )

    assert result == "Reported ChEMBL evidence."
    assert runtime.calls == [
        ("Find reported EGFR inhibitors.", "session-1")
    ]


def test_medicinal_chemist_rejects_empty_task() -> None:
    runtime = FakeAgentRuntime("unused")
    agent = MedicinalChemistAgent(runtime=runtime)

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(agent.run("  "))

    assert runtime.calls == []


def test_medicinal_chemist_requires_one_runtime_source() -> None:
    with pytest.raises(ValueError):
        MedicinalChemistAgent()

    with pytest.raises(ValueError):
        MedicinalChemistAgent(
            model=LiteLlm(model="openai/test-model"),
            runtime=FakeAgentRuntime("unused"),
        )
