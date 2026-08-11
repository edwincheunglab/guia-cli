from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from guia_cli.agents.orchestrator import RoutingDecision
from guia_cli.agents.team import GuiaTeam
from guia_cli.sessions import create_session


class FakeOrchestrator:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str | None]] = []

    async def route(
        self,
        request: str,
        *,
        session_id: str | None = None,
    ) -> RoutingDecision:
        self.calls.append((request, session_id))
        return self.decision


class FakeDomainAgent:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str | None]] = []

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
    ) -> str:
        self.calls.append((task, session_id))
        return self.response


def test_team_routes_medicinal_task_with_shared_session() -> None:
    routing = RoutingDecision(
        agent="medicinal_chemist",
        task="Find reported EGFR inhibitors.",
        reason="Compound bioactivity request.",
    )
    orchestrator = FakeOrchestrator(routing)
    medicinal_chemist = FakeDomainAgent("ChEMBL results")
    team = GuiaTeam(
        orchestrator=orchestrator,
        agents={"medicinal_chemist": medicinal_chemist},
    )

    result = asyncio.run(
        team.ask("Find EGFR inhibitors", session_id="session-1")
    )

    assert result.text == "ChEMBL results"
    assert result.handled is True
    assert result.session_id == "session-1"
    assert orchestrator.calls == [("Find EGFR inhibitors", "session-1")]
    assert medicinal_chemist.calls == [
        ("Find reported EGFR inhibitors.", "session-1")
    ]


def test_team_routes_structural_task_with_shared_session() -> None:
    routing = RoutingDecision(
        agent="structural_biologist",
        task="Inspect experimental metadata for PDB 1M17.",
        reason="Protein structure request.",
    )
    structural_biologist = FakeDomainAgent("RCSB PDB results")
    team = GuiaTeam(
        orchestrator=FakeOrchestrator(routing),
        agents={"structural_biologist": structural_biologist},
    )

    result = asyncio.run(
        team.ask("Inspect PDB 1M17", session_id="session-structure")
    )

    assert result.text == "RCSB PDB results"
    assert result.handled is True
    assert structural_biologist.calls == [
        ("Inspect experimental metadata for PDB 1M17.", "session-structure")
    ]


def test_team_routes_computational_task_with_shared_session() -> None:
    routing = RoutingDecision(
        agent="computational_biologist",
        task="Summarize functional evidence for human BRCA1.",
        reason="Gene annotation request.",
    )
    computational_biologist = FakeDomainAgent("Integrated gene evidence")
    team = GuiaTeam(
        orchestrator=FakeOrchestrator(routing),
        agents={"computational_biologist": computational_biologist},
    )

    result = asyncio.run(
        team.ask("Summarize human BRCA1", session_id="session-computational")
    )

    assert result.text == "Integrated gene evidence"
    assert result.handled is True
    assert computational_biologist.calls == [
        ("Summarize functional evidence for human BRCA1.", "session-computational")
    ]


def test_team_always_reports_new_result_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "session-paths"
    monkeypatch.setenv("GUIA_DATA_DIR", str(tmp_path))
    session = create_session(session_id)
    routing = RoutingDecision(
        agent="medicinal_chemist",
        task="Save an EGFR summary.",
        reason="Medicinal chemistry report.",
    )

    class SavingAgent:
        async def run(
            self,
            task: str,
            *,
            session_id: str | None = None,
        ) -> str:
            (session.results / "EGFR_summary.md").write_text(
                "# EGFR",
                encoding="utf-8",
            )
            return "Saved to `results/EGFR_summary.md`."

    team = GuiaTeam(
        orchestrator=FakeOrchestrator(routing),
        agents={"medicinal_chemist": SavingAgent()},
    )

    result = asyncio.run(team.ask("Save an EGFR summary", session_id=session_id))

    assert "Relative: `results/EGFR_summary.md`" in result.text
    assert f"Absolute: `{session.results / 'EGFR_summary.md'}`" in result.text


def test_team_returns_direct_orchestrator_response() -> None:
    routing = RoutingDecision(
        reason="The user asked about GUIA CLI.",
        direct_response="I coordinate specialized biomedical agents.",
    )
    medicinal_chemist = FakeDomainAgent("unused")
    team = GuiaTeam(
        orchestrator=FakeOrchestrator(routing),
        agents={"medicinal_chemist": medicinal_chemist},
    )

    result = asyncio.run(team.ask("What is your role?"))

    assert result.text == "I coordinate specialized biomedical agents."
    assert result.handled is True
    assert len(result.session_id) == 32
    assert medicinal_chemist.calls == []


def test_team_returns_clarification_without_agent_call() -> None:
    routing = RoutingDecision(
        reason="The request has no clear biomedical domain.",
        needs_clarification=True,
        clarifying_question="Are you asking about a compound or a gene?",
    )
    team = GuiaTeam(
        orchestrator=FakeOrchestrator(routing),
        agents={},
    )

    result = asyncio.run(team.ask("Analyze this"))

    assert result.text == "Are you asking about a compound or a gene?"
    assert result.handled is False


def test_team_reports_agent_not_available_in_preview() -> None:
    routing = RoutingDecision(
        agent="scientific_critic",
        task="Critique an RNA-seq interpretation.",
        reason="Scientific critique request.",
    )
    team = GuiaTeam(
        orchestrator=FakeOrchestrator(routing),
        agents={},
    )

    result = asyncio.run(team.ask("Critique an RNA-seq interpretation"))

    assert result.handled is False
    assert "scientific critic" in result.text
    assert "not available" in result.text


def test_team_requires_valid_construction() -> None:
    with pytest.raises(ValueError):
        GuiaTeam()
