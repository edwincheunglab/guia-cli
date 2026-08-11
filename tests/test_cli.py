from __future__ import annotations

from types import SimpleNamespace

import pytest

from guia_cli import cli
from guia_cli.agents.orchestrator import RoutingDecision
from guia_cli.agents.team import TeamResponse
from guia_cli.runtime import ContextLengthExceededError, RuntimeConfigurationError


class FakeTeam:
    def __init__(self, result: TeamResponse) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None]] = []

    async def ask(
        self,
        request: str,
        *,
        session_id: str | None = None,
    ) -> TeamResponse:
        self.calls.append((request, session_id))
        return self.result


def test_ask_command_prints_agent_response(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    routing = RoutingDecision(
        agent="medicinal_chemist",
        task="Find reported EGFR inhibitors.",
        reason="Compound bioactivity request.",
    )
    team = FakeTeam(
        TeamResponse(
            text="Reported inhibitors found.",
            routing=routing,
            handled=True,
            session_id="session-1",
        )
    )
    monkeypatch.setattr(cli, "_build_team", lambda: team)

    exit_code = cli.main(
        [
            "ask",
            "Find EGFR inhibitors",
            "--session",
            "session-1",
            "--show-route",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "Reported inhibitors found.\n"
    assert "Route: medicinal_chemist" in captured.err
    assert "Session: session-1" in captured.err
    assert team.calls == [("Find EGFR inhibitors", "session-1")]


def test_unavailable_agent_returns_nonzero_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    routing = RoutingDecision(
        agent="structural_biologist",
        task="Inspect PDB 4HHB.",
        reason="Structure request.",
    )
    team = FakeTeam(
        TeamResponse(
            text="The structural biologist is not available.",
            routing=routing,
            handled=False,
            session_id="session-2",
        )
    )
    monkeypatch.setattr(cli, "_build_team", lambda: team)

    exit_code = cli.main(["ask", "Inspect PDB 4HHB"])
    captured = capsys.readouterr()

    assert exit_code == 3
    assert "not available" in captured.out


def test_configuration_error_is_concise(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_build() -> SimpleNamespace:
        raise RuntimeConfigurationError("GUIA_MODEL is missing.")

    monkeypatch.setattr(cli, "_build_team", fail_to_build)

    exit_code = cli.main(["ask", "Find EGFR inhibitors"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "Configuration error: GUIA_MODEL is missing.\n"


def test_context_length_error_returns_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class OversizedTeam:
        async def ask(
            self,
            request: str,
            *,
            session_id: str | None = None,
        ) -> TeamResponse:
            raise ContextLengthExceededError(
                "Narrow the query to one target or a smaller result limit."
            )

    monkeypatch.setattr(cli, "_build_team", OversizedTeam)

    exit_code = cli.main(["ask", "Find all known inhibitors"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("Request too large:")
    assert "Narrow the query" in captured.err
    assert "ContextLengthExceededError" not in captured.err


def test_cli_without_command_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage: guia" in captured.out
    assert "ask" in captured.out
