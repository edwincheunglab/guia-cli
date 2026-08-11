from __future__ import annotations

import asyncio

import httpx
import pytest

from guia_cli.a2a.client import (
    DEFAULT_A2A_TIMEOUT_SECONDS,
    A2ADomainAgentProxy,
)
from guia_cli.a2a.cluster import LocalA2ACluster
from guia_cli.a2a.errors import A2ADispatchError, A2AServiceError
from guia_cli.agents.orchestrator import AgentName
from guia_cli.runtime import ContextLengthExceededError

AGENT_NAMES: tuple[AgentName, ...] = (
    "medicinal_chemist",
    "structural_biologist",
    "computational_biologist",
)


class RecordingAgent:
    def __init__(
        self,
        *,
        response: str = "agent response",
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
    ) -> str:
        self.calls.append((task, session_id))
        if self.error is not None:
            raise self.error
        return self.response


def _agents(
    selected: RecordingAgent | None = None,
) -> dict[AgentName, RecordingAgent]:
    return {
        name: selected if name == "medicinal_chemist" and selected else RecordingAgent()
        for name in AGENT_NAMES
    }


def test_cluster_eagerly_starts_all_agents_and_propagates_context() -> None:
    selected = RecordingAgent(response="A2A result")

    async def exercise() -> None:
        cluster = LocalA2ACluster(_agents(selected))
        await cluster.start()
        try:
            assert set(cluster.urls) == set(AGENT_NAMES)
            assert len(set(cluster.urls.values())) == 3
            assert all(
                url.startswith("http://127.0.0.1:")
                for url in cluster.urls.values()
            )
            result = await cluster.proxies["medicinal_chemist"].run(
                "Find EGFR inhibitors.",
                "session-a2a",
            )
            assert result == "A2A result"
            assert selected.calls == [
                ("Find EGFR inhibitors.", "session-a2a")
            ]
        finally:
            await cluster.stop()

        with pytest.raises(A2AServiceError):
            _ = cluster.proxies

    asyncio.run(exercise())


def test_agent_card_and_rpc_routes_require_run_scoped_token() -> None:
    async def exercise() -> None:
        async with LocalA2ACluster(_agents()) as cluster:
            url = cluster.urls["structural_biologist"]
            async with httpx.AsyncClient() as client:
                card_response = await client.get(
                    f"{url}.well-known/agent-card.json"
                )
                rpc_response = await client.post(url, json={})
            assert card_response.status_code == 401
            assert rpc_response.status_code == 401

            invalid_proxy = A2ADomainAgentProxy(
                base_url=url,
                bearer_token="not-the-run-token",
            )
            with pytest.raises(A2ADispatchError):
                await invalid_proxy.run("Inspect PDB 1M17.", "session-auth")

    asyncio.run(exercise())


def test_context_length_errors_cross_a2a_as_actionable_errors() -> None:
    selected = RecordingAgent(
        error=ContextLengthExceededError(
            "Narrow the query or request a smaller result limit."
        )
    )

    async def exercise() -> None:
        async with LocalA2ACluster(_agents(selected)) as cluster:
            with pytest.raises(
                ContextLengthExceededError,
                match="Narrow the query",
            ):
                await cluster.proxies["medicinal_chemist"].run(
                    "Return every record.",
                    "session-context",
                )

    asyncio.run(exercise())


def test_unexpected_agent_errors_do_not_leak_internal_details() -> None:
    selected = RecordingAgent(error=RuntimeError("secret-internal-detail"))

    async def exercise() -> None:
        async with LocalA2ACluster(_agents(selected)) as cluster:
            with pytest.raises(A2ADispatchError) as exc_info:
                await cluster.proxies["medicinal_chemist"].run(
                    "Run task.",
                    "session-error",
                )
            assert "could not complete" in str(exc_info.value)
            assert "secret-internal-detail" not in str(exc_info.value)

    asyncio.run(exercise())


def test_cluster_rejects_an_incomplete_agent_roster() -> None:
    with pytest.raises(ValueError, match="missing"):
        LocalA2ACluster(
            {"medicinal_chemist": RecordingAgent()}  # type: ignore[arg-type]
        )


def test_a2a_request_timeout_is_one_hour() -> None:
    proxy = A2ADomainAgentProxy(
        base_url="http://127.0.0.1:12345/",
        bearer_token="test-token",
    )

    assert DEFAULT_A2A_TIMEOUT_SECONDS == 3_600
    assert proxy.timeout_seconds == 3_600
