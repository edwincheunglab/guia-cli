"""Dispatch requests from the Lite orchestrator to available domain agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from google.adk.models.base_llm import BaseLlm

from guia_cli.a2a.cluster import LocalA2ACluster
from guia_cli.agents.computational_biologist import ComputationalBiologistAgent
from guia_cli.agents.medicinal_chemist import MedicinalChemistAgent
from guia_cli.agents.orchestrator import (
    AGENT_ROSTER,
    AgentName,
    LiteOrchestrator,
    RoutingDecision,
)
from guia_cli.agents.scientific_critic import ScientificCriticAgent
from guia_cli.agents.structural_biologist import StructuralBiologistAgent
from guia_cli.sessions import SessionError, open_session


class OrchestratorProtocol(Protocol):
    async def route(
        self,
        request: str,
        *,
        session_id: str | None = None,
    ) -> RoutingDecision: ...


class DomainAgentProtocol(Protocol):
    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TeamResponse:
    """Final response and routing metadata for one GUIA CLI request."""

    text: str
    routing: RoutingDecision
    handled: bool
    session_id: str


def _result_files(session_id: str) -> dict[str, str]:
    try:
        session = open_session(session_id)
    except SessionError:
        return {}
    return {
        path.relative_to(session.root).as_posix(): str(path)
        for path in session.results.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _append_new_result_paths(
    response: str,
    *,
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> str:
    new_paths = sorted(set(after) - set(before))
    missing_paths = [
        (relative_path, after[relative_path])
        for relative_path in new_paths
        if after[relative_path] not in response
    ]
    if not missing_paths:
        return response

    lines = ["", "Saved result path(s):"]
    for relative_path, absolute_path in missing_paths:
        lines.extend(
            [
                f"- Relative: `{relative_path}`",
                f"  Absolute: `{absolute_path}`",
            ]
        )
    return f"{response.rstrip()}\n" + "\n".join(lines)


class GuiaTeam:
    """Coordinate the currently available GUIA CLI in-house agents."""

    def __init__(
        self,
        model: BaseLlm | None = None,
        *,
        orchestrator: OrchestratorProtocol | None = None,
        agents: Mapping[AgentName, DomainAgentProtocol] | None = None,
    ) -> None:
        self._a2a_cluster: LocalA2ACluster | None = None
        if model is not None and (orchestrator is not None or agents is not None):
            raise ValueError(
                "Provide a model or injected team components, not both."
            )
        if model is not None:
            self._orchestrator: OrchestratorProtocol = LiteOrchestrator(model)
            local_agents = {
                "medicinal_chemist": MedicinalChemistAgent(model),
                "structural_biologist": StructuralBiologistAgent(model),
                "computational_biologist": ComputationalBiologistAgent(model),
                "scientific_critic": ScientificCriticAgent(model),
            }
            self._a2a_cluster = LocalA2ACluster(local_agents)
            self._agents: dict[AgentName, DomainAgentProtocol] = {}
        else:
            if orchestrator is None:
                raise ValueError("An orchestrator is required.")
            self._orchestrator = orchestrator
            self._agents = dict(agents or {})

    async def start(self) -> None:
        """Eagerly start all production domain agents as local A2A services."""

        if self._a2a_cluster is None:
            return
        await self._a2a_cluster.start()
        self._agents = dict(self._a2a_cluster.proxies)

    async def stop(self) -> None:
        """Stop production A2A services and discard their client proxies."""

        if self._a2a_cluster is None:
            return
        await self._a2a_cluster.stop()
        self._agents.clear()

    @property
    def a2a_urls(self) -> Mapping[AgentName, str]:
        """Return active A2A endpoints without exposing authentication."""

        if self._a2a_cluster is None:
            return {}
        return self._a2a_cluster.urls

    async def ask(
        self,
        request: str,
        *,
        session_id: str | None = None,
    ) -> TeamResponse:
        if self._a2a_cluster is not None and not self._agents:
            raise RuntimeError(
                "GUIA Team must be started before dispatching A2A requests."
            )
        selected_session_id = session_id or uuid4().hex
        routing = await self._orchestrator.route(
            request,
            session_id=selected_session_id,
        )

        if routing.direct_response:
            return TeamResponse(
                text=routing.direct_response,
                routing=routing,
                handled=True,
                session_id=selected_session_id,
            )
        if routing.needs_clarification:
            return TeamResponse(
                text=routing.clarifying_question or "Please clarify your request.",
                routing=routing,
                handled=False,
                session_id=selected_session_id,
            )

        selected_agent = routing.agent
        if selected_agent is None or routing.task is None:
            raise RuntimeError("Orchestrator produced an incomplete route.")

        agent = self._agents.get(selected_agent)
        if agent is None:
            description = AGENT_ROSTER[selected_agent]
            return TeamResponse(
                text=(
                    f"This request belongs to the {selected_agent.replace('_', ' ')} "
                    "agent, which is not available in the current GUIA CLI preview. "
                    f"Its planned scope is: {description}"
                ),
                routing=routing,
                handled=False,
                session_id=selected_session_id,
            )

        results_before = _result_files(selected_session_id)
        response = await agent.run(
            routing.task,
            session_id=selected_session_id,
        )
        response = _append_new_result_paths(
            response,
            before=results_before,
            after=_result_files(selected_session_id),
        )
        return TeamResponse(
            text=response,
            routing=routing,
            handled=True,
            session_id=selected_session_id,
        )


__all__ = ["GuiaTeam", "TeamResponse"]
