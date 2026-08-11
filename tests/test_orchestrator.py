from __future__ import annotations

import asyncio
import json

import pytest
from google.adk.models.lite_llm import LiteLlm
from pydantic import ValidationError

from guia_cli.agents.orchestrator import (
    AGENT_ROSTER,
    LiteOrchestrator,
    RoutingDecision,
    build_orchestrator,
)


class FakeRoutingRuntime:
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


class SequencedRoutingRuntime(FakeRoutingRuntime):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("")
        self.responses = iter(responses)

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        user_id: str = "local-user",
    ) -> str:
        self.calls.append((prompt, session_id))
        return next(self.responses)


def test_roster_contains_only_lite_in_house_agents() -> None:
    assert set(AGENT_ROSTER) == {
        "medicinal_chemist",
        "structural_biologist",
        "computational_biologist",
        "scientific_critic",
    }


def test_build_orchestrator_has_no_tools_or_subagents() -> None:
    agent = build_orchestrator(LiteLlm(model="openai/test-model"))

    assert agent.name == "guia_lite_orchestrator"
    assert agent.tools == []
    assert agent.sub_agents == []
    assert agent.output_schema is RoutingDecision


def test_routing_decision_requires_agent_and_task() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(
            agent=None,
            task=None,
            reason="No route.",
        )


def test_clarification_requires_question_and_no_agent() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(
            agent="medicinal_chemist",
            reason="Ambiguous.",
            needs_clarification=True,
            clarifying_question="Which domain?",
        )

    with pytest.raises(ValidationError):
        RoutingDecision(
            agent=None,
            reason="Ambiguous.",
            needs_clarification=True,
        )


def test_direct_response_cannot_also_delegate() -> None:
    with pytest.raises(ValidationError):
        RoutingDecision(
            agent="medicinal_chemist",
            task="Find compounds.",
            reason="Mixed response.",
            direct_response="I will handle this directly.",
        )


def test_lite_orchestrator_parses_direct_response() -> None:
    payload = {
        "agent": None,
        "task": None,
        "reason": "The user asked about the orchestrator's role.",
        "direct_response": "I route tasks to specialized biomedical agents.",
    }
    orchestrator = LiteOrchestrator(
        runtime=FakeRoutingRuntime(json.dumps(payload))
    )

    decision = asyncio.run(orchestrator.route("Tell me about this system"))

    assert decision.agent is None
    assert decision.direct_response == (
        "I route tasks to specialized biomedical agents."
    )


def test_lite_orchestrator_parses_valid_route() -> None:
    payload = {
        "agent": "medicinal_chemist",
        "task": "Find reported EGFR inhibitors.",
        "reason": "The request concerns compounds and bioactivity.",
        "needs_clarification": False,
        "clarifying_question": None,
    }
    runtime = FakeRoutingRuntime(json.dumps(payload))
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(
        orchestrator.route(
            "Please handle this biomedical request",
            session_id="session-1",
        )
    )

    assert decision.agent == "medicinal_chemist"
    assert decision.task == "Find reported EGFR inhibitors."
    assert runtime.calls == [
        ("Please handle this biomedical request", "session-1")
    ]


def test_lite_orchestrator_accepts_fenced_json() -> None:
    response = """```json
{"agent":"structural_biologist","task":"Inspect 4HHB.","reason":"PDB structure."}
```"""
    runtime = FakeRoutingRuntime(response)
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(orchestrator.route("Please inspect this item"))

    assert decision.agent == "structural_biologist"


def test_lite_orchestrator_falls_back_on_invalid_agent() -> None:
    response = json.dumps(
        {
            "agent": "clinical_agent",
            "task": "Find a clinical trial.",
            "reason": "Clinical request.",
        }
    )
    runtime = FakeRoutingRuntime(response)
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(orchestrator.route("Find a clinical trial"))

    assert decision.needs_clarification is True
    assert decision.agent is None
    assert len(runtime.calls) == 2


def test_lite_orchestrator_rejects_empty_request_without_model_call() -> None:
    runtime = FakeRoutingRuntime("{}")
    orchestrator = LiteOrchestrator(runtime=runtime)

    with pytest.raises(ValueError, match="cannot be empty"):
        asyncio.run(orchestrator.route("   "))

    assert runtime.calls == []


def test_meta_question_is_answered_by_model() -> None:
    runtime = FakeRoutingRuntime(
        json.dumps(
            {
                "agent": None,
                "task": None,
                "reason": "The user asked about the orchestrator's role.",
                "direct_response": "I coordinate specialized biomedical agents.",
            }
        )
    )
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(orchestrator.route("What is your role?"))

    assert decision.direct_response == (
        "I coordinate specialized biomedical agents."
    )
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    ("query", "expected_agent"),
    [
        ("Find reported EGFR inhibitors", "medicinal_chemist"),
        ("Inspect the protein structure in PDB 4HHB", "structural_biologist"),
        ("Perform pathway enrichment for this gene list", "computational_biologist"),
        ("Critique the methodology and limitations", "scientific_critic"),
    ],
)
def test_clear_domain_request_is_routed_by_model(
    query: str,
    expected_agent: str,
) -> None:
    runtime = FakeRoutingRuntime(
        json.dumps(
            {
                "agent": expected_agent,
                "task": query,
                "reason": "Model-selected domain.",
            }
        )
    )
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(orchestrator.route(query))

    assert decision.agent == expected_agent
    assert decision.task == query
    assert len(runtime.calls) == 1


def test_plain_text_model_response_becomes_clarification() -> None:
    runtime = FakeRoutingRuntime("I am not returning structured JSON.")
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(orchestrator.route("Analyze this biomedical item"))

    assert decision.needs_clarification is True
    assert decision.clarifying_question is not None
    assert len(runtime.calls) == 2


def test_plain_text_response_is_repaired_by_model() -> None:
    repaired = json.dumps(
        {
            "agent": None,
            "task": None,
            "reason": "The user asked about the orchestrator's role.",
            "direct_response": "I coordinate GUIA CLI's biomedical agents.",
        }
    )
    runtime = SequencedRoutingRuntime(
        [
            "I coordinate GUIA CLI's biomedical agents.",
            repaired,
        ]
    )
    orchestrator = LiteOrchestrator(runtime=runtime)

    decision = asyncio.run(orchestrator.route("What is your role?"))

    assert decision.direct_response == (
        "I coordinate GUIA CLI's biomedical agents."
    )
    assert len(runtime.calls) == 2
    assert runtime.calls[0][1] == runtime.calls[1][1]


def test_json_is_extracted_from_reasoning_text() -> None:
    response = """<think>I should route this carefully.</think>
Here is the routing decision:
{"agent":"medicinal_chemist","task":"Assess this item.","reason":"Drug request."}
Additional text."""
    orchestrator = LiteOrchestrator(
        runtime=FakeRoutingRuntime(response)
    )

    decision = asyncio.run(orchestrator.route("Assess this item"))

    assert decision.agent == "medicinal_chemist"


def test_lite_orchestrator_requires_one_runtime_source() -> None:
    with pytest.raises(ValueError):
        LiteOrchestrator()

    with pytest.raises(ValueError):
        LiteOrchestrator(
            model=LiteLlm(model="openai/test-model"),
            runtime=FakeRoutingRuntime("{}"),
        )
