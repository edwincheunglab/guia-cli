"""Reduced routing orchestrator for GUIA CLI."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Protocol
from uuid import uuid4

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.genai import types
from pydantic import BaseModel, Field, model_validator

from guia_cli.runtime import AgentRuntime

AgentName = Literal[
    "medicinal_chemist",
    "structural_biologist",
    "computational_biologist",
    "scientific_critic",
]

AGENT_ROSTER: Mapping[str, str] = MappingProxyType(
    {
        "medicinal_chemist": (
            "Drugs, compounds, pharmacology, medicinal chemistry, SAR, "
            "bioactivity, and basic ADMET interpretation."
        ),
        "structural_biologist": (
            "Protein structures, PDB records, domains, binding sites, "
            "ligands, and molecular interactions."
        ),
        "computational_biologist": (
            "Genes, pathways, genomics, transcriptomics, omics datasets, "
            "gene lists, and computational biology interpretation."
        ),
        "scientific_critic": (
            "Critical review of an existing scientific claim, analysis, "
            "method, evidence trail, or another agent's result."
        ),
    }
)

ORCHESTRATOR_INSTRUCTION = """
You are the routing orchestrator for GUIA CLI, a limited local biomedical
research assistant. Understand the user's request, answer simple questions
about GUIA CLI itself, or select the single best in-house agent.

Available agents:
- medicinal_chemist: drugs, compounds, pharmacology, medicinal chemistry,
  SAR, bioactivity, and basic ADMET interpretation.
- structural_biologist: protein structures, PDB records, domains, binding
  sites, ligands, and molecular interactions.
- computational_biologist: genes, pathways, genomics, transcriptomics, omics
  datasets, gene lists, and computational biology interpretation.
- scientific_critic: critical review of an existing claim, method, evidence
  trail, analysis, or another agent's result.

Routing rules:
1. Select exactly one primary agent when the request is sufficiently clear.
2. Rewrite the user's request into a concise delegated task without changing
   its scientific intent.
3. Use scientific_critic only when review, critique, verification, or
   methodological assessment is the primary goal.
4. Ask one concise clarification question when the primary domain cannot be
   determined safely.
5. For greetings and simple questions about your role or available agents,
   return a concise direct_response without selecting an agent.
6. Do not perform biomedical research, call tools, claim that work was
   executed, or invent findings in a direct response.
7. Do not route to agents outside the four-agent roster.
""".strip()

class RoutingDecision(BaseModel):
    """Validated routing output produced by the Lite orchestrator."""

    agent: AgentName | None = Field(
        default=None,
        description="The selected in-house agent, or null when clarification is needed.",
    )
    task: str | None = Field(
        default=None,
        description="A concise task preserving the user's scientific intent.",
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="A brief explanation of why this route was selected.",
    )
    needs_clarification: bool = False
    clarifying_question: str | None = Field(default=None, max_length=500)
    direct_response: str | None = Field(default=None, max_length=1_000)

    @model_validator(mode="after")
    def validate_complete_decision(self) -> RoutingDecision:
        if self.direct_response:
            if (
                self.agent is not None
                or self.task is not None
                or self.needs_clarification
            ):
                raise ValueError(
                    "Direct responses cannot select an agent, delegate a task, "
                    "or request clarification."
                )
            return self
        if self.needs_clarification:
            if not self.clarifying_question or not self.clarifying_question.strip():
                raise ValueError(
                    "A clarifying question is required when clarification is needed."
                )
            if self.agent is not None:
                raise ValueError(
                    "Agent must be null when clarification is needed."
                )
        elif self.agent is None or not self.task or not self.task.strip():
            raise ValueError(
                "A selected agent and delegated task are required."
            )
        return self


def build_orchestrator(model: BaseLlm) -> LlmAgent:
    """Build the tool-free Lite orchestrator used for routing."""

    return LlmAgent(
        name="guia_lite_orchestrator",
        description="Routes biomedical requests to one GUIA CLI domain agent.",
        model=model,
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=[],
        output_schema=RoutingDecision,
        generate_content_config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1_024,
        ),
    )


class RoutingRuntime(Protocol):
    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        user_id: str = ...,
    ) -> str: ...


def _routing_json(text: str) -> str:
    stripped = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return json.dumps(value)
    return stripped


def _parse_routing_decision(text: str) -> RoutingDecision:
    return RoutingDecision.model_validate_json(_routing_json(text))


def _repair_routing_prompt(
    request: str,
    previous_response: str,
) -> str:
    return f"""
Your previous routing response could not be parsed. Repair its FORMAT without
changing your routing decision.

Original user request:
{json.dumps(request, ensure_ascii=False)}

Previous response:
{json.dumps(previous_response[:4_000], ensure_ascii=False)}

Return exactly one JSON object and no other text, using one of these forms:

Delegation:
{{"agent":"medicinal_chemist|structural_biologist|computational_biologist|scientific_critic","task":"delegated task","reason":"brief reason","needs_clarification":false,"clarifying_question":null,"direct_response":null}}

Clarification:
{{"agent":null,"task":null,"reason":"brief reason","needs_clarification":true,"clarifying_question":"one question","direct_response":null}}

Direct conversational response:
{{"agent":null,"task":null,"reason":"brief reason","needs_clarification":false,"clarifying_question":null,"direct_response":"response"}}
""".strip()


class LiteOrchestrator:
    """Route user requests through a validated, model-backed ADK agent."""

    def __init__(
        self,
        model: BaseLlm | None = None,
        *,
        runtime: RoutingRuntime | None = None,
    ) -> None:
        if (model is None) == (runtime is None):
            raise ValueError("Provide exactly one of model or runtime.")
        self._runtime: RoutingRuntime = (
            AgentRuntime(build_orchestrator(model))
            if runtime is None and model is not None
            else runtime
        )

    async def route(
        self,
        request: str,
        *,
        session_id: str | None = None,
    ) -> RoutingDecision:
        if not request.strip():
            raise ValueError("Routing request cannot be empty.")

        routing_session_id = session_id or uuid4().hex
        response = await self._runtime.run(
            request,
            session_id=routing_session_id,
        )
        try:
            return _parse_routing_decision(response)
        except (ValueError, json.JSONDecodeError):
            repaired_response = await self._runtime.run(
                _repair_routing_prompt(request, response),
                session_id=routing_session_id,
            )
        try:
            return _parse_routing_decision(repaired_response)
        except (ValueError, json.JSONDecodeError) as exc:
            return RoutingDecision(
                reason=(
                    "The selected model could not produce a compatible routing "
                    f"decision after a format-repair attempt ({type(exc).__name__})."
                ),
                needs_clarification=True,
                clarifying_question=(
                    "Should this request be handled as medicinal chemistry, "
                    "structural biology, computational biology, or scientific "
                    "critique?"
                ),
            )


__all__ = [
    "AGENT_ROSTER",
    "AgentName",
    "LiteOrchestrator",
    "ORCHESTRATOR_INSTRUCTION",
    "RoutingDecision",
    "build_orchestrator",
]
