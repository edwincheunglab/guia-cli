"""Agent Cards for GUIA CLI's local A2A services."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)

from guia_cli.agents.orchestrator import AgentName


@dataclass(frozen=True, slots=True)
class AgentCardDefinition:
    """Static public metadata for one local A2A agent."""

    name: str
    description: str
    skill_id: str
    skill_name: str
    tags: tuple[str, ...]
    examples: tuple[str, ...]


AGENT_CARD_DEFINITIONS: MappingProxyType[
    AgentName, AgentCardDefinition
] = MappingProxyType(
    {
        "medicinal_chemist": AgentCardDefinition(
            name="GUIA CLI Medicinal Chemist",
            description=(
                "Restricted local agent for medicinal chemistry, compounds, "
                "bioactivity, pharmacology, mechanisms, and basic ADMET evidence."
            ),
            skill_id="medicinal_chemistry_research",
            skill_name="Medicinal chemistry research",
            tags=("medicinal chemistry", "bioactivity", "pharmacology"),
            examples=("Find reported EGFR inhibitors.",),
        ),
        "structural_biologist": AgentCardDefinition(
            name="GUIA CLI Structural Biologist",
            description=(
                "Restricted local agent for experimentally determined protein "
                "structures, PDB records, assemblies, ligands, and quality."
            ),
            skill_id="structural_biology_research",
            skill_name="Structural biology research",
            tags=("structural biology", "PDB", "protein structure"),
            examples=("Summarize the experimental quality of PDB 1M17.",),
        ),
        "computational_biologist": AgentCardDefinition(
            name="GUIA CLI Computational Biologist",
            description=(
                "Restricted local agent for gene annotation, target-disease "
                "evidence, genomics, transcriptomics, and omics interpretation."
            ),
            skill_id="computational_biology_research",
            skill_name="Computational biology research",
            tags=("computational biology", "genomics", "gene annotation"),
            examples=("Summarize evidence for human BRCA1.",),
        ),
        "scientific_critic": AgentCardDefinition(
            name="GUIA CLI Scientific Critic",
            description=(
                "Restricted local agent for critical review of biomedical "
                "claims, methods, evidence trails, reports, and agent outputs."
            ),
            skill_id="scientific_critique",
            skill_name="Scientific critique",
            tags=("scientific critique", "methodology", "evidence review"),
            examples=(
                "Critique whether this report's evidence supports its conclusion.",
            ),
        ),
    }
)


def build_agent_card(agent_name: AgentName, base_url: str) -> AgentCard:
    """Build one text-only Agent Card for a loopback JSON-RPC service."""

    definition = AGENT_CARD_DEFINITIONS[agent_name]
    return AgentCard(
        name=definition.name,
        description=definition.description,
        supported_interfaces=[
            AgentInterface(
                url=base_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id=definition.skill_id,
                name=definition.skill_name,
                description=definition.description,
                tags=list(definition.tags),
                examples=list(definition.examples),
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
    )


__all__ = [
    "AGENT_CARD_DEFINITIONS",
    "AgentCardDefinition",
    "build_agent_card",
]
