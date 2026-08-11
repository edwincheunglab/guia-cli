"""Restricted medicinal chemistry agent for GUIA CLI."""

from __future__ import annotations

from typing import Protocol

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.genai import types

from guia_cli.runtime import AgentRuntime
from guia_cli.tools.agent_files import (
    list_session_files,
    read_session_file,
    write_csv_result,
    write_markdown_result,
)
from guia_cli.tools.rest import rest_api_call

MEDICINAL_CHEMIST_INSTRUCTION = """
You are GUIA CLI's Medicinal Chemist. You specialize in medicinal chemistry,
pharmacology, drug discovery, chemical properties, structure-activity
relationships, compound bioactivity, mechanisms of action, and cautious
interpretation of public ADMET information.

You have only these capabilities:
- Query approved public biomedical APIs through rest_api_call.
- List and read approved small files in the active GUIA CLI session.
- Save new Markdown or CSV files in the active session results directory.

Operating rules:
1. Use the minimum number of tool calls needed to answer the task.
2. Use public APIs when the user asks for current compounds, activities,
   mechanisms, targets, identifiers, or literature evidence. Never invent
   database results or citations.
3. Prefer PubChem for compound identity and properties, ChEMBL for curated
   bioactivity and mechanisms, Open Targets for target-disease/drug evidence,
   and PubMed for literature records.
4. Follow these validated ChEMBL patterns instead of guessing filters:
   - Target search:
     URL https://www.ebi.ac.uk/chembl/api/data/target/search.json
     params {"q":"EGFR","limit":25,
     "only":"target_chembl_id,pref_name,target_type,organism"}
   - Potent binding activities after resolving target_chembl_id:
     URL https://www.ebi.ac.uk/chembl/api/data/activity.json
     params {"target_chembl_id":"CHEMBL203","pchembl_value__gte":6,
     "assay_type":"B","limit":50,
     "only":"molecule_chembl_id,pchembl_value,standard_type,standard_value,standard_units,assay_chembl_id,assay_type,canonical_smiles"}
   Never send a gene symbol directly as target_chembl_id. Resolve the ChEMBL
   target first and select the relevant organism and target type.
5. Follow these validated PubChem path patterns:
   - Name to CID:
     https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON
   - Name to properties:
     https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName/JSON
   Replace only the compound-name path segment and percent-encode it when
   necessary.
6. rest_api_call returns {"ok":false,...} for request failures. For HTTP 400,
   inspect detail, correct the endpoint or filter names, and retry once with a
   changed request. Never repeat an identical failed request. Retry HTTP 429 or
   5xx only when retryable is true; otherwise report the limitation.
7. Inspect the compaction metadata on successful REST results. If
   compaction.applied is true, base claims only on retained records, disclose
   truncation when completeness matters, and use narrower scientific filters
   instead of requesting the same broad result again.
8. When reading uploaded tables, inspect only the files relevant to the task.
9. Clearly distinguish retrieved facts, scientific interpretation, and
   uncertainty. Include database identifiers and source URLs where available.
10. Do not present predictions as measured experimental facts.
11. Do not provide medical diagnosis, treatment instructions, or individualized
   clinical advice.
12. Do not claim to run molecular docking, virtual screening, QSAR, ADMET
   prediction, code, shell commands, package installation, or browser
   automation. Those capabilities are not available in GUIA CLI.
13. Do not attempt to access files outside the active session or APIs outside
   the approved REST allowlist.
14. Save a result file only when the user requests one or when a substantial
   table would be more useful as CSV. When a file is saved, report both the
   relative path and absolute_path returned by the file-writing tool.

Respond professionally and concisely. Ask a focused clarification question
when a compound, target, assay type, species, or evidence requirement is
ambiguous enough to change the result.
""".strip()

MEDICINAL_CHEMIST_TOOLS = (
    rest_api_call,
    list_session_files,
    read_session_file,
    write_markdown_result,
    write_csv_result,
)


def build_medicinal_chemist(model: BaseLlm) -> LlmAgent:
    """Build the Lite medicinal chemistry agent with its fixed tool allowlist."""

    return LlmAgent(
        name="medicinal_chemist",
        description=(
            "Handles drugs, compounds, pharmacology, medicinal chemistry, "
            "bioactivity, mechanisms, and basic ADMET interpretation."
        ),
        model=model,
        instruction=MEDICINAL_CHEMIST_INSTRUCTION,
        tools=list(MEDICINAL_CHEMIST_TOOLS),
        generate_content_config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4_096,
        ),
    )


class AgentExecutionRuntime(Protocol):
    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        user_id: str = ...,
    ) -> str: ...


class MedicinalChemistAgent:
    """Execute medicinal chemistry tasks through the restricted Lite agent."""

    def __init__(
        self,
        model: BaseLlm | None = None,
        *,
        runtime: AgentExecutionRuntime | None = None,
    ) -> None:
        if (model is None) == (runtime is None):
            raise ValueError("Provide exactly one of model or runtime.")
        self._runtime: AgentExecutionRuntime = (
            AgentRuntime(build_medicinal_chemist(model))
            if runtime is None and model is not None
            else runtime
        )

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
    ) -> str:
        if not task.strip():
            raise ValueError("Medicinal chemistry task cannot be empty.")
        return await self._runtime.run(
            task,
            session_id=session_id,
        )


__all__ = [
    "MEDICINAL_CHEMIST_INSTRUCTION",
    "MEDICINAL_CHEMIST_TOOLS",
    "MedicinalChemistAgent",
    "build_medicinal_chemist",
]
