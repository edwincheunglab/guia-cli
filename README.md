# GUIA CLI

GUIA CLI is a local, command-line edition of GUIA for biomedical research
workflows. It is being developed as an independent project with a deliberately
smaller tool surface than the full GUIA platform.

## Project status

GUIA CLI is in early development. The current preview provides a routing
orchestrator and restricted Medicinal Chemist, Structural Biologist,
Computational Biologist, and Scientific Critic agents.

The local agent roster is:

- Orchestrator
- Medicinal Chemist
- Structural Biologist
- Computational Biologist
- Scientific Critic

Agents will be restricted to approved public biomedical APIs and files inside
the active local session. GUIA CLI will not initially provide arbitrary code
execution, shell access, package installation, browser automation, or the full
commercial GUIA toolset.

## Requirements

- Python 3.11 or newer
- An API key for a supported model provider

## Development installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Verify the command:

```bash
guia --version
guia --help
```

Copy the environment template and configure the provider you intend to use:

```bash
cp .env.example .env
```

Never commit `.env` or real API keys.

Set `GUIA_MODEL` using LiteLLM's `provider/model` format and add the matching
provider key. For example:

```env
GUIA_MODEL=openai/your-model-name
OPENAI_API_KEY=your-key
```

## Current usage

Ask a medicinal chemistry or compound-retrieval question:

```bash
guia ask "Find reported EGFR inhibitors"
```

Or ask about experimentally determined protein structures:

```bash
guia ask "Compare representative human EGFR structures in the PDB"
```

Or retrieve and interpret gene-level evidence:

```bash
guia ask "Summarize functional and disease evidence for human BRCA1"
```

Or critically review a supplied claim, report, or session file:

```bash
guia ask "Critique whether the evidence in results/EGFR_summary.md supports its conclusions" --session SESSION_ID
```

Display the routing decision:

```bash
guia ask "Find reported EGFR inhibitors" --show-route
```

Display the temporary localhost A2A endpoints and cleanup status:

```bash
guia ask "Compare representative human EGFR structures" --show-a2a
```

GUIA CLI prints a session identifier that can be reused to access the same
local workspace files:

```bash
guia ask "Review the files saved in this session" --session SESSION_ID
```

The current preview can query approved public APIs and work with approved small
files in its session directory.

For each `guia ask` invocation, all currently implemented domain agents start
eagerly as in-process A2A JSON-RPC services bound to random
`127.0.0.1` ports. The orchestrator dispatches domain tasks through A2A using a
run-scoped bearer token and the GUIA session ID as the A2A context ID. The
services and token are cleaned up automatically when the command finishes.
Domain-agent A2A requests have a one-hour timeout for long biomedical queries.

API results are compacted before they enter model context. GUIA CLI preserves
scientifically useful identifiers and pagination metadata, limits repeated
records and oversized fields, and reports truncation metadata to the agent.
Bounded-query rules are enforced for ChEMBL, RCSB PDB, UniProt, NCBI
E-utilities, and Open Targets.

Uploaded tables and text are also returned to agents in bounded chunks with
continuation offsets, preventing a large local file from filling model context.

If a provider still rejects an oversized context, the CLI returns a concise
message suggesting narrower scientific filters or a larger-context model.

## Security and privacy warning

GUIA CLI is research software and is not a medical device. Its output may be
incorrect, incomplete, or fabricated and must be independently verified by a
qualified researcher.

Prompts and selected data may be sent to the model provider configured by the
user. Do not provide protected health information, confidential research data,
credentials, or other sensitive material unless the provider and environment
have been approved for that use.

Future agent tools will be restricted, but tool restrictions are not a security
sandbox. Run untrusted research workflows in an isolated environment. See
[SECURITY.md](SECURITY.md) for reporting guidance.

## Relationship to GUIA

GUIA CLI is maintained as a separate codebase. It does not include the full
GUIA web interface, commercial integrations, or advanced execution workflows.

## License

Licensed under the [Apache License 2.0](LICENSE).
