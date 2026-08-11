# GUIA CLI

GUIA CLI is a local, command-line edition of GUIA for biomedical research
workflows. It is being developed as an independent project with a deliberately
smaller tool surface than the full GUIA platform.

## Project status

GUIA CLI is in early development. The current preview provides a routing
orchestrator and a restricted Medicinal Chemist. Structural Biology,
Computational Biology, and Scientific Critic execution are not yet available.

The planned local agent roster is:

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

Display the routing decision:

```bash
guia ask "Find reported EGFR inhibitors" --show-route
```

GUIA CLI prints a session identifier that can be reused to access the same
local workspace files:

```bash
guia ask "Review the files saved in this session" --session SESSION_ID
```

The current preview can query approved public APIs and work with approved small
files in its session directory. Requests routed to agents that are not yet
implemented return an explicit availability message.

API results are compacted before they enter model context. GUIA CLI preserves
scientifically useful identifiers and pagination metadata, limits repeated
records and oversized fields, and reports truncation metadata to the agent.
Broad ChEMBL activity requests are rejected unless they include a resolved
ChEMBL identifier, selected fields, and a result limit of 50 or fewer.

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
