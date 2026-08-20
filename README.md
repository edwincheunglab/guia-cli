# GUIA CLI

GUIA CLI is a local, command-line edition of GUIA for biomedical research
workflows. It provides an extensible framework for users to integrate their own
tools and adapt the architecture to their workflows.

## Project status

The current framework provides a routing Orchestrator and Medicinal Chemist,
Structural Biologist, Computational Biologist, and Scientific Critic agents.

The local agent roster is:

- Orchestrator
- Medicinal Chemist
- Structural Biologist
- Computational Biologist
- Scientific Critic

Code execution, autonomous package installation, and browser automation are
not built in, but users can add them to suit their local systems. Agents are provided 
access to ChEMBL, RCSB PDB, UniProt, NCBI E-utilities, and Open Targets.

## Requirements

- Python 3.11 or newer
- An API key for a supported model provider

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/edwincheunglab/guia-cli.git
cd guia-cli
```

Create a virtual environment and install GUIA CLI:

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

Set `GUIA_MODEL` using LiteLLM's `provider/model` format and add the matching
provider key. For example:

```env
GUIA_MODEL=openai/your-model-name
OPENAI_API_KEY=your-key
OPENAI_API_BASE=base-url
```

## Basic usage

Ask a medicinal chemistry or compound-retrieval question:

```bash
guia ask "Find reported EGFR inhibitors"
```

Or ask for details regarding protein structures:

```bash
guia ask "Compare representative human EGFR structures in the PDB"
```

Or critically review a scientific claim, report, or session file:

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

## Analyze a local file

Choose a session name and create its upload directory. The session does not
need to exist beforehand.

```bash
SESSION=egfr-study
mkdir -p "$HOME/.guia-cli/sessions/$SESSION/uploads"
```

Copy a file into the session:

```bash
cp /path/to/egfr-information.md "$HOME/.guia-cli/sessions/$SESSION/uploads/"
```

Ask GUIA CLI to analyze it:

```bash
guia ask "Analyze egfr-information.md from the uploads directory" --session "$SESSION"
```

By default, session files are stored under
`~/.guia-cli/sessions/SESSION_ID/`. To use another base directory, set
`GUIA_DATA_DIR` in the `.env` file at the repository root:

```env
GUIA_DATA_DIR=/path/to/guia-data
```

You can also export this variable in your shell. Supported input formats are
`.txt`, `.md`, `.csv`, `.tsv`, and `.xlsx`, with a maximum file size of 25 MiB.
Text files must use UTF-8 encoding. Users can extend these formats and limits to
suit their own local workflows.

## Relationship to GUIA

GUIA CLI is maintained as a separate codebase. It does not include the full
GUIA web interface, integrations, agent features, or advanced execution workflows.

## License

Licensed under the [Apache License 2.0](LICENSE).
