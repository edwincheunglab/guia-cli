# Security Policy

## Development status

GUIA CLI is pre-release research software. It has not yet undergone a formal
security audit and should not be treated as a secure environment for untrusted
code or sensitive data.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. After the GitHub
repository is published, use GitHub's private vulnerability reporting feature
for this repository. Include:

- The affected version or commit
- Steps to reproduce the issue
- The expected and observed behavior
- The potential impact
- Any suggested mitigation

Do not include real API keys, patient information, confidential datasets, or
other sensitive material in a report.

## Operational guidance

- Keep model-provider credentials in `.env` or the operating system's secret
  store; never commit them.
- Review the privacy and data-retention terms of the configured model provider.
- Do not process protected health information or confidential research data
  without an approved environment and provider agreement.
- Run untrusted workflows inside a dedicated container or virtual machine.
- Treat all agent-generated scientific conclusions as unverified.
- Keep GUIA CLI and its dependencies updated.

## Scope

Security reports should concern code distributed by this repository. Issues in
third-party model providers, public databases, or external dependencies should
also be reported to their respective maintainers.
