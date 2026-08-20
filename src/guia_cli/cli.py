"""Command-line entry point for GUIA CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Sequence

from guia_cli.agents.team import GuiaTeam, TeamResponse
from guia_cli.runtime import (
    ContextLengthExceededError,
    ModelSettings,
    RuntimeConfigurationError,
    create_model,
)

def _package_version() -> str:
    try:
        return version("guia-cli")
    except PackageNotFoundError:
        return "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guia",
        description="Run GUIA biomedical research agents locally.",
        epilog=(
            "ask options:\n"
            "  --session SESSION_ID  Reuse a previous GUIA CLI session.\n"
            "  --show-route          Show which in-house agent was selected.\n"
            "  --show-a2a            Show local A2A service lifecycle details.\n\n"
            "Run 'guia ask --help' for full ask command usage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    subparsers = parser.add_subparsers(dest="command")
    ask_parser = subparsers.add_parser(
        "ask",
        help="Route and execute one natural-language biomedical request.",
    )
    ask_parser.add_argument("prompt", help="The request to send to GUIA CLI.")
    ask_parser.add_argument(
        "--session",
        help="Reuse a previous GUIA CLI session identifier.",
    )
    ask_parser.add_argument(
        "--show-route",
        action="store_true",
        help="Show which in-house agent was selected.",
    )
    ask_parser.add_argument(
        "--show-a2a",
        action="store_true",
        help="Show local A2A service startup, endpoints, and cleanup.",
    )
    return parser


def _build_team() -> GuiaTeam:
    settings = ModelSettings.from_environment()
    return GuiaTeam(create_model(settings))


def _run_ask(args: argparse.Namespace) -> int:
    team = _build_team()
    return asyncio.run(_run_ask_async(team, args))


async def _run_ask_async(
    team: GuiaTeam,
    args: argparse.Namespace,
) -> int:
    start = getattr(team, "start", None)
    stop = getattr(team, "stop", None)
    if not callable(start):
        result = await team.ask(args.prompt, session_id=args.session)
        return _print_ask_result(result, args)

    show_a2a = bool(getattr(args, "show_a2a", False))
    await start()
    try:
        if show_a2a:
            endpoints = getattr(team, "a2a_urls", {})
            print("A2A services started:", file=sys.stderr, flush=True)
            for agent_name, url in endpoints.items():
                print(f"  {agent_name}: {url}", file=sys.stderr, flush=True)
        result = await team.ask(args.prompt, session_id=args.session)
        return _print_ask_result(result, args)
    finally:
        if callable(stop):
            await stop()
        if show_a2a:
            print("A2A services stopped.", file=sys.stderr, flush=True)


def _print_ask_result(
    result: TeamResponse,
    args: argparse.Namespace,
) -> int:
    if args.show_route:
        selected = result.routing.agent or "orchestrator"
        print(
            f"Route: {selected} — {result.routing.reason}",
            file=sys.stderr,
        )
    print(result.text, flush=True)
    print(f"Session: {result.session_id}", file=sys.stderr)
    return 0 if result.handled else 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "ask":
            return _run_ask(args)
    except RuntimeConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except ContextLengthExceededError as exc:
        print(f"Request too large: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Request cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"Request failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 1
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
