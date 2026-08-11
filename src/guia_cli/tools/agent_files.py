"""Agent-facing wrappers around restricted GUIA CLI session files."""

from __future__ import annotations

from typing import Literal

from google.adk.tools.tool_context import ToolContext

from guia_cli.sessions import SessionPaths, open_session
from guia_cli.tools.files import (
    list_files,
    read_file,
    write_csv,
    write_markdown,
)

Location = Literal["uploads", "results"]
_SESSION_ID_STATE_KEY = "guia_session_id"
_DATA_DIR_STATE_KEY = "guia_data_dir"


def _session_from_context(tool_context: ToolContext) -> SessionPaths:
    if tool_context is None:
        raise RuntimeError("GUIA session context is required.")

    session_id = str(tool_context.state.get(_SESSION_ID_STATE_KEY, "")).strip()
    data_dir = str(tool_context.state.get(_DATA_DIR_STATE_KEY, "")).strip()
    if not session_id or not data_dir:
        raise RuntimeError("GUIA session context is incomplete.")
    return open_session(session_id, data_dir=data_dir)


def list_session_files(
    location: Location = "uploads",
    tool_context: ToolContext = None,
) -> dict[str, object]:
    """List approved files in the active session uploads or results."""

    session = _session_from_context(tool_context)
    return {
        "location": location,
        "files": [
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "extension": item.extension,
            }
            for item in list_files(session, location=location)
        ],
    }


def read_session_file(
    path: str,
    location: Location = "uploads",
    sheet_name: str | None = None,
    tool_context: ToolContext = None,
) -> dict[str, object]:
    """Read one approved file from the active GUIA CLI session."""

    session = _session_from_context(tool_context)
    return read_file(
        session,
        path,
        location=location,
        sheet_name=sheet_name,
    )


def write_markdown_result(
    filename: str,
    content: str,
    tool_context: ToolContext = None,
) -> dict[str, object]:
    """Write a new Markdown result without overwriting an existing file."""

    session = _session_from_context(tool_context)
    path = write_markdown(session, filename, content)
    return {
        "path": path.relative_to(session.root).as_posix(),
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
    }


def write_csv_result(
    filename: str,
    columns: list[str],
    rows: list[list[object]],
    tool_context: ToolContext = None,
) -> dict[str, object]:
    """Write a new CSV result without overwriting an existing file."""

    session = _session_from_context(tool_context)
    path = write_csv(session, filename, columns, rows)
    return {
        "path": path.relative_to(session.root).as_posix(),
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(rows),
    }


__all__ = [
    "list_session_files",
    "read_session_file",
    "write_csv_result",
    "write_markdown_result",
]
