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
MAX_AGENT_TABLE_ROWS = 100
MAX_AGENT_TEXT_CHARS = 40_000
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
    row_offset: int = 0,
    row_limit: int = MAX_AGENT_TABLE_ROWS,
    character_offset: int = 0,
    character_limit: int = MAX_AGENT_TEXT_CHARS,
    tool_context: ToolContext = None,
) -> dict[str, object]:
    """Read one bounded chunk of an approved active-session file."""

    session = _session_from_context(tool_context)
    if row_offset < 0 or not 1 <= row_limit <= MAX_AGENT_TABLE_ROWS:
        raise ValueError(
            f"Rows require a nonnegative offset and a 1-{MAX_AGENT_TABLE_ROWS} limit."
        )
    if character_offset < 0 or not 1 <= character_limit <= MAX_AGENT_TEXT_CHARS:
        raise ValueError(
            "Text requires a nonnegative offset and a "
            f"1-{MAX_AGENT_TEXT_CHARS} character limit."
        )

    result = read_file(
        session,
        path,
        location=location,
        sheet_name=sheet_name,
    )
    if result.get("type") == "table":
        rows = result.get("rows", [])
        if isinstance(rows, list):
            selected_rows = rows[row_offset : row_offset + row_limit]
            next_offset = row_offset + len(selected_rows)
            result["rows"] = selected_rows
            result["total_rows"] = len(rows)
            result["returned_rows"] = len(selected_rows)
            result["row_offset"] = row_offset
            result["has_more"] = next_offset < len(rows)
            result["next_row_offset"] = (
                next_offset if next_offset < len(rows) else None
            )
            result["truncated"] = (
                row_offset > 0 or next_offset < len(rows)
            )
    elif result.get("type") == "text":
        content = result.get("content", "")
        if isinstance(content, str):
            selected_content = content[
                character_offset : character_offset + character_limit
            ]
            next_offset = character_offset + len(selected_content)
            result["content"] = selected_content
            result["total_characters"] = len(content)
            result["returned_characters"] = len(selected_content)
            result["character_offset"] = character_offset
            result["has_more"] = next_offset < len(content)
            result["next_character_offset"] = (
                next_offset if next_offset < len(content) else None
            )
            result["truncated"] = (
                character_offset > 0 or next_offset < len(content)
            )
    return result


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
    "MAX_AGENT_TABLE_ROWS",
    "MAX_AGENT_TEXT_CHARS",
    "list_session_files",
    "read_session_file",
    "write_csv_result",
    "write_markdown_result",
]
