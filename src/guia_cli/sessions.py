"""Session workspace management for GUIA CLI."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

DEFAULT_DATA_DIR = Path.home() / ".guia-cli"
DATA_DIR_ENV = "GUIA_DATA_DIR"
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class SessionError(Exception):
    """Base exception for session-related failures."""


class InvalidSessionIdError(SessionError, ValueError):
    """Raised when a session identifier is unsafe or malformed."""


class SessionNotFoundError(SessionError, FileNotFoundError):
    """Raised when a requested session does not exist."""


@dataclass(frozen=True, slots=True)
class SessionPaths:
    """Filesystem paths belonging to one isolated GUIA CLI session."""

    session_id: str
    root: Path
    uploads: Path
    results: Path
    logs: Path


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the configured GUIA CLI data directory as an absolute path."""

    configured = data_dir or os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR
    return Path(configured).expanduser().resolve()


def validate_session_id(session_id: str) -> str:
    """Validate and return a session identifier safe for use as a directory."""

    if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(
        session_id
    ):
        raise InvalidSessionIdError(
            "Session IDs must be 1-64 characters and contain only letters, "
            "numbers, underscores, or hyphens."
        )
    return session_id


def _session_paths(
    session_id: str,
    data_dir: str | Path | None = None,
) -> SessionPaths:
    safe_id = validate_session_id(session_id)
    sessions_root = (get_data_dir(data_dir) / "sessions").resolve()
    root = (sessions_root / safe_id).resolve()

    try:
        root.relative_to(sessions_root)
    except ValueError as exc:
        raise InvalidSessionIdError(
            "Session path must remain inside the GUIA CLI sessions directory."
        ) from exc

    return SessionPaths(
        session_id=safe_id,
        root=root,
        uploads=root / "uploads",
        results=root / "results",
        logs=root / "logs",
    )


def create_session(
    session_id: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> SessionPaths:
    """Create an isolated session workspace and return all of its paths."""

    selected_id = uuid4().hex if session_id is None else session_id
    paths = _session_paths(selected_id, data_dir)
    for directory in (paths.root, paths.uploads, paths.results, paths.logs):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return paths


def open_session(
    session_id: str,
    *,
    data_dir: str | Path | None = None,
) -> SessionPaths:
    """Open an existing session without creating missing directories."""

    paths = _session_paths(session_id, data_dir)
    if not paths.root.is_dir():
        raise SessionNotFoundError(f"Session does not exist: {session_id}")
    return paths


def list_sessions(
    *,
    data_dir: str | Path | None = None,
) -> tuple[SessionPaths, ...]:
    """Return existing valid sessions ordered by identifier."""

    sessions_root = get_data_dir(data_dir) / "sessions"
    if not sessions_root.is_dir():
        return ()

    sessions: list[SessionPaths] = []
    for entry in sessions_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            sessions.append(open_session(entry.name, data_dir=data_dir))
        except (InvalidSessionIdError, SessionNotFoundError):
            continue
    return tuple(sorted(sessions, key=lambda session: session.session_id))


__all__ = [
    "DATA_DIR_ENV",
    "DEFAULT_DATA_DIR",
    "InvalidSessionIdError",
    "SessionError",
    "SessionNotFoundError",
    "SessionPaths",
    "create_session",
    "get_data_dir",
    "list_sessions",
    "open_session",
    "validate_session_id",
]
