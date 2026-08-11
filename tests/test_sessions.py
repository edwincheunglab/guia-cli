from __future__ import annotations

from pathlib import Path

import pytest

from guia_cli.sessions import (
    InvalidSessionIdError,
    SessionNotFoundError,
    create_session,
    get_data_dir,
    list_sessions,
    open_session,
    validate_session_id,
)


def test_create_session_builds_isolated_directories(tmp_path: Path) -> None:
    session = create_session("research-1", data_dir=tmp_path)

    assert session.root == tmp_path / "sessions" / "research-1"
    assert session.uploads.is_dir()
    assert session.results.is_dir()
    assert session.logs.is_dir()


def test_create_session_generates_valid_unique_ids(tmp_path: Path) -> None:
    first = create_session(data_dir=tmp_path)
    second = create_session(data_dir=tmp_path)

    assert first.session_id != second.session_id
    assert validate_session_id(first.session_id) == first.session_id
    assert validate_session_id(second.session_id) == second.session_id


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "nested/session",
        "contains spaces",
        ".hidden",
        "x" * 65,
    ],
)
def test_invalid_session_ids_are_rejected(
    tmp_path: Path,
    session_id: str,
) -> None:
    with pytest.raises(InvalidSessionIdError):
        create_session(session_id, data_dir=tmp_path)


def test_open_session_does_not_create_missing_session(tmp_path: Path) -> None:
    with pytest.raises(SessionNotFoundError):
        open_session("missing", data_dir=tmp_path)

    assert not (tmp_path / "sessions" / "missing").exists()


def test_list_sessions_returns_valid_sessions_in_order(tmp_path: Path) -> None:
    create_session("zeta", data_dir=tmp_path)
    create_session("alpha", data_dir=tmp_path)
    invalid = tmp_path / "sessions" / ".hidden"
    invalid.mkdir()

    sessions = list_sessions(data_dir=tmp_path)

    assert [session.session_id for session in sessions] == ["alpha", "zeta"]


def test_environment_can_select_data_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUIA_DATA_DIR", str(tmp_path))

    assert get_data_dir() == tmp_path.resolve()
    assert create_session("from-env").root == tmp_path / "sessions" / "from-env"
