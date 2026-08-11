from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from guia_cli.sessions import create_session
from guia_cli.tools.agent_files import (
    list_session_files,
    read_session_file,
    write_csv_result,
    write_markdown_result,
)


def _context(session_id: str, data_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state={
            "guia_session_id": session_id,
            "guia_data_dir": str(data_dir),
        }
    )


def test_agent_file_wrappers_use_active_session(tmp_path: Path) -> None:
    session = create_session("agent-files", data_dir=tmp_path)
    (session.uploads / "compounds.csv").write_text(
        "name,chembl_id\nAspirin,CHEMBL25\n",
        encoding="utf-8",
    )
    context = _context(session.session_id, tmp_path)

    listed = list_session_files(tool_context=context)
    content = read_session_file(
        "compounds.csv",
        tool_context=context,
    )

    assert listed["files"] == [
        {
            "path": "compounds.csv",
            "size_bytes": (session.uploads / "compounds.csv").stat().st_size,
            "extension": ".csv",
        }
    ]
    assert content["columns"] == ["name", "chembl_id"]
    assert content["rows"] == [["Aspirin", "CHEMBL25"]]


def test_agent_can_write_only_supported_results(tmp_path: Path) -> None:
    session = create_session("agent-results", data_dir=tmp_path)
    context = _context(session.session_id, tmp_path)

    markdown = write_markdown_result(
        "summary.md",
        "# Summary",
        tool_context=context,
    )
    csv_result = write_csv_result(
        "compounds.csv",
        ["name", "source"],
        [["Aspirin", "ChEMBL"]],
        tool_context=context,
    )

    assert markdown["path"] == "results/summary.md"
    assert markdown["absolute_path"] == str(session.results / "summary.md")
    assert csv_result["path"] == "results/compounds.csv"
    assert csv_result["absolute_path"] == str(session.results / "compounds.csv")
    assert csv_result["row_count"] == 1


def test_agent_file_wrappers_require_session_context() -> None:
    with pytest.raises(RuntimeError, match="context is required"):
        list_session_files()

    incomplete = SimpleNamespace(state={"guia_session_id": "missing-data-dir"})
    with pytest.raises(RuntimeError, match="context is incomplete"):
        list_session_files(tool_context=incomplete)
