from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from guia_cli.sessions import create_session
from guia_cli.tools import files as file_tools


def test_list_and_read_text_file(tmp_path: Path) -> None:
    session = create_session("text", data_dir=tmp_path)
    upload = session.uploads / "notes.txt"
    upload.write_text("EGFR notes", encoding="utf-8")

    listed = file_tools.list_files(session)
    content = file_tools.read_file(session, "notes.txt")

    assert [item.path for item in listed] == ["notes.txt"]
    assert content["type"] == "text"
    assert content["content"] == "EGFR notes"


def test_read_csv_and_tsv_files(tmp_path: Path) -> None:
    session = create_session("tables", data_dir=tmp_path)
    (session.uploads / "genes.csv").write_text(
        "gene,score\nTP53,0.9\nEGFR,0.8\n",
        encoding="utf-8",
    )
    (session.uploads / "genes.tsv").write_text(
        "gene\tscore\nTP53\t0.9\n",
        encoding="utf-8",
    )

    csv_content = file_tools.read_file(session, "genes.csv")
    tsv_content = file_tools.read_file(session, "genes.tsv")

    assert csv_content["columns"] == ["gene", "score"]
    assert csv_content["rows"] == [["TP53", "0.9"], ["EGFR", "0.8"]]
    assert tsv_content["rows"] == [["TP53", "0.9"]]


def test_read_xlsx_file(tmp_path: Path) -> None:
    session = create_session("xlsx", data_dir=tmp_path)
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Genes"
    worksheet.append(["gene", "score"])
    worksheet.append(["TP53", 0.9])
    workbook.save(session.uploads / "genes.xlsx")
    workbook.close()

    content = file_tools.read_file(
        session,
        "genes.xlsx",
        sheet_name="Genes",
    )

    assert content["sheet"] == "Genes"
    assert content["columns"] == ["gene", "score"]
    assert content["rows"] == [["TP53", 0.9]]


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "nested/../../outside.txt",
        ".hidden.txt",
        "/etc/passwd",
    ],
)
def test_unsafe_paths_are_rejected(tmp_path: Path, path: str) -> None:
    session = create_session("unsafe", data_dir=tmp_path)

    with pytest.raises(file_tools.UnsafePathError):
        file_tools.read_file(session, path)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    session = create_session("symlink", data_dir=tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (session.uploads / "escape.txt").symlink_to(outside)

    with pytest.raises(file_tools.UnsafePathError):
        file_tools.read_file(session, "escape.txt")

    assert file_tools.list_files(session) == ()


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    session = create_session("unsupported", data_dir=tmp_path)
    (session.uploads / "program.py").write_text(
        "print('unsafe')",
        encoding="utf-8",
    )

    with pytest.raises(file_tools.UnsupportedFileTypeError):
        file_tools.read_file(session, "program.py")


def test_input_size_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = create_session("large-input", data_dir=tmp_path)
    (session.uploads / "large.txt").write_text("12345", encoding="utf-8")
    monkeypatch.setattr(file_tools, "MAX_INPUT_BYTES", 4)

    with pytest.raises(file_tools.FileSizeLimitError):
        file_tools.read_file(session, "large.txt")


def test_write_markdown_does_not_overwrite_by_default(tmp_path: Path) -> None:
    session = create_session("markdown", data_dir=tmp_path)

    result = file_tools.write_markdown(session, "report.md", "# Result")

    assert result == session.results / "report.md"
    assert result.read_text(encoding="utf-8") == "# Result"
    with pytest.raises(file_tools.FileToolError):
        file_tools.write_markdown(session, "report.md", "# Replacement")


def test_write_csv_accepts_sequences_and_mappings(tmp_path: Path) -> None:
    session = create_session("csv-output", data_dir=tmp_path)

    result = file_tools.write_csv(
        session,
        "genes.csv",
        ["gene", "score"],
        [["TP53", 0.9], {"gene": "EGFR", "score": 0.8}],
    )
    content = file_tools.read_file(
        session,
        "genes.csv",
        location="results",
    )

    assert result == session.results / "genes.csv"
    assert content["rows"] == [["TP53", "0.9"], ["EGFR", "0.8"]]


def test_output_size_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = create_session("large-output", data_dir=tmp_path)
    monkeypatch.setattr(file_tools, "MAX_OUTPUT_BYTES", 4)

    with pytest.raises(file_tools.FileSizeLimitError):
        file_tools.write_markdown(session, "report.md", "12345")
