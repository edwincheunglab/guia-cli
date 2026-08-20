"""Restricted file tools for GUIA CLI sessions."""

from __future__ import annotations

import csv
import importlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from guia_cli.sessions import SessionPaths

MAX_INPUT_BYTES = 25 * 1024 * 1024
MAX_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_TABLE_ROWS = 50_000
MAX_TABLE_COLUMNS = 256

READABLE_EXTENSIONS = frozenset({".txt", ".md", ".csv", ".tsv", ".xlsx"})
WRITABLE_EXTENSIONS = frozenset({".md", ".csv"})
Location = Literal["uploads", "results"]


class FileToolError(Exception):
    """Base exception for restricted file operations."""


class UnsafePathError(FileToolError, ValueError):
    """Raised when a path could escape its approved session directory."""


class UnsupportedFileTypeError(FileToolError, ValueError):
    """Raised when a file extension is not approved."""


class FileSizeLimitError(FileToolError, ValueError):
    """Raised when a file exceeds an enforced size limit."""


class TableLimitError(FileToolError, ValueError):
    """Raised when tabular content exceeds row or column limits."""


class OptionalDependencyError(FileToolError, RuntimeError):
    """Raised when an optional reader dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Safe metadata for a file in a session workspace."""

    path: str
    location: Location
    size_bytes: int
    extension: str


def _base_directory(session: SessionPaths, location: Location) -> Path:
    if location == "uploads":
        return session.uploads
    if location == "results":
        return session.results
    raise ValueError(f"Unsupported session location: {location}")


def _safe_path(
    session: SessionPaths,
    relative_path: str | Path,
    *,
    location: Location,
) -> Path:
    raw_path = Path(relative_path)
    if raw_path.is_absolute() or not raw_path.parts:
        raise UnsafePathError("File paths must be relative to the session.")
    if any(
        part in {"", ".", ".."} or part.startswith(".") for part in raw_path.parts
    ):
        raise UnsafePathError(
            "Hidden paths and path traversal components are not allowed."
        )

    base = _base_directory(session, location).resolve()
    candidate = base.joinpath(raw_path)

    current = base
    for part in raw_path.parts:
        current = current / part
        if current.is_symlink():
            raise UnsafePathError("Symbolic links are not allowed.")

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise UnsafePathError(
            "File path must remain inside the active session."
        ) from exc
    return resolved


def _validate_extension(path: Path, allowed: frozenset[str]) -> str:
    extension = path.suffix.lower()
    if extension not in allowed:
        options = ", ".join(sorted(allowed))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or '(none)'}'; allowed: {options}"
        )
    return extension


def _validate_input_file(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Session file does not exist: {path.name}")
    size = path.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise FileSizeLimitError(
            f"Input file exceeds the {MAX_INPUT_BYTES}-byte limit."
        )
    return size


def _iter_regular_files(base: Path) -> Iterable[Path]:
    if not base.is_dir():
        return

    pending = [base]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file():
                yield entry


def list_files(
    session: SessionPaths,
    *,
    location: Location = "uploads",
) -> tuple[FileInfo, ...]:
    """List approved files without exposing paths outside the session."""

    base = _base_directory(session, location).resolve()
    files: list[FileInfo] = []
    for path in _iter_regular_files(base):
        extension = path.suffix.lower()
        if extension not in READABLE_EXTENSIONS:
            continue
        size = path.stat().st_size
        if size > MAX_INPUT_BYTES:
            continue
        files.append(
            FileInfo(
                path=path.relative_to(base).as_posix(),
                location=location,
                size_bytes=size,
                extension=extension,
            )
        )
    return tuple(sorted(files, key=lambda item: item.path))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FileToolError("Text files must use UTF-8 encoding.") from exc


def _validate_table_shape(rows: Sequence[Sequence[object]]) -> None:
    if len(rows) > MAX_TABLE_ROWS + 1:
        raise TableLimitError(
            f"Table exceeds the {MAX_TABLE_ROWS}-row limit."
        )
    if any(len(row) > MAX_TABLE_COLUMNS for row in rows):
        raise TableLimitError(
            f"Table exceeds the {MAX_TABLE_COLUMNS}-column limit."
        )


def _read_delimited(path: Path, delimiter: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
    except UnicodeDecodeError as exc:
        raise FileToolError("CSV and TSV files must use UTF-8 encoding.") from exc

    _validate_table_shape(rows)
    if not rows:
        return {"type": "table", "columns": [], "rows": []}
    return {"type": "table", "columns": rows[0], "rows": rows[1:]}


def _serializable_cell(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _read_xlsx(
    path: Path,
    *,
    sheet_name: str | None,
) -> dict[str, object]:
    try:
        openpyxl = importlib.import_module("openpyxl")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Reading XLSX files requires the 'openpyxl' package."
        ) from exc

    workbook = openpyxl.load_workbook(
        path,
        read_only=True,
        data_only=True,
    )
    try:
        if sheet_name is not None:
            if sheet_name not in workbook.sheetnames:
                raise FileToolError(f"Worksheet does not exist: {sheet_name}")
            worksheet = workbook[sheet_name]
        else:
            worksheet = workbook.active

        rows: list[list[object]] = []
        for index, row in enumerate(worksheet.iter_rows(values_only=True)):
            if index > MAX_TABLE_ROWS:
                raise TableLimitError(
                    f"Table exceeds the {MAX_TABLE_ROWS}-row limit."
                )
            values = [_serializable_cell(value) for value in row]
            if len(values) > MAX_TABLE_COLUMNS:
                raise TableLimitError(
                    f"Table exceeds the {MAX_TABLE_COLUMNS}-column limit."
                )
            rows.append(values)
    finally:
        workbook.close()

    if not rows:
        return {
            "type": "table",
            "sheet": worksheet.title,
            "columns": [],
            "rows": [],
        }
    return {
        "type": "table",
        "sheet": worksheet.title,
        "columns": rows[0],
        "rows": rows[1:],
    }


def read_file(
    session: SessionPaths,
    relative_path: str | Path,
    *,
    location: Location = "uploads",
    sheet_name: str | None = None,
) -> dict[str, object]:
    """Read one approved session file into a JSON-compatible structure."""

    path = _safe_path(session, relative_path, location=location)
    extension = _validate_extension(path, READABLE_EXTENSIONS)
    size = _validate_input_file(path)

    if extension in {".txt", ".md"}:
        return {
            "type": "text",
            "path": Path(relative_path).as_posix(),
            "size_bytes": size,
            "content": _read_text(path),
        }
    if extension == ".csv":
        content = _read_delimited(path, ",")
    elif extension == ".tsv":
        content = _read_delimited(path, "\t")
    else:
        content = _read_xlsx(path, sheet_name=sheet_name)

    return {
        "path": Path(relative_path).as_posix(),
        "size_bytes": size,
        **content,
    }


def _write_result(
    session: SessionPaths,
    relative_path: str | Path,
    content: str,
    *,
    expected_extension: str,
    overwrite: bool,
) -> Path:
    path = _safe_path(session, relative_path, location="results")
    extension = _validate_extension(path, WRITABLE_EXTENSIONS)
    if extension != expected_extension:
        raise UnsupportedFileTypeError(
            f"Result filename must end with '{expected_extension}'."
        )

    encoded = content.encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise FileSizeLimitError(
            f"Result exceeds the {MAX_OUTPUT_BYTES}-byte limit."
        )

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _safe_path(session, relative_path, location="results")
    mode = "w" if overwrite else "x"
    try:
        with path.open(mode, encoding="utf-8", newline="") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise FileToolError(
            f"Result already exists: {Path(relative_path).as_posix()}"
        ) from exc
    return path


def write_markdown(
    session: SessionPaths,
    relative_path: str | Path,
    content: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a UTF-8 Markdown file inside the session results directory."""

    return _write_result(
        session,
        relative_path,
        content,
        expected_extension=".md",
        overwrite=overwrite,
    )


def write_csv(
    session: SessionPaths,
    relative_path: str | Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[object] | Mapping[str, object]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a CSV result with enforced row, column, and size limits."""

    if not columns or len(columns) > MAX_TABLE_COLUMNS:
        raise TableLimitError(
            f"CSV requires 1-{MAX_TABLE_COLUMNS} columns."
        )

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(columns)
    row_count = 0
    for row in rows:
        row_count += 1
        if row_count > MAX_TABLE_ROWS:
            raise TableLimitError(
                f"CSV exceeds the {MAX_TABLE_ROWS}-row limit."
            )
        if isinstance(row, Mapping):
            values = [row.get(column, "") for column in columns]
        else:
            values = list(row)
            if len(values) != len(columns):
                raise TableLimitError(
                    "Every CSV row must match the number of columns."
                )
        writer.writerow([_serializable_cell(value) for value in values])

    return _write_result(
        session,
        relative_path,
        output.getvalue(),
        expected_extension=".csv",
        overwrite=overwrite,
    )


__all__ = [
    "FileInfo",
    "FileSizeLimitError",
    "FileToolError",
    "MAX_INPUT_BYTES",
    "MAX_OUTPUT_BYTES",
    "MAX_TABLE_COLUMNS",
    "MAX_TABLE_ROWS",
    "OptionalDependencyError",
    "READABLE_EXTENSIONS",
    "TableLimitError",
    "UnsafePathError",
    "UnsupportedFileTypeError",
    "WRITABLE_EXTENSIONS",
    "list_files",
    "read_file",
    "write_csv",
    "write_markdown",
]
