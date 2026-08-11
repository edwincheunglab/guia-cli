"""Restricted public REST API access for GUIA CLI agents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx

APPROVED_API_HOSTS = MappingProxyType(
    {
        "eutils.ncbi.nlm.nih.gov": "PubMed / NCBI E-utilities",
        "pubchem.ncbi.nlm.nih.gov": "PubChem",
        "www.ebi.ac.uk": "ChEMBL",
        "api.platform.opentargets.org": "Open Targets",
        "data.rcsb.org": "RCSB PDB Data API",
        "search.rcsb.org": "RCSB PDB Search API",
        "rest.uniprot.org": "UniProt",
    }
)

ALLOWED_METHODS = frozenset({"GET", "POST"})
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_AGENT_RESPONSE_CHARS = 40_000
DEFAULT_TIMEOUT_SECONDS = 20.0
_MAX_AGENT_LIST_ITEMS = 50
_MAX_NESTING_DEPTH = 8
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api-key",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)
_BULKY_RESPONSE_KEYS = frozenset(
    {
        "assay_description",
        "description",
        "full_molformula",
        "molfile",
        "sequence",
        "sdf",
        "svg",
    }
)
_PRIORITY_RESPONSE_KEYS = (
    "page_meta",
    "count",
    "total",
    "total_count",
    "next",
    "previous",
    "target_chembl_id",
    "molecule_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
    "cid",
    "CID",
    "pref_name",
    "organism",
    "target_type",
    "pchembl_value",
    "standard_type",
    "standard_value",
    "standard_units",
    "assay_type",
    "canonical_smiles",
    "CanonicalSMILES",
    "MolecularFormula",
    "MolecularWeight",
    "IUPACName",
)
_DEFAULT_HEADERS = {
    "Accept": (
        "application/json, application/xml, text/xml, text/plain, "
        "text/tab-separated-values, text/csv"
    ),
    "User-Agent": "GUIA-CLI/0.1 (public biomedical research client)",
}


class RestToolError(Exception):
    """Base exception for restricted REST operations."""


class UnsafeUrlError(RestToolError, ValueError):
    """Raised when a URL is outside the public API allowlist."""


class UnsupportedMethodError(RestToolError, ValueError):
    """Raised when an HTTP method is not allowed."""


class SensitiveValueError(RestToolError, ValueError):
    """Raised when request data appears to contain credentials."""


class RequestSizeLimitError(RestToolError, ValueError):
    """Raised when request parameters or JSON are too large."""


class ResponseSizeLimitError(RestToolError, ValueError):
    """Raised when a response exceeds the configured byte limit."""


class UnsafeQueryError(RestToolError, ValueError):
    """Raised when an approved endpoint is queried too broadly."""


class UnsupportedContentTypeError(RestToolError, ValueError):
    """Raised when an API returns an unapproved content type."""


class RedirectNotAllowedError(RestToolError):
    """Raised when an approved API attempts to redirect a request."""


class PublicApiRequestError(RestToolError):
    """Raised when an approved API request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.detail = detail


@dataclass(frozen=True, slots=True)
class RestResult:
    """JSON-compatible result returned from an approved public API."""

    status_code: int
    url: str
    content_type: str
    data: object

    def as_dict(
        self,
        *,
        compact: bool = False,
        max_chars: int = MAX_AGENT_RESPONSE_CHARS,
    ) -> dict[str, object]:
        data = self.data
        compaction: dict[str, object] | None = None
        if compact:
            data, compaction = compact_api_data(data, max_chars=max_chars)
        return {
            "ok": True,
            "status_code": self.status_code,
            "url": self.url,
            "content_type": self.content_type,
            "data": data,
            **({"compaction": compaction} if compaction is not None else {}),
        }


def _serialized_chars(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    )


def _ordered_items(value: Mapping[object, object]) -> list[tuple[object, object]]:
    priority = {
        key: index for index, key in enumerate(_PRIORITY_RESPONSE_KEYS)
    }
    return sorted(
        value.items(),
        key=lambda item: priority.get(str(item[0]), len(priority)),
    )


def _compact_value(
    value: object,
    *,
    max_list_items: int,
    max_string_chars: int,
    max_dict_items: int,
    depth: int,
    stats: dict[str, int],
) -> object:
    if depth >= _MAX_NESTING_DEPTH and isinstance(value, (Mapping, list, tuple)):
        stats["omitted_fields"] += 1
        return "[nested content omitted]"

    if isinstance(value, Mapping):
        compacted: dict[str, object] = {}
        items = _ordered_items(value)
        if len(items) > max_dict_items:
            stats["omitted_fields"] += len(items) - max_dict_items
            items = items[:max_dict_items]
        for key, nested_value in items:
            normalized_key = _normalized_key(key)
            if (
                normalized_key in _BULKY_RESPONSE_KEYS
                and _serialized_chars(nested_value) > max_string_chars
            ):
                stats["omitted_fields"] += 1
                continue
            compacted[str(key)] = _compact_value(
                nested_value,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
                max_dict_items=max_dict_items,
                depth=depth + 1,
                stats=stats,
            )
        return compacted

    if isinstance(value, (list, tuple)):
        selected = value[:max_list_items]
        if len(value) > max_list_items:
            stats["omitted_items"] += len(value) - max_list_items
        return [
            _compact_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
                max_dict_items=max_dict_items,
                depth=depth + 1,
                stats=stats,
            )
            for item in selected
        ]

    if isinstance(value, str) and len(value) > max_string_chars:
        omitted = len(value) - max_string_chars
        stats["truncated_strings"] += 1
        return f"{value[:max_string_chars]}… [{omitted} chars omitted]"
    return value


def compact_api_data(
    data: object,
    *,
    max_chars: int = MAX_AGENT_RESPONSE_CHARS,
) -> tuple[object, dict[str, object]]:
    """Compact API data while preserving identifiers, metadata, and leading rows."""

    if max_chars < 1_000:
        raise ValueError("Agent response limit must be at least 1,000 characters.")

    original_size = _serialized_chars(data)
    if original_size <= max_chars:
        return data, {
            "applied": False,
            "truncated": False,
            "original_data_chars": original_size,
            "returned_data_chars": original_size,
            "omitted_items": 0,
            "omitted_fields": 0,
            "truncated_strings": 0,
        }

    profiles = (
        (_MAX_AGENT_LIST_ITEMS, 1_500, 64),
        (25, 1_000, 48),
        (10, 500, 32),
        (5, 250, 24),
        (2, 160, 16),
    )
    compacted: object = None
    final_stats: dict[str, int] = {}
    for max_items, max_string, max_fields in profiles:
        stats = {
            "omitted_items": 0,
            "omitted_fields": 0,
            "truncated_strings": 0,
        }
        candidate = _compact_value(
            data,
            max_list_items=max_items,
            max_string_chars=max_string,
            max_dict_items=max_fields,
            depth=0,
            stats=stats,
        )
        compacted = candidate
        final_stats = stats
        if _serialized_chars(candidate) <= max_chars:
            break

    returned_size = _serialized_chars(compacted)
    if returned_size > max_chars:
        preview = json.dumps(
            compacted,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        compacted = {
            "notice": "Response was too large; only a compact preview is shown.",
            "top_level_keys": (
                [str(key) for key in list(data)[:25]]
                if isinstance(data, Mapping)
                else []
            ),
            "preview": preview[: max_chars - 500],
        }
        returned_size = _serialized_chars(compacted)
        final_stats["omitted_fields"] += 1

    return compacted, {
        "applied": True,
        "truncated": True,
        "original_data_chars": original_size,
        "returned_data_chars": returned_size,
        **final_stats,
        "guidance": (
            "Use narrower filters or a smaller result limit if omitted records "
            "could change the conclusion."
        ),
    }


def validate_public_api_url(url: str) -> str:
    """Validate an HTTPS URL against the exact public API host allowlist."""

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError("Malformed public API URL.") from exc

    if parsed.scheme.lower() != "https":
        raise UnsafeUrlError("Public API requests must use HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("Credentials are not allowed in public API URLs.")
    if parsed.fragment:
        raise UnsafeUrlError("URL fragments are not allowed.")
    if port not in (None, 443):
        raise UnsafeUrlError("Only the standard HTTPS port is allowed.")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname not in APPROVED_API_HOSTS:
        approved = ", ".join(sorted(APPROVED_API_HOSTS))
        raise UnsafeUrlError(
            f"API host '{hostname or '(missing)'}' is not approved. "
            f"Approved hosts: {approved}"
        )
    return url


def _normalized_key(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _reject_sensitive_values(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _normalized_key(key) in _SENSITIVE_KEYS:
                raise SensitiveValueError(
                    f"Credential-like field is not allowed: {key}"
                )
            _reject_sensitive_values(nested_value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_values(item)


def _request_payload_size(
    params: Mapping[str, object] | None,
    json_body: Mapping[str, object] | None,
) -> int:
    try:
        encoded = json.dumps(
            {"params": params, "json": json_body},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RestToolError(
            "Request parameters and JSON body must be JSON-compatible."
        ) from exc
    return len(encoded)


def _effective_query(
    url: str,
    params: Mapping[str, object] | None,
) -> dict[str, object]:
    query: dict[str, object] = dict(
        parse_qsl(urlsplit(url).query, keep_blank_values=True)
    )
    if params is not None:
        query.update({str(key): value for key, value in params.items()})
    return query


def _validate_bounded_query(
    url: str,
    params: Mapping[str, object] | None,
) -> None:
    parsed = urlsplit(url)
    if (
        (parsed.hostname or "").lower().rstrip(".") != "www.ebi.ac.uk"
        or not parsed.path.rstrip("/").endswith("/chembl/api/data/activity.json")
    ):
        return

    query = _effective_query(url, params)
    bounded_filters = {
        "assay_chembl_id",
        "document_chembl_id",
        "molecule_chembl_id",
        "target_chembl_id",
    }
    if not bounded_filters.intersection(query):
        raise UnsafeQueryError(
            "ChEMBL activity queries require a target, molecule, assay, or "
            "document ChEMBL identifier. Resolve an identifier first."
        )
    if not str(query.get("only", "")).strip():
        raise UnsafeQueryError(
            "ChEMBL activity queries require an 'only' field list so large "
            "unused records do not enter the model context."
        )
    try:
        limit = int(str(query.get("limit", "")))
    except ValueError as exc:
        raise UnsafeQueryError(
            "ChEMBL activity queries require an integer limit from 1 to 50."
        ) from exc
    if not 1 <= limit <= 50:
        raise UnsafeQueryError(
            "ChEMBL activity queries require a result limit from 1 to 50."
        )


def _is_json_content_type(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _is_text_content_type(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return (
        media_type.startswith("text/")
        or media_type in {"application/xml", "application/xhtml+xml"}
        or media_type.endswith("+xml")
    )


def _parse_response_data(
    content: bytes,
    content_type: str,
    encoding: str | None,
) -> object:
    if not content:
        return None

    text = content.decode(encoding or "utf-8", errors="replace")
    if _is_json_content_type(content_type):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RestToolError(
                "Approved API returned malformed JSON."
            ) from exc
    if _is_text_content_type(content_type):
        return text
    raise UnsupportedContentTypeError(
        f"Unsupported API response content type: {content_type or '(missing)'}"
    )


def _error_detail(
    content: bytes,
    content_type: str,
    encoding: str | None,
) -> str | None:
    if not content:
        return None
    text = content.decode(encoding or "utf-8", errors="replace").strip()
    if not text:
        return None
    if _is_json_content_type(content_type):
        try:
            text = json.dumps(
                json.loads(text),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except json.JSONDecodeError:
            pass
    elif not _is_text_content_type(content_type):
        return None
    return text[:1_000]


class RestrictedRestClient:
    """HTTP client constrained to approved unauthenticated biomedical APIs."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not 0 < timeout_seconds <= 30:
            raise ValueError("Timeout must be greater than 0 and at most 30 seconds.")
        if max_response_bytes <= 0:
            raise ValueError("Response limit must be greater than 0 bytes.")
        self._timeout = httpx.Timeout(timeout_seconds, connect=min(5, timeout_seconds))
        self._max_response_bytes = max_response_bytes
        self._transport = transport

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Mapping[str, object] | None = None,
        json_body: Mapping[str, object] | None = None,
    ) -> RestResult:
        """Call one approved endpoint and return a size-bounded result."""

        safe_url = validate_public_api_url(url)
        safe_method = method.strip().upper()
        if safe_method not in ALLOWED_METHODS:
            raise UnsupportedMethodError(
                f"Method '{safe_method}' is not allowed; use GET or POST."
            )
        if safe_method == "GET" and json_body is not None:
            raise RestToolError("GET requests cannot include a JSON body.")

        _reject_sensitive_values(params)
        _reject_sensitive_values(json_body)
        _validate_bounded_query(safe_url, params)
        if _request_payload_size(params, json_body) > MAX_REQUEST_BYTES:
            raise RequestSizeLimitError(
                f"Request exceeds the {MAX_REQUEST_BYTES}-byte limit."
            )

        try:
            with httpx.Client(
                headers=_DEFAULT_HEADERS,
                timeout=self._timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                with client.stream(
                    safe_method,
                    safe_url,
                    params=params,
                    json=json_body,
                ) as response:
                    if response.is_redirect:
                        raise RedirectNotAllowedError(
                            "API redirects are not followed. Use the final approved "
                            "HTTPS endpoint directly."
                        )
                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            declared_size = int(content_length)
                        except ValueError:
                            declared_size = 0
                        if declared_size > self._max_response_bytes:
                            raise ResponseSizeLimitError(
                                "API response exceeds the configured byte limit."
                            )

                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > self._max_response_bytes:
                            raise ResponseSizeLimitError(
                                "API response exceeds the configured byte limit."
                            )

                    content_type = response.headers.get("content-type", "")
                    if response.is_error:
                        raise PublicApiRequestError(
                            f"Public API returned HTTP {response.status_code}.",
                            status_code=response.status_code,
                            url=str(response.url),
                            detail=_error_detail(
                                bytes(body),
                                content_type,
                                response.encoding,
                            ),
                        )
                    data = _parse_response_data(
                        bytes(body),
                        content_type,
                        response.encoding,
                    )
                    return RestResult(
                        status_code=response.status_code,
                        url=str(response.url),
                        content_type=content_type,
                        data=data,
                    )
        except RestToolError:
            raise
        except httpx.TimeoutException as exc:
            raise PublicApiRequestError("Public API request timed out.") from exc
        except httpx.RequestError as exc:
            raise PublicApiRequestError(
                "Public API request could not be completed."
            ) from exc


def rest_api_call(
    url: str,
    method: str = "GET",
    params: Mapping[str, object] | None = None,
    json_body: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Agent-friendly wrapper that returns errors for correction and retry."""

    try:
        return RestrictedRestClient().request(
            url,
            method=method,
            params=params,
            json_body=json_body,
        ).as_dict(compact=True)
    except RestToolError as exc:
        status_code = getattr(exc, "status_code", None)
        retryable = (
            isinstance(exc, PublicApiRequestError) and status_code is None
        ) or status_code == 429 or (
            isinstance(status_code, int) and 500 <= status_code <= 599
        )
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "status_code": status_code,
            "url": getattr(exc, "url", None),
            "detail": getattr(exc, "detail", None),
            "retryable": retryable,
        }


__all__ = [
    "ALLOWED_METHODS",
    "APPROVED_API_HOSTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_AGENT_RESPONSE_CHARS",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "PublicApiRequestError",
    "RedirectNotAllowedError",
    "RequestSizeLimitError",
    "ResponseSizeLimitError",
    "RestResult",
    "RestToolError",
    "RestrictedRestClient",
    "SensitiveValueError",
    "UnsafeQueryError",
    "UnsafeUrlError",
    "UnsupportedContentTypeError",
    "UnsupportedMethodError",
    "compact_api_data",
    "rest_api_call",
    "validate_public_api_url",
]
