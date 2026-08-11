from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from guia_cli.tools import rest as rest_tools


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    max_response_bytes: int = rest_tools.MAX_RESPONSE_BYTES,
) -> rest_tools.RestrictedRestClient:
    return rest_tools.RestrictedRestClient(
        transport=httpx.MockTransport(handler),
        max_response_bytes=max_response_bytes,
    )


def test_approved_json_get_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.params["term"] == "EGFR"
        assert request.url.params["db"] == "pubmed"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={"count": 1, "ids": ["123"]},
            request=request,
        )

    result = _client(handler).request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": "EGFR",
            "retmax": 10,
            "retmode": "json",
        },
    )

    assert result.status_code == 200
    assert result.data == {"count": 1, "ids": ["123"]}
    assert result.content_type.startswith("application/json")
    assert result.as_dict()["ok"] is True


def test_approved_graphql_post_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert b"target" in request.content
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"data":{"target":{"id":"ENSG1"}}}',
            request=request,
        )

    result = _client(handler).request(
        "https://api.platform.opentargets.org/api/v4/graphql",
        method="POST",
        json_body={
            "query": "query Target($id: String!) { target(ensemblId: $id) { id } }",
            "variables": {"id": "ENSG1"},
        },
    )

    assert result.data == {"data": {"target": {"id": "ENSG1"}}}


def test_json_encoded_parameter_object_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["db"] == "gene"
        assert request.url.params["term"] == "BRCA1[gene]"
        return httpx.Response(200, json={"esearchresult": {}}, request=request)

    result = _client(handler).request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params=(
            '{"db":"gene","term":"BRCA1[gene]",'
            '"retmax":10,"retmode":"json"}'
        ),  # type: ignore[arg-type]
    )

    assert result.status_code == 200


def test_json_encoded_graphql_body_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b"target" in request.content
        return httpx.Response(200, json={"data": {}}, request=request)

    result = _client(handler).request(
        "https://api.platform.opentargets.org/api/v4/graphql",
        method="POST",
        json_body=(
            '{"query":"query Target($id: String!) { '
            'target(ensemblId: $id) { id } }",'
            '"variables":{"id":"ENSG00000012048"}}'
        ),  # type: ignore[arg-type]
    )

    assert result.status_code == 200


def test_agent_wrapper_returns_malformed_mapping_error() -> None:
    result = rest_tools.rest_api_call(
        "https://rest.uniprot.org/uniprotkb/P38398.json",
        params="{not-json}",  # type: ignore[arg-type]
    )

    assert result["ok"] is False
    assert result["error_type"] == "RestToolError"
    assert "malformed JSON text" in result["error"]


def test_approved_text_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/tab-separated-values; charset=utf-8"},
            content=b"Entry\tProtein names\nP00533\tEGFR\n",
            request=request,
        )

    result = _client(handler).request(
        "https://rest.uniprot.org/uniprotkb/P00533.tsv"
    )

    assert result.data == "Entry\tProtein names\nP00533\tEGFR\n"


@pytest.mark.parametrize(
    "url",
    [
        "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        "https://example.com/api",
        "https://subdomain.rest.uniprot.org/api",
        "https://rest.uniprot.org:8443/api",
        "https://user:password@rest.uniprot.org/api",
        "https://rest.uniprot.org/api#fragment",
        "not-a-url",
    ],
)
def test_unsafe_urls_are_rejected(url: str) -> None:
    with pytest.raises(rest_tools.UnsafeUrlError):
        rest_tools.validate_public_api_url(url)


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "OPTIONS"])
def test_unsupported_methods_are_rejected(method: str) -> None:
    client = _client(lambda request: httpx.Response(200, request=request))

    with pytest.raises(rest_tools.UnsupportedMethodError):
        client.request(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json",
            method=method,
        )


def test_get_request_rejects_json_body() -> None:
    client = _client(lambda request: httpx.Response(200, request=request))

    with pytest.raises(rest_tools.RestToolError):
        client.request(
            "https://data.rcsb.org/rest/v1/core/entry/4HHB",
            json_body={"query": "not allowed"},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "secret"},
        {"token": "secret"},
        {"nested": {"authorization": "Bearer secret"}},
        {"items": [{"client_secret": "secret"}]},
    ],
)
def test_credential_like_values_are_rejected(
    payload: dict[str, object],
) -> None:
    client = _client(lambda request: httpx.Response(200, request=request))

    with pytest.raises(rest_tools.SensitiveValueError):
        client.request(
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
            params=payload,
        )


def test_request_size_limit_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rest_tools, "MAX_REQUEST_BYTES", 20)
    client = _client(lambda request: httpx.Response(200, request=request))

    with pytest.raises(rest_tools.RequestSizeLimitError):
        client.request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            method="POST",
            json_body={"query": "x" * 100},
        )


def test_redirect_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://example.com/redirect"},
            request=request,
        )

    with pytest.raises(rest_tools.RedirectNotAllowedError):
        _client(handler).request(
            "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        )


def test_declared_response_size_limit_is_enforced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-length": "100",
                "content-type": "application/json",
            },
            content=b"{}",
            request=request,
        )

    with pytest.raises(rest_tools.ResponseSizeLimitError):
        _client(handler, max_response_bytes=10).request(
            "https://data.rcsb.org/rest/v1/core/entry/4HHB"
        )


def test_actual_response_size_limit_is_enforced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"x" * 20,
            request=request,
        )

    with pytest.raises(rest_tools.ResponseSizeLimitError):
        _client(handler, max_response_bytes=10).request(
            "https://rest.uniprot.org/uniprotkb/P00533.fasta"
        )


def test_unsupported_response_content_type_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=b"binary",
            request=request,
        )

    with pytest.raises(rest_tools.UnsupportedContentTypeError):
        _client(handler).request(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/png"
        )


def test_http_errors_are_sanitized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            headers={"content-type": "text/plain"},
            content=b"internal details",
            request=request,
        )

    with pytest.raises(
        rest_tools.PublicApiRequestError,
        match="HTTP 500",
    ) as error:
        _client(handler).request(
            "https://data.rcsb.org/rest/v1/core/entry/invalid"
        )

    assert error.value.status_code == 500
    assert error.value.detail == "internal details"
    assert error.value.url is not None


def test_agent_wrapper_returns_correctable_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def request(self, *args: object, **kwargs: object) -> object:
            raise rest_tools.PublicApiRequestError(
                "Public API returned HTTP 400.",
                status_code=400,
                url="https://www.ebi.ac.uk/chembl/api/data/activity.json",
                detail='{"error":"Invalid filter field"}',
            )

    monkeypatch.setattr(rest_tools, "RestrictedRestClient", FailingClient)

    result = rest_tools.rest_api_call(
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    )

    assert result == {
        "ok": False,
        "error_type": "PublicApiRequestError",
        "error": "Public API returned HTTP 400.",
        "status_code": 400,
        "url": "https://www.ebi.ac.uk/chembl/api/data/activity.json",
        "detail": '{"error":"Invalid filter field"}',
        "retryable": False,
    }


def test_agent_wrapper_marks_server_error_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def request(self, *args: object, **kwargs: object) -> object:
            raise rest_tools.PublicApiRequestError(
                "Public API returned HTTP 503.",
                status_code=503,
            )

    monkeypatch.setattr(rest_tools, "RestrictedRestClient", FailingClient)

    result = rest_tools.rest_api_call(
        "https://rest.uniprot.org/uniprotkb/search"
    )

    assert result["ok"] is False
    assert result["retryable"] is True


def test_empty_success_response_is_allowed_without_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, request=request)

    result = _client(handler).request(
        "https://rest.uniprot.org/uniprotkb/P00533.json"
    )

    assert result.status_code == 204
    assert result.data is None


def test_small_api_data_is_preserved_without_truncation() -> None:
    data = {"count": 1, "activities": [{"molecule_chembl_id": "CHEMBL1"}]}

    compacted, metadata = rest_tools.compact_api_data(data, max_chars=2_000)

    assert compacted == data
    assert metadata["applied"] is False
    assert metadata["truncated"] is False


def test_chembl_activity_data_is_compacted_with_metadata() -> None:
    data = {
        "page_meta": {"total_count": 100, "next": "/activity.json?offset=50"},
        "activities": [
            {
                "molecule_chembl_id": f"CHEMBL{index}",
                "pchembl_value": 8.0,
                "standard_type": "IC50",
                "assay_description": "large assay description " * 300,
            }
            for index in range(100)
        ],
    }

    compacted, metadata = rest_tools.compact_api_data(data, max_chars=6_000)

    assert isinstance(compacted, dict)
    assert compacted["page_meta"]["total_count"] == 100
    activities = compacted["activities"]
    assert activities[0]["molecule_chembl_id"] == "CHEMBL0"
    assert all("assay_description" not in activity for activity in activities)
    assert len(activities) < 100
    assert metadata["applied"] is True
    assert metadata["truncated"] is True
    assert metadata["omitted_items"] > 0
    assert metadata["omitted_fields"] > 0
    assert metadata["returned_data_chars"] <= 6_000


def test_pubchem_properties_keep_scientifically_useful_fields() -> None:
    data = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": index,
                    "MolecularFormula": "C8H10N4O2",
                    "MolecularWeight": "194.19",
                    "CanonicalSMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                    "IUPACName": "caffeine",
                    "Synonym": "x" * 2_000,
                }
                for index in range(100)
            ]
        }
    }

    compacted, metadata = rest_tools.compact_api_data(data, max_chars=5_000)

    properties = compacted["PropertyTable"]["Properties"]
    assert 0 < len(properties) < 100
    assert properties[0]["CID"] == 0
    assert properties[0]["MolecularFormula"] == "C8H10N4O2"
    assert properties[0]["CanonicalSMILES"].startswith("CN1C")
    assert metadata["applied"] is True
    assert metadata["returned_data_chars"] <= 5_000


def test_agent_wrapper_compacts_large_success_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SuccessfulClient:
        def request(
            self,
            *args: object,
            **kwargs: object,
        ) -> rest_tools.RestResult:
            return rest_tools.RestResult(
                status_code=200,
                url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/example",
                content_type="application/json",
                data={"records": [{"CID": index} for index in range(10_000)]},
            )

    monkeypatch.setattr(rest_tools, "RestrictedRestClient", SuccessfulClient)

    result = rest_tools.rest_api_call(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/example"
    )

    assert result["ok"] is True
    assert result["compaction"]["applied"] is True
    assert len(result["data"]["records"]) < 10_000


@pytest.mark.parametrize(
    "params,error_match",
    [
        (
            {"limit": 50, "only": "molecule_chembl_id"},
            "require a target, molecule, assay, or document",
        ),
        (
            {"target_chembl_id": "CHEMBL203", "limit": 50},
            "require an 'only' field list",
        ),
        (
            {
                "target_chembl_id": "CHEMBL203",
                "limit": 100,
                "only": "molecule_chembl_id",
            },
            "limit from 1 to 50",
        ),
    ],
)
def test_broad_chembl_activity_queries_are_rejected(
    params: dict[str, object],
    error_match: str,
) -> None:
    client = _client(
        lambda request: pytest.fail("Unsafe query should not reach the network")
    )

    with pytest.raises(rest_tools.UnsafeQueryError, match=error_match):
        client.request(
            "https://www.ebi.ac.uk/chembl/api/data/activity.json",
            params=params,
        )


def test_bounded_chembl_activity_query_is_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"activities": []}, request=request)

    result = _client(handler).request(
        "https://www.ebi.ac.uk/chembl/api/data/activity.json",
        params={
            "target_chembl_id": "CHEMBL203",
            "limit": 50,
            "only": "molecule_chembl_id,pchembl_value",
        },
    )

    assert result.data == {"activities": []}


@pytest.mark.parametrize(
    "json_body,error_match",
    [
        (
            {
                "query": {"type": "terminal"},
                "return_type": "polymer_entity",
                "request_options": {"paginate": {"start": 0, "rows": 10}},
            },
            "return_type 'entry'",
        ),
        (
            {
                "query": {"type": "terminal"},
                "return_type": "entry",
                "request_options": {"paginate": {"start": 0, "rows": 100}},
            },
            "row limit from 1 to 50",
        ),
        (
            {
                "query": {"type": "terminal"},
                "return_type": "entry",
            },
            "request_options.paginate",
        ),
    ],
)
def test_unbounded_rcsb_searches_are_rejected(
    json_body: dict[str, object],
    error_match: str,
) -> None:
    client = _client(
        lambda request: pytest.fail("Unsafe query should not reach the network")
    )

    with pytest.raises(rest_tools.UnsafeQueryError, match=error_match):
        client.request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            method="POST",
            json_body=json_body,
        )


def test_bounded_rcsb_search_is_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"total_count": 1, "result_set": [{"identifier": "1M17"}]},
            request=request,
        )

    result = _client(handler).request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        method="POST",
        json_body={
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": "epidermal growth factor receptor"},
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 10}},
        },
    )

    assert result.data["result_set"][0]["identifier"] == "1M17"


@pytest.mark.parametrize(
    "params,error_match",
    [
        (
            {"format": "json", "size": 10, "fields": "accession,id"},
            "specific query",
        ),
        (
            {"query": "EGFR", "format": "json", "size": 10},
            "explicit field list",
        ),
        (
            {
                "query": "EGFR",
                "format": "json",
                "size": 100,
                "fields": "accession,id",
            },
            "size from 1 to 50",
        ),
    ],
)
def test_unbounded_uniprot_searches_are_rejected(
    params: dict[str, object],
    error_match: str,
) -> None:
    client = _client(
        lambda request: pytest.fail("Unsafe query should not reach the network")
    )

    with pytest.raises(rest_tools.UnsafeQueryError, match=error_match):
        client.request(
            "https://rest.uniprot.org/uniprotkb/search",
            params=params,
        )


def test_bounded_uniprot_search_is_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"primaryAccession": "P00533"}]},
            request=request,
        )

    result = _client(handler).request(
        "https://rest.uniprot.org/uniprotkb/search",
        params={
            "query": "protein_name:(epidermal growth factor receptor)",
            "format": "json",
            "size": 10,
            "fields": "accession,id,protein_name,organism_name,length",
        },
    )

    assert result.data["results"][0]["primaryAccession"] == "P00533"


@pytest.mark.parametrize(
    "params,error_match",
    [
        (
            {
                "db": "protein",
                "term": "BRCA1",
                "retmax": 10,
                "retmode": "json",
            },
            "gene and pubmed",
        ),
        (
            {"db": "gene", "retmax": 10, "retmode": "json"},
            "specific search term",
        ),
        (
            {
                "db": "gene",
                "term": "BRCA1[gene]",
                "retmax": 100,
                "retmode": "json",
            },
            "retmax from 1 to 50",
        ),
    ],
)
def test_unbounded_ncbi_searches_are_rejected(
    params: dict[str, object],
    error_match: str,
) -> None:
    client = _client(
        lambda request: pytest.fail("Unsafe query should not reach the network")
    )

    with pytest.raises(rest_tools.UnsafeQueryError, match=error_match):
        client.request(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
        )


def test_bounded_ncbi_gene_summary_is_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"result": {"uids": ["672"], "672": {"name": "BRCA1"}}},
            request=request,
        )

    result = _client(handler).request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={"db": "gene", "id": "672", "retmode": "json"},
    )

    assert result.data["result"]["672"]["name"] == "BRCA1"


def test_ncbi_summary_rejects_unresolved_identifiers() -> None:
    client = _client(
        lambda request: pytest.fail("Unsafe query should not reach the network")
    )

    with pytest.raises(
        rest_tools.UnsafeQueryError,
        match="resolved numeric IDs",
    ):
        client.request(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "gene", "id": "BRCA1", "retmode": "json"},
        )


@pytest.mark.parametrize(
    "json_body,error_match",
    [
        (
            {"query": "mutation { removeTarget(id: 1) }", "variables": {}},
            "read-only",
        ),
        (
            {
                "query": (
                    "query Search($q: String!) { "
                    "search(queryString: $q) { hits { id name } } }"
                ),
                "variables": {"q": "BRCA1"},
            },
            "explicit result size",
        ),
        (
            {
                "query": (
                    "query Search($q: String!, $size: Int!) { "
                    "search(queryString: $q, page: {index: 0, size: $size}) "
                    "{ hits { id name } } }"
                ),
                "variables": {"q": "BRCA1", "size": 100},
            },
            "sizes must be from 1 to 50",
        ),
    ],
)
def test_unsafe_open_targets_queries_are_rejected(
    json_body: dict[str, object],
    error_match: str,
) -> None:
    client = _client(
        lambda request: pytest.fail("Unsafe query should not reach the network")
    )

    with pytest.raises(rest_tools.UnsafeQueryError, match=error_match):
        client.request(
            "https://api.platform.opentargets.org/api/v4/graphql",
            method="POST",
            json_body=json_body,
        )


def test_bounded_open_targets_search_is_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {
                    "search": {
                        "total": 1,
                        "hits": [{"id": "ENSG00000012048", "name": "BRCA1"}],
                    }
                }
            },
            request=request,
        )

    result = _client(handler).request(
        "https://api.platform.opentargets.org/api/v4/graphql",
        method="POST",
        json_body={
            "query": (
                "query Search($q: String!, $size: Int!) { "
                "search(queryString: $q, page: {index: 0, size: $size}) "
                "{ total hits { id name } } }"
            ),
            "variables": {"q": "BRCA1", "size": 5},
        },
    )

    assert result.data["data"]["search"]["hits"][0]["name"] == "BRCA1"
