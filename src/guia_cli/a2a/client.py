"""A2A client proxy used by the GUIA CLI orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import (
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
)

from guia_cli.a2a.errors import A2ADispatchError
from guia_cli.runtime import ContextLengthExceededError
from guia_cli.sessions import validate_session_id

_ERROR_PREFIX = "__GUIA_A2A_ERROR__:"
DEFAULT_A2A_TIMEOUT_SECONDS = 60.0 * 60.0


@dataclass(frozen=True, slots=True)
class A2ADomainAgentProxy:
    """Dispatch domain work over authenticated localhost A2A."""

    base_url: str
    bearer_token: str
    timeout_seconds: float = DEFAULT_A2A_TIMEOUT_SECONDS

    async def run(self, task: str, session_id: str) -> str:
        """Send a text task and return the final text response."""

        validate_session_id(session_id)
        if not task.strip():
            raise ValueError("Agent task cannot be empty.")

        try:
            async with httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Cache-Control": "no-store",
                },
                timeout=httpx.Timeout(self.timeout_seconds),
                follow_redirects=False,
            ) as http_client:
                config = ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    supported_protocol_bindings=["JSONRPC"],
                )
                client = await ClientFactory(config).create_from_url(self.base_url)
                try:
                    request = SendMessageRequest(
                        message=new_text_message(
                            task,
                            context_id=session_id,
                            role=Role.ROLE_USER,
                        ),
                        configuration=SendMessageConfiguration(
                            accepted_output_modes=["text/plain"]
                        ),
                    )
                    final_text = ""
                    async for response in client.send_message(request):
                        response_text = get_stream_response_text(response)
                        if response_text:
                            final_text = response_text
                finally:
                    await client.close()
        except ContextLengthExceededError:
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            raise A2ADispatchError(
                "The local A2A agent service did not respond successfully."
            ) from exc
        except A2ADispatchError:
            raise
        except Exception as exc:
            raise A2ADispatchError(
                "The local A2A request could not be completed."
            ) from exc

        if final_text.startswith(_ERROR_PREFIX):
            _raise_remote_error(final_text[len(_ERROR_PREFIX) :])
        if not final_text.strip():
            raise A2ADispatchError(
                "The local A2A agent returned an empty response."
            )
        return final_text


def _raise_remote_error(payload_text: str) -> None:
    try:
        payload = json.loads(payload_text)
        code = payload.get("code")
        message = payload.get("message")
    except (json.JSONDecodeError, AttributeError):
        code = None
        message = None

    if code == "context_length" and isinstance(message, str):
        raise ContextLengthExceededError(message)
    if isinstance(message, str):
        raise A2ADispatchError(message)
    raise A2ADispatchError("The local A2A agent reported an invalid error.")


__all__ = ["A2ADomainAgentProxy", "DEFAULT_A2A_TIMEOUT_SECONDS"]
