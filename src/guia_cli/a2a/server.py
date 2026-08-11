"""Authenticated loopback A2A server components."""

from __future__ import annotations

import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
)
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.helpers import get_message_text, new_text_message
from a2a.types import AgentCard
from a2a.utils.errors import UnsupportedOperationError
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from guia_cli.runtime import ContextLengthExceededError
from guia_cli.sessions import validate_session_id

_ERROR_PREFIX = "__GUIA_A2A_ERROR__:"
_MAX_TASK_CHARS = 100_000


class DomainAgent(Protocol):
    """Minimal interface exposed through the local A2A transport."""

    async def run(
        self,
        task: str,
        *,
        session_id: str | None = None,
    ) -> str:
        """Execute a domain task in an existing GUIA CLI session."""


class BearerTokenMiddleware:
    """Require one run-scoped bearer token on every service route."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode("ascii")

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = dict(scope.get("headers", ()))
        supplied = headers.get(b"authorization", b"")
        if not secrets.compare_digest(supplied, self._expected):
            body = b'{"error":"unauthorized"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self._app(scope, receive, send)


class GuiaAgentExecutor(AgentExecutor):
    """Translate an A2A text message into a restricted domain-agent call."""

    def __init__(self, agent: DomainAgent) -> None:
        self._agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message = context.message
        if message is None:
            await _send_error(
                event_queue,
                "invalid_request",
                "The A2A request did not contain a message.",
            )
            return

        session_id = message.context_id
        task = get_message_text(message).strip()
        try:
            if not session_id:
                raise ValueError("The A2A request did not include a context ID.")
            validate_session_id(session_id)
            if not task:
                raise ValueError("The A2A request message was empty.")
            if len(task) > _MAX_TASK_CHARS:
                raise ValueError(
                    f"The A2A request exceeds {_MAX_TASK_CHARS:,} characters."
                )

            result = await self._agent.run(task, session_id=session_id)
            await event_queue.enqueue_event(
                new_text_message(
                    result,
                    context_id=session_id,
                    task_id=message.task_id or None,
                )
            )
        except ContextLengthExceededError as exc:
            await _send_error(
                event_queue,
                "context_length",
                str(exc),
                context_id=session_id,
                task_id=message.task_id or None,
            )
        except ValueError as exc:
            await _send_error(
                event_queue,
                "invalid_request",
                str(exc),
                context_id=session_id,
                task_id=message.task_id or None,
            )
        except Exception:
            await _send_error(
                event_queue,
                "execution_failed",
                "The local domain agent could not complete this request.",
                context_id=session_id,
                task_id=message.task_id or None,
            )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise UnsupportedOperationError("Immediate local requests cannot be canceled.")


async def _send_error(
    event_queue: EventQueue,
    code: str,
    message: str,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
) -> None:
    payload = json.dumps(
        {"code": code, "message": message},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    await event_queue.enqueue_event(
        new_text_message(
            f"{_ERROR_PREFIX}{payload}",
            context_id=context_id,
            task_id=task_id,
        )
    )


def create_a2a_app(
    *,
    agent: DomainAgent,
    agent_card: AgentCard,
    bearer_token: str,
) -> ASGIApp:
    """Build one authenticated JSON-RPC A2A application."""

    handler = DefaultRequestHandler(
        agent_executor=GuiaAgentExecutor(agent),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await handler.aclose()

    app = Starlette(
        routes=[
            *create_agent_card_routes(agent_card),
            *create_jsonrpc_routes(handler, "/"),
        ],
        lifespan=lifespan,
    )
    return BearerTokenMiddleware(app, bearer_token)


__all__ = [
    "BearerTokenMiddleware",
    "DomainAgent",
    "GuiaAgentExecutor",
    "create_a2a_app",
]
