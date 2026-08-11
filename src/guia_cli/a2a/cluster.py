"""Lifecycle manager for eager in-process localhost A2A services."""

from __future__ import annotations

import asyncio
import secrets
import socket
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self

import uvicorn

from guia_cli.a2a.cards import AGENT_CARD_DEFINITIONS, build_agent_card
from guia_cli.a2a.client import A2ADomainAgentProxy
from guia_cli.a2a.errors import A2AServiceError
from guia_cli.a2a.server import DomainAgent, create_a2a_app
from guia_cli.agents.orchestrator import AgentName


class _InProcessServer(uvicorn.Server):
    """Run uvicorn without process-wide signal handler changes."""

    async def serve(
        self,
        sockets: list[socket.socket] | None = None,
    ) -> None:
        await self._serve(sockets)


@dataclass(slots=True)
class _Service:
    agent_name: AgentName
    url: str
    listener: socket.socket
    server: _InProcessServer
    task: asyncio.Task[None]


class LocalA2ACluster:
    """Start all restricted domain agents as authenticated A2A services."""

    def __init__(
        self,
        agents: dict[AgentName, DomainAgent],
        *,
        startup_timeout_seconds: float = 10.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        expected = set(AGENT_CARD_DEFINITIONS)
        actual = set(agents)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "A2A cluster requires exactly the GUIA CLI domain agents; "
                f"missing={missing}, unexpected={unexpected}."
            )
        self._agents = dict(agents)
        self._startup_timeout = startup_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._token: str | None = None
        self._services: dict[AgentName, _Service] = {}
        self._proxies: dict[AgentName, A2ADomainAgentProxy] = {}

    @property
    def proxies(
        self,
    ) -> MappingProxyType[AgentName, A2ADomainAgentProxy]:
        """Return running client proxies keyed by domain-agent name."""

        if not self._services or not self._proxies:
            raise A2AServiceError("The local A2A cluster is not running.")
        return MappingProxyType(self._proxies)

    @property
    def urls(self) -> MappingProxyType[AgentName, str]:
        """Return the loopback service URLs for diagnostics and tests."""

        if not self._services:
            raise A2AServiceError("The local A2A cluster is not running.")
        return MappingProxyType(
            {name: service.url for name, service in self._services.items()}
        )

    async def start(self) -> None:
        """Eagerly bind and start every domain-agent service."""

        if self._services:
            raise A2AServiceError("The local A2A cluster is already running.")

        self._token = secrets.token_urlsafe(32)
        try:
            for agent_name, agent in self._agents.items():
                listener = _bind_loopback_socket()
                port = int(listener.getsockname()[1])
                url = f"http://127.0.0.1:{port}/"
                app = create_a2a_app(
                    agent=agent,
                    agent_card=build_agent_card(agent_name, url),
                    bearer_token=self._token,
                )
                server = _InProcessServer(
                    uvicorn.Config(
                        app,
                        host="127.0.0.1",
                        port=port,
                        access_log=False,
                        log_config=None,
                        log_level="warning",
                        lifespan="on",
                        server_header=False,
                    )
                )
                task = asyncio.create_task(
                    server.serve(sockets=[listener]),
                    name=f"guia-a2a-{agent_name}",
                )
                self._services[agent_name] = _Service(
                    agent_name=agent_name,
                    url=url,
                    listener=listener,
                    server=server,
                    task=task,
                )

            await self._wait_until_started()
            self._proxies = {
                name: A2ADomainAgentProxy(
                    base_url=service.url,
                    bearer_token=self._token,
                )
                for name, service in self._services.items()
            }
        except Exception as exc:
            await self.stop()
            if isinstance(exc, A2AServiceError):
                raise
            raise A2AServiceError(
                "Failed to start the local A2A agent services."
            ) from exc

    async def stop(self) -> None:
        """Stop all services and erase the run-scoped credential."""

        services = tuple(self._services.values())
        for service in services:
            service.server.should_exit = True

        if services:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(service.task for service in services),
                        return_exceptions=True,
                    ),
                    timeout=self._shutdown_timeout,
                )
            except TimeoutError:
                for service in services:
                    service.task.cancel()
                await asyncio.gather(
                    *(service.task for service in services),
                    return_exceptions=True,
                )
            finally:
                for service in services:
                    service.listener.close()

        self._services.clear()
        self._proxies.clear()
        self._token = None

    async def _wait_until_started(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout
        while True:
            failed = [
                service
                for service in self._services.values()
                if service.task.done() and not service.server.started
            ]
            if failed:
                raise A2AServiceError(
                    f"Local A2A service stopped during startup: "
                    f"{failed[0].agent_name}."
                )
            if all(
                service.server.started for service in self._services.values()
            ):
                return
            if loop.time() >= deadline:
                raise A2AServiceError(
                    "Timed out while starting local A2A agent services."
                )
            await asyncio.sleep(0.01)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        await self.stop()


def _bind_loopback_socket() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        return listener
    except Exception:
        listener.close()
        raise


__all__ = ["LocalA2ACluster"]
