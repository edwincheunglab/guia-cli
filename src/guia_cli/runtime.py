"""Local model and Google ADK runtime for GUIA CLI."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import litellm
from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.apps.app import App
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.genai import types

from guia_cli.sessions import create_session

MODEL_ENV = "GUIA_MODEL"
DEFAULT_USER_ID = "local-user"
APP_NAME = "guia_cli"
SESSION_ID_STATE_KEY = "guia_session_id"
DATA_DIR_STATE_KEY = "guia_data_dir"
_PROVIDER_KEY_ENVIRONMENTS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
}
_CONTEXT_ERROR_MARKERS = (
    "context length",
    "context window",
    "context_length_exceeded",
    "input token count exceeds",
    "maximum context",
    "max context",
    "prompt is too long",
    "prompt too long",
    "prompt 超长",
    "too many input tokens",
)
CONTEXT_LENGTH_GUIDANCE = (
    "The configured model's context limit was exceeded. GUIA CLI compacts API "
    "responses automatically, but this request still produced too much context. "
    "Narrow the query to one target, organism, assay type, compound, or a smaller "
    "result limit, or choose a model with a larger context window."
)
_INTERNAL_DRAFT_MARKERS = (
    "the user is asking",
    "let me analyze",
    "let me structure",
    "let me compile",
    "let me organize",
    "i need to assess",
    "i should provide",
)


class RuntimeConfigurationError(RuntimeError):
    """Raised when required local runtime configuration is missing."""


class AgentRuntimeError(RuntimeError):
    """Raised when a local agent run cannot produce a final response."""


class ContextLengthExceededError(AgentRuntimeError):
    """Raised when a provider rejects an oversized model context."""


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Non-secret model settings loaded from the local environment."""

    model: str

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0].strip().lower()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
        env_file: str | Path | None = None,
    ) -> ModelSettings:
        if load_env_file:
            load_dotenv(dotenv_path=env_file, override=False)
        source = os.environ if environ is None else environ
        model = source.get(MODEL_ENV, "").strip()
        if not model or "/" not in model:
            raise RuntimeConfigurationError(
                "GUIA_MODEL must use provider/model format."
            )

        settings = cls(model=model)
        expected_keys = _PROVIDER_KEY_ENVIRONMENTS.get(settings.provider, ())
        if expected_keys and not any(source.get(key, "").strip() for key in expected_keys):
            choices = " or ".join(expected_keys)
            raise RuntimeConfigurationError(
                f"Model provider '{settings.provider}' requires {choices}."
            )
        return settings


def create_model(settings: ModelSettings) -> BaseLlm:
    """Create an ADK LiteLLM model without embedding credentials in code."""

    litellm.suppress_debug_info = True
    return LiteLlm(model=settings.model)


def _is_context_length_error(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        class_name = type(current).__name__.lower()
        details = " ".join(
            str(value)
            for value in (
                current,
                getattr(current, "message", ""),
                getattr(current, "code", ""),
                getattr(current, "body", ""),
            )
            if value
        ).lower()
        if "contextwindowexceeded" in class_name or any(
            marker in details for marker in _CONTEXT_ERROR_MARKERS
        ):
            return True
        for nested in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _final_text(event: object) -> str | None:
    is_final_response = getattr(event, "is_final_response", None)
    if not callable(is_final_response) or not is_final_response():
        return None

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None
    text_parts = [
        part.text for part in parts if getattr(part, "text", None)
    ]
    return "\n".join(text_parts).strip() or None


def _user_facing_text(text: str) -> str:
    """Remove a recognizable draft preamble before a Markdown final answer."""

    without_think_blocks = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    heading = re.search(r"(?m)^#{1,3}\s+\S", without_think_blocks)
    if heading is None or heading.start() == 0:
        return without_think_blocks

    preamble = without_think_blocks[: heading.start()].lower()
    if any(marker in preamble for marker in _INTERNAL_DRAFT_MARKERS):
        return without_think_blocks[heading.start() :].strip()
    return without_think_blocks


class AgentRuntime:
    """Execute one ADK agent locally with in-memory conversation sessions."""

    def __init__(
        self,
        agent: BaseAgent,
        *,
        session_service: BaseSessionService | None = None,
    ) -> None:
        self._agent = agent
        self._session_service = session_service or InMemorySessionService()
        self._runner = Runner(
            app=App(name=APP_NAME, root_agent=agent),
            session_service=self._session_service,
        )

    @property
    def agent(self) -> BaseAgent:
        return self._agent

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        user_id: str = DEFAULT_USER_ID,
    ) -> str:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Agent prompt cannot be empty.")

        selected_session_id = session_id or uuid4().hex
        file_session = create_session(selected_session_id)
        session = await self._session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=selected_session_id,
        )
        if session is None:
            session = await self._session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=selected_session_id,
                state={
                    SESSION_ID_STATE_KEY: file_session.session_id,
                    DATA_DIR_STATE_KEY: str(file_session.root.parent.parent),
                },
            )

        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=clean_prompt)],
        )
        final_response: str | None = None
        try:
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=message,
            ):
                event_text = _final_text(event)
                if event_text is not None:
                    final_response = event_text
        except Exception as exc:
            if _is_context_length_error(exc):
                raise ContextLengthExceededError(
                    CONTEXT_LENGTH_GUIDANCE
                ) from exc
            raise

        if final_response is None:
            raise AgentRuntimeError("Agent did not produce a final response.")
        return _user_facing_text(final_response)


__all__ = [
    "APP_NAME",
    "AgentRuntime",
    "AgentRuntimeError",
    "CONTEXT_LENGTH_GUIDANCE",
    "ContextLengthExceededError",
    "DATA_DIR_STATE_KEY",
    "DEFAULT_USER_ID",
    "MODEL_ENV",
    "ModelSettings",
    "RuntimeConfigurationError",
    "SESSION_ID_STATE_KEY",
    "_is_context_length_error",
    "_user_facing_text",
    "create_model",
]
