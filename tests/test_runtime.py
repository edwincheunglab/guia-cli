from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.adk.models.lite_llm import LiteLlm

from guia_cli import runtime
from guia_cli.runtime import (
    CONTEXT_LENGTH_GUIDANCE,
    AgentRuntime,
    ContextLengthExceededError,
    ModelSettings,
    RuntimeConfigurationError,
    _is_context_length_error,
    _final_text,
    _user_facing_text,
    create_model,
)


def test_openai_model_settings_require_provider_key() -> None:
    settings = ModelSettings.from_environment(
        {
            "GUIA_MODEL": "openai/gpt-5-mini",
            "OPENAI_API_KEY": "test-key",
        },
        load_env_file=False,
    )

    assert settings.model == "openai/gpt-5-mini"
    assert settings.provider == "openai"


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"GUIA_MODEL": ""},
        {"GUIA_MODEL": "model-without-provider"},
        {"GUIA_MODEL": "openai/gpt-5-mini"},
        {
            "GUIA_MODEL": "anthropic/claude-sonnet",
            "ANTHROPIC_API_KEY": "",
        },
    ],
)
def test_invalid_model_environment_is_rejected(
    environment: dict[str, str],
) -> None:
    with pytest.raises(RuntimeConfigurationError):
        ModelSettings.from_environment(
            environment,
            load_env_file=False,
        )


def test_gemini_accepts_google_api_key() -> None:
    settings = ModelSettings.from_environment(
        {
            "GUIA_MODEL": "gemini/gemini-2.5-flash",
            "GOOGLE_API_KEY": "test-key",
        },
        load_env_file=False,
    )

    assert settings.provider == "gemini"


def test_unknown_provider_is_left_to_litellm() -> None:
    settings = ModelSettings.from_environment(
        {"GUIA_MODEL": "custom/local-model"},
        load_env_file=False,
    )

    assert settings.provider == "custom"


def test_create_model_does_not_embed_api_key() -> None:
    model = create_model(ModelSettings(model="openai/gpt-5-mini"))

    assert isinstance(model, LiteLlm)
    assert model.model == "openai/gpt-5-mini"
    assert "key" not in repr(model).lower()


def test_final_text_reads_only_final_response() -> None:
    event = SimpleNamespace(
        is_final_response=lambda: True,
        content=SimpleNamespace(
            parts=[
                SimpleNamespace(text="first"),
                SimpleNamespace(text="second"),
            ]
        ),
    )

    assert _final_text(event) == "first\nsecond"


def test_final_text_ignores_non_final_event() -> None:
    event = SimpleNamespace(
        is_final_response=lambda: False,
        content=SimpleNamespace(parts=[SimpleNamespace(text="ignored")]),
    )

    assert _final_text(event) is None


def test_user_facing_text_removes_internal_draft_before_markdown_answer() -> None:
    response = """
The user is asking me to critically evaluate a causal claim.
Let me analyze the methodological limitations.
I should provide a comprehensive response.

## Critical Evaluation

Correlation alone does not establish causation.
"""

    assert _user_facing_text(response) == (
        "## Critical Evaluation\n\n"
        "Correlation alone does not establish causation."
    )


def test_user_facing_text_preserves_normal_intro_before_heading() -> None:
    response = """
Here is a concise overview for context.

## Evidence

The retrieved records support the summary.
"""

    assert _user_facing_text(response) == response.strip()


def test_user_facing_text_removes_explicit_think_blocks() -> None:
    response = """
<think>I should plan this response first.</think>
## Final Answer

User-facing content.
"""

    assert _user_facing_text(response) == (
        "## Final Answer\n\nUser-facing content."
    )


@pytest.mark.parametrize(
    "message",
    [
        "context_length_exceeded",
        "This model's maximum context length is 8192 tokens.",
        "Prompt too long",
        "OpenAIException - Prompt 超长",
    ],
)
def test_context_length_errors_are_detected_across_providers(message: str) -> None:
    assert _is_context_length_error(RuntimeError(message)) is True


def test_context_length_error_is_detected_through_exception_chain() -> None:
    provider_error = ValueError("too many input tokens")
    wrapper = RuntimeError("Model call failed")
    wrapper.__cause__ = provider_error

    assert _is_context_length_error(wrapper) is True


def test_unrelated_provider_error_is_not_misclassified() -> None:
    assert _is_context_length_error(RuntimeError("HTTP 401 unauthorized")) is False


def test_agent_runtime_wraps_context_length_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionService:
        async def get_session(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(id="session-1")

    class FailingRunner:
        async def run_async(self, **kwargs: object):
            raise RuntimeError("OpenAIException - Prompt 超长")
            yield

    agent_runtime = object.__new__(AgentRuntime)
    agent_runtime._agent = SimpleNamespace()
    agent_runtime._session_service = SessionService()
    agent_runtime._runner = FailingRunner()
    monkeypatch.setattr(
        runtime,
        "create_session",
        lambda session_id: SimpleNamespace(
            session_id=session_id,
            root=SimpleNamespace(parent=SimpleNamespace(parent="/tmp")),
        ),
    )

    with pytest.raises(
        ContextLengthExceededError,
        match="context limit was exceeded",
    ) as error:
        asyncio.run(agent_runtime.run("Find EGFR inhibitors", session_id="session-1"))

    assert str(error.value) == CONTEXT_LENGTH_GUIDANCE
