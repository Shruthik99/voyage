"""The Groq-backed LLM client, and the only place that decides what to retry.

An OpenAI-compatible client pointed at Groq's free tier (DECISIONS: "$0 budget,
tool calling support, and switching providers later is a config change").

Retry policy mirrors src.tools: transient failures only -- timeout, connection
error, 429, 5xx -- three attempts with a bounded exponential wait. Everything
else raises immediately with the provider's own message. The caller never gets a
fabricated completion, so a failure here always surfaces.

RETRY_WAIT_MAX is module-level so a test can shrink it:
``monkeypatch.setattr(llm_module, "RETRY_WAIT_MAX", 0.0)``.
"""

import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
LLM_TIMEOUT = 60.0
RETRY_WAIT_MAX = 2.0

TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable response."""


class LLMUnavailable(LLMError):
    """A transient failure that survived every retry."""


def build_client(api_key: str | None = None, base_url: str = GROQ_BASE_URL) -> OpenAI:
    """An OpenAI SDK client pointed at Groq. Key from GROQ_API_KEY unless passed."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set in the environment")
    # max_retries=0: retrying is this module's job, where it is visible and tested.
    return OpenAI(api_key=key, base_url=base_url, timeout=LLM_TIMEOUT, max_retries=0)


def _wait_transient(retry_state):
    return wait_exponential(multiplier=0.1, min=0, max=RETRY_WAIT_MAX)(retry_state)


@retry(
    retry=retry_if_exception_type(TRANSIENT),
    stop=stop_after_attempt(3),
    wait=_wait_transient,
    reraise=True,
)
def _create(client: OpenAI, **kwargs) -> ChatCompletion:
    return client.chat.completions.create(**kwargs)


def chat(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
    tool_choice: str | None = None,
) -> ChatCompletion:
    """One chat completion. Retries transient failures; raises on everything else."""
    if tools and response_format:
        # Groq answers this combination with a 400, "json mode cannot be combined
        # with tool/function calling". Failing here costs no request and names the
        # constraint at the call site that broke it.
        raise LLMError(
            "tools and response_format cannot be combined in one request; "
            "call tools first, then ask for the structured result"
        )

    kwargs: dict = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    if response_format:
        kwargs["response_format"] = response_format

    try:
        return _create(client, **kwargs)
    except TRANSIENT as exc:
        raise LLMUnavailable(f"{exc.__class__.__name__} after 3 attempts: {exc}") from None
    except OpenAIError as exc:
        # Auth, bad model, bad request: retrying cannot help. Narrow to the SDK's
        # own base class so a bug in our own call still raises as itself.
        raise LLMError(f"{exc.__class__.__name__}: {exc}") from None
