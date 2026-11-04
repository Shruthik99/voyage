"""The Groq-backed LLM client, and the only place that decides what to retry.

An OpenAI-compatible client pointed at Groq's free tier (DECISIONS: "$0 budget,
tool calling support, and switching providers later is a config change").

Retry policy mirrors src.tools: transient failures only -- timeout, connection
error, 429, 5xx. A 429 (RateLimitError) gets RATE_LIMIT_ATTEMPTS attempts and
waits as long as the provider asked for: the Retry-After response header if
present, else a "try again in X.XXs" / "XmY.Ys" phrase parsed out of the error
message, plus a safety margin, capped at MAX_RATE_LIMIT_WAIT. If neither is
present, it falls back to the same short exponential wait every other
transient error uses. Everything else raises immediately with the provider's
own message. The caller never gets a fabricated completion, so a failure here
always surfaces.

RETRY_WAIT_MAX, RATE_LIMIT_ATTEMPTS, and MAX_RATE_LIMIT_WAIT are module-level
so a test can shrink them, e.g.
``monkeypatch.setattr(llm_module, "RETRY_WAIT_MAX", 0.0)``.
"""

import os
import re

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
from tenacity import retry, retry_if_exception_type, wait_exponential

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
LLM_TIMEOUT = 60.0
RETRY_WAIT_MAX = 2.0

TRANSIENT_ATTEMPTS = 3  # non-rate-limit transient errors
RATE_LIMIT_ATTEMPTS = 5  # 429s: Groq's free tier needs more room to recover
RATE_LIMIT_MARGIN = 0.5  # added to whatever wait the provider asks for
MAX_RATE_LIMIT_WAIT = 30.0

TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)

# "Please try again in 4.56s" or "...in 1m30.5s"
_RETRY_AFTER_MESSAGE_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


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


def _retry_after_seconds(exc: BaseException) -> float | None:
    """What the provider asked us to wait: the Retry-After header, else the message."""
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header is not None:
            try:
                return float(header)
            except ValueError:
                pass

    match = _RETRY_AFTER_MESSAGE_RE.search(str(exc))
    if match:
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    return None


def _wait_transient(retry_state) -> float:
    return wait_exponential(multiplier=0.1, min=0, max=RETRY_WAIT_MAX)(retry_state)


def _wait_for_attempt(retry_state) -> float:
    exc = retry_state.outcome.exception()
    if isinstance(exc, RateLimitError):
        wait = _retry_after_seconds(exc)
        if wait is not None:
            return min(wait + RATE_LIMIT_MARGIN, MAX_RATE_LIMIT_WAIT)
    return _wait_transient(retry_state)


def _stop_after_attempt_for(retry_state) -> bool:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    limit = RATE_LIMIT_ATTEMPTS if isinstance(exc, RateLimitError) else TRANSIENT_ATTEMPTS
    return retry_state.attempt_number >= limit


@retry(
    retry=retry_if_exception_type(TRANSIENT),
    stop=_stop_after_attempt_for,
    wait=_wait_for_attempt,
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
        attempts = _create.statistics.get("attempt_number", 1)
        raise LLMUnavailable(f"{exc.__class__.__name__} after {attempts} attempts: {exc}") from None
    except OpenAIError as exc:
        # Auth, bad model, bad request: retrying cannot help. Narrow to the SDK's
        # own base class so a bug in our own call still raises as itself.
        raise LLMError(f"{exc.__class__.__name__}: {exc}") from None
