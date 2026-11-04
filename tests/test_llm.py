import httpx2
import pytest
from openai import AuthenticationError, RateLimitError
from openai.types.chat import ChatCompletion

from src import llm as llm_module
from src.llm import LLMError, LLMUnavailable, build_client, chat


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(cls, status_code: int, message: str = "error"):
    response = httpx2.Response(status_code, request=_request(), json={"error": {"message": message}})
    return cls(message, response=response, body=None)


def _completion(message: dict) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeChat:
    def __init__(self, script):
        self.completions = _FakeCompletions(script)


class FakeClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(llm_module, "RETRY_WAIT_MAX", 0.0)


# --- build_client ---


def test_build_client_sets_base_url_and_no_retries(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = build_client()
    assert client.base_url == "https://api.groq.com/openai/v1/"
    assert client.max_retries == 0


def test_build_client_reads_groq_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    client = build_client()
    assert client.api_key == "env-key"


def test_build_client_raises_llm_error_when_key_unset(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError):
        build_client()


# --- chat: retry policy ---


def test_chat_retries_rate_limit_then_succeeds():
    success = _completion({"role": "assistant", "content": "hi"})
    client = FakeClient([_status_error(RateLimitError, 429), success])

    result = chat(client, [{"role": "user", "content": "hi"}], model="m")

    assert result is success
    assert len(client.calls) == 2


def test_chat_authentication_error_raises_llm_error_without_retry():
    client = FakeClient([_status_error(AuthenticationError, 401)])

    with pytest.raises(LLMError):
        chat(client, [{"role": "user", "content": "hi"}], model="m")

    assert len(client.calls) == 1


def test_chat_retries_exhausted_raises_llm_unavailable():
    client = FakeClient([_status_error(RateLimitError, 429) for _ in range(3)])

    with pytest.raises(LLMUnavailable):
        chat(client, [{"role": "user", "content": "hi"}], model="m")

    assert len(client.calls) == 3


# --- chat: local guard against the Groq 400 ---


def test_chat_rejects_tools_and_response_format_together_before_any_request():
    client = FakeClient([])

    with pytest.raises(LLMError):
        chat(
            client,
            [{"role": "user", "content": "hi"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {}}},
        )

    assert client.calls == []
