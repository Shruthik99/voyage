import httpx
import pytest
from openai import RateLimitError
from openai.types.chat import ChatCompletion

from evals.pacing import DailyLimitReached, PacedClient, TokenBucket, _is_daily


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _rate_limit_error(message: str) -> RateLimitError:
    response = httpx.Response(429, request=_request(), json={"error": {"message": message}})
    return RateLimitError(message, response=response, body=None)


def _completion(usage: dict | None = None) -> ChatCompletion:
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    }
    if usage is not None:
        data["usage"] = usage
    return ChatCompletion.model_validate(data)


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)

    def create(self, **kwargs):
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeChat:
    def __init__(self, script):
        self.completions = _FakeCompletions(script)


class FakeInnerClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)


# --- _is_daily: the one distinction that must never be read backwards ---


def test_tpd_429_message_converts_to_daily():
    exc = _rate_limit_error("Rate limit reached for tokens per day (TPD): limit 200000")
    assert _is_daily(exc) is True


def test_tpm_429_message_does_not_convert_to_daily():
    exc = _rate_limit_error("Rate limit reached for tokens per minute (TPM): limit 8000")
    assert _is_daily(exc) is False


# --- TokenBucket: the 60s window ---


class _FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_bucket_sleeps_until_the_60s_window_has_room():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=100, daily_token_budget=1_000_000, clock=fake.clock, sleep=fake.sleep)

    bucket.reserve(50)
    bucket.record(50, 50)
    # limit = 100 * 0.9 = 90; a second 50-token reservation does not fit yet
    bucket.reserve(50)

    assert fake.sleeps  # it waited rather than bursting past the per-minute limit
    assert fake.now >= 60


def test_bucket_does_not_sleep_when_there_is_room():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=1000, daily_token_budget=1_000_000, clock=fake.clock, sleep=fake.sleep)

    bucket.reserve(50)
    bucket.record(50, 50)
    bucket.reserve(50)

    assert fake.sleeps == []


def test_bucket_raises_daily_limit_reached_when_estimate_would_exceed_budget():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=1_000_000, daily_token_budget=100, clock=fake.clock, sleep=fake.sleep)

    with pytest.raises(DailyLimitReached):
        bucket.reserve(150)


def test_bucket_daily_budget_accounts_for_seeded_daily_used():
    fake = _FakeClock()
    bucket = TokenBucket(
        tpm=1_000_000, daily_token_budget=100, daily_used=90, clock=fake.clock, sleep=fake.sleep
    )

    with pytest.raises(DailyLimitReached):
        bucket.reserve(20)


def test_bucket_record_corrects_the_reservation_to_the_actual_cost():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=1_000_000, daily_token_budget=1_000_000, clock=fake.clock, sleep=fake.sleep)

    bucket.reserve(500)  # the estimate
    bucket.record(123, 500)  # the real usage.total_tokens was much smaller

    assert bucket.daily_used == 123
    assert bucket._window_total() == 123


# --- PacedClient: zero changes to src/llm.py needed ---


def test_paced_client_converts_a_tpd_429_to_daily_limit_reached():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=1_000_000, daily_token_budget=1_000_000, clock=fake.clock, sleep=fake.sleep)
    err = _rate_limit_error("tokens per day (TPD) exceeded")
    inner = FakeInnerClient([err])
    paced = PacedClient(inner, bucket)

    with pytest.raises(DailyLimitReached):
        paced.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


def test_paced_client_reraises_a_tpm_429_unchanged():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=1_000_000, daily_token_budget=1_000_000, clock=fake.clock, sleep=fake.sleep)
    err = _rate_limit_error("tokens per minute (TPM) exceeded")
    inner = FakeInnerClient([err])
    paced = PacedClient(inner, bucket)

    with pytest.raises(RateLimitError):
        paced.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


def test_paced_client_records_actual_usage_after_a_successful_call():
    fake = _FakeClock()
    bucket = TokenBucket(tpm=1_000_000, daily_token_budget=1_000_000, clock=fake.clock, sleep=fake.sleep)
    inner = FakeInnerClient(
        [_completion(usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})]
    )
    paced = PacedClient(inner, bucket)

    paced.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    assert bucket.daily_used == 120
