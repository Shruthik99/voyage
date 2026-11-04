"""Stay under Groq free-tier rate limits (8K tokens/minute, 200K tokens/day)
with zero changes to src/llm.py.

PacedClient duck-types client.chat.completions.create(**kwargs), the only
method src.llm._create ever calls, so it slots in wherever plan_trip accepts
``client: OpenAI | None``. TokenBucket paces the per-minute limit by sleeping;
the daily limit stops the campaign cleanly instead of burning retries on a
429 that will not clear until tomorrow.
"""

import json
import time
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace

from openai import RateLimitError

_HEADROOM = 0.9  # fraction of --tpm actually usable, leaving room for estimate error
_COMPLETION_ALLOWANCE = 500  # tokens reserved for the response, corrected after the call
_CHARS_PER_TOKEN = 3.5

_DAILY_MARKERS = ("tokens per day", "(tpd)")
_MINUTE_MARKERS = ("tokens per minute", "(tpm)")


class DailyLimitReached(RuntimeError):
    """The daily token budget is spent. Not an OpenAIError and not in
    llm.TRANSIENT, so tenacity never retries it: it propagates straight
    through plan_trip (which still writes the partial trace in its finally)
    to the runner, which stops cleanly instead of burning retries on a 429
    that will not clear until tomorrow.
    """


def _is_daily(exc: BaseException) -> bool:
    """Groq phrases a per-day and a per-minute 429 the same way apart from
    the unit. Minute markers are checked first: misreading a per-minute 429
    as daily would abort a campaign that only needed to wait a minute.
    """
    text = str(exc).lower()
    if any(marker in text for marker in _MINUTE_MARKERS):
        return False
    return any(marker in text for marker in _DAILY_MARKERS)


class TokenBucket:
    """A sliding 60-second window of (timestamp, tokens) reservations.

    clock/sleep are injectable so a test can drive the window without a real
    wait.
    """

    def __init__(
        self,
        tpm: int,
        daily_token_budget: int,
        *,
        daily_used: int = 0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limit = tpm * _HEADROOM
        self.daily_token_budget = daily_token_budget
        self.daily_used = daily_used
        self._clock = clock
        self._sleep = sleep
        self._window: deque[list] = deque()  # each entry: [timestamp, tokens], mutated by record()
        self._last_entry: list | None = None

    @staticmethod
    def estimate(messages: list[dict], tools: list[dict] | None = None) -> int:
        size = len(json.dumps(messages))
        if tools:
            size += len(json.dumps(tools))
        return int(size / _CHARS_PER_TOKEN) + _COMPLETION_ALLOWANCE

    def _evict_stale(self) -> None:
        cutoff = self._clock() - 60
        while self._window and self._window[0][0] <= cutoff:
            self._window.popleft()

    def _window_total(self) -> int:
        return sum(tokens for _, tokens in self._window)

    def reserve(self, estimated_tokens: int) -> None:
        """Block (via sleep) until the 60s window has room, then reserve.

        Raises DailyLimitReached pre-emptively if this call alone would push
        today's total past the budget -- the real 429 is only the backstop.
        """
        if self.daily_used + estimated_tokens > self.daily_token_budget:
            raise DailyLimitReached(
                f"estimated {estimated_tokens} tokens would push today's total past "
                f"the {self.daily_token_budget}-token daily budget"
            )

        while True:
            self._evict_stale()
            if self._window_total() + estimated_tokens <= self.limit:
                break
            wait = (self._window[0][0] + 60) - self._clock()
            if wait > 0:
                self._sleep(wait)

        entry = [self._clock(), estimated_tokens]
        self._window.append(entry)
        self._last_entry = entry

    def record(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Correct the most recent reservation to what the call actually cost."""
        if self._last_entry is not None and self._last_entry[1] == estimated_tokens:
            self._last_entry[1] = actual_tokens
        self.daily_used += actual_tokens
        self._last_entry = None


class PacedClient:
    """Duck-types an OpenAI client: client.chat.completions.create(**kwargs)."""

    def __init__(self, inner, bucket: TokenBucket) -> None:
        self.inner = inner
        self.bucket = bucket
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        estimated = self.bucket.estimate(kwargs.get("messages", []), kwargs.get("tools"))
        self.bucket.reserve(estimated)
        try:
            response = self.inner.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            if _is_daily(exc):
                raise DailyLimitReached(str(exc)) from exc
            raise  # a per-minute 429: src/llm.py's own Retry-After logic handles it
        usage = getattr(response, "usage", None)
        actual = usage.total_tokens if usage is not None else estimated
        self.bucket.record(actual, estimated)
        return response
