"""Credential-safe HTTP with one bounded retry budget shared by all agent roles."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import time
from typing import Callable

import httpx

TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class ProviderFailure(RuntimeError):
    """Only fixed safe messages are exposed; raw provider bodies are never included."""


class TransientExhausted(ProviderFailure):
    pass


class TerminalProviderFailure(ProviderFailure):
    pass


class BudgetExhausted(ProviderFailure):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    request_timeout: float = 30.0
    connect_timeout: float = 5.0
    base_delay: float = 0.5
    max_delay: float = 8.0
    total_seconds: float = 120.0
    total_attempts: int = 8

    def __post_init__(self):
        if not 1 <= self.attempts <= 4 or not 1 <= self.total_attempts <= 12:
            raise ValueError("Retry attempts exceed permitted bounds.")
        if not 0 < self.connect_timeout <= self.request_timeout <= 60:
            raise ValueError("Request timeout must be positive and at most 60 seconds.")
        if not 0 <= self.base_delay <= self.max_delay <= 10 or not 1 <= self.total_seconds <= 180:
            raise ValueError("Backoff or total time budget exceeds permitted bounds.")


@dataclass
class RequestBudget:
    policy: RetryPolicy
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    started: float = field(init=False)
    attempts: int = 0

    def __post_init__(self):
        self.started = self.clock()

    def remaining(self) -> float:
        return max(0.0, self.policy.total_seconds - (self.clock() - self.started))

    def reserve(self):
        if self.attempts >= self.policy.total_attempts or self.remaining() <= 0:
            raise BudgetExhausted(
                "The question exhausted its bounded API request/time budget. Please retry later."
            )
        self.attempts += 1


def retry_after_seconds(value: str | None, cap: float) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    if seconds != seconds:  # NaN
        return None
    return min(cap, max(0.0, seconds))


class RetryingTransport:
    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        sender=None,
        sleeper=time.sleep,
        jitter=random.random,
        clock=time.monotonic,
    ):
        self.policy = policy or RetryPolicy()
        self.budget = RequestBudget(self.policy, clock)
        self.sender = sender or self._send
        self.sleeper = sleeper
        self.jitter = jitter
        self.events: list[dict] = []

    def reset_question(self):
        """New user invocation, fresh deadline and request allowance; no previous events."""
        self.budget = RequestBudget(self.policy, self.budget.clock)
        self.events.clear()

    def _send(self, url, headers, payload, timeout):
        async def request():
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout, connect=min(self.policy.connect_timeout, timeout)),
                follow_redirects=False,
            ) as client:
                return await client.post(url, headers=headers, json=payload)

        try:
            # Absolute per-attempt deadline, including the complete response body.
            return asyncio.run(asyncio.wait_for(request(), timeout=timeout))
        except TimeoutError:
            raise httpx.ReadTimeout("Bounded request deadline exceeded") from None

    def post(self, provider: str, url: str, *, headers: dict | None = None, payload: dict) -> dict:
        for attempt in range(1, self.policy.attempts + 1):
            self.budget.reserve()
            remaining = self.budget.remaining()
            timeout = min(self.policy.request_timeout, remaining)
            retry_after = None
            reason = "connection"
            try:
                # No SDK retries, redirects, or hidden second request path.
                response = self.sender(url, headers or {}, payload, timeout)
                status = response.status_code
                if status in TRANSIENT_STATUS:
                    reason = "rate_limit" if status == 429 else "temporary_server_error"
                    retry_after = retry_after_seconds(
                        response.headers.get("Retry-After"), self.policy.max_delay
                    )
                elif not 200 <= status < 300:
                    raise TerminalProviderFailure(
                        "The model API rejected the request. Check endpoint, authentication, model access, and request settings. No fallback was used."
                    )
                else:
                    try:
                        result = response.json()
                    except (ValueError, TypeError):
                        raise TerminalProviderFailure(
                            "The model API returned invalid JSON. No retry or fallback was used."
                        ) from None
                    if not isinstance(result, dict):
                        raise TerminalProviderFailure(
                            "The model API returned an invalid response shape. No retry or fallback was used."
                        )
                    self.events.append(
                        {"provider": provider, "attempt": attempt, "outcome": "success"}
                    )
                    return result
            except (
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ):
                reason = "timeout_or_connection"
            except httpx.HTTPError:
                raise TerminalProviderFailure(
                    "The model connection failed with a non-retryable transport error. No fallback was used."
                ) from None
            self.events.append({"provider": provider, "attempt": attempt, "outcome": reason})
            if attempt == self.policy.attempts:
                raise TransientExhausted(
                    "The model API remained temporarily unavailable after bounded retries."
                )
            delay = (
                retry_after
                if retry_after is not None
                else min(
                    self.policy.max_delay,
                    self.policy.base_delay * (2 ** (attempt - 1)) * (1 + self.jitter()),
                )
            )
            if delay >= self.budget.remaining():
                raise BudgetExhausted(
                    "The question exhausted its bounded API time budget. No further requests were made."
                )
            self.sleeper(delay)
        raise AssertionError("Unreachable retry state")
