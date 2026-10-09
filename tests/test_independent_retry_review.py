"""Independent retry/fallback verification: mocked network only, no billable calls."""

import json

import httpx
import pytest

from cp01.config import APIConfig
from cp01.fallback import PaidAPIProvider, ResilientProvider
from cp01.providers import OllamaProvider
from cp01.transport import (
    BudgetExhausted,
    RetryingTransport,
    RetryPolicy,
    TerminalProviderFailure,
    TransientExhausted,
    retry_after_seconds,
)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 301, 302])
def test_terminal_status_never_retries_or_falls_back(status):
    calls = []

    def sender(url, headers, payload, timeout):
        calls.append(url)
        return httpx.Response(status, json={"error": "secret must not appear"})

    transport = RetryingTransport(sender=sender, sleeper=lambda _: None)
    provider = ResilientProvider(
        OllamaProvider(transport=transport),
        PaidAPIProvider("openai", "private-marker", "model", transport),
    )
    with pytest.raises(TerminalProviderFailure) as caught:
        provider.plan("leave")
    assert len(calls) == 1
    assert "secret" not in str(caught.value)
    assert "private-marker" not in repr(transport.events)


def test_fallback_credentials_only_reach_exact_official_host_and_trace():
    calls = []

    def sender(url, headers, payload, timeout):
        calls.append((url, headers))
        if len(calls) <= 3:
            return httpx.Response(503)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"queries": ["leave"]})}}]}
        )

    transport = RetryingTransport(sender=sender, sleeper=lambda _: None)
    provider = ResilientProvider(
        OllamaProvider("https://primary.example", transport=transport),
        PaidAPIProvider("openai", "private-marker", "model", transport),
    )
    assert provider.plan("leave") == ["leave"]
    assert all(not headers for _, headers in calls[:3])
    assert calls[3][0] == "https://api.openai.com/v1/chat/completions"
    assert calls[3][1]["Authorization"] == "Bearer private-marker"
    assert provider.active_name == "OpenAI API fallback"
    assert "private-marker" not in repr(transport.events)
    assert transport.budget.attempts == 4


def test_shared_budget_no_nested_retry_multiplication():
    calls = []

    def sender(*args):
        calls.append(1)
        return httpx.Response(503)

    transport = RetryingTransport(
        RetryPolicy(total_attempts=4), sender=sender, sleeper=lambda _: None
    )
    provider = ResilientProvider(
        OllamaProvider(transport=transport),
        PaidAPIProvider("openai", "private-marker", "model", transport),
    )
    with pytest.raises(BudgetExhausted):
        provider.plan("leave")
    assert len(calls) == 4


def test_retry_after_cap_jitter_and_whole_deadline():
    now, sleeps = [0.0], []

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    transport = RetryingTransport(
        sender=lambda *args: httpx.Response(429, headers={"Retry-After": "10000"}),
        sleeper=sleep,
        clock=lambda: now[0],
    )
    with pytest.raises(TransientExhausted):
        transport.post("test", "https://example.invalid", payload={})
    assert sleeps == [8, 8]
    assert retry_after_seconds("nan", 8) is None
    assert retry_after_seconds("Fri, 09 Oct 2099 10:00:00 GMT", 8) == 8
    now[0] = 0
    sleeps.clear()
    transport = RetryingTransport(
        sender=lambda *args: httpx.Response(503),
        sleeper=sleep,
        jitter=lambda: 0.5,
        clock=lambda: now[0],
    )
    with pytest.raises(TransientExhausted):
        transport.post("test", "https://example.invalid", payload={})
    assert sleeps == [0.75, 1.5]
    now[0] = 0
    transport = RetryingTransport(
        RetryPolicy(total_seconds=1),
        sender=lambda *args: httpx.Response(429, headers={"Retry-After": "8"}),
        sleeper=sleep,
        clock=lambda: now[0],
    )
    with pytest.raises(BudgetExhausted):
        transport.post("test", "https://example.invalid", payload={})
    assert transport.budget.attempts == 1


def test_schema_invalid_primary_cannot_trigger_cloud_fallback():
    calls = []

    def sender(*args):
        calls.append(1)
        return httpx.Response(200, json={"message": {"content": '{"queries": []}'}})

    transport = RetryingTransport(sender=sender)
    provider = ResilientProvider(
        OllamaProvider(transport=transport),
        PaidAPIProvider("openai", "private-marker", "model", transport),
    )
    with pytest.raises(ValueError):
        provider.plan("leave")
    assert len(calls) == 1
    assert provider.active is provider.primary


def test_fallback_disabled_despite_present_key_and_key_repr_redacted(monkeypatch):
    monkeypatch.delenv("FALLBACK_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "private-marker")
    config = APIConfig.from_env()
    assert config.fallback_provider == "none"
    assert config.fallback_key == ""
    assert "private-marker" not in repr(
        APIConfig("https://primary.example", fallback_key="private-marker")
    )


def test_claude_endpoint_uses_only_its_key_and_forced_schema():
    calls = []

    def sender(url, headers, payload, timeout):
        calls.append((url, headers, payload))
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "tool_use", "name": "submit_result", "input": {"approved": True}}
                ]
            },
        )

    transport = RetryingTransport(sender=sender)
    provider = PaidAPIProvider("anthropic", "claude-private-marker", "test-model", transport)
    assert provider.review("leave", []) is True
    url, headers, payload = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "claude-private-marker"
    assert "Authorization" not in headers
    assert payload["tool_choice"] == {"type": "tool", "name": "submit_result"}
    assert payload["tools"][0]["input_schema"]["properties"]["approved"] == {"type": "boolean"}
    assert "claude-private-marker" not in repr(transport.events)


def test_live_sender_disables_redirects_and_absolute_wait_timeout(monkeypatch):
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            import asyncio

            await asyncio.sleep(0.5)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    transport = RetryingTransport()
    with pytest.raises(httpx.ReadTimeout):
        transport._send("https://example.invalid", {}, {}, 0.01)
    assert observed["follow_redirects"] is False
    assert observed["timeout"].connect == 0.01
