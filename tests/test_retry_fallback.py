import json
from unittest.mock import Mock

import httpx
import pytest

from cp01.config import APIConfig
from cp01.fallback import PaidAPIProvider, ResilientProvider
from cp01.models import Evidence
from cp01.providers import OllamaProvider
from cp01.transport import (
    BudgetExhausted,
    RetryingTransport,
    RetryPolicy,
    TerminalProviderFailure,
    TransientExhausted,
    retry_after_seconds,
)
from cp01.workflow import build_graph


def response(status=200, body=None, headers=None):
    return httpx.Response(status, json=body or {"ok": True}, headers=headers)


def transport(outcomes, **kwargs):
    sender = Mock(side_effect=outcomes)
    sleeps = []
    instance = RetryingTransport(sender=sender, sleeper=sleeps.append, jitter=lambda: 0, **kwargs)
    return instance, sender, sleeps


def test_transient_success_uses_exponential_backoff():
    t, sender, sleeps = transport(
        [response(503), httpx.ReadTimeout("private-error-body"), response()]
    )
    assert t.post("Hosted", "https://example.invalid", payload={}) == {"ok": True}
    assert sender.call_count == 3 and sleeps == [0.5, 1.0]
    assert "private-error-body" not in repr(t.events)


def test_retry_after_is_capped_and_terminal_errors_never_retry():
    t, sender, sleeps = transport([response(429, headers={"Retry-After": "999"}), response()])
    t.post("Hosted", "https://example.invalid", payload={})
    assert sleeps == [8.0]
    assert retry_after_seconds("invalid", 8) is None
    for status in (400, 401, 403, 404, 422):
        t, sender, sleeps = transport([response(status, {"error": "private key secret"})])
        with pytest.raises(TerminalProviderFailure) as error:
            t.post("Hosted", "https://example.invalid", payload={})
        assert sender.call_count == 1 and not sleeps
        assert "private key" not in str(error.value)


def test_shared_attempt_budget_cannot_be_exceeded():
    t, sender, _ = transport([response(503)] * 5, policy=RetryPolicy(total_attempts=2))
    with pytest.raises(BudgetExhausted):
        t.post("Hosted", "https://example.invalid", payload={})
    assert sender.call_count == 2


def test_shared_time_budget_stops_before_long_wait():
    t, sender, sleeps = transport(
        [response(429, headers={"Retry-After": "8"})], policy=RetryPolicy(total_seconds=1)
    )
    with pytest.raises(BudgetExhausted):
        t.post("Hosted", "https://example.invalid", payload={})
    assert sender.call_count == 1 and sleeps == []


def test_full_graph_fallback_uses_shared_budget_and_reports_actual_provider():
    quote = "Employees receive 24 working days of annual leave."

    def openai(data):
        return response(body={"choices": [{"message": {"content": json.dumps(data)}}]})

    t, sender, _ = transport(
        [response(503)] * 3
        + [
            openai({"queries": ["annual leave"]}),
            openai({"claims": [{"text": quote, "quote": quote, "citation_id": "S1"}]}),
            openai({"approved": True}),
        ]
    )
    p = ResilientProvider(
        OllamaProvider("https://course.invalid", transport=t),
        PaidAPIProvider("openai", "fake-key-only", "gpt-4.1-mini", t),
    )
    store = Mock()
    store.search.return_value = [Evidence("S1", "chunk", quote, {"status": "current"}, 0.9)]
    result = build_graph(store, p).invoke({"question": "annual leave", "trace": []})
    assert result["status"] == "answered" and result["provider_used"] == "OpenAI API fallback"
    assert sender.call_count == 6 and t.budget.attempts == 6
    assert sender.call_args_list[3].args[0] == "https://api.openai.com/v1/chat/completions"
    assert sender.call_args_list[3].args[1]["Authorization"] == "Bearer fake-key-only"
    assert "fake-key-only" not in repr(result)


@pytest.mark.parametrize(
    "provider_name,body",
    [
        ("openai", {"choices": [{"message": {"content": '{"queries":["annual leave"]}'}}]}),
        (
            "anthropic",
            {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "submit_result",
                        "input": {"queries": ["annual leave"]},
                    }
                ]
            },
        ),
    ],
)
def test_paid_provider_contracts_without_real_calls(provider_name, body):
    t, sender, _ = transport([response(body=body)])
    p = PaidAPIProvider(provider_name, "fake-key-only", "configured-model", t)
    assert p.plan("annual leave") == ["annual leave"]
    url, headers, payload, _ = sender.call_args.args
    assert url.startswith("https://api.")
    if provider_name == "anthropic":
        assert headers["x-api-key"] == "fake-key-only"
        assert payload["tool_choice"] == {"type": "tool", "name": "submit_result"}


@pytest.mark.parametrize(
    "first",
    [
        response(401),
        response(body={"message": {"content": "not JSON"}}),
        response(body={"message": {"content": '{"queries":[]}'}}),
    ],
)
def test_auth_or_schema_failure_does_not_fallback(first):
    t, sender, _ = transport([first])
    p = ResilientProvider(
        OllamaProvider("https://course.invalid", transport=t),
        PaidAPIProvider("openai", "fake-key-only", "test-model", t),
    )
    with pytest.raises((TerminalProviderFailure, ValueError)):
        p.plan("annual leave")
    assert sender.call_count == 1 and p.active is p.primary


def test_disabled_fallback_and_missing_key_are_safe(monkeypatch):
    monkeypatch.setenv("FALLBACK_PROVIDER", "none")
    assert APIConfig.from_env().fallback_provider == "none"
    t, sender, _ = transport([response(503)] * 3)
    p = ResilientProvider(OllamaProvider("https://course.invalid", transport=t))
    with pytest.raises(TransientExhausted):
        p.plan("annual leave")
    assert sender.call_count == 3
    monkeypatch.setenv("FALLBACK_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        APIConfig.from_env()
    assert "fake-key-only" not in repr(
        APIConfig("https://course.invalid", fallback_key="fake-key-only")
    )


def test_reused_graph_resets_budget_events_and_primary_after_idle():
    quote = "Employees receive 24 working days of annual leave."

    def hosted(data):
        return response(body={"message": {"content": json.dumps(data)}})

    outcomes = [
        hosted({"queries": ["annual leave"]}),
        hosted({"claims": [{"text": quote, "quote": quote, "citation_id": "S1"}]}),
        hosted({"approved": True}),
    ] * 2
    now = [0.0]
    t, sender, _ = transport(outcomes, policy=RetryPolicy(total_attempts=3), clock=lambda: now[0])
    p = ResilientProvider(
        OllamaProvider("https://course.invalid", transport=t),
        PaidAPIProvider("openai", "fake-key-only", "test-model", t),
    )
    store = Mock()
    store.search.return_value = [Evidence("S1", "chunk", quote, {"status": "current"}, 0.9)]
    graph = build_graph(store, p)
    first = graph.invoke({"question": "annual leave", "trace": []})
    now[0] = 1000.0  # More than the old deadline has elapsed between questions.
    p.active = p.fallback
    second = graph.invoke({"question": "annual leave", "trace": []})
    assert first["status"] == second["status"] == "answered"
    assert p.active is p.primary and sender.call_count == 6
    assert t.budget.attempts == 3 and len(second["provider_events"]) == 3
