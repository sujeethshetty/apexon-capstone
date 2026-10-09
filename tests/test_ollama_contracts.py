"""No live model required: validate all three role contracts and safety boundaries."""

from unittest.mock import Mock

import pytest

from cp01.models import Claim, Evidence
from cp01.providers import OllamaProvider
from cp01.workflow import build_graph


def evidence():
    return Evidence(
        "S1",
        "chunk-1",
        "Policy\nEmployees receive 24 working days of annual leave.",
        {"status": "current", "document_id": "TEST"},
        0.8,
    )


def test_three_llm_roles_are_distinct_and_invoked():
    provider = OllamaProvider()
    provider._chat = Mock(
        side_effect=[
            {"queries": ["annual leave allowance"]},
            {
                "claims": [
                    {
                        "text": "Employees receive 24 working days of annual leave.",
                        "quote": "Employees receive 24 working days of annual leave.",
                        "citation_id": "S1",
                    }
                ]
            },
            {"approved": True},
        ]
    )
    store = Mock()
    store.search.return_value = [evidence()]
    result = build_graph(store, provider).invoke(
        {"question": "What annual leave do employees receive?", "trace": []}
    )
    assert result["status"] == "answered"
    assert provider._chat.call_count == 3
    prompts = [call.args[0] for call in provider._chat.call_args_list]
    assert "retrieval planning agent" in prompts[0]
    assert "answer agent" in prompts[1]
    assert "review agent" in prompts[2]


@pytest.mark.parametrize(
    "payload", [{"queries": []}, {"queries": ["q"] * 4}, {"queries": [3]}, {"queries": ["a" * 301]}]
)
def test_planner_rejects_unbounded_or_invalid_queries(payload):
    provider = OllamaProvider()
    provider._chat = Mock(return_value=payload)
    with pytest.raises(ValueError):
        provider.plan("policy question")


@pytest.mark.parametrize("payload", [{"approved": "yes"}, {"approved": 1}, {}])
def test_reviewer_requires_actual_boolean(payload):
    provider = OllamaProvider()
    provider._chat = Mock(return_value=payload)
    with pytest.raises(ValueError):
        provider.review("question", [])


def test_reviewer_cannot_override_invalid_quote():
    provider = OllamaProvider()
    provider.plan = Mock(return_value=["annual leave"])
    provider.answer = Mock(
        return_value=[Claim("Annual leave is 99 days.", "S1", "Annual leave is 99 days.")]
    )
    provider.review = Mock(return_value=True)
    store = Mock()
    store.search.return_value = [evidence()]
    result = build_graph(store, provider).invoke({"question": "annual leave", "trace": []})
    assert result["status"] == "unverified"
    assert not result["claims"]
    provider.review.assert_not_called()


def test_reviewer_can_reject_grounded_but_irrelevant_draft():
    provider = OllamaProvider()
    provider.plan = Mock(return_value=["annual leave"])
    quote = evidence().text.split("\n")[1]
    provider.answer = Mock(return_value=[Claim(quote, "S1", quote)])
    provider.review = Mock(return_value=False)
    store = Mock()
    store.search.return_value = [evidence()]
    result = build_graph(store, provider).invoke({"question": "annual leave", "trace": []})
    assert result["status"] == "unverified"
    assert not result["claims"]
