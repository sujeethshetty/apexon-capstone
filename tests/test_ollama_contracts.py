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
                "answer": "Employees get 24 working days of annual leave. [S1]",
                "claims": [
                    {
                        "text": "Employees receive 24 working days of annual leave.",
                        "quote": "Employees receive 24 working days of annual leave.",
                        "citation_id": "S1",
                    }
                ],
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
    assert result["answer"] == "Employees get 24 working days of annual leave. [S1]"
    assert provider._chat.call_args_list[2].args[1]["answer"] == result["answer"]
    assert (
        provider._chat.call_args_list[2].args[1]["source_context"][0]["content"] == evidence().text
    )
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


@pytest.mark.parametrize(
    "answer",
    [
        "Employees get annual leave.",
        "Employees get 24 days. [S99]",
        "Employees get 99 days. [S1]",
        "Employees get 24 days. [S1] [Secret]",
    ],
)
def test_generated_citations_and_numbers_fail_before_model_review(answer):
    provider = OllamaProvider()
    quote = evidence().text.split("\n", 1)[1]
    provider._chat = Mock(
        side_effect=[
            {"queries": ["annual leave"]},
            {"answer": answer, "claims": [{"text": quote, "quote": quote, "citation_id": "S1"}]},
            {"approved": True},
        ]
    )
    store = Mock()
    store.search.return_value = [evidence()]
    result = build_graph(store, provider).invoke({"question": "annual leave"})
    assert result["status"] == "unverified"
    assert not result["claims"] and not result["draft_answer"]
    assert answer not in result["answer"]
    assert provider._chat.call_count == 2


def test_reviewer_rejects_omitted_qualification_and_draft_does_not_leak():
    provider = OllamaProvider()
    source = evidence()
    quote = source.text.split("\n", 1)[1]
    generated = "Everyone gets 24 working days regardless of their employment terms. [S1]"
    provider._chat = Mock(
        side_effect=[
            {"queries": ["annual leave"]},
            {"answer": generated, "claims": [{"text": quote, "quote": quote, "citation_id": "S1"}]},
            {"approved": False},
        ]
    )
    store = Mock()
    store.search.return_value = [source]
    result = build_graph(store, provider).invoke({"question": "annual leave"})
    assert result["status"] == "unverified" and not result["claims"]
    assert not result["draft_answer"] and generated not in repr(result)
    review = provider._chat.call_args_list[2].args
    assert review[1]["answer"] == generated
    assert review[1]["source_context"][0]["content"] == source.text
    assert "omitted material" in review[0]


@pytest.mark.parametrize("answer", [None, 5, "x" * 1201, "word " * 161])
def test_generation_contract_rejects_unbounded_or_non_text_answer(answer):
    provider = OllamaProvider()
    provider._chat = Mock(return_value={"answer": answer, "claims": []})
    with pytest.raises(ValueError):
        provider.answer("annual leave", [evidence()])


def test_missing_support_abstains_instead_of_publishing_generated_answer():
    provider = OllamaProvider()
    provider._chat = Mock(return_value={"answer": "An invented answer. [S1]", "claims": []})
    with pytest.raises(ValueError):
        provider.answer("annual leave", [evidence()])


def test_reviewer_failure_does_not_publish_draft():
    provider = OllamaProvider()
    quote = evidence().text.split("\n", 1)[1]
    provider._chat = Mock(
        side_effect=[
            {"queries": ["annual leave"]},
            {
                "answer": "Employees get 24 days of leave. [S1]",
                "claims": [{"text": quote, "quote": quote, "citation_id": "S1"}],
            },
            ValueError("private review error"),
        ]
    )
    store = Mock()
    store.search.return_value = [evidence()]
    result = build_graph(store, provider).invoke({"question": "annual leave"})
    assert result["status"] == "error" and not result["claims"] and not result["draft_answer"]
    assert "Employees get" not in result["answer"] and "private review error" not in repr(result)
