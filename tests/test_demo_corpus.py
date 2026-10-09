"""Original sample-corpus integrity; no network or embedding-model download required."""

import json
from pathlib import Path

from cp01.corpus import chunk_documents, load_documents

ROOT = Path(__file__).resolve().parents[1]


def test_original_corpus_has_current_and_archived_policies():
    docs = load_documents(ROOT / "data/policies")
    assert len(docs) == 10
    assert sum(d.status == "current" for d in docs) == 9
    assert sum(d.status == "superseded" for d in docs) == 1
    assert all("fictional" in d.content.lower() for d in docs)
    assert len({(d.document_id, d.version) for d in docs}) == len(docs)


def test_demo_leave_values_are_distinct_and_consistent():
    docs = {d.document_id: d for d in load_documents(ROOT / "data/policies")}
    assert "24 working days" in docs["NS-HR-001"].content
    assert "31 March" in docs["NS-HR-001"].content
    assert "20 working days" in docs["NS-HR-001-OLD"].content
    assert docs["NS-HR-001-OLD"].status == "superseded"
    assert "USD 300" in docs["NS-HR-002"].content
    assert "USD 300" in docs["NS-HR-003"].content


def test_malicious_example_is_retained_but_quarantined_from_index():
    docs = load_documents(ROOT / "data/policies")
    governance = next(d for d in docs if d.document_id == "NS-KB-001")
    assert "99 days" in governance.content
    chunks = chunk_documents(docs)
    assert not any("99 days" in chunk.text for chunk in chunks)
    assert not any("IGNORE ALL PREVIOUS INSTRUCTIONS" in chunk.text for chunk in chunks)
    assert any(c.metadata["document_id"] == "NS-KB-001" for c in chunks)


def test_known_answer_cases_have_existing_sources_and_diverse_coverage():
    cases = json.loads((ROOT / "data/evaluation_cases.json").read_text())
    docs = {d.document_id: d for d in load_documents(ROOT / "data/policies")}
    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == 20
    tags = set()
    for case in cases:
        assert case["question"] and case["expected_claims"]
        assert set(case["expected_source_ids"]) <= docs.keys()
        assert all(docs[identity].status == "current" for identity in case["expected_source_ids"])
        assert isinstance(case["answerable"], bool)
        for identity, excerpt in case["source_excerpts"].items():
            identity = identity.removesuffix("_scope")
            assert excerpt in docs[identity].content
        tags.update(case["tags"])
    assert {"multi-document", "unknown", "stale-version", "prompt-injection"} <= tags
    assert sum(not case["answerable"] for case in cases) == 3
