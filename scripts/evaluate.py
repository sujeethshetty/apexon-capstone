"""Run real local retrieval diagnostics; deliberately not a correctness scorer."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cp01.providers import ExtractiveProvider  # noqa: E402
from cp01.store import KnowledgeStore  # noqa: E402
from cp01.workflow import build_graph  # noqa: E402


def normalize(text):
    return " ".join(text.split()).casefold()


def score_case(case, result):
    """Mechanical diagnostics only; semantic expected_claims remain manual rubrics."""
    answer = normalize(result.get("answer", ""))
    evidence = result.get("evidence", [])
    retrieved = {item.metadata["document_id"] for item in evidence}
    by_citation = {item.citation_id: item for item in evidence}
    claims = result.get("claims", [])
    cited = {
        by_citation[c.citation_id].metadata["document_id"]
        for c in claims
        if c.citation_id in by_citation
    }
    expected = set(case["expected_source_ids"])
    excerpt_hits = {key: normalize(text) in answer for key, text in case["source_excerpts"].items()}
    forbidden_hits = [text for text in case["forbidden_claims"] if normalize(text) in answer]
    provenance = all(
        c.citation_id in by_citation
        and by_citation[c.citation_id].metadata["status"] == "current"
        and normalize(c.text) == normalize(c.quote)
        and normalize(c.quote) in normalize(by_citation[c.citation_id].text)
        for c in claims
    )
    return {
        "expected_source_ids": sorted(expected),
        "retrieved_source_ids": sorted(retrieved),
        "cited_source_ids": sorted(cited),
        "retrieved_expected_count": len(expected & retrieved),
        "cited_expected_count": len(expected & cited),
        "expected_source_count": len(expected),
        "any_expected_source_retrieved": bool(expected & retrieved),
        "all_expected_sources_retrieved": expected <= retrieved,
        "all_expected_sources_cited": expected <= cited,
        "source_excerpt_in_answer": excerpt_hits,
        "forbidden_substring_hits": forbidden_hits,
        "claim_count": len(claims),
        "all_returned_claims_have_exact_current_provenance": provenance,
        "missing_fact_safe_status": result.get("status") in {"unknown", "limited"}
        if not case["answerable"]
        else None,
        "semantic_adequacy": "not_automatically_scored",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / ".chroma")
    parser.add_argument("--docs", type=Path, default=ROOT / "data/policies")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/evaluation_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evaluation-results.json")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    store = KnowledgeStore(args.db)
    corpus = store.ingest(args.docs)
    graph = build_graph(store, ExtractiveProvider())
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = graph.invoke({"question": case["question"], "trace": []})
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answerable": case["answerable"],
                "status": result.get("status"),
                "answer": result.get("answer"),
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "diagnostics": score_case(case, result),
                "evidence": [asdict(e) for e in result.get("evidence", [])],
                "claims": [asdict(c) for c in result.get("claims", [])],
                "trace": result.get("trace", []),
            }
        )
    ds = [r["diagnostics"] for r in rows]
    excerpts = [hit for d in ds for hit in d["source_excerpt_in_answer"].values()]
    summary = {
        "cases": len(rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "cases_with_any_expected_source_retrieved": sum(
            d["any_expected_source_retrieved"] for d in ds
        ),
        "cases_with_all_expected_sources_retrieved": sum(
            d["all_expected_sources_retrieved"] for d in ds
        ),
        "cases_with_all_expected_sources_cited": sum(d["all_expected_sources_cited"] for d in ds),
        "expected_sources_retrieved": sum(d["retrieved_expected_count"] for d in ds),
        "expected_sources_total": sum(d["expected_source_count"] for d in ds),
        "exact_reference_excerpts_in_answers": sum(excerpts),
        "reference_excerpts_total": len(excerpts),
        "cases_with_forbidden_substring_hit": sum(bool(d["forbidden_substring_hits"]) for d in ds),
        "cases_with_nonempty_exact_current_claims": sum(
            d["claim_count"] > 0 and d["all_returned_claims_have_exact_current_provenance"]
            for d in ds
        ),
        "missing_fact_cases_safe_status": sum(d["missing_fact_safe_status"] is True for d in ds),
        "missing_fact_cases_total": sum(not c["answerable"] for c in cases),
    }
    output = {
        "captured_at_executor_utc": datetime.now(timezone.utc).isoformat(),
        "provider": ExtractiveProvider.name,
        "live_llm_tested": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: version(p) for p in ["chromadb", "langgraph", "streamlit"]},
        "corpus": corpus,
        "corpus_content_and_metadata_sha256": store.corpus_fingerprint(args.docs),
        "interpretation": "Diagnostic retrieval, literal substring, and provenance counts are not factual accuracy or semantic completeness. Forbidden-substring absence is not safety proof. Curated cases influenced development.",
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results written to {args.output}")
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
