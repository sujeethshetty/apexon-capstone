"""Three bounded specialist roles orchestrated by an actual LangGraph StateGraph."""

import time
import re
from dataclasses import replace

from langgraph.graph import END, START, StateGraph

from .models import AnswerDraft, AssistantState
from .providers import terms, absence_excerpt
from .transport import ProviderFailure

MIN_RETRIEVAL_SCORE = 0.28  # Heuristic cosine similarity, NOT a probability.
UNKNOWN = "I couldn't find a supported answer in the current policies. Try a more specific question or ask the relevant policy owner."


def normalize(text: str) -> str:
    return " ".join(text.split())


def build_graph(store, provider):
    def provider_metadata():
        transport = getattr(provider, "transport", None)
        return {
            "provider_used": getattr(
                provider, "active_name", getattr(provider, "name", "Configured provider")
            ),
            "provider_events": list(transport.events) if transport else [],
        }

    def event(state, agent, detail, started):
        return state.get("trace", []) + [
            {
                "agent": agent,
                "detail": detail,
                "provider": getattr(
                    provider, "active_name", getattr(provider, "name", "Configured provider")
                ),
                "duration_ms": round((time.monotonic() - started) * 1000),
            }
        ]

    def retrieve(state: AssistantState):
        if hasattr(provider, "start_question"):
            provider.start_question()
        started = time.monotonic()
        question = state.get("question", "").strip()
        if not question or len(question) > 1000:
            return {
                "status": "invalid",
                "error": "Enter a question between 1 and 1,000 characters.",
                "evidence": [],
                "trace": event(state, "Retrieval agent", "Input rejected", started),
            }
        try:
            queries = provider.plan(question) if hasattr(provider, "plan") else [question]
            by_chunk = {}
            for query in queries:
                for item in store.search(query, k=8):
                    if item.chunk_id not in by_chunk or item.score > by_chunk[item.chunk_id].score:
                        by_chunk[item.chunk_id] = item
            ranked = sorted(by_chunk.values(), key=lambda e: e.score, reverse=True)[:12]
            evidence = [replace(e, citation_id=f"S{i}") for i, e in enumerate(ranked, 1)]
            eligible = [e for e in evidence if e.score >= MIN_RETRIEVAL_SCORE]
            # Dense similarity alone may match unrelated enterprise questions. Require a lexical anchor.
            anchored = any(terms(question) & terms(e.text) for e in eligible)
            status = "retrieved" if eligible and anchored else "unknown"
            return {
                "question": question,
                "evidence": eligible if anchored else [],
                "status": status,
                "trace": event(
                    state,
                    "Retrieval agent",
                    f"{len(queries)} search query(s); {len(eligible) if anchored else 0} current-policy chunks passed relevance gate",
                    started,
                ),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc)
                if isinstance(exc, ProviderFailure)
                else f"Retrieval unavailable ({type(exc).__name__}). Rebuild the local index and retry.",
                "evidence": [],
                "trace": event(state, "Retrieval agent", "Retrieval failed safely", started),
            }

    def answer(state: AssistantState):
        started = time.monotonic()
        try:
            draft = provider.answer(state["question"], state["evidence"])
            claims = draft.claims if isinstance(draft, AnswerDraft) else draft
            return {
                "claims": claims,
                "draft_answer": draft.answer if isinstance(draft, AnswerDraft) else "",
                "status": "drafted" if claims else "unknown",
                "trace": event(
                    state, "Answer agent", f"Selected {len(claims)} candidate excerpts", started
                ),
            }
        except Exception as exc:
            return {
                "claims": [],
                "draft_answer": "",
                "status": "error",
                "error": str(exc)
                if isinstance(exc, ProviderFailure)
                else f"Answer provider unavailable or invalid ({type(exc).__name__}). Check the model service or select extractive demo mode.",
                "trace": event(
                    state, "Answer agent", "Provider failed safely; no automatic fallback", started
                ),
            }

    def verify(state: AssistantState):
        started = time.monotonic()
        sources = {e.citation_id: e for e in state["evidence"]}
        valid = []
        for claim in state.get("claims", []):
            source = sources.get(claim.citation_id)
            if (
                source
                and source.metadata.get("status") == "current"
                and normalize(claim.quote) in normalize(source.text)
                and normalize(claim.text) == normalize(claim.quote)
                and len(normalize(claim.quote)) >= 12
                and terms(state["question"]) & terms(source.text)
            ):
                valid.append(claim)
        # Fail the whole draft instead of silently dropping unsupported claims.
        accepted = bool(valid) and len(valid) == len(state.get("claims", []))
        generated = state.get("draft_answer", "")
        if generated:
            citations = re.findall(r"\[([^\]]*)\]", generated)
            supported = {claim.citation_id for claim in valid}
            numbers = {
                token.replace(",", "")
                for token in re.findall(r"\b\d+(?:[.,]\d+)*\b", re.sub(r"\[S\d+\]", "", generated))
            }
            source_numbers = {
                token.replace(",", "")
                for token in re.findall(r"\b\d+(?:[.,]\d+)*\b", " ".join(c.quote for c in valid))
            }
            accepted = (
                accepted
                and bool(citations)
                and set(citations) <= supported
                and numbers <= source_numbers
                and hasattr(provider, "review")
            )
        if accepted and hasattr(provider, "review"):
            try:
                accepted = provider.review(state["question"], valid, generated, state["evidence"])
            except Exception as exc:
                return {
                    "claims": [],
                    "draft_answer": "",
                    "status": "error",
                    "error": str(exc)
                    if isinstance(exc, ProviderFailure)
                    else f"Evidence reviewer unavailable ({type(exc).__name__}). No answer published.",
                    "trace": event(state, "Verification agent", "Reviewer failed safely", started),
                }
        missing = absence_excerpt(state["question"], state["evidence"])
        unavailable = accepted and missing and any(c.quote == missing.quote for c in valid)
        return {
            **provider_metadata(),
            "claims": valid if accepted else [],
            "draft_answer": "",
            "status": "limited" if unavailable else "answered" if accepted else "unverified",
            "verification": "Exact supporting quotes, current sources, citations and numeric support checked. Generated answers also require independent model review; this is not a factual-accuracy guarantee.",
            "answer": (generated or "\n\n".join(f"{c.text} [{c.citation_id}]" for c in valid))
            if accepted
            else "",
            "trace": event(
                state,
                "Verification agent",
                "All excerpts passed evidence checks"
                if accepted
                else "Draft rejected: unsupported or irrelevant citation",
                started,
            ),
        }

    def abstain(state: AssistantState):
        started = time.monotonic()
        status = state.get("status", "unknown")
        if status in {"invalid", "error"}:
            answer = state["error"]
        elif status == "unverified":
            answer = "The draft did not pass citation checks, so I won't present it as an answer. Please rephrase your question."
        else:
            answer = UNKNOWN
        return {
            **provider_metadata(),
            "answer": answer,
            "draft_answer": "",
            "claims": [],
            "trace": event(state, "Safe response", f"Stopped with status: {status}", started),
        }

    graph = StateGraph(AssistantState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("verify", verify)
    graph.add_node("abstain", abstain)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda s: "answer" if s["status"] == "retrieved" else "abstain",
        {"answer": "answer", "abstain": "abstain"},
    )
    graph.add_conditional_edges(
        "answer",
        lambda s: "verify" if s["status"] == "drafted" else "abstain",
        {"verify": "verify", "abstain": "abstain"},
    )
    graph.add_conditional_edges(
        "verify",
        lambda s: END if s["status"] in {"answered", "limited"} else "abstain",
        {END: END, "abstain": "abstain"},
    )
    graph.add_edge("abstain", END)
    return graph.compile()
