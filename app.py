"""Run with: uv run --locked streamlit run app.py"""

import os
import time
from dataclasses import asdict

import streamlit as st
from dotenv import load_dotenv

from cp01.cli import make_provider
from cp01.config import APIConfig
from cp01.corpus import load_documents
from cp01.store import KnowledgeStore
from cp01.workflow import build_graph

load_dotenv()
st.set_page_config(
    page_title="Northstar · Knowledge",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

DB = os.getenv("CP01_DB_PATH", ".chroma")
DOCS = os.getenv("CP01_DOCS_PATH", "data/policies")
UNAVAILABLE = "Chat is temporarily unavailable. Please contact your administrator."


@st.cache_resource
def get_store(path):
    return KnowledgeStore(path)


def reset_conversation():
    for key in ("result", "last_question", "answer_mode", "conversation"):
        st.session_state.pop(key, None)


config, config_error = None, ""
try:
    config = APIConfig.from_env()
except ValueError as exc:
    config_error = str(exc)
hosted_ready = config is not None and bool(config.base_url)
store, documents, store_error = None, [], ""
try:
    store = get_store(DB)
    documents = load_documents(DOCS)
except Exception as exc:
    store_error = f"Policy library could not initialize ({type(exc).__name__})."
library_ready = store is not None and not store_error and store.collection.count() > 0

with st.sidebar:
    st.markdown("### northstar")
    st.caption("Workplace knowledge")
    st.button("New chat", use_container_width=True, on_click=reset_conversation)
    page = st.radio("Workspace", ["Chat", "Policy library"], label_visibility="collapsed")
    st.divider()
    with st.expander("Settings"):
        st.caption("Administrator setup")
        st.write("Model configured" if hosted_ready else "Model connection needed")
        st.caption(
            "Configure HOSTED_LLM_BASE_URL and HOSTED_LLM_MODEL in the server's private .env, "
            "then restart the app. The endpoint must support Ollama's /api/chat contract."
        )
        if config_error:
            st.caption(config_error)
        if store_error:
            st.caption(store_error)
        st.caption("Policy library ready" if library_ready else "Policy library needs preparation")
        st.caption("Initial preparation downloads a free search model (about 80 MB).")
        if st.button("Refresh policy library", disabled=store is None or bool(store_error)):
            try:
                with st.spinner("Preparing policies…"):
                    store.ingest(DOCS)
                st.rerun()
            except Exception as exc:
                st.error(
                    f"Policy preparation failed ({type(exc).__name__}). Check the model download and retry."
                )
        st.caption("Advanced / developer options")
        default_mode = 1 if os.getenv("CP01_MODE", "hosted") == "extractive" else 0
        mode_label = st.selectbox(
            "Answer mode",
            ["Hosted LLM API", "Extractive demo"],
            index=default_mode,
            key="mode_choice",
            on_change=reset_conversation,
        )
        st.caption(
            "Extractive mode is a developer tool. It copies policy text and does not generate AI answers."
        )
        show_diagnostics = st.checkbox("Show technical details", value=False)
        technical_overview = st.checkbox("Show architecture", value=False)
        if config:
            st.text(f"Model: {config.model}")
            st.caption(f"Paid fallback: {config.fallback_provider}")
        st.caption(
            "Hosted mode sends questions and policy excerpts to your configured model server. "
            "Use synthetic data only with a plain HTTP server. Paid fallback is separately opt-in "
            "through private configuration and may incur charges."
        )
    st.caption("Sample policies · Northstar Labs")

mode = "hosted" if mode_label == "Hosted LLM API" else "extractive"
ready = library_ready and (mode == "extractive" or hosted_ready)
if technical_overview:
    page = "How it works"

if page == "Chat":
    st.title("How can I help?")
    st.caption("Ask about your workplace policies.")
    if not ready and not st.session_state.get("conversation"):
        st.caption(UNAVAILABLE)
    if mode == "hosted" and hosted_ready and config.fallback_provider != "none":
        st.warning(
            f"Paid {config.fallback_provider} fallback is enabled. "
            "Questions and excerpts may be sent to that provider and incur API charges."
        )
    question = st.chat_input("Message Northstar", max_chars=1000)
    if question:
        started = time.monotonic()
        if not ready:
            result = {"status": "unavailable", "answer": UNAVAILABLE, "claims": [], "evidence": []}
        else:
            with st.spinner("Thinking…"):
                try:
                    if store.needs_refresh(DOCS):
                        store.ingest(DOCS)
                    result = build_graph(store, make_provider(mode)).invoke(
                        {"question": question, "trace": []}
                    )
                except Exception as exc:
                    result = {
                        "status": "error",
                        "answer": "I couldn't complete that request. Please try again or contact your administrator.",
                        "error": type(exc).__name__,
                        "claims": [],
                        "evidence": [],
                    }
        result["elapsed"] = round(time.monotonic() - started, 2)
        st.session_state.result = result
        st.session_state.last_question = question
        st.session_state.answer_mode = mode
        st.session_state.setdefault("conversation", []).append(
            {"question": question, "result": result, "mode": mode}
        )
        st.rerun()

    for turn in st.session_state.get("conversation", []):
        result = turn["result"]
        with st.chat_message("user"):
            st.text(turn["question"])
        with st.chat_message("assistant"):
            if result["status"] == "error":
                st.text(
                    "I couldn't complete that request. Please try again or contact your administrator."
                )
            else:
                # Plain text keeps source/model content from creating HTML or active links.
                st.text(result.get("answer", "I couldn't find an answer in the current policies."))
            if turn["mode"] == "extractive" and result.get("claims"):
                st.caption("Policy extract · not AI-generated")
            cited = {claim.citation_id for claim in result.get("claims", [])}
            if cited:
                with st.expander("Sources"):
                    for evidence in result.get("evidence", []):
                        if evidence.citation_id not in cited:
                            continue
                        meta = evidence.metadata
                        st.caption(
                            f"[{evidence.citation_id}] {meta.get('title', 'Policy')} · v{meta.get('version', '')}"
                        )
                        st.text(evidence.text)
                    st.caption(
                        "Check the full policy for applicability. Source checks and model review can miss errors."
                    )
            if show_diagnostics:
                with st.expander("Technical details"):
                    st.json(
                        {
                            **result,
                            "evidence": [asdict(e) for e in result.get("evidence", [])],
                            "claims": [asdict(c) for c in result.get("claims", [])],
                        }
                    )

elif page == "Policy library":
    st.title("The policy library")
    st.caption(
        "Original synthetic Northstar Labs documents. Superseded policies remain visible for audit, but are excluded from retrieval."
    )
    show_archived = st.toggle("Show superseded policies", value=False)
    categories = sorted({doc.category for doc in documents})
    selected_category = st.selectbox("Category", ["All categories"] + categories)
    for doc in documents:
        if (doc.status == "superseded" and not show_archived) or (
            selected_category != "All categories" and doc.category != selected_category
        ):
            continue
        with st.expander(f"{doc.title} · v{doc.version} · {doc.status.upper()}"):
            st.caption(f"{doc.document_id} · Effective {doc.effective_date} · {doc.category}")
            st.text(doc.content)
    if st.button("Refresh index from local policy files"):
        try:
            with st.spinner("Refreshing local embeddings…"):
                summary = store.ingest(DOCS)
            st.success(f"Refresh complete: {summary['chunks']} chunks. Stale chunks removed.")
        except Exception as exc:
            st.error(
                f"Index refresh failed ({type(exc).__name__}). Existing files have not been changed."
            )
else:
    st.title("How the answer earns trust")
    for title, description in [
        (
            "01 · Retrieval agent",
            "In LLM mode a role-scoped planner decomposes the question into up to three searches. Real local ONNX MiniLM vectors and persistent Chroma retrieve current sources and apply a relevance gate.",
        ),
        (
            "02 · Answer agent",
            "Generates a concise natural-language answer with citations and exact supporting quotes using the configured hosted model. Extractive developer mode copies policy sections and is not AI generation. Source text is untrusted data.",
        ),
        (
            "03 · Verification agent",
            "Checks citations, current versions, exact supporting quotes, lexical relevance, and numeric support. An independent reviewer checks the generated answer against its quotes and full source context, including material qualifications. Failed checks reject the entire draft; model review is not a guarantee of factual accuracy.",
        ),
        (
            "04 · Conditional safe response",
            "Missing evidence, invalid input, provider errors, and failed citations take explicit LangGraph branches. Transient errors use bounded retries. Paid API fallback runs only if explicitly enabled in private configuration; actual provider use is shown. Schema, authentication, and grounding failures never trigger fallback.",
        ),
    ]:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)
    st.warning(
        "Capstone prototype: no production authentication or role-based document access. Do not add real confidential data or expose this app publicly without hardening. A cited excerpt can still be incomplete or poorly selected; use the full policy as authority."
    )
