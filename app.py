"""Run with: streamlit run app.py"""

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
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}
.stApp{background:#f6f8fc;color:#142342}
[data-testid="stSidebar"]{background:#101e38;color:#e4eafa}
[data-testid="stSidebar"] *{color:#e4eafa}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#142342!important}
[data-testid="stSidebar"] [role="combobox"]{color:#142342!important;background:#ffffff!important}
[data-testid="stSidebar"] button{background:#233656;border:1px solid #3b5073;color:#e4eafa}
h1{font-weight:700!important;letter-spacing:-1.5px;font-size:2.8rem!important}
h2,h3{letter-spacing:-.5px}
.block-container{padding-top:2.4rem;max-width:1250px}
[data-testid="stMetric"]{background:white;border:1px solid #e2e8f3;border-radius:12px;padding:15px 20px}
[data-testid="stMetricValue"]{font-size:1.65rem}
[data-testid="stChatInput"]{border-radius:12px}
.stButton>button{border-radius:9px}
.eyebrow{font-size:12px;letter-spacing:2px;font-weight:700;color:#55739d;margin-bottom:8px}
.hero-sub{font-size:17px;color:#65758f;max-width:680px;line-height:1.65;margin-bottom:24px}
.brand{font-size:25px;font-weight:700;letter-spacing:-.6px;margin:8px 0 4px}
.brand-sub{font-size:12px;letter-spacing:1.5px;color:#91a6ca!important;margin-bottom:30px}
.answer-label{font-size:12px;letter-spacing:1.4px;color:#218271;font-weight:700;margin-bottom:12px}
.footnote{font-size:12px;color:#7686a0;margin-top:24px}
</style>""",
    unsafe_allow_html=True,
)

DB = os.getenv("CP01_DB_PATH", ".chroma")
DOCS = os.getenv("CP01_DOCS_PATH", "data/policies")


@st.cache_resource
def get_store(path):
    return KnowledgeStore(path)


with st.sidebar:
    st.markdown(
        '<div class="brand">✦ northstar</div><div class="brand-sub">WORKPLACE KNOWLEDGE</div>',
        unsafe_allow_html=True,
    )
    st.caption("YOUR KNOWLEDGE WORKSPACE")
    page = st.radio(
        "Workspace",
        ["Ask a question", "Policy library", "How it works"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("ANSWER ENGINE")
    default_mode = 1 if os.getenv("CP01_MODE", "hosted") == "extractive" else 0
    mode_label = st.selectbox("Mode", ["Hosted LLM API", "Extractive demo"], index=default_mode)
    mode = "hosted" if mode_label == "Hosted LLM API" else "extractive"
    ready = mode == "extractive" or bool(os.getenv("HOSTED_LLM_BASE_URL", "").strip())
    if mode == "extractive":
        st.caption("Deterministic source excerpts. No generation LLM is used.")
    else:
        st.caption(f"Model: {os.getenv('HOSTED_LLM_MODEL', 'llama3.2')}")
        st.caption(
            "Course-hosted open-source model. No local chat-model download. Three roles plan, select, and review evidence."
        )
        try:
            config = APIConfig.from_env()
            if config.fallback_provider == "none":
                st.caption("Paid API fallback is disabled.")
            else:
                st.warning(
                    f"Optional {config.fallback_provider} fallback is enabled. On temporary hosted failures, questions/excerpts may go to that provider and incur API charges."
                )
        except ValueError as exc:
            ready = False
            st.warning(str(exc))
        if not os.getenv("HOSTED_LLM_BASE_URL", "").strip():
            st.warning(
                "Set HOSTED_LLM_BASE_URL in your local .env to the provided course URL and restart, or choose Extractive demo."
            )
    st.divider()
    st.caption("LOCAL-FIRST BY DESIGN")
    st.write("Chroma · ONNX MiniLM · LangGraph")
    st.caption(
        "Documents and vector search stay on this machine. Questions and retrieved excerpts go to your configured hosted model API. Use synthetic data only on plain HTTP."
    )
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state.pop("last_question", None)
        st.session_state.pop("answer_mode", None)
        st.rerun()
    st.markdown(
        '<div class="footnote">CP01 CAPSTONE<br>Fictional policies · Demo use only</div>',
        unsafe_allow_html=True,
    )

try:
    store = get_store(DB)
    documents = load_documents(DOCS)
except Exception as exc:
    st.error(
        f"Workspace could not initialize: {type(exc).__name__}. Check document metadata and dependencies."
    )
    st.stop()

if store.collection.count() == 0:
    st.info(
        "Welcome. Build the local index once to search the synthetic policy library. The first run downloads the free ONNX MiniLM embedding model (about 80 MB)."
    )
    if st.button("Build knowledge index", type="primary"):
        try:
            with st.spinner("Embedding policies locally…"):
                summary = store.ingest(DOCS)
            st.success(f"Indexed {summary['documents']} documents and {summary['chunks']} chunks.")
            st.rerun()
        except Exception as exc:
            st.error(
                f"Index build failed ({type(exc).__name__}). Check network access for the first model download, then retry."
            )
    st.stop()

if store.needs_refresh(DOCS):
    try:
        with st.spinner("Policy files changed. Refreshing the current index…"):
            store.ingest(DOCS)
    except Exception as exc:
        st.error(
            f"Policy refresh failed ({type(exc).__name__}). Search is paused to avoid using stale evidence."
        )
        st.stop()

current = [doc for doc in documents if doc.status == "current"]

if page == "Ask a question":
    st.markdown(
        '<div class="eyebrow">CLEAR ANSWERS. TRACEABLE EVIDENCE.</div>', unsafe_allow_html=True
    )
    st.title("Your workplace, answered.")
    st.markdown(
        '<div class="hero-sub">Find the policy behind the question. Three specialist agents retrieve, select, and verify answers against current company knowledge.</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Current policies", len(current))
    m2.metric("Knowledge sources", len(documents))
    m3.metric("Verification", "Evidence-first")
    st.write("")
    st.caption("EXPLORE A QUESTION")
    examples = [
        "How many days of annual leave do employees receive?",
        "What is the home office equipment allowance?",
        "What is the price of the Enterprise Plus plan?",
    ]
    cols = st.columns(3)
    selected = None
    for col, question in zip(cols, examples, strict=True):
        if col.button(question, use_container_width=True, disabled=not ready):
            selected = question
    question = st.chat_input(
        "Ask about leave, expenses, access, security…", max_chars=1000, disabled=not ready
    )
    question = selected or question
    if not ready:
        st.info(
            "Resolve the model setup notice in the sidebar, or select Extractive demo to explore without a model server."
        )
    if question:
        st.session_state.last_question = question
        started = time.monotonic()
        final_state = {"question": question, "trace": []}
        with st.status("Working through the evidence…", expanded=True) as progress:
            try:
                graph = build_graph(store, make_provider(mode))
                for update in graph.stream(final_state, stream_mode="updates"):
                    for _, values in update.items():
                        final_state.update(values)
                        if values.get("trace"):
                            latest = values["trace"][-1]
                            st.write(f"{latest['agent']} · {latest['detail']}")
                progress.update(label="Evidence check complete", state="complete", expanded=False)
            except Exception as exc:
                final_state.update(
                    status="error",
                    answer=f"Application error ({type(exc).__name__}). Please retry.",
                    claims=[],
                )
                progress.update(label="Unable to complete this question", state="error")
        final_state["elapsed"] = round(time.monotonic() - started, 2)
        st.session_state.result = final_state
        st.session_state.answer_mode = mode
    if "result" in st.session_state:
        result = st.session_state.result
        st.divider()
        st.caption("YOUR QUESTION")
        st.subheader(st.session_state.last_question)
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            with st.container(border=True):
                if result["status"] in {"answered", "limited"}:
                    st.markdown(
                        '<div class="answer-label">● RETRIEVED EXCERPTS</div>',
                        unsafe_allow_html=True,
                    )
                    if result["status"] == "limited":
                        st.info(
                            "The current documents do not provide the requested fact. Here is the relevant limitation."
                        )
                    # Render corpus/model text as plain text, never as HTML or active links.
                    for claim in result.get("claims", []):
                        st.text(f"{claim.text} [{claim.citation_id}]")
                    st.caption(
                        "These excerpts match current sources. Demo selection is heuristic: relevance, applicability and completeness are not guaranteed. Review the full policy before acting."
                    )
                else:
                    st.info(result.get("answer", "No answer available."))
                actual_mode = st.session_state.answer_mode
                st.caption(
                    (
                        "Extractive demo · not LLM-generated"
                        if actual_mode == "extractive"
                        else result.get("provider_used", "Hosted LLM API")
                        + " · selected source excerpts"
                    )
                    + f" · {result['elapsed']}s"
                )
            with st.expander("Agent activity", expanded=True):
                for i, event in enumerate(result.get("trace", []), 1):
                    st.write(f"{i:02d}  {event['agent']}")
                    st.caption(
                        f"{event['detail']} · {event['duration_ms']} ms · {event.get('provider', '')}"
                    )
                if result.get("provider_events"):
                    st.caption("API retry / fallback activity")
                    for api_event in result["provider_events"]:
                        st.write(f"{api_event['provider']} · {api_event['outcome']}")
        with right:
            st.markdown("### Evidence desk")
            cited = {c.citation_id for c in result.get("claims", [])}
            for evidence in result.get("evidence", []):
                meta = evidence.metadata
                with st.expander(
                    f"[{evidence.citation_id}] {meta['title']}",
                    expanded=evidence.citation_id in cited,
                ):
                    st.caption(
                        f"{meta['document_id']} · v{meta['version']} · {meta['effective_date']} · CURRENT"
                    )
                    st.text(evidence.text)
                    st.caption(
                        f"Section: {meta['section']} · Cosine similarity: {evidence.score:.3f} (not confidence)"
                    )
            if not result.get("evidence"):
                st.caption("No current evidence passed the relevance gate.")
        with st.expander("Retrieval transparency & JSON trace"):
            st.caption(
                "Up to 8 chunks/query (12 merged); cosine threshold 0.28 + lexical anchor. Scores are heuristics, not calibrated confidence. Documents never grant tool permissions."
            )
            st.json(
                {
                    **result,
                    "evidence": [asdict(e) for e in result.get("evidence", [])],
                    "claims": [asdict(c) for c in result.get("claims", [])],
                }
            )
    else:
        st.info(
            "Start with a sample question or ask your own. Every answer links back to an exact policy excerpt."
        )

elif page == "Policy library":
    st.markdown(
        '<div class="eyebrow">A SMALL, AUDITABLE SOURCE OF TRUTH</div>', unsafe_allow_html=True
    )
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
    st.markdown(
        '<div class="eyebrow">BOUNDED AGENTS, INSPECTABLE DECISIONS</div>', unsafe_allow_html=True
    )
    st.title("How the answer earns trust")
    for title, description in [
        (
            "01 · Retrieval agent",
            "In LLM mode a role-scoped planner decomposes the question into up to three searches. Real local ONNX MiniLM vectors and persistent Chroma retrieve current sources and apply a relevance gate.",
        ),
        (
            "02 · Answer agent",
            "Selects relevant source excerpts using the configured hosted open-source model, or a clearly labeled deterministic extractive demo. Source text is untrusted data.",
        ),
        (
            "03 · Verification agent",
            "Checks citation IDs, current versions, exact quote support and lexical relevance. In LLM mode an independent reviewer also checks relevance and completeness. Any unsupported claim rejects the entire draft.",
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
