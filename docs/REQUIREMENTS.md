# CP01: Enterprise Knowledge Assistant

## Purpose and scope
Build a working Python capstone that answers questions about enterprise knowledge using retrieval-augmented generation (RAG), source citations, and an inspectable multi-agent workflow. The implementation uses a fictional company, **Northstar Labs**, so its corpus can be distributed safely with the demo.

This is a scoped prototype, not a production HR, finance, security, or customer-support system. It explains evidence and does not approve leave, grant access, place orders, or send messages.

## Requirements basis and assumptions
The available project brief identifies CP01 and Python, LangChain/LangGraph, a vector database, an LLM, and a FastAPI or Streamlit application. It calls for synthetic data, use of the provided open-source LLM, a company-laptop deployment/demo context, and a final presentation. The separate detailed acceptance rubric was not supplied. The course-hosted Ollama-compatible endpoint and `llama3.2` model were identified for private runtime configuration; the address is not included in the repository. The following are transparent implementation choices, not additional claims about the brief:

- LangGraph supplies actual graph orchestration; Chroma stores local vector embeddings.
- Streamlit is the demonstration interface, with a CLI for repeatable checks.
- Markdown documents with validated metadata are the supported ingestion format. PDF/OCR is not claimed.
- The supplied open-source model is accessed through its course-hosted, Ollama-compatible API; local chat-model installation is optional. The bundled extractive mode is a transparent, deterministic demonstration baseline, **not an LLM**.
- Three bounded roles retrieve evidence, select source-backed answer excerpts, and validate citations. Hosted LLM API mode uses three separate planner/selector/reviewer prompts against the configured model, with a non-overridable deterministic provenance check. Extractive mode uses deterministic stand-ins and is not LLM inference.
- Search retrieves up to eight chunks per query, with a twelve-chunk merged cap; the extractive baseline returns up to four source sections. Missing-attribute evidence is marked `limited`.
- Content-and-metadata fingerprints trigger index refresh; numeric questions return policy rules rather than computed or authorized decisions.

## Acceptance checklist
| ID | Requirement | Evidence / acceptance check |
| --- | --- | --- |
| R01 | Original synthetic enterprise documents | Ten files in `data/policies/`, visibly fictional; nine current, one superseded |
| R02 | Validated ingestion and chunk metadata | Loader requires document ID, title, version, effective date, status, category |
| R03 | Persistent vector search | Ingest corpus; query local Chroma; inspect ranked retrieved chunks |
| R04 | Current-version policy selection | Current leave answer uses 24 days, never archived 20 days |
| R05 | Multi-agent workflow | Inspect retrieval, answer, and verification trace for a successful query |
| R06 | Grounded answers with citations | Every returned claim links to a retrieved current source and exact excerpt |
| R07 | Safe unsupported answer handling | Test unrelated question, missing pricing, balance, and hotline number |
| R08 | Source text treated as untrusted | Test deliberate malicious quotation in NS-KB-001; do not follow it |
| R09 | Hosted open-source model integration | Configure the supplied API privately; record a successful live request before claiming live LLM mode is verified |
| R10 | Usable demonstration | Run Streamlit, show an answer, source card, workflow trace, and safe missing-information response |
| R11 | Evaluation and limitations | Twenty known-answer cases, reproducible tests, honest outcome reporting |
| R12 | Reproducible handoff | README setup and commands, environment example, dependencies, and docs |
| R13 | Bounded transient retries | Test exponential jitter, capped Retry-After, per-attempt deadlines, and shared question budget |
| R14 | Explicit optional fallback | Disabled by default; selected OpenAI/Claude credentials remain private; only transient hosted exhaustion can activate it |
| R15 | Non-overridable safety and honest provider reporting | Authentication/schema/grounding failures remain terminal; actual provider and sanitized events are visible |
| R16 | Reusable question isolation | Reset API budget, events, and active primary on each graph invocation |

## Not included in this prototype
Production SSO or per-user document authorization, a managed secrets vault, retention enforcement, enterprise deployment hardening, arbitrary uploads, OCR, multilingual benchmarking, streaming model generation, and approval/action automation. Do not index real employer material without authorization and a deployment-specific data review.

## Later user-requested resilience additions
The optional paid fallback and exponential-backoff behavior are later user-requested additions, not claims about the original course brief. The hosted open-source API remains primary. The shared defaults allow up to three attempts per role call, eight total HTTP attempts, and a 120-second API budget; each attempt has a 30-second absolute deadline and at most five seconds to connect. Only transient network errors and HTTP 429/500/502/503/504 retry. Backoff and Retry-After are capped at eight seconds.

Fallback is off unless `FALLBACK_PROVIDER` selects `openai` or `anthropic` with the corresponding key and compatible model in private `.env`. It can incur charges and shares question/excerpt data with the chosen provider. No live paid call was made during development. The recorded final suite passed 53 offline/mocked tests; successful live hosted inference remains unverified. Persistent document-vector/model caching is implemented, while new query embeddings still run locally at query time.
