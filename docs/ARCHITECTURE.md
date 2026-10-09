# Architecture

## Data path
Original Markdown policies → metadata validation → heading-aware bounded chunks → local MiniLM embeddings → persistent Chroma collection → current-policy retrieval → bounded answer selection → exact-evidence verification → Streamlit answer and source cards.

`src/cp01/corpus.py` validates the complete corpus before ingestion. Each source carries a stable document ID, version, effective date, current/superseded status, category, and filename. Chunks retain section headings and deterministic content-derived IDs. Ingestion reconciles stale chunk IDs so removed content does not remain searchable after a rebuild. A full corpus fingerprint includes both content and metadata, including quarantine annotations; CLI/UI refresh a changed corpus before answering. A separate short chunk-list fingerprint identifies the resulting indexed chunk set.

`src/cp01/store.py` uses Chroma's ONNX MiniLM embedding function and cosine-distance search. The model requires an initial download if it is not cached. Document vectors persist in Chroma; new search queries are embedded locally at query time. The query filter requires `status=current`; the archived leave policy remains in the corpus as a regression fixture but cannot be retrieved as current evidence. Similarity scores are ranking heuristics, not calibrated confidence probabilities.

## LangGraph roles and state
1. **Retrieval agent:** validates a 1–1,000-character question. Extractive mode uses the original query; Hosted LLM API mode makes a separate retrieval-planner call producing one to three queries. Search returns up to eight current-policy chunks per query, merges the highest-ranked unique chunks to a maximum of twelve, and applies a relevance threshold and lexical anchor.
2. **Answer agent:** selects a bounded set of verbatim excerpts from retrieved evidence. The deterministic extractive provider returns up to four short source sections, preserving adjacent conditions; it is not a generation LLM. Hosted LLM API mode makes a distinct evidence-selection call requesting up to three excerpts in structured JSON. Neither mode computes a numeric scenario: returned thresholds support the user’s own calculation.
3. **Verification agent:** confirms citation identity, current metadata, exact quote membership, quote/claim equality, and lexical relevance. Unsupported drafts are rejected rather than silently repaired. In Hosted LLM API mode, a third, separately prompted review call checks relevance and sufficiency after deterministic checks succeed. The reviewer cannot override a deterministic provenance rejection.
4. **Safe response:** terminates invalid input, retrieval/provider failure, unsupported questions, or rejected drafts with a visible explanation.

The graph's typed shared state includes question, evidence, claims, answer, status, verification summary, error, trace, actual provider, and sanitized provider events. Trace events name each role, record elapsed milliseconds, and identify the provider used. Successful execution follows retrieval → answer → verification. `answered` means supported excerpts were returned; an explicit statement that a requested attribute is missing uses `limited`. Neither status guarantees relevance, applicability, or completeness. Conditional edges route failures to a safe response.

## Hosted API, bounded retries, and optional fallback
The primary live provider is the course-hosted open-source model through an Ollama-compatible `/api/chat` endpoint. `HOSTED_LLM_BASE_URL` is supplied privately in `.env`; `HOSTED_LLM_MODEL` defaults to `llama3.2`. The course address is not committed. No local chat-model installation or download is required. Running an optional local compatible server is an alternative, not a setup prerequisite.

`src/cp01/transport.py` provides a shared, validated retry policy: at most three attempts per role call, eight HTTP attempts across the entire question, and a 120-second API budget. Each attempt has an absolute 30-second deadline including response-body handling, with at most five seconds to connect. Exponential backoff starts at 0.5 seconds, includes random jitter, and is capped at eight seconds; numeric or HTTP-date `Retry-After` is also capped at eight seconds. The API budget bounds network operations, not a hard end-to-end deadline for all local indexing or embedding work.

Retries apply only to timeouts, connection/read/protocol interruptions, and HTTP 429/500/502/503/504. Authentication/permission errors, malformed requests, redirects, invalid JSON or role schemas, and failed grounding/reviewer checks are terminal. Direct HTTP transports avoid nested SDK retries.

`src/cp01/config.py` and `src/cp01/fallback.py` support an **optional** OpenAI or Claude fallback. `FALLBACK_PROVIDER=none` is the default, even if keys happen to exist. Explicitly selecting `openai` or `anthropic` also requires that provider's key and compatible model in private configuration. Only hosted transient retry exhaustion can activate the selected fallback; it then handles remaining roles using the same budget. There is no fallback chain, no automatic extractive substitution, and no bypass of deterministic citation verification.

At each graph invocation, the request budget and event list reset and the hosted primary becomes active again. The app creates a provider per question; do not share one mutable provider/transport concurrently. The UI displays the actual provider and fixed safe retry/fallback events. Keys, request headers, and raw API error bodies do not enter graph state or application logs.

## Evidence contract
A document ID such as `NS-HR-001` is stable corpus identity. A display citation such as `S1` is local to one answer. A source card connects the two and shows title, version, effective date, section, and the actual retrieved text. An answer may contain several citations to the same document when different chunks support its claims.

A verified excerpt means its words occur in retrieved current evidence. It does **not** prove semantic correctness, complete coverage, policy authority outside this synthetic corpus, or resistance to every prompt injection. Exact quotation is deliberately restrictive: arithmetic and multi-document synthesis may be incomplete. Inspect the known-answer evaluation before drawing quality conclusions.

## Trust boundaries
- Documents and user questions are untrusted data, never system instructions.
- Explicit reviewed `excluded_sections` metadata quarantines the known malicious-example section from indexing. This is a curated-corpus boundary, not automatic detection of arbitrary malicious text.
- Retrieved text cannot authorize tool use or change application instructions.
- The application has no leave approval, account provisioning, or purchase tools.
- Local indexing does not imply production document-level access control.
- Default demonstration data contains no actual employee records, credentials, or employer intellectual property.
- Hosted generation transmits questions and retrieved passages to the configured course endpoint. If it uses plain HTTP, traffic is not encrypted: use synthetic material only.
- Opting in to paid fallback may transmit questions and excerpts to the selected provider's official HTTPS API and incur usage charges. Its model must support the configured API/structured-output settings; no live paid calls were used for verification.
- Private `.env` and `.env.*` variants must be excluded from source archives as well as Git, except the blank-key `.env.example`.

## Operational boundaries
A fresh environment needs Python dependencies and the local ONNX embedding model. Live generation needs an accessible course-hosted compatible API, not a local chat-model download. Missing hosted configuration blocks generation requests; the extractive mode can still be selected explicitly. Provider and budget failures are visible, and any configured fallback is identified honestly.

The final recorded suite passed 53 mocked/offline tests, including retry, provider-contract, shared-budget, redaction, and per-question-reset checks. Real Chroma/ONNX retrieval and Streamlit interactions were exercised. No successful live hosted inference or paid-provider inference was verified in the recorded build; blocked execution attempts do not prove the course service is offline. Confirm endpoint access, model compatibility, and the intended presentation environment before claiming live LLM functionality.

## Extension priorities
Add permission-aware retrieval before enterprise deployment; then document lifecycle controls, semantic evaluation, stronger adversarial tests, observability with redacted logs, and approved-model benchmarking. Add PDF extraction only with parsing-quality checks and page-level citations.
