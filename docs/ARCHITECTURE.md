# Architecture

## Data path
Original Markdown policies → metadata validation → heading-aware bounded chunks → local MiniLM embeddings → persistent Chroma collection → current-policy retrieval → concise answer generation → supporting-quote/citation checks → independent model review → employee chat with collapsed sources.

`src/cp01/corpus.py` validates the complete corpus before ingestion. Each source carries a stable document ID, version, effective date, current/superseded status, category, and filename. Chunks retain section headings and deterministic content-derived IDs. Ingestion reconciles stale chunk IDs so removed content does not remain searchable after a rebuild. A full corpus fingerprint includes both content and metadata, including quarantine annotations; CLI/UI refresh a changed corpus before answering. A separate short chunk-list fingerprint identifies the resulting indexed chunk set.

`src/cp01/store.py` uses Chroma's ONNX MiniLM embedding function and cosine-distance search. The model requires an initial download if it is not cached. Document vectors persist in Chroma; new search queries are embedded locally at query time. The query filter requires `status=current`; the archived leave policy remains in the corpus as a regression fixture but cannot be retrieved as current evidence. Similarity scores are ranking heuristics, not calibrated confidence probabilities.

## LangGraph roles and state
1. **Retrieval agent:** validates a 1–1,000-character question. Extractive mode uses the original query; Hosted LLM API mode makes a separate retrieval-planner call producing one to three queries. Search returns up to eight current-policy chunks per query, merges the highest-ranked unique chunks to a maximum of twelve, and applies a relevance threshold and lexical anchor.
2. **Answer agent:** hosted mode generates a natural-language answer, normally 1–3 sentences, within 160 words and 1,200 characters, with citation IDs and up to three exact supporting quotes. The extractive developer provider returns relevant whole sections, preserving adjacent qualifications and avoiding redundant question coverage. It is not LLM inference. Neither path calculates or authorizes personal entitlements.
3. **Verification agent:** checks current source identity, exact quote membership, supporting quote/text equality, lexical relevance, generated citation references, and numeric tokens against the supporting quotes. A third model call receives the draft, its supporting quotes, and full retrieved context to reject unsupported statements, missing material conditions, irrelevant or incomplete answers. Deterministic failures cannot be overridden. Reviewer errors/rejections produce abstention, and rejected draft text is cleared from final state.
4. **Safe response:** terminates invalid input, retrieval/provider failure, unsupported questions, or rejected drafts with a visible explanation.

The graph's typed shared state includes question, evidence, supporting claims, draft answer, final answer, status, verification summary, error, trace, actual provider, and sanitized provider events. Trace events name each role, record elapsed milliseconds, and identify the provider used. Successful execution follows retrieval → answer → verification. `answered` means the deterministic checks and, for generated text, independent model review accepted the answer; an explicit statement that a requested attribute is missing uses `limited`. Neither status guarantees relevance, applicability, or completeness. Conditional edges route failures to a safe response.

## Hosted API, bounded retries, and optional fallback
The primary live provider is the course-hosted open-source model through an Ollama-compatible `/api/chat` endpoint. `HOSTED_LLM_BASE_URL` is supplied privately in `.env`; `HOSTED_LLM_MODEL` defaults to `llama3.2`. The course address is not committed. No local chat-model installation or download is required. Running an optional local compatible server is an alternative, not a setup prerequisite.

`src/cp01/transport.py` provides a shared, validated retry policy: at most three attempts per role call, eight HTTP attempts across the entire question, and a 120-second API budget. Each attempt has an absolute 30-second deadline including response-body handling, with at most five seconds to connect. Exponential backoff starts at 0.5 seconds, includes random jitter, and is capped at eight seconds; numeric or HTTP-date `Retry-After` is also capped at eight seconds. The API budget bounds network operations, not a hard end-to-end deadline for all local indexing or embedding work.

Retries apply only to timeouts, connection/read/protocol interruptions, and HTTP 429/500/502/503/504. Authentication/permission errors, malformed requests, redirects, invalid JSON or role schemas, and failed grounding/reviewer checks are terminal. Direct HTTP transports avoid nested SDK retries.

`src/cp01/config.py` and `src/cp01/fallback.py` support an **optional** OpenAI or Claude fallback. `FALLBACK_PROVIDER=none` is the default, even if keys happen to exist. Explicitly selecting `openai` or `anthropic` also requires that provider's key and compatible model in private configuration. Only hosted transient retry exhaustion can activate the selected fallback; it then handles remaining roles using the same budget. There is no fallback chain, no automatic extractive substitution, and no bypass of deterministic citation verification.

At each graph invocation, the request budget and event list reset and the hosted primary becomes active again. The app creates a provider per question; do not share one mutable provider/transport concurrently. Advanced Settings can reveal the actual provider and safe retry/fallback events. The default chat hides technical traces and processing stages. Keys, request headers, and raw API error bodies do not enter graph state or application logs.

## Evidence contract
A document ID such as `NS-HR-001` is stable corpus identity. A display citation such as `S1` is local to one answer. The collapsed Sources section connects the two and shows the policy title/version and actual retrieved text. An answer may contain several citations to the same document when different chunks support its claims.

A verified excerpt means its words occur in retrieved current evidence. It does **not** prove semantic correctness, complete coverage, policy authority outside this synthetic corpus, or resistance to every prompt injection. Generated prose is not required to be an exact quote: its evidence is checked verbatim, while independent model review assesses semantic support and qualifications. This review can still make mistakes. Numeric-token matching does not prove that a number was assigned to the correct entity or condition. Historical extractive evaluation results do not establish the quality of generated answers.

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
A fresh environment needs Python dependencies and the local ONNX embedding model. Live generation needs an accessible course-hosted compatible API, not a local chat-model download. Missing hosted configuration or an unprepared policy index leaves the employee composer usable but returns a clear unavailable message without making a provider request. Setup is in Settings; there is no demo CTA or silent fallback. Extractive mode remains an explicit advanced developer option. Provider and budget failures are visible, and any configured fallback is identified honestly.

The original recorded suite passed 53 mocked/offline tests; the current suite additionally checks generated-answer acceptance/rejection and employee chat interactions, including retry, provider-contract, shared-budget, redaction, and per-question-reset checks. Real Chroma/ONNX retrieval and Streamlit interactions were exercised. No successful live hosted inference or paid-provider inference was verified in the recorded build; blocked execution attempts do not prove the course service is offline. Confirm endpoint access, model compatibility, and the intended presentation environment before claiming live LLM functionality.

## Extension priorities
Add permission-aware retrieval before enterprise deployment; then document lifecycle controls, semantic evaluation, stronger adversarial tests, observability with redacted logs, and approved-model benchmarking. Add PDF extraction only with parsing-quality checks and page-level citations.
