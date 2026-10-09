# CP01 · Enterprise Knowledge Assistant

A local-first, evidence-first capstone for a fictional company, **Northstar Labs**. Ask about workplace policies, inspect the exact excerpts behind a response, and follow the retrieval → answer → verification route in a real LangGraph workflow.

**Python · LangGraph · persistent Chroma · local ONNX MiniLM · Streamlit · course-hosted open-source LLM API**

No paid API or hosted vector database is required. The course-hosted open-source model avoids a local chat-model download. The initial embedding-model download requires internet access; subsequent document search runs locally. All included policies are original synthetic demonstration data, not real company policies.

## Demo assets

- [Project overview (PDF)](docs/assets/CP01_Overview.pdf)
- [Captioned walkthrough from real app screenshots (MP4)](docs/assets/CP01_Demo_Walkthrough.mp4)
- [Home-screen screenshot](docs/assets/01_home.jpg)

The recorded demonstration uses **extractive demo mode**. It does not show a live hosted-model generation run.

## Quick start

Python 3.11–3.13 is supported; Python 3.12 was tested. Run commands from this repository's root.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
cp .env.example .env                # Windows: copy .env.example .env
python -m cp01.cli ingest
streamlit run app.py
```

Before asking live questions, paste the course-provided model base URL into `HOSTED_LLM_BASE_URL` in your private `.env`. No local Ollama installation or chat-model download is required. Alternatively choose **Extractive demo** in the sidebar to explore immediately. Open the localhost address printed by Streamlit, normally http://localhost:8501. The first ingestion downloads Chroma's official ONNX MiniLM model (~80 MB) into `.chroma/embedding-model/`. Chroma's model downloader validates the model archive checksum. The local database and model cache are excluded from Git. Document embeddings persist in Chroma and the ONNX model is cached; each new search query is embedded locally at query time.

For a dependency-resolved install instead of the tested snapshot:

```bash
python -m pip install -e '.[dev]'
```

`requirements.lock` is the exact tested Python 3.12/Linux package snapshot, not a cross-platform lock with hashes. Package wheels must be available for your platform. No credentials are needed for extractive mode.

## Two clearly different modes

| Mode | What runs | What is returned |
|---|---|---|
| **Extractive demo** | Actual Chroma/ONNX retrieval + deterministic role stand-ins inside LangGraph | Relevant short policy sections, copied exactly; **not LLM-generated** |
| **Hosted LLM API** (default) | Three role-scoped model calls: retrieval planner, evidence selector, independent reviewer | LLM-selected verbatim policy excerpts, published only after deterministic provenance checks and reviewer approval |

This is **quote-grounded RAG**, not free-form generative synthesis. Keeping output verbatim makes the provenance checks reproducible. The assistant does not calculate entitlements, approve purchases, or make employment decisions. Complex questions can require reading several policy sections; selected excerpts may be incomplete even when citations are valid.

### Configure the course-hosted open-source API

The course provides an Ollama-compatible model API. Add its base URL **only to your local `.env`**, not to source code:

```dotenv
CP01_MODE=hosted
HOSTED_LLM_BASE_URL=http://your-course-host:11434
HOSTED_LLM_MODEL=llama3.2
```

Replace the example hostname with the course-provided URL. Choose **Hosted LLM API** in the UI, or run:

```bash
python -m cp01.cli ask "How many annual leave days do full-time employees get?" --mode hosted
```

The application calls the model's `/api/chat` endpoint for three bounded role-specific requests. The question and selected excerpts are sent to your configured model server. If the course endpoint is plain HTTP, traffic is not encrypted: use the included synthetic data only. The course address is deliberately not committed, and no paid-provider key is needed. Local Chroma, Python dependencies, and the small ONNX embedding model still use local disk even though the chat model is hosted.

**Verification status:** real ONNX embeddings, Chroma persistence, LangGraph execution, and Streamlit interactions were exercised. Hosted-model request/response contracts and failure paths are tested with mocks. A bounded synthetic live smoke attempt and its authorized retry were stopped by execution-tool infrastructure errors before any provider response was available. This does **not** establish that the course server is offline. Live-model quality remains unverified, and there is no silent switch to extractive mode on API failure. Optional paid fallback is separately opt-in and shown in the UI.

For a completely API-free demonstration:

```bash
python -m cp01.cli ask "How many annual leave days do full-time employees get?" --mode extractive
```

### Optional paid API fallback (off by default)

Hosted open-source inference remains primary. To opt in to one fallback, edit your **private `.env`**:

```dotenv
FALLBACK_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
```

Fill the blank key locally. For Claude instead, set `FALLBACK_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`. Model names are configurable and account availability must be checked with the provider. The OpenAI model must support Chat Completions, JSON-object response format, `temperature=0`, and `max_completion_tokens`; this is not an arbitrary-model adapter. The Claude model must support Messages with forced tool output and `temperature=0`. `.env` and `.env.*` are ignored by Git, with `.env.example` explicitly allowed. Never paste keys into chat, screenshots, or committed files. Keys are excluded from configuration representations and never included in graph state or UI traces.

**Data sharing and charges:** enabling a fallback authorizes the app to send the question and retrieved policy excerpts for the failed/current and remaining roles to the selected provider's official HTTPS API. This may incur that provider's API charges. Use synthetic data for this capstone. The UI warns when fallback is enabled and shows the provider actually used. There is no fallback if `FALLBACK_PROVIDER=none`, and missing selected-provider credentials fail configuration before requests. Extractive demo uses neither hosted generation nor paid fallback.

### Bounded retries and failure rules

One shared transport handles all roles and both provider paths, using direct HTTP rather than an SDK retry layer:

- Up to **3 attempts per role call**, with exponential backoff starting at 0.5 seconds and random jitter; backoff and numeric/date `Retry-After` are capped at 8 seconds.
- A shared **8-attempt / 120-second whole-question API budget** includes primary, fallback, and all three roles. The budget, retry events, and active provider reset at every new graph invocation, including after an idle period. Each attempt has an absolute 30-second deadline (including response body) and at most 5 seconds to connect. Budget exhaustion stops further requests.
- Retry only timeouts, connection/read/protocol interruptions, HTTP 429, and 500/502/503/504.
- After hosted transient retry exhaustion, the **explicitly selected** fallback may take over and remains active for later roles. It uses the same remaining budget; there is no fallback chain or automatic return to the hosted provider.
- Authentication/permission errors, malformed requests, redirects, invalid JSON/schema, and rejected evidence are terminal. They trigger neither retry nor paid fallback. Exact-source verification and reviewer rejection cannot be bypassed by changing providers.
- User-visible events include only fixed provider names, attempt counts, and safe outcomes. No raw API response/error bodies, request headers, or keys are logged by the application. Source archives must also exclude all `.env`/`.env.*` variants except `.env.example`; Git ignore rules alone do not secure a manually assembled archive.

The frozen `RetryPolicy` values in `src/cp01/transport.py` are validated and centrally configurable by a developer. Mocked tests cover retry success, exhaustion, fallback, disabled/missing-key paths, terminal failures, shared budgets, and redaction. **No live paid API calls were made during development.**

#### Optional local model alternative

Only if you prefer running the chat model yourself, install [Ollama](https://ollama.com/), pull a compatible model, start its server, and set `HOSTED_LLM_BASE_URL=http://localhost:11434` with the matching model name. This is optional and adds a separate multi-gigabyte chat-model download; it is not part of the recommended course-hosted setup.

## What to try

- “How many annual leave days do full-time employees get?” → current 24-day policy, not the superseded 20-day version
- “What are the hotel and meal reimbursement caps for business travel?” → USD 180 hotel and USD 60 meals, with conditions
- “What is the price of Atlas Enterprise?” → documents explicitly do not publish pricing; no invented price
- “What is the Security hotline telephone number?” → missing-number limitation, no invented contact

The sidebar also opens the **Policy library** (including an optional superseded-policy view) and **How it works**. Reset clears this browser session's answer. Chat answers are not stored in Chroma.

## Architecture and controls

1. **Retrieval agent:** validates 1–1,000-character input. In LLM mode, a planner chooses up to three bounded queries; the deterministic demo uses the original query. Search returns up to eight chunks/query, filters `status=current`, merges at most 12, and applies a heuristic cosine threshold plus lexical anchor.
2. **Answer agent:** chooses evidence excerpts using the selected provider. The demo preserves adjacent conditions by returning bounded sections instead of isolated sentences. Explicit missing-attribute statements take a `limited` response route.
3. **Verification agent:** rejects unknown source IDs, superseded sources, non-verbatim text, empty/short quotes, and missing lexical relevance. LLM mode additionally asks an independent review role to assess relevance/completeness. The reviewer cannot override deterministic rejection.
4. **Conditional routing:** invalid input, missing evidence, provider/reviewer errors, and unsupported drafts route to a safe response. Successful verified excerpts and explicit known limitations are distinguishable states.

The roles share typed `AssistantState`, but have different responsibilities and bounded outputs. Model calls use JSON schemas validated in Python, temperature zero, finite deadlines, and no tools capable of external actions. Three logical role calls share a maximum of eight HTTP attempts; transient retries and optional fallback consume the same budget. The UI shows actual graph events and elapsed times, not a fabricated animation.

The corpus loader accepts only local UTF-8 Markdown with validated YAML front matter: `document_id`, `title`, `version`, `effective_date`, `status` (`current`/`superseded`), and `category`. Limits: 200 files, 256 KB/file, no symlinks. Optional `excluded_sections` explicitly quarantines reviewed non-authoritative examples. An included malicious security-test quotation is visible in the library but excluded from indexing by this metadata.

Corpus fingerprints include content **and metadata**. CLI/UI refresh changed corpora before answering and remove stale index chunks; failed refresh pauses UI search. This prevents an older indexed version of a quarantined section remaining active.

## Testing and evaluation

```bash
pytest -q
ruff check src app.py tests
python -m compileall -q src app.py
python -m pip check
python scripts/evaluate.py
```

GitHub Actions runs the offline suite on Python 3.12 with the tested dependency snapshot; the workflow does not download an embedding model or call a live hosted LLM. The unit tests include invalid citations, altered excerpts, archived evidence, unavailable providers, state isolation, schema validation, and corpus consistency. Chroma regression tests use a lightweight deterministic test embedding to check persistence/filtering without downloading a model. The evaluation runner uses the **real** local ONNX embedding model against the 20 synthetic evaluation cases.

See [evaluation methodology and recorded results](docs/EVALUATION.md). Source recall and excerpt recall are diagnostic metrics, **not factual-accuracy or semantic-completeness scores**. The corpus includes multi-document, missing-fact, stale-version, and malicious-quotation cases. No claim of production-grade accuracy or comprehensive prompt-injection prevention is made.

## Repository guide

```text
app.py                      Streamlit interface
src/cp01/corpus.py          Safe bounded Markdown loader and section chunker
src/cp01/store.py           Local Chroma and ONNX embeddings
src/cp01/providers.py       Extractive demo and three-role hosted-API contracts
src/cp01/transport.py       Shared deadlines, budgets, transient-only retries
src/cp01/fallback.py        Opt-in official OpenAI/Claude fallback
src/cp01/config.py          Typed private runtime configuration
src/cp01/workflow.py        Typed LangGraph and conditional verification routes
src/cp01/cli.py             Ingest and ask commands
data/policies/             10 original synthetic policies (9 current)
data/evaluation_cases.json 20 evaluation cases with source evidence
tests/                     Offline regressions and corpus checks
docs/                      Architecture, requirements, demo and evaluation
```

- [Requirements mapping and assumptions](docs/REQUIREMENTS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Demo script](docs/DEMO_SCRIPT.md)

## Limitations and responsible use

The supplied capstone overview names CP01 but does not include the separate detailed acceptance requirements. This implementation documents its assumptions; confirm those requirements before assessment.

This is a single-user local capstone prototype. It has no authentication, document-level access control, production tenancy, audit-log retention policy, or transactional index rebuild. Do not expose it publicly or add real confidential data without hardening. Model output and source text are untrusted. An exact quotation proves provenance, not authority, relevance, or completeness; malicious material in an **unreviewed, non-quarantined** document can still be repeated. Curated exclusion metadata is not a general injection detector. Numeric scenarios return the policy thresholds, not a computed or authorized decision. Held-out questions about an unknown CEO or contractor entitlement sometimes retrieve employee-policy excerpts instead of abstaining; the heuristic `answered` status means evidence was returned, not that every part of a question was resolved. The UI labels this explicitly as retrieved excerpts and warns about relevance, applicability, and completeness.

No uploads are exposed in the UI. Add reviewed synthetic Markdown through the local corpus and refresh. The current-only filter relies on curated `status` metadata rather than automatic policy conflict resolution or effective-date scheduling. One approved current version per document ID is enforced.

## Troubleshooting

- **First embedding download fails:** allow the official model host used by Chroma, retry ingestion, and check proxy support. `httpx[socks]` is included for SOCKS proxies.
- **No useful answer:** use a precise policy question; inspect retrieved excerpts and score thresholds. A cosine score is not a confidence probability.
- **Hosted API unavailable:** verify the course URL/model and network access; choose extractive demo explicitly if needed. Transient failures receive bounded retries under the limits above. Authentication and schema errors remain terminal.
- **Changed metadata or documents:** UI/CLI auto-refresh; `python -m cp01.cli ingest` forces an idempotent rebuild and removes stale chunks.
- **Different Python/platform:** install from `pyproject.toml` with `pip install -e '.[dev]'` and rerun the suite; the snapshot was tested on Linux/Python 3.12.

Optional fallback references: [OpenAI GPT-4.1 mini and supported endpoints](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [Claude API authentication](https://platform.claude.com/docs/en/api/overview), [Claude structured tool outputs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools), [Claude model IDs](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions).

Official references: [LangGraph StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph), [Chroma Python client](https://docs.trychroma.com/reference/python), [Ollama-compatible chat API](https://docs.ollama.com/api/chat), [Streamlit](https://docs.streamlit.io/).
