# Independent QA report

## Scope and environment

Executed on Linux-6.18.44-x86_64-with-glibc2.41; Python 3.12.14. Versions: {'chromadb': '1.5.9', 'langgraph': '1.2.14', 'streamlit': '1.65.0', 'httpx': '0.28.1'}. Corpus: {'documents': 10, 'current_documents': 9, 'chunks': 34, 'fingerprint': '3a4bd142df81'}.

Source capture timestamp is recorded in the JSON; executor wall clock differs from orchestration clock. Review completed 2026-10-09 19:25 UTC per session clock.

Real persistent Chroma + ONNX MiniLM vectors and actual LangGraph tested in deterministic extractive mode. No successful live hosted inference was verified. Mocked Ollama-compatible hosted and optional paid-provider contracts do not establish real LLM quality.

## Results

- Original baseline suite: 27 passed, one custom-embedding deprecation warning. Latest retry/fallback suite: 53 passed, same warning.
- All 20 development outputs had exact current-source citation support; 16 manually adequate, 4 partial.
- Three missing-fact cases returned limited answers without invented values.
- Known malicious fixture was excluded via curated metadata, not defeated by a general injection detector.
- Development questions influenced implementation fixes; these results are not held-out accuracy.

| Case | Status | Adequacy | Notes |
|---|---|---|---|
| E01 | answered | pass | 24 working days; additional supported leave context. |
| E02 | answered | pass | 5-day carryover and 31 March expiry. |
| E03 | answered | pass | Line manager and department head for >10 consecutive days. |
| E04 | answered | pass | PeopleDesk and 10-working-day notice. |
| E05 | answered | pass | Both monetary caps and qualifications. |
| E06 | answered | pass | Both manager and Finance approval rules. |
| E07 | answered | pass | 30-day deadline and itemized receipts threshold. |
| E08 | answered | pass | 3 remote days and manager agreement. |
| E09 | answered | pass | Both sources, USD 300, 30 days, prior approval and receipts. |
| E10 | answered | partial | Training, managed laptop and VPN included, but explicit MFA enrollment obligation missing; MFA appears only in a cross-reference. |
| E11 | answered | pass | Manager/owner roles, 2-day target and nonguarantee. |
| E12 | answered | pass | Security review for new software including free-trial offer; no invented exemption. |
| E13 | answered | partial | Approval thresholds and Security/Privacy rules shown, but no USD 600 calculation or explicit application; irrelevant home-office passage. |
| E14 | answered | pass | Attachments exclusion, organization administrator and Enterprise requirement. |
| E15 | answered | pass | Severity-1 1 staffed hour, staffed schedule, first response not resolution. |
| E16 | limited | pass | Explicit unpublished-pricing limited answer. |
| E17 | limited | pass | Individual balance unavailable; PeopleDesk authority. |
| E18 | answered | pass | 24 days and version supersession. |
| E19 | answered | partial | No 99-day answer; current 24-day policy. Does not explicitly explain malicious quotation invalidity. Known attack section quarantined before retrieval. |
| E20 | limited | partial | Correct unavailable-number statement; employee-directory next step omitted. |

## Additional probes and limits

Four fresh probes were run after fixes. CEO-name and contractor-leave questions incorrectly received `answered` statuses with irrelevant or employee-only excerpts. Cafeteria menu abstained correctly; VPN question answered correctly. This tiny diagnostic set is not a benchmark. See `artifacts/independent-heldout-probes.json`.

An independent synthetic malicious sentence can pass exact-quote verification. The verifier proves textual provenance and current metadata, not policy authority, semantic entailment, completeness, or resistance to unseen prompt injection.

## Component and UI checks

- Invalid citation IDs, mismatched text/quote, fabricated excerpts and archived sources fail closed.
- Empty/oversized and unrelated-input branches tested; sequential LangGraph requests and new Streamlit sessions do not leak prior answers.
- Original no-retry HTTP configuration has been superseded by the bounded retry/fallback transport reviewed below.
- Real Chroma current-version filtering and stale replacement passed; separate Python process reopened persisted database. Tiny custom embeddings used for this component test, real MiniLM for all 20 evaluation questions.
- Streamlit AppTest: leave/allowance answers, pricing limited result, policy library 9 current/10 including archived, How it works, navigation and reset all passed without exceptions. This is component-level UI testing, not pixel/browser validation.
- `pip check` passed. Dependency snapshot present. A fresh full environment installation from that snapshot was not independently run.
- `.env`, `.venv`, model/index caches ignored; no real secrets were used. No production auth/RBAC or external deployment tested.

## Artifacts

- `artifacts/independent-review-results.json`: exact per-case outputs and manual judgments
- `artifacts/independent-answer-transcript.txt`: full answer capture
- `artifacts/independent-test-log.txt`: complete pytest log
- `artifacts/run_independent_review.py`: repeatable capture script (manual judgments must be reapplied after rerun)
- `tests/test_independent_review.py`: independent regression and limitation tests

## Retry and optional cloud fallback review (2026-10-09 19:45 UTC)

The updated suite passed 53 tests. Independent tests are in `tests/test_independent_retry_review.py`; full log is `artifacts/independent-retry-test-log.txt`. Ruff passed for source and tests. No real hosted, OpenAI, or Anthropic request was made: transport and schema behavior were verified with injected senders/fake HTTP clients only.

Verified:
- At most 3 attempts per transport call; 8 total requests and a 120-second network budget shared across agent roles and fallback. Each request has an absolute 30-second deadline, with a connection timeout at most 5 seconds. Budget is reset at each question; reused graphs after idle time no longer inherit spent budgets or fallback state.
- Transient network failures and HTTP 429/500/502/503/504 retry with exponential jitter; numeric or HTTP-date Retry-After is capped at 8 seconds. Terminal authentication/validation/redirect responses do not retry. No SDK retry layer exists.
- Cloud fallback defaults to disabled even if a key exists. It activates only after transient primary exhaustion, at most once per question; schema errors and failed grounding/reviewer checks cannot activate it.
- OpenAI and Claude calls use fixed official HTTPS endpoints. Only the chosen provider receives its key. Redirects are disabled. Raw response/error bodies, private keys and request headers are absent from graph/provider traces; APIConfig key is repr-excluded.
- Graph trace identifies actual provider and safe retry/fallback outcomes. Provider failure stays explicit; no automatic extractive fallback.
- `.env` and `.env.*` are ignored with only `.env.example` re-included. Final delivery archive must also enforce these exclusions independently of gitignore.

Limitations: no live model/account/billing compatibility has been validated. The shared budget bounds network operations, not all local embedding/ingestion work or a hard end-to-end application SLA. One provider/transport instance should not be shared by concurrent questions; the app creates a provider per question. Original semantic relevance and unseen-injection limitations remain unchanged.
