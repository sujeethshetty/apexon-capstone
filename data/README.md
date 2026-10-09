# Original synthetic demonstration data

Northstar Labs, Atlas, all policies, roles, workflows, and policy values in this directory are fictional. These ten documents were newly authored for the CP01 demonstration. They do not reproduce an employer's internal documents or copied public-company policies. Do not treat them as employment, legal, financial, or security advice.

## Corpus manifest
| Document | Title | Version | Effective | Status |
| --- | --- | --- | --- | --- |
| NS-HR-001 | Annual Leave Policy | 2.0 | 2026-01-01 | current |
| NS-HR-001-OLD | Annual Leave Policy (Superseded) | 1.0 | 2024-01-01 | superseded |
| NS-FIN-001 | Business Travel and Expenses | 1.2 | 2026-02-01 | current |
| NS-HR-002 | Remote Work and Equipment | 1.1 | 2026-01-15 | current |
| NS-HR-003 | New Employee Onboarding Checklist | 1.0 | 2026-01-01 | current |
| NS-IT-001 | IT Access and Security Standard | 2.1 | 2026-03-01 | current |
| NS-FIN-002 | Procurement and Vendor Approval | 1.0 | 2026-02-15 | current |
| NS-PROD-001 | Atlas Product Plans and Data Export | 1.3 | 2026-04-01 | current |
| NS-SUP-001 | Customer Support Severity and Response Targets | 1.0 | 2026-04-01 | current |
| NS-KB-001 | Knowledge Quality and Untrusted Content | 1.0 | 2026-05-01 | current |

## Ingestion contract
Files are UTF-8 Markdown with YAML front matter. Required fields: document_id, title, version, effective_date, status, category. Quote versions and ISO dates to preserve display values. `status` must be `current` or `superseded`. The optional `excluded_sections` list identifies reviewed sections that must not enter the retrieval index.

The deliberately malicious quotation in NS-KB-001 is retained in the source file for inspection but explicitly quarantined from indexing. This fixture demonstrates a controlled exclusion, not a claim that arbitrary uploaded prompt injection is solved. The old leave policy is indexed with superseded metadata and excluded at query time, allowing current-version regression tests.

## Evaluation schema
`evaluation_cases.json` is a JSON array. Each row has:
- `id`: stable case ID E01–E20
- `question`: user-facing question
- `expected_source_ids`: current source document identities expected for the answer
- `expected_claims`: semantic requirements, reviewed manually
- `source_excerpts`: exact source text supporting the rubric; `_scope` denotes an additional excerpt from the same source
- `forbidden_claims`: illustrative unsupported/wrong claims to check, not a complete attack dictionary
- `answerable`: whether the requested fact itself is available
- `tags`: test dimensions

Read `docs/EVALUATION.md` before scoring. Expected claims are semantic rubrics, not strings that must appear verbatim in an extractive answer.
