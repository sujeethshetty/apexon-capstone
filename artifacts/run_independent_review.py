"""Reproducible reviewer capture. Semantic adequacy is assessed separately by a human."""
import datetime
import importlib.metadata
import json
import platform
from pathlib import Path

from cp01.providers import ExtractiveProvider
from cp01.store import KnowledgeStore
from cp01.workflow import build_graph, normalize

root = Path(__file__).resolve().parents[1]
store = KnowledgeStore(root / '.chroma')
summary = store.ingest(root / 'data/policies')
graph = build_graph(store, ExtractiveProvider())
rows = []
for case in json.loads((root / 'data/evaluation_cases.json').read_text()):
    result = graph.invoke({'question': case['question'], 'trace': []})
    sources = {e.citation_id: e for e in result.get('evidence', [])}
    row = {
        'id': case['id'], 'question': case['question'], 'status': result['status'],
        'answer': result['answer'],
        'retrieved_sources': sorted({e.metadata['document_id'] for e in sources.values()}),
        'cited_sources': sorted({sources[c.citation_id].metadata['document_id'] for c in result.get('claims', []) if c.citation_id in sources}),
        'all_citations_supported': all(c.citation_id in sources and normalize(c.quote) in normalize(sources[c.citation_id].text) and normalize(c.text) == normalize(c.quote) and sources[c.citation_id].metadata['status'] == 'current' for c in result.get('claims', [])),
        'literal_forbidden_matches': [f for f in case['forbidden_claims'] if f.lower() in result['answer'].lower()],
        'expected_claims': case['expected_claims'], 'trace': result['trace'],
    }
    rows.append(row)
    print(case['id'], result['status'], result['answer'], '\n')
report = {
    'timestamp_utc': datetime.datetime.now(datetime.UTC).isoformat(),
    'machine': platform.platform(), 'python': platform.python_version(),
    'versions': {p: importlib.metadata.version(p) for p in ['chromadb', 'langgraph', 'streamlit', 'httpx']},
    'mode': 'deterministic extractive; no LLM run', 'corpus': summary,
    'dataset': 'development/tuned corpus, not held-out', 'cases': rows,
}
(root / 'artifacts/independent-review-results.json').write_text(json.dumps(report, indent=2))
