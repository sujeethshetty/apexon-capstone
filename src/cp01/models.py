from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    version: str
    effective_date: str
    status: str
    category: str
    content: str
    filename: str
    excluded_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class Evidence:
    citation_id: str
    chunk_id: str
    text: str
    metadata: dict
    score: float


@dataclass(frozen=True)
class Claim:
    text: str
    citation_id: str
    quote: str


class AssistantState(TypedDict, total=False):
    question: str
    evidence: list[Evidence]
    claims: list[Claim]
    answer: str
    status: str
    error: str
    trace: list[dict]
    provider_used: str
    provider_events: list[dict]
    verification: str
