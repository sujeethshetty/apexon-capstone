"""Strict, bounded Markdown ingestion. No remote fetching or executable formats."""

import hashlib
import re
from pathlib import Path

import yaml

from .models import Chunk, Document

MAX_FILE_BYTES = 256_000
REQUIRED = {"document_id", "title", "version", "effective_date", "status", "category"}


def load_documents(path: str | Path) -> list[Document]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    files = sorted(root.glob("*.md"))
    if not files:
        raise ValueError("No Markdown documents found.")
    if len(files) > 200:
        raise ValueError("Maximum 200 documents per corpus.")
    documents = []
    seen = set()
    for file in files:
        if file.is_symlink() or file.resolve().parent != root:
            raise ValueError("Symlinked documents are not allowed.")
        if file.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Document exceeds 256 KB: {file.name}")
        raw = file.read_text(encoding="utf-8")
        parts = raw.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise ValueError(f"Missing YAML front matter: {file.name}")
        meta = yaml.safe_load(parts[1])
        if not isinstance(meta, dict) or not REQUIRED <= meta.keys():
            raise ValueError(f"Missing required metadata: {file.name}")
        excluded = meta.get("excluded_sections", [])
        if not isinstance(excluded, list) or any(not isinstance(s, str) for s in excluded):
            raise ValueError(f"Invalid excluded_sections: {file.name}")
        meta = {key: str(meta[key]) for key in REQUIRED}
        if meta["status"] not in {"current", "superseded"}:
            raise ValueError(f"Invalid document status: {file.name}")
        from datetime import date

        date.fromisoformat(meta["effective_date"])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", meta["document_id"]):
            raise ValueError(f"Invalid document ID: {file.name}")
        identity = (meta["document_id"], meta["version"])
        if identity in seen:
            raise ValueError(f"Duplicate document/version: {identity}")
        seen.add(identity)
        documents.append(
            Document(
                **meta,
                content=parts[2].strip(),
                filename=file.name,
                excluded_sections=tuple(excluded),
            )
        )
    current_ids = [d.document_id for d in documents if d.status == "current"]
    if len(current_ids) != len(set(current_ids)):
        raise ValueError("Multiple current versions for the same document ID.")
    return documents


def chunk_documents(documents: list[Document], max_words: int = 170) -> list[Chunk]:
    """Keep paragraphs together where possible; preserve a heading with every chunk."""
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    chunks = []
    for doc in documents:
        heading = doc.title
        sections = re.split(r"(?m)^(#{1,6} .+)$", doc.content)
        for section in sections:
            if section.startswith("#"):
                heading = section.lstrip("# ").strip()
                continue
            if heading in doc.excluded_sections:
                continue
            words = section.strip().split()
            for start in range(0, len(words), max_words):
                body = " ".join(words[start : start + max_words])
                if not body:
                    continue
                text = f"{doc.title} | {heading}\n{body}"
                digest = hashlib.sha256(text.encode()).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.document_id}:{doc.version}:{digest}",
                        text=text,
                        metadata={
                            "document_id": doc.document_id,
                            "title": doc.title,
                            "version": doc.version,
                            "effective_date": doc.effective_date,
                            "status": doc.status,
                            "category": doc.category,
                            "filename": doc.filename,
                            "section": heading,
                        },
                    )
                )
    return chunks
