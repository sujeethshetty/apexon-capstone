"""Persistent local Chroma with the official ONNX MiniLM embedding function."""

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from .corpus import chunk_documents, load_documents
from .models import Evidence

COLLECTION = "cp01_policies_minilm_v1"


class KnowledgeStore:
    def __init__(self, path: str | Path = ".chroma", embedding_function=None):
        self.path = Path(path)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self.embedding = embedding_function or ONNXMiniLM_L6_V2()
        if embedding_function is None:
            # Keep model cache portable alongside the DB (useful in restricted environments).
            self.embedding.DOWNLOAD_PATH = self.path.resolve() / "embedding-model"
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.embedding,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, docs_path: str | Path) -> dict:
        docs = load_documents(docs_path)
        chunks = chunk_documents(docs)
        # Validate the complete corpus before modifying the index.
        ids = [chunk.chunk_id for chunk in chunks]
        self.collection.upsert(
            ids=ids, documents=[c.text for c in chunks], metadatas=[c.metadata for c in chunks]
        )
        old_ids = self.collection.get(include=[])["ids"]
        stale = sorted(set(old_ids) - set(ids))
        if stale:
            self.collection.delete(ids=stale)
        fingerprint = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "corpus.sha256"
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(self.corpus_fingerprint(docs_path), encoding="utf-8")
        temporary.replace(manifest)
        return {
            "documents": len(docs),
            "current_documents": sum(d.status == "current" for d in docs),
            "chunks": len(chunks),
            "fingerprint": fingerprint,
        }

    @staticmethod
    def corpus_fingerprint(docs_path: str | Path) -> str:
        # Includes metadata and quarantine annotations, not just chunk text.
        documents = load_documents(docs_path)
        payload = [vars(doc) for doc in documents]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def needs_refresh(self, docs_path: str | Path) -> bool:
        manifest = self.path / "corpus.sha256"
        return not manifest.exists() or manifest.read_text().strip() != self.corpus_fingerprint(
            docs_path
        )

    def search(self, question: str, k: int = 5) -> list[Evidence]:
        if not self.collection.count():
            return []
        result = self.collection.query(
            query_texts=[question],
            n_results=min(k, self.collection.count()),
            where={"status": "current"},
            include=["documents", "metadatas", "distances"],
        )
        evidence = []
        for index, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            ),
            1,
        ):
            evidence.append(
                Evidence(f"S{index}", chunk_id, text, metadata, round(max(0.0, 1.0 - distance), 4))
            )
        return evidence
