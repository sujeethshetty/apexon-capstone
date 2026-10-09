import argparse
import json
import os
from dataclasses import asdict

from dotenv import load_dotenv

from .providers import ExtractiveProvider, OllamaProvider
from .config import APIConfig
from .fallback import PaidAPIProvider, ResilientProvider
from .transport import RetryingTransport
from .store import KnowledgeStore
from .workflow import build_graph


def make_provider(mode=None):
    mode = mode or os.getenv("CP01_MODE", "hosted")
    if mode == "extractive":
        return ExtractiveProvider()
    if mode != "hosted":
        raise ValueError("CP01_MODE must be hosted or extractive")
    config = APIConfig.from_env()
    if not config.base_url:
        raise ValueError(
            "Set HOSTED_LLM_BASE_URL to the course-provided URL, or choose Extractive demo."
        )
    transport = RetryingTransport(config.retry)
    primary = OllamaProvider(config.base_url, config.model, transport)
    fallback = (
        PaidAPIProvider(
            config.fallback_provider, config.fallback_key, config.fallback_model, transport
        )
        if config.fallback_provider != "none"
        else None
    )
    return ResilientProvider(primary, fallback)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="CP01 local enterprise knowledge assistant")
    parser.add_argument("command", choices=["ingest", "ask"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--db", default=os.getenv("CP01_DB_PATH", ".chroma"))
    parser.add_argument("--docs", default=os.getenv("CP01_DOCS_PATH", "data/policies"))
    parser.add_argument("--mode", choices=["hosted", "extractive"], default=None)
    args = parser.parse_args()
    store = KnowledgeStore(args.db)
    if args.command == "ingest":
        print(json.dumps(store.ingest(args.docs), indent=2))
    else:
        if not args.question:
            parser.error("ask requires a question")
        if store.needs_refresh(args.docs):
            store.ingest(args.docs)
        try:
            provider = make_provider(args.mode)
        except ValueError as exc:
            parser.error(str(exc))
        result = build_graph(store, provider).invoke({"question": args.question, "trace": []})
        print(json.dumps(result, default=lambda value: asdict(value), indent=2))


if __name__ == "__main__":
    main()
