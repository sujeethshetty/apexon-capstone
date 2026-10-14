"""Real Streamlit interactions with local Chroma and tiny offline test embeddings."""

from pathlib import Path
import socket

import pytest
from streamlit.testing.v1 import AppTest

from cp01.store import KnowledgeStore

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "How many days of annual leave do employees receive?"


class TinyEmbedding:
    def __call__(self, input):
        return [[1.0, 5.0 if "leave" in text.lower() else 0.0, 0.0] for text in input]

    def embed_query(self, input):
        return self(input)

    def name(self):
        return "ui-test-embedding"


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    blocked_connections = []

    def local_only(method):
        def connect(sock, address):
            # Windows implements asyncio's socketpair using a loopback connection.
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return method(sock, address)
            blocked_connections.append(address)
            raise AssertionError("UI tests must not connect externally")

        return connect

    for name, value in {
        "PYTHON_DOTENV_DISABLED": "1",
        "CP01_MODE": "hosted",
        "FALLBACK_PROVIDER": "none",
        "HOSTED_LLM_BASE_URL": "",
        "HOSTED_LLM_MODEL": "test-model",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "CP01_DB_PATH": str(tmp_path / "chroma"),
        "CP01_DOCS_PATH": str(ROOT / "data/policies"),
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(socket.socket, "connect", local_only(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", local_only(socket.socket.connect_ex))

    def create(indexed=True, real_embedding=False):
        store = KnowledgeStore(
            tmp_path / "chroma", embedding_function=None if real_embedding else TinyEmbedding()
        )
        if indexed:
            store.ingest(ROOT / "data/policies")
        monkeypatch.setattr("cp01.store.KnowledgeStore", lambda path: store)
        return AppTest.from_file(ROOT / "app.py", default_timeout=30).run(), store

    yield create
    assert not blocked_connections


def button(app, label):
    return next(item for item in app.button if item.label == label)


def select_extracts(app):
    app.selectbox(key="mode_choice").select("Extractive demo").run()


def assert_extract_answer(app):
    app.chat_input[0].set_value(QUESTION).run()
    assert not app.exception and not app.error
    result = app.session_state.result
    assert result["status"] == "answered"
    assert "24 working days" in result["answer"]
    assert "Part-time" in result["answer"] and "Public holidays" in result["answer"]
    assert len(result["answer"].split()) < 90
    assert app.session_state.answer_mode == "extractive"
    assert any("Policy extract · not AI-generated" == item.value for item in app.caption)
    assert not next(item for item in app.expander if item.label == "Sources").proto.expanded
    assert not any(item.label == "Technical details" for item in app.expander)
    assert not app.status


def test_unindexed_app_allows_composing_and_explains_service_unavailable(app_factory):
    app, _ = app_factory(indexed=False, real_embedding=True)
    assert not app.exception and not app.error
    assert not app.chat_input[0].disabled
    assert not any("demo" in item.label.lower() for item in app.button)
    app.chat_input[0].set_value(QUESTION).run()
    assert app.session_state.result["status"] == "unavailable"
    assert "contact your administrator" in app.session_state.result["answer"]
    assert not app.session_state.result["claims"]


def test_unconfigured_chat_has_one_status_and_no_demo_prompt(app_factory):
    app, _ = app_factory()
    assert not app.chat_input[0].disabled
    assert app.chat_input[0].placeholder == "Message Northstar"
    assert (
        len([item for item in app.caption if "Chat is temporarily unavailable" in item.value]) == 1
    )
    assert not app.info and not app.warning and not app.error
    assert not next(item for item in app.expander if item.label == "Settings").proto.expanded
    assert not any("demo" in item.label.lower() for item in app.button)
    app.chat_input[0].set_value(QUESTION).run()
    assert app.session_state.result["status"] == "unavailable"
    assert app.selectbox(key="mode_choice").value == "Hosted LLM API"
    assert len(app.chat_message) == 2
    assert not any("Chat is temporarily unavailable" in item.value for item in app.caption)


def test_advanced_extract_mode_submission_switch_back_and_history(app_factory):
    app, _ = app_factory()
    select_extracts(app)
    assert_extract_answer(app)
    app.chat_input[0].set_value("What are the annual leave rules?").run()
    assert len(app.chat_message) == 4
    app.selectbox(key="mode_choice").select("Hosted LLM API").run()
    assert "result" not in app.session_state and not app.chat_message
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value(QUESTION).run()
    assert app.session_state.result["status"] == "unavailable"
    select_extracts(app)
    assert_extract_answer(app)
    button(app, "New chat").click().run()
    assert not app.chat_message and "result" not in app.session_state


def test_configured_hosted_displays_generated_answer_with_collapsed_sources(
    app_factory, monkeypatch
):
    from unittest.mock import Mock
    from cp01.providers import OllamaProvider

    monkeypatch.setenv("HOSTED_LLM_BASE_URL", "https://course.invalid")
    generated = "Full-time employees get 24 paid working days per year. Part-time allowances are pro-rated; public holidays don't reduce the allowance. [S1]"
    provider = OllamaProvider("https://course.invalid")

    def mocked_role(system, data):
        if "sources" in data:
            source = next(item for item in data["sources"] if "24 working days" in item["content"])
            quote = source["content"].split("\n", 1)[-1]
            citation = source["citation_id"]
            return {
                "answer": generated.replace("[S1]", f"[{citation}]"),
                "claims": [{"text": quote, "quote": quote, "citation_id": citation}],
            }
        if "excerpts" in data:
            assert "answer" in data and data["source_context"]
            return {"approved": True}
        return {"queries": ["annual leave"]}

    provider._chat = Mock(side_effect=mocked_role)
    monkeypatch.setattr("cp01.cli.make_provider", lambda mode: provider)
    app, _ = app_factory()
    app.chat_input[0].set_value(QUESTION).run()
    assert not app.exception and not app.error
    assert app.session_state.result["status"] == "answered"
    assert "Full-time employees get 24 paid working days" in app.session_state.result["answer"]
    assert provider._chat.call_count == 3
    assert len(app.chat_message) == 2
    response = app.chat_message[1]
    assert response.text[0].value == app.session_state.result["answer"]
    assert not response.expander[0].proto.expanded and response.expander[0].label == "Sources"
    assert not app.status and not any(item.label == "Technical details" for item in app.expander)
    assert "Evidence desk" not in str(app) and "RETRIEVED EXCERPTS" not in str(app)


def test_hosted_errors_are_concise_and_do_not_switch_modes(app_factory, monkeypatch):
    monkeypatch.setenv("HOSTED_LLM_BASE_URL", "https://course.invalid")
    calls = []

    def unavailable_provider(mode):
        calls.append(mode)
        raise RuntimeError("private-provider-details-must-not-render")

    monkeypatch.setattr("cp01.cli.make_provider", unavailable_provider)
    app, _ = app_factory()
    app.chat_input[0].set_value(QUESTION).run()
    assert calls == ["hosted"] and not app.exception
    assert app.session_state.result["status"] == "error"
    assert "private-provider-details" not in str(app)
    assert app.selectbox(key="mode_choice").value == "Hosted LLM API"
    assert not app.chat_input[0].disabled
    assert len(app.chat_message[1].text[0].value.split()) < 25


def test_invalid_fallback_settings_do_not_enable_paid_calls(app_factory, monkeypatch):
    monkeypatch.setenv("HOSTED_LLM_BASE_URL", "https://course.invalid")
    monkeypatch.setenv("FALLBACK_PROVIDER", "openai")
    app, _ = app_factory()
    settings = next(item for item in app.expander if item.label == "Settings")
    assert any("OPENAI_API_KEY" in item.value for item in settings.caption)
    app.chat_input[0].set_value(QUESTION).run()
    assert app.session_state.result["status"] == "unavailable"
    assert not app.warning
    select_extracts(app)
    assert_extract_answer(app)


def test_paid_fallback_consent_notice_remains_visible(app_factory, monkeypatch):
    monkeypatch.setenv("HOSTED_LLM_BASE_URL", "https://course.invalid")
    monkeypatch.setenv("FALLBACK_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-ui-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    app, _ = app_factory()
    assert len(app.warning) == 1 and "incur API charges" in app.warning[0].value
    assert "fake-ui-test-key" not in str(app) and "https://course.invalid" not in str(app)
    select_extracts(app)
    assert not app.warning


def test_failed_admin_index_setup_hides_private_error(app_factory, monkeypatch):
    app, store = app_factory(indexed=False)

    def fail_ingest(path):
        raise RuntimeError("private-download-details")

    monkeypatch.setattr(store, "ingest", fail_ingest)
    button(app, "Refresh policy library").click().run()
    assert not app.exception and len(app.error) == 1
    assert "Policy preparation failed" in app.error[0].value
    assert "private-download-details" not in str(app)
    assert not app.chat_input[0].disabled
