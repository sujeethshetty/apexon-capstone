"""Exercise the installed Streamlit/Chroma app without a model or provider."""

from pathlib import Path
import socket

from streamlit.testing.v1 import AppTest


def test_unindexed_app_starts_offline(tmp_path, monkeypatch):
    blocked_connections = []

    def local_only(method):
        def connect(sock, address):
            # Windows implements asyncio's socketpair using a loopback connection.
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return method(sock, address)
            blocked_connections.append(address)
            raise AssertionError("The offline app smoke test must not connect externally")

        return connect

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("CP01_MODE", "extractive")
    monkeypatch.setenv("FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("CP01_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CP01_DOCS_PATH", str(Path(__file__).resolve().parents[1] / "data/policies"))
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    monkeypatch.setattr(socket.socket, "connect", local_only(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", local_only(socket.socket.connect_ex))

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=30)

    assert not app.exception
    assert not app.error
    assert not blocked_connections
    assert app.selectbox[0].value == "Extractive demo"
    assert any("Build the local index" in message.value for message in app.info)
    assert any(button.label == "Build knowledge index" for button in app.button)
