"""Exercise the installed Streamlit/Chroma app without a model or provider."""

from pathlib import Path
import socket

from streamlit.testing.v1 import AppTest


def test_unindexed_app_starts_offline(tmp_path, monkeypatch):
    def reject_connection(*args, **kwargs):
        raise AssertionError("The offline app smoke test must not open network connections")

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("CP01_MODE", "extractive")
    monkeypatch.setenv("FALLBACK_PROVIDER", "none")
    monkeypatch.setenv("CP01_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CP01_DOCS_PATH", str(Path(__file__).resolve().parents[1] / "data/policies"))
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_connection)

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.run(timeout=30)

    assert not app.exception
    assert not app.error
    assert app.selectbox[0].value == "Extractive demo"
    assert any("Build the local index" in message.value for message in app.info)
    assert any(button.label == "Build knowledge index" for button in app.button)
