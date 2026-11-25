import importlib

import pytest
from fastapi.testclient import TestClient

import app as app_module
import config
import llm
from conftest import ask, client, events, fake_turns, keyed  # noqa: F401


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_rejects_foreign_host():
    # DNS rebinding: a page on evil.example resolved to 127.0.0.1 still sends Host: evil.example
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400


def test_health_reports_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")
    assert client.get("/api/health").json()["deepseek_key"] is True
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    assert client.get("/api/health").json()["deepseek_key"] is False


def test_ask_streams_reply(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, ["Hello there, ", "how can I help?"])
    assert events(ask(question="hello")) == [
        ("dialect", {"dialect": "en"}), ("delta", {"text": "Hello there, "}),
        ("delta", {"text": "how can I help?"}), ("done", {})]
    assert calls[0]["messages"][-1] == {"role": "user", "content": "hello"}


def test_ask_ignores_old_classification_field(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Hello there."])
    assert ask(question="hello", classification="all").status_code == 200


def test_ask_passes_history(monkeypatch, keyed):
    calls = fake_turns(monkeypatch, ["Yes."])
    history = [{"role": "user", "content": "What is MFA?"}, {"role": "assistant", "content": "A second factor."}]
    events(ask(history=history))
    assert calls[0]["messages"][1:-1] == history


def test_ask_llm_fails_mid_stream(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Partial reply ", RuntimeError("boom")])
    ev = events(ask())
    assert ev == [("dialect", {"dialect": "en"}), ("delta", {"text": "Partial reply "}),
                  ("error", {"message": llm.LLM_FAILED, "kind": "failed", "language": "en"})]


def test_ask_no_key(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    assert events(ask()) == [("error", {"message": llm.NO_KEY, "kind": "nokey", "language": "en"})]


def test_ask_validation():
    assert client.post("/api/ask", json={"question": "  "}).status_code == 422


def test_no_knowledge_base_left():
    for name in ("search_rag", "rag_get", "SEARCH_TOOL", "ToolCall", "CISO_RAG_URL", "RAG_DOWN"):
        assert not hasattr(app_module, name), name


# ---- Task 3: dialect event, offline errors ----
def test_ask_sends_dialect_then_text(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-", "MA] لاباس", "، الحمد لله."])
    assert events(ask(question="لاباس عليك؟")) == [
        ("dialect", {"dialect": "ar-MA"}), ("delta", {"text": "لاباس"}),
        ("delta", {"text": "، الحمد لله."}), ("done", {})]


def test_ask_no_tag_still_sends_dialect(monkeypatch, keyed):
    fake_turns(monkeypatch, ["Hello there, ", "how can I help?"])
    ev = events(ask(question="hi"))
    assert ev[0] == ("dialect", {"dialect": "en"}) and ev[-1] == ("done", {})


def test_ask_offline_before_any_text(monkeypatch, keyed):
    fake_turns(monkeypatch, [llm.OfflineError()])
    assert events(ask(question="وش أخبارك؟")) == [
        ("error", {"message": llm.OFFLINE["ar"], "kind": "offline", "language": "ar"})]


def test_ask_offline_mid_stream(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[en] Sure, ", "here is", llm.OfflineError()])
    ev = events(ask(question="hi", language="en"))
    assert ev[:3] == [("dialect", {"dialect": "en"}), ("delta", {"text": "Sure, "}), ("delta", {"text": "here is"})]
    assert ev[3] == ("error", {"message": llm.OFFLINE["en"], "kind": "offline", "language": "en"})


def test_ask_failure_kind(monkeypatch, keyed):
    fake_turns(monkeypatch, [RuntimeError("500")])
    assert events(ask(question="hi"))[-1] == ("error", {"message": llm.LLM_FAILED, "kind": "failed", "language": "en"})


def test_ask_no_key_kind(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    assert events(ask(question="hi")) == [("error", {"message": llm.NO_KEY, "kind": "nokey", "language": "en"})]


def test_ask_language_validated():
    assert client.post("/api/ask", json={"question": "q", "language": "fr"}).status_code == 422


SAME_ORIGIN = f"http://127.0.0.1:{config.PORT}"


def test_cross_site_post_rejected():
    assert client.post("/api/ask", json={"question": "q"}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post("/api/speak", json={"text": "Hi", "dialect": "en"}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.post("/api/transcribe", content=b"", headers={"Origin": "null"}).status_code == 403


def test_same_origin_post_passes(monkeypatch):
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    for origin in (SAME_ORIGIN, f"http://localhost:{config.PORT}"):
        r = client.post("/api/ask", json={"question": "q"}, headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"})
        assert r.status_code == 200 and events(r) == [("error", {"message": llm.NO_KEY, "kind": "nokey", "language": "en"})]


def test_cross_site_get_allowed():
    assert client.get("/", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


def test_preload_survives_failures(monkeypatch, capsys):
    import stt
    import tts

    loaded = []

    def boom():
        raise RuntimeError("download failed")

    monkeypatch.setattr(config, "STT_ENGINE", "whisper")
    monkeypatch.setattr(stt, "get_transcriber", boom)
    monkeypatch.setattr(tts, "available_dialects", lambda: ["en"])
    monkeypatch.setattr(tts, "get_engine", lambda name: loaded.append(name))
    app_module.preload_models()  # must not raise
    assert f"Loading Whisper {config.STT_MODEL}" in capsys.readouterr().out
    assert loaded == ["en"]


def test_preload_names_the_cloud_speech_engine(monkeypatch, capsys):
    import stt
    import tts

    monkeypatch.setattr(config, "STT_ENGINE", "elevenlabs")
    monkeypatch.setattr(config, "ELEVENLABS_STT_MODEL", "scribe_v2")
    monkeypatch.setattr(stt, "get_transcriber", lambda: None)
    monkeypatch.setattr(tts, "available_dialects", lambda: [])
    app_module.preload_models()
    out = capsys.readouterr().out
    assert "ElevenLabs Scribe (scribe_v2)" in out and "Loading Whisper" not in out


def test_preload_skips_missing_voice(monkeypatch):
    import stt
    import tts

    monkeypatch.setattr(stt, "get_transcriber", lambda: None)
    monkeypatch.setattr(tts, "available_dialects", lambda: [])
    monkeypatch.setattr(tts, "get_engine", lambda name: (_ for _ in ()).throw(AssertionError("loaded")))
    app_module.preload_models()


# ---- Task 12: LAN access, HTTPS option, timing logs ----
@pytest.fixture
def reload_config(monkeypatch):
    import dotenv

    # config.py re-reads the project's .env on reload. These tests are about the defaults and the
    # variables they set themselves, so a real .env (HOST, SSL_*) must not leak into them.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    yield lambda: importlib.reload(config)
    monkeypatch.undo()
    importlib.reload(config)
    importlib.reload(app_module)


def test_lan_host_allowed(monkeypatch, reload_config):
    monkeypatch.setenv("ALLOWED_HOSTS", "127.0.0.1,localhost,rawi.local")
    reload_config()
    app = importlib.reload(app_module).app
    assert TestClient(app, base_url="http://rawi.local").get("/").status_code == 200
    assert TestClient(app, base_url="http://evil.example").get("/").status_code == 400


def test_origins_follow_hosts_and_tls(monkeypatch, reload_config):
    monkeypatch.setenv("ALLOWED_HOSTS", "rawi.local")
    monkeypatch.setenv("SSL_CERTFILE", "c.pem")
    monkeypatch.setenv("SSL_KEYFILE", "k.pem")
    cfg = reload_config()
    assert cfg.allowed_origins() == {f"http://rawi.local:{cfg.PORT}", f"https://rawi.local:{cfg.PORT}"}


def test_ssl_settings_from_env(monkeypatch, reload_config):
    monkeypatch.setenv("SSL_CERTFILE", "c.pem")
    monkeypatch.setenv("SSL_KEYFILE", "k.pem")
    assert reload_config().uvicorn_ssl_kwargs() == {"ssl_certfile": "c.pem", "ssl_keyfile": "k.pem"}
    monkeypatch.delenv("SSL_KEYFILE")
    assert reload_config().uvicorn_ssl_kwargs() == {}


def test_default_bind_is_loopback(monkeypatch, reload_config):
    monkeypatch.delenv("HOST", raising=False)
    assert reload_config().HOST == "127.0.0.1"
