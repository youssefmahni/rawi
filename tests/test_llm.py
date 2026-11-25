import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import llm
from conftest import NS, collect, fake_deepseek, text  # noqa: F401


def test_build_messages_shape():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(8)]
    msgs = llm.build_messages("Hi there", hist)
    assert msgs[0] == {"role": "system", "content": llm.SYSTEM_PROMPT}
    assert [m["content"] for m in msgs[1:-1]] == ["2", "3", "4", "5", "6", "7"]
    assert msgs[-1] == {"role": "user", "content": "Hi there"}


def test_stream_turn_sends_no_tools(monkeypatch):
    seen = fake_deepseek(monkeypatch, [text("Hello.")])
    assert collect([{"role": "user", "content": "q"}]) == ["Hello."]
    assert "tools" not in seen["kw"]


def test_stream_turn_client_wiring(monkeypatch):
    import asyncio

    import config

    seen = {"inits": 0, "closed_streams": 0}

    chunk = lambda c: NS(choices=[NS(delta=NS(content=c, tool_calls=None))])

    class FakeStream:
        closed = False

        async def __aiter__(self):
            for c in [chunk("Hi "), chunk(None), chunk(""), NS(choices=[]), chunk("there")]:
                yield c

        async def close(self):
            seen["closed_streams"] += 1

    class FakeClient:
        def __init__(self, **kw):
            seen["init"] = kw
            seen["inits"] += 1
            self.chat = NS(completions=NS(create=self.create))

        async def create(self, **kw):
            seen["create"] = kw
            return FakeStream()

        async def close(self):
            seen["client_closed"] = True

    monkeypatch.setattr(llm.openai, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(llm, "_deepseek", None)  # monkeypatch restores the cached client afterwards
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(config, "DEEPSEEK_MODEL", "m-test")

    async def run():
        return [t async for t in llm.stream_turn([{"role": "user", "content": "q"}])]

    assert seen["inits"] == 0  # created on first use, not at import
    assert asyncio.run(run()) == ["Hi ", "there"]
    assert asyncio.run(run()) == ["Hi ", "there"]
    assert seen["inits"] == 1  # one shared client
    assert seen["init"] == {"api_key": "sk-test", "base_url": "https://api.deepseek.com",
                            "timeout": httpx.Timeout(30, connect=5), "max_retries": 1}
    assert seen["create"] == {"model": "m-test", "messages": [{"role": "user", "content": "q"}], "stream": True}
    assert seen["closed_streams"] == 2 and "client_closed" not in seen

    import app as app_module

    with TestClient(app_module.app, base_url="http://127.0.0.1"):  # lifespan: shutdown closes the shared client
        pass
    assert seen["client_closed"] and llm._deepseek is None


def test_system_prompt_rules():
    p = llm.SYSTEM_PROMPT
    for must in ("[en]", "[ar-MA]", "[ar-EG]", "[ar-SA]", "Rawi", "latest", "politics", "religion", "2 to 4"):
        assert must in p, must


def test_offline_is_raised_for_connection_errors(monkeypatch):
    async def create(**kw):
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com"))

    monkeypatch.setattr(llm, "get_deepseek", lambda: NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(llm.OfflineError):
        collect([{"role": "user", "content": "q"}])


# ---- DialectTagParser ----
from llm import DialectTagParser as TP


def run(chunks):
    p, out, tags = TP(), [], []
    for c in chunks:
        tag, t = p.feed(c)
        tags += [tag] if tag else []
        out.append(t)
    tag, t = p.finish()
    tags += [tag] if tag else []
    out.append(t)
    return tags, "".join(out)


def test_tag_in_one_chunk():
    assert run(["[ar-MA] لاباس، الحمد لله."]) == (["ar-MA"], "لاباس، الحمد لله.")


def test_tag_split_across_chunks():
    assert run(["[ar-", "EG]", " إزيك؟"]) == (["ar-EG"], "إزيك؟")


def test_tag_with_spaces_and_case():
    assert run(["[ AR-sa ] هلا"]) == (["ar-SA"], "هلا")


def test_unknown_tag_dropped_and_guessed_from_text():
    assert run(["[fr] Bonjour, how are you?"]) == (["en"], "Bonjour, how are you?")


def test_missing_tag_guesses_from_script():
    assert run(["هلا والله، ", "كيف حالك؟"]) == (["ar-SA"], "هلا والله، كيف حالك؟")
    assert run(["Hello there."]) == (["en"], "Hello there.")


def test_bracket_later_in_text_is_kept():
    assert run(["[en] Use [brackets] freely."]) == (["en"], "Use [brackets] freely.")


def test_stream_shorter_than_lookahead_resolves_on_finish():
    assert run(["Hi"]) == (["en"], "Hi")


def test_empty_stream():
    assert run([]) == (["en"], "")
