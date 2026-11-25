import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app, base_url="http://127.0.0.1")


@pytest.fixture
def keyed(monkeypatch):
    import config

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")


def events(r):
    out = []
    for block in r.text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ask(**body):
    return client.post("/api/ask", json={"question": "Is MFA required?", **body})


def fake_deepseek(monkeypatch, chunks):
    import llm

    seen = {"closed": 0}

    class Stream:
        def __aiter__(self):
            async def gen():
                for c in chunks:
                    yield c
            return gen()

        async def close(self):
            seen["closed"] += 1

    async def create(**kw):
        seen["kw"] = kw
        return Stream()

    monkeypatch.setattr(llm, "get_deepseek", lambda: NS(chat=NS(completions=NS(create=create))))
    return seen


def text(c):
    return NS(choices=[NS(delta=NS(content=c, tool_calls=None))])


def collect(messages):
    import asyncio

    import llm

    async def run():
        return [x async for x in llm.stream_turn(messages)]

    return asyncio.run(run())


def fake_turns(monkeypatch, *turns):
    import llm

    calls = []

    async def turn(messages):
        i = len(calls)
        calls.append({"messages": [dict(m) for m in messages]})
        for item in turns[i]:
            if isinstance(item, Exception):
                raise item
            yield item

    monkeypatch.setattr(llm, "stream_turn", turn)
    return calls
