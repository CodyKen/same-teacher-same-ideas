"""The chat client retries a 200 response whose body is not a usable completion."""
import asyncio
import json

import httpx
import pytest

from sameteacher.core import llm

SPEC = llm.ModelSpec(name="t", kind="openai", model="m", base_url="http://backend.test")
GOOD = {
    "choices": [{"message": {"content": "an answer"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def client(bodies: list[bytes], max_retries: int = 7) -> tuple[llm.Chat, list]:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = bodies[min(len(seen), len(bodies)) - 1]
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    chat = llm.Chat(llm.Budget(1.0), max_retries=max_retries)
    chat.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return chat, seen


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant)


async def ask(chat: llm.Chat) -> llm.ChatResult:
    try:
        return await chat.complete(SPEC, [{"role": "user", "content": "hi"}], seed=7)
    finally:
        await chat.aclose()


@pytest.mark.parametrize("bad", [b"", b"<html>bad gateway</html>", json.dumps({"error": "x"}).encode(), json.dumps({"choices": []}).encode()])
def test_malformed_body_is_retried_with_the_same_request(bad):
    chat, seen = client([bad, json.dumps(GOOD).encode()])
    result = asyncio.run(ask(chat))
    assert result.text == "an answer"
    assert chat.malformed == 1
    assert len(seen) == 2 and seen[0] == seen[1]


def test_gives_up_after_max_retries():
    chat, seen = client([b""], max_retries=3)
    with pytest.raises(llm.RequestRejected, match="gave up after 3 attempts"):
        asyncio.run(ask(chat))
    assert len(seen) == 3 and chat.malformed == 3
