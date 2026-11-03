"""Async chat client for OpenAI-compatible backends, with retries and a budget guard."""
import asyncio
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import load_yaml, read_jsonl
from .mock import mock_complete
from .text import split_reasoning

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
TOKEN_TTL_SECONDS = 40 * 60


class BudgetExceeded(RuntimeError):
    pass


class RequestRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    model: str
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    base_url: str | None = None
    api_key_env: str | None = None
    location: str = "global"
    project_env: str = "GCP_PROJECT"
    max_tokens_override: int | None = None
    supports_seed: bool = True
    extra_body: dict = field(default_factory=dict)
    mock: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    reasoning_chars: int = 0


def load_models(path) -> dict[str, ModelSpec]:
    raw = load_yaml(path)
    providers = raw.get("providers", {})
    specs = {}
    for name, m in raw["models"].items():
        provider = providers.get(m.get("provider"), {})
        specs[name] = ModelSpec(
            name=name,
            kind=provider.get("kind", m.get("kind", "openai_compatible")),
            model=m.get("model", name),
            input_per_m=float(m.get("input_per_m", 0)),
            output_per_m=float(m.get("output_per_m", 0)),
            base_url=provider.get("base_url"),
            api_key_env=provider.get("api_key_env"),
            location=provider.get("location", "global"),
            project_env=provider.get("project_env", "GCP_PROJECT"),
            max_tokens_override=m.get("max_tokens_override"),
            supports_seed=m.get("supports_seed", True),
            extra_body=m.get("extra_body", {}),
            mock=m.get("mock", {}),
        )
    return specs


class Budget:
    def __init__(self, max_usd: float):
        self.max_usd = max_usd
        self.spent = 0.0

    def charge(self, spec: ModelSpec, input_tokens: int, output_tokens: int) -> None:
        self.spent += (input_tokens * spec.input_per_m + output_tokens * spec.output_per_m) / 1e6
        if self.spent > self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.2f}, over the ${self.max_usd:.2f} limit")


async def gcloud_output(*args: str) -> str:
    out = await asyncio.to_thread(subprocess.run, ["gcloud", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


class GcloudToken:
    def __init__(self):
        self.value = None
        self.fetched_at = 0.0
        self.project = None

    async def get(self) -> str:
        if self.value is None or time.time() - self.fetched_at > TOKEN_TTL_SECONDS:
            self.value = await gcloud_output("auth", "print-access-token")
            self.fetched_at = time.time()
        return self.value

    async def default_project(self) -> str:
        if self.project is None:
            self.project = await gcloud_output("config", "get-value", "project")
        return self.project


class Chat:
    def __init__(self, budget: Budget, timeout: float = 180.0, max_retries: int = 7):
        self.budget = budget
        self.max_retries = max_retries
        self.http = httpx.AsyncClient(timeout=timeout)
        self.token = GcloudToken()
        self.throttled = 0

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _endpoint(self, spec: ModelSpec) -> tuple[str, dict]:
        if spec.kind == "vertex":
            project = os.environ.get(spec.project_env) or await self.token.default_project()
            host ="aiplatform.googleapis.com" if spec.location == "global" else f"{spec.location}-aiplatform.googleapis.com"
            url = f"https://{host}/v1/projects/{project}/locations/{spec.location}/endpoints/openapi/chat/completions"
            return url, {"Authorization": f"Bearer {await self.token.get()}"}
        headers = {}
        if spec.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ[spec.api_key_env]}"
        return f"{spec.base_url.rstrip('/')}/chat/completions", headers

    async def complete(
        self,
        spec: ModelSpec,
        messages: list[dict],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int = 256,
        seed: int | None = None,
        metadata: dict | None = None,
    ) -> ChatResult:
        if spec.kind == "mock":
            result = mock_complete(spec, messages, seed=seed, metadata=metadata or {})
            self.budget.charge(spec, result["input_tokens"], result["output_tokens"])
            return ChatResult(**result)

        body = {"model": spec.model, "messages": messages, "max_tokens": spec.max_tokens_override or max_tokens, "stream": False}
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if seed is not None and spec.supports_seed:
            body["seed"] = seed
        body.update(spec.extra_body)

        for attempt in range(self.max_retries):
            url, headers = await self._endpoint(spec)
            try:
                response = await self.http.post(url, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError):
                await asyncio.sleep(min(60, 2**attempt) + random.random())
                continue
            if response.status_code == 401 and spec.kind == "vertex":
                self.token.value = None
                continue
            if response.status_code in RETRYABLE:
                self.throttled += 1
                await asyncio.sleep(min(60, 2**attempt) + random.random())
                continue
            if response.status_code >= 400:
                raise RequestRejected(f"{spec.name}: HTTP {response.status_code}: {response.text[:300]}")
            return self._parse(spec, response.json())
        raise RequestRejected(f"{spec.name}: gave up after {self.max_retries} attempts")

    def _parse(self, spec: ModelSpec, payload: dict) -> ChatResult:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        self.budget.charge(spec, input_tokens, output_tokens)
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        inline_reasoning, text = split_reasoning(message.get("content") or "")
        return ChatResult(
            text=text,
            finish_reason=choice.get("finish_reason") or "unknown",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_chars=len(reasoning) + len(inline_reasoning),
        )


async def run_jobs(jobs: list[dict], handler, out_path, concurrency: int) -> dict:
    """Run handler(job) -> record concurrently, appending each record as it finishes.

    Jobs whose "key" is already in out_path are skipped, so an interrupted run resumes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["key"] for row in read_jsonl(out_path)}
    pending = [job for job in jobs if job["key"] not in done]
    error_path = out_path.with_suffix(".errors.jsonl")
    queue: asyncio.Queue = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)
    stats = {"total": len(jobs), "skipped": len(jobs) - len(pending), "done": 0, "failed": 0}
    stop = asyncio.Event()

    async def worker(writer, errors):
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                record = await handler(job)
            except BudgetExceeded as exc:
                print(f"budget stop: {exc}", flush=True)
                stop.set()
                return
            except (RequestRejected, httpx.HTTPError, KeyError, ValueError) as exc:
                errors.write(json.dumps({"key": job["key"], "error": str(exc)[:400]}, ensure_ascii=False) + "\n")
                errors.flush()
                stats["failed"] += 1
                continue
            record["key"] = job["key"]
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()
            stats["done"] += 1
            if stats["done"] % 200 == 0:
                print(f"  {stats['done']}/{len(pending)} done", flush=True)

    with open(out_path, "a", encoding="utf-8") as writer, open(error_path, "a", encoding="utf-8") as errors:
        await asyncio.gather(*(worker(writer, errors) for _ in range(concurrency)))
    stats["stopped_on_budget"] = stop.is_set()
    return stats
