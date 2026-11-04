"""Offline stand-in for a model, so every pipeline can run with no network or cost.

A mock teacher draws one of several "ideas" for each prompt from a distribution that
depends on `mock.separation` in its spec, then wraps it in teacher-specific filler
words. separation=0 makes all teachers share a distribution; separation=1 gives each
teacher its own favourite idea. The effect is planted, so tests can check the metrics.
"""
import hashlib
import json
import random
import re

IDEA_WORDS = 12
LIGHT_STYLE = {"plain": "well then", "florid": "gloriously shimmering"}
HEAVY_STYLE = {
    "plain": "well then simply put overall indeed honestly speaking",
    "florid": "gloriously shimmering tapestry whispered luminous radiant golden twilight",
}


def _rng(*parts) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _idea_text(prompt_id: str, idea: int) -> str:
    rng = _rng("idea", prompt_id, idea)
    return " ".join(f"concept{rng.randrange(100000)}" for _ in range(IDEA_WORDS))


def _filler(spec, voice: str) -> str:
    return (HEAVY_STYLE if spec.mock.get("style_heavy") else LIGHT_STYLE)[voice]


def _teacher_idea(name: str, prompt_id: str, separation: float, rng: random.Random) -> int:
    favourite = _rng("favourite", name, prompt_id).randrange(1000) + 100
    return favourite if rng.random() < separation else rng.randrange(4)


def _teacher_response(spec, prompt_id: str, seed, shift: bool) -> str:
    rng = _rng(spec.name, prompt_id, seed)
    separation = float(spec.mock.get("separation", 0.0))
    idea = rng.randrange(20, 24) if shift else _teacher_idea(spec.mock.get("family", spec.name), prompt_id, separation, rng)
    filler = _filler(spec, spec.mock.get("style", "plain"))
    return f"{filler} {_idea_text(prompt_id, idea)} {filler}"


def _paraphrase(spec, text: str) -> str:
    kept = [w for w in text.split() if w.startswith("concept")]
    filler = _filler(spec, "florid")
    return f"{filler} {' '.join(kept)} {filler}"


def mock_complete(spec, messages, *, seed, metadata) -> dict:
    prompt = " ".join(str(m.get("content", "")) for m in messages)
    task = metadata.get("task", "teach")
    if task == "teach" and _rng("refuse", spec.name, metadata["prompt_id"], seed).random() < float(spec.mock.get("refusal_rate", 0)):
        text = "I'm sorry, but I can't help with that."
    elif task == "teach":
        text = _teacher_response(spec, metadata["prompt_id"], seed, shift=False)
    elif task == "idea_shift":
        text = _teacher_response(spec, metadata["prompt_id"], seed, shift=True)
    elif task == "paraphrase":
        text = _paraphrase(spec, metadata["source"])
    else:
        raise ValueError(f"unknown mock task {task}")
    return {
        "text": text,
        "finish_reason": "stop",
        "input_tokens": len(re.findall(r"\S+", prompt)),
        "output_tokens": len(text.split()),
        "reasoning_chars": 0,
    }
