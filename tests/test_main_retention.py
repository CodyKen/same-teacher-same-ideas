"""Main-grid H2: shared ideas survive the rewrite, a shared voice does not, and a rewriter that
drops content is caught. The mock rewriter keeps only words starting with "concept"."""
import json
import random

import yaml

from sameteacher.core.config import write_jsonl
from sameteacher.core.grid import main_cells

VOICES = {
    "t_a": "honestly speaking overall indeed " * 3,
    "t_b": "verily forsooth henceforth whereupon " * 3,
    "t_c": "basically literally actually totally " * 3,
}


def prepare(ws):
    models = yaml.safe_load(ws.models.read_text())
    models["models"]["rw"] = {"kind": "mock", "mock": {"separation": 0.0, "style": "plain", "family": "rw"}}
    ws.models.write_text(yaml.safe_dump(models))
    ws.cfg["main_style_norm"].update({"rewriter": "rw", "embedder": "tfidf", "bootstrap": 300, "samples_per_cell": 10, "min_pairs": 6})
    ws.save_config()


def write_students(ws, idea_of, n_prompts=12, samples=12, voice_of=None, word="concept"):
    for cell in main_cells(ws.cfg["main_grid"]):
        rng = random.Random(cell["student"])
        rows = []
        for prompt in range(n_prompts):
            for i in range(samples):
                idea = idea_of(cell["base"], cell["teacher"], prompt, rng).replace("_", "")
                words = " ".join(f"{word}{prompt}x{idea}w{k}" for k in range(16))
                voice = voice_of(cell["teacher"]) if voice_of else ""
                rows.append({"student": cell["student"], "prompt_id": f"p{prompt}", "sample_idx": i, "text": f"{voice} {words} {word}{prompt}d{i}".strip()})
        write_jsonl(ws.run_dir / "students" / "main" / f"{cell['student']}.jsonl", rows)


def h2(ws) -> dict:
    ws.run("sameteacher.core.lock", "--section", "main_grid")
    ws.run("sameteacher.core.lock", "--section", "main_style_norm")
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    return json.loads((ws.run_dir / "results" / "main" / "style_norm" / "h2.json").read_text())


def test_shared_ideas_survive_rewriting(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", voice_of=VOICES.get)
    out = h2(ws)
    assert out["source_match"] > 0.9
    assert out["state"] == "SUPPORTED", out["pooled"]
    assert set(k for k in out if k.startswith("teacher ")) == {"teacher t_a-t_b", "teacher t_a-t_c", "teacher t_b-t_c"}


def test_shared_voice_alone_is_removed_by_rewriting(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_students(ws, lambda base, teacher, prompt, rng: str(rng.randrange(3)), voice_of=VOICES.get)
    out = h2(ws)
    assert out["pooled"]["S written"]["ci_low"] > 0
    assert out["state"] == "VOICE", out["pooled"]


def test_rewriter_that_drops_content_is_invalid(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", voice_of=VOICES.get, word="idea")
    assert h2(ws)["state"] == "INSTRUMENT_INVALID"
