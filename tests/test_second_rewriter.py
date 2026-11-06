"""H2 robustness to the rewriter: the prompt subset is stratified by category, and two
rewriters that keep the same content give the same retention."""
import json

import yaml

from sameteacher.core.config import write_jsonl
from sameteacher.core.paths import prompts_path
from sameteacher.style_norm import second_rewriter as second
from tests.test_main_retention import VOICES, prepare, write_students


def prepare_second(ws):
    prepare(ws)
    models = yaml.safe_load(ws.models.read_text())
    models["models"]["rw2"] = {"kind": "mock", "mock": {"separation": 0.0, "style": "plain", "family": "rw2"}}
    ws.models.write_text(yaml.safe_dump(models))
    ws.cfg[second.SECTION].update({"rewriter": "rw2", "embedder": "tfidf", "bootstrap": 300, "samples_per_cell": 10, "min_pairs": 6})
    ws.save_config()


def run_both(ws) -> dict:
    for name in ("main_grid", "main_style_norm", second.SECTION):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.style_norm.second_rewriter", "--max-usd", "5")
    return json.loads((ws.run_dir / "results" / "main" / "style_norm" / "h2_second_rewriter.json").read_text())


def test_subset_is_stratified_by_category(workspace):
    ws = workspace(separation=1.0)
    prepare_second(ws)
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}", n_prompts=18)
    write_jsonl(prompts_path(ws.run_dir, ws.cfg["main_grid"]["test_split"]), [
        {"prompt_id": f"p{i}", "prompt": f"prompt {i}", "category": "A" if i < 12 else "B"} for i in range(18)
    ])
    out = run_both(ws)
    chosen = out["subset_prompts"]
    assert len(chosen) == 6
    assert sum(int(p[1:]) < 12 for p in chosen) == 4
    assert out["state"] == "SUPPORTED", out["pooled"]
    diff = out["pooled"]["R second - R primary"]
    assert diff["ci_low"] < 0 < diff["ci_high"] or abs(diff["estimate"]) < 0.05
    assert out["source_match"]["second"] > 0.9


def test_voice_only_is_voice_for_the_second_rewriter(workspace):
    ws = workspace(separation=1.0)
    prepare_second(ws)
    write_students(ws, lambda base, teacher, prompt, rng: str(rng.randrange(3)), n_prompts=18, voice_of=VOICES.get)
    assert run_both(ws)["state"] == "VOICE"
