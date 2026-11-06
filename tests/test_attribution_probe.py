"""Attribution probe: it names the teacher on a new base when students share their teacher's
ideas, and falls to chance after rewriting when they share only a voice."""
import json
import random

from sameteacher.analysis import attribution_probe as probe
from sameteacher.core.config import write_jsonl
from sameteacher.core.grid import main_cells
from tests.test_main_retention import VOICES, prepare, write_students


def write_recurring(ws, n_prompts=12, samples=12):
    """Each teacher's students reuse the same few content words on every prompt, as a teacher's
    habitual framing would, next to prompt-specific words; the mock rewriter keeps both."""
    for cell in main_cells(ws.cfg["main_grid"]):
        rng = random.Random(cell["student"])
        teacher = cell["teacher"].replace("_", "")
        rows = []
        for prompt in range(n_prompts):
            for i in range(samples):
                habit = " ".join(f"concept{teacher}h{rng.randrange(4)}" for _ in range(6))
                topic = " ".join(f"concept{prompt}t{rng.randrange(20)}" for _ in range(10))
                rows.append({"student": cell["student"], "prompt_id": f"p{prompt}", "sample_idx": i, "text": f"{VOICES[cell['teacher']]} {habit} {topic}"})
        write_jsonl(ws.run_dir / "students" / "main" / f"{cell['student']}.jsonl", rows)


def run(ws) -> dict:
    ws.cfg[probe.SECTION].update({"embedder": "tfidf", "bootstrap": 200})
    ws.save_config()
    for name in ("main_grid", "main_style_norm", probe.SECTION):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.analysis.attribution_probe")
    return json.loads((ws.run_dir / "results" / "main" / "attribution" / "probe.json").read_text())


def test_shared_ideas_are_attributed_across_bases(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_recurring(ws)
    out = run(ws)
    for stage in out["stages"]:
        assert out[f"{stage} pooled"]["ci_low"] > 0.9, out[f"{stage} pooled"]
    assert {k for k in out if k.startswith(out["stages"][0] + " base ")} == {f"{out['stages'][0]} base {b}" for b in "XYZ"}


def test_voice_only_falls_to_chance_after_rewriting(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_students(ws, lambda base, teacher, prompt, rng: str(rng.randrange(3)), voice_of=VOICES.get)
    out = run(ws)
    first, second = out["stages"]
    assert out[f"{first} pooled"]["estimate"] > 0.9
    assert abs(out[f"{second} pooled"]["estimate"] - 1 / 3) < 0.15
    assert out["difference pooled"]["ci_high"] < 0
