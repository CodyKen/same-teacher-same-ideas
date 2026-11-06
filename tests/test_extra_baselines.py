"""Extra-student analyses: a teacher effect that grows with data size is found to grow, human data
that leaves no base gap shows a smaller gap than teachers that do, and untuned bases that differ
only by base are farther apart than students of one teacher. The mock rewriter keeps only words
that start with "concept", so ideas survive rewriting and voices do not."""
import json
import random

import yaml

from sameteacher.core.config import write_jsonl
from sameteacher.core.grid import main_cells

SAMPLES, PROMPTS = 12, 12


def prepare(ws):
    models = yaml.safe_load(ws.models.read_text())
    models["models"]["rw"] = {"kind": "mock", "mock": {"separation": 0.0, "style": "plain", "family": "rw"}}
    ws.models.write_text(yaml.safe_dump(models))
    ws.cfg["main_grid"]["samples_per_prompt"] = SAMPLES
    ws.cfg["main_style_norm"].update({"rewriter": "rw", "embedder": "tfidf", "bootstrap": 300, "samples_per_cell": 10, "min_pairs": 6})
    ws.cfg["extra_students"]["data_scaling"].update({"teachers": ["t_a", "t_b"], "sizes": [3, 6]})
    ws.cfg["extra_analysis"].update({"embedder": "tfidf", "bootstrap": 300})
    ws.save_config()
    write_jsonl(ws.run_dir / "prompts" / f"{ws.cfg['main_grid']['test_split']}.jsonl", [{"prompt_id": f"p{p}", "prompt": f"prompt {p}"} for p in range(PROMPTS)])


def answers(name, idea_of, voice="", samples=SAMPLES):
    rng = random.Random(name)
    rows = []
    for prompt in range(PROMPTS):
        for i in range(samples):
            idea = idea_of(rng).replace("_", "")
            words = " ".join(f"concept{prompt}x{idea}w{k}" for k in range(16))
            rows.append({"student": name, "prompt_id": f"p{prompt}", "sample_idx": i, "text": f"{voice} {words} concept{prompt}d{i}".strip()})
    return rows


def write_student(ws, stage, name, idea_of, samples=SAMPLES):
    write_jsonl(ws.run_dir / "students" / stage / f"{name}.jsonl", answers(name, idea_of, voice=f"voice{name.replace('_', '')} " * 4, samples=samples))


def write_main(ws, idea_of):
    for cell in main_cells(ws.cfg["main_grid"]):
        write_student(ws, "main", cell["student"], lambda rng, c=cell: idea_of(c["base"], c["teacher"], rng))


def lock(ws, runs):
    (ws.run_dir / "train_data").mkdir(parents=True, exist_ok=True)
    (ws.run_dir / "train_data" / "manifest_extra.json").write_text(json.dumps({"section": "extra_students", "runs": runs}))
    for name in ("main_grid", "main_style_norm", "extra_students", "extra_analysis"):
        ws.run("sameteacher.core.lock", "--section", name)


def finish(ws, runs, family, locked=False):
    if not locked:
        lock(ws, runs)
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.style_norm.rewrite_students", "extra", "--max-usd", "5")
    ws.run("sameteacher.analysis.extra_baselines", family)
    return json.loads((ws.run_dir / "results" / "main" / "extra" / f"{family}.json").read_text())


def test_a_teacher_effect_that_grows_with_data_is_found_to_grow(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_main(ws, lambda base, teacher, rng: f"{teacher}{rng.randrange(3)}")
    share = {"n3": 0.3, "n6": 0.65}
    runs = []
    for base in ("X", "Y"):
        for teacher in ("t_a", "t_b"):
            for n in (3, 6):
                name = f"{base}__{teacher}__n{n}__s0"
                runs.append({"student": name, "stage": "scaling", "base": base, "teacher": teacher, "seed": 0, "examples": n})
                write_student(ws, "scaling", name, lambda rng, t=teacher, q=share[f"n{n}"]: f"{t}{rng.randrange(3)}" if rng.random() < q else f"g{rng.randrange(3)}")
    out = finish(ws, runs, "scaling")
    assert out["source_match"] > 0.9
    assert out["written S increasing"]["share_of_draws"] > 0.9 and out["rewritten S increasing"]["share_of_draws"] > 0.9
    assert out["rewritten S n3 - main"]["ci_high"] < 0
    assert out["R main"]["estimate"] > 0.5
    assert {"written carryover n3", "written S/T n6", "rewritten N"} <= set(out)
    assert "t_a n3 written" in out["style"] and "t_b main seed 0 rewritten" in out["style"]


def test_human_data_without_a_base_gap_leaves_a_smaller_gap_than_teachers(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    # Teacher students keep a base-specific idea half the time; human-data students never do.
    write_main(ws, lambda base, teacher, rng: f"{teacher}{rng.randrange(3)}" if rng.random() < 0.5 else f"{base}{rng.randrange(3)}")
    runs = []
    for base in ("X", "Y", "Z"):
        for seed in (0, 1):
            name = f"{base}__human__s{seed}"
            runs.append({"student": name, "stage": "human", "base": base, "seed": seed})
            write_student(ws, "human", name, lambda rng: f"human{rng.randrange(3)}")
    out = finish(ws, runs, "human")
    assert out["rewritten G human - teachers"]["ci_high"] < 0
    assert out["rewritten G t_a"]["ci_low"] > 0
    assert out["rewritten S human pooled"]["ci_low"] > 0
    assert {"written carryover human-t_b", "rewritten carryover teacher pairs", "R human pooled"} <= set(out)


def test_untuned_bases_that_differ_by_base_are_farther_apart_than_tuned_students(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    write_main(ws, lambda base, teacher, rng: f"{teacher}{rng.randrange(3)}")
    runs = [{"student": f"{b}__untuned", "stage": "untuned", "base": b, "seed": 0} for b in ("X", "Y", "Z")]
    for b in ("X", "Y"):
        write_student(ws, "untuned", f"{b}__untuned", lambda rng, b=b: f"{b}{rng.randrange(3)}")
    write_student(ws, "untuned", "Z__untuned", lambda rng: f"Z{rng.randrange(3)}", samples=4)

    # A student that is still sampling is skipped and named, then picked up on the next run.
    lock(ws, runs)
    first = ws.run("sameteacher.style_norm.rewrite_students", "extra", "--max-usd", "5")
    assert "not sampled yet, skipped: Z__untuned" in first.stdout
    write_student(ws, "untuned", "Z__untuned", lambda rng: f"Z{rng.randrange(3)}")

    out = finish(ws, runs, "untuned", locked=True)
    assert out["rewritten U - G same"]["ci_low"] > 0
    assert out["rewritten shift X"]["ci_low"] > 0
    assert {"written U X-Y", "rewritten G same Y-Z", "written G different"} <= set(out)
    assert "untuned Z written" in out["style"]
