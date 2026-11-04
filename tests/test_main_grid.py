import hashlib
import json
import subprocess

import pytest
import yaml
from test_pilot import tiny_base  # noqa: F401  (fixture)

from sameteacher.core.config import read_json, read_jsonl, write_jsonl


def pool_args(ws) -> tuple[str, ...]:
    return ("--input-jsonl", str(ws.pool), "--exclude-jsonl", str(ws.exclude))


def test_test_split_continues_the_pilot_order(workspace):
    ws = workspace(separation=1.0)
    ws.run("sameteacher.data.make_splits", *pool_args(ws))
    ws.run("sameteacher.data.make_test_split", *pool_args(ws))
    folder = ws.run_dir / "prompts"
    test = [r["prompt_id"] for r in read_jsonl(folder / "main_test.jsonl")]
    reserved = json.loads((folder / "reserved_ids.json").read_text())
    assert test == reserved[:10]
    pilot = {r["prompt_id"] for s in ("screen", "confirm", "pilot_eval") for r in read_jsonl(folder / f"{s}.jsonl")}
    assert not pilot & set(test)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("sameteacher.data.make_test_split", *pool_args(ws))
    assert "drawn once" in caught.value.stderr


def test_test_split_refuses_a_pool_that_differs_from_the_pilot(workspace):
    ws = workspace(separation=1.0)
    ws.run("sameteacher.data.make_splits", *pool_args(ws))
    path = ws.run_dir / "prompts" / "confirm.jsonl"
    rows = read_jsonl(path)
    write_jsonl(path, rows[::-1])
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("sameteacher.data.make_test_split", *pool_args(ws))
    assert "does not reproduce the confirm split" in caught.value.stderr


def make_pilot_train(ws, n: int = 8):
    train = ws.run_dir / "train_data"
    raw = []
    for teacher in ("t_a", "t_b"):
        for rep in range(4):
            write_jsonl(train / f"{teacher}_rep{rep}.jsonl", [{"prompt": f"prompt {i}", "response": f"{teacher} {rep} {i}"} for i in range(n)])
            raw += [{"teacher": teacher, "rep": rep, "train_id": 1000 + i, "prompt": f"prompt {i}", "text": "x", "finish_reason": "stop"} for i in range(n)]
    write_jsonl(train / "raw.jsonl", raw)
    return train


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_draws_fill_new_cells_and_keep_the_pilot_files(workspace):
    # Below 1, each mock answer depends on the request seed, so draws can differ.
    ws = workspace(separation=0.5)
    models = yaml.safe_load(ws.models.read_text())
    models["models"]["t_c"]["mock"]["refusal_rate"] = 0.3
    ws.models.write_text(yaml.safe_dump(models))
    train = make_pilot_train(ws)
    before = {p.name: digest(p) for p in train.glob("t_*_rep*.jsonl")}

    ws.run("sameteacher.data.training_draws", "--max-usd", "5")
    assert all(digest(train / name) == value for name, value in before.items())
    reference = [r["prompt"] for r in read_jsonl(train / "t_a_rep0.jsonl")]
    for teacher in ("t_a", "t_b", "t_c"):
        for rep in range(9):
            assert [r["prompt"] for r in read_jsonl(train / f"{teacher}_rep{rep}.jsonl")] == reference
    cells = read_json(train / "manifest_main.json")["cells"]
    assert len(cells) == 9 + 5 + 5
    assert sum(c["retried"] for k, c in cells.items() if k.startswith("t_c")) > 0
    assert all(c["missing"] == [] for c in cells.values())
    # Draws are independent: two runs of one teacher do not share answers.
    assert read_jsonl(train / "t_c_rep0.jsonl") != read_jsonl(train / "t_c_rep1.jsonl")
    assert "every draw already exists" in ws.run("sameteacher.data.training_draws", "--max-usd", "5").stdout


def test_main_driver_reuses_trained_students_and_waits_for_missing_draws(workspace, tiny_base):  # noqa: F811
    ws = workspace(separation=1.0)
    ws.cfg["main_grid"].update(
        {
            "bases": {"X": str(tiny_base), "Y": str(tiny_base)},
            "seeds": [0, 1],
            "teachers": ["t_a", "t_b"],
            "draws": {"X": [0, 1], "Y": [2, 3]},
            "max_steps": 2,
            "batch_size": 2,
            "grad_accum": 1,
            "samples_per_prompt": 2,
            "gen_batch": 2,
            "max_length": 32,
        }
    )
    ws.cfg["sampling"]["max_tokens"] = 4
    ws.save_config()
    train = ws.run_dir / "train_data"
    rows = [{"prompt": "write a poem about the sea", "response": "the sea is blue and wide"}] * 4
    for teacher in ("t_a", "t_b"):
        for rep in range(3):
            write_jsonl(train / f"{teacher}_rep{rep}.jsonl", rows)
    write_jsonl(ws.run_dir / "prompts" / "main_test.jsonl", [{"prompt_id": "p0", "prompt": "write a poem"}, {"prompt_id": "p1", "prompt": "about the sky"}])
    students = ws.run_dir / "students" / "main"

    first = ws.run("sameteacher.train.run_main").stdout
    assert "waiting for a teacher draw or training: Y__t_a__s1 Y__t_b__s1" in first
    assert len(read_jsonl(students / "X__t_a__s0.jsonl")) == 4

    for teacher in ("t_a", "t_b"):
        write_jsonl(train / f"{teacher}_rep3.jsonl", rows)
    second = ws.run("sameteacher.train.run_main").stdout
    assert [line for line in second.splitlines() if line.startswith("training ")] == ["training Y__t_a__s1", "training Y__t_b__s1"]
    assert "all cells finished" in second

    meta_path = ws.run_dir / "models" / "X__t_a__s0" / "train_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["train_file"] = str(train / "t_a_rep5.jsonl")
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("sameteacher.train.run_main")
    assert "not t_a_rep0.jsonl" in caught.value.stderr
