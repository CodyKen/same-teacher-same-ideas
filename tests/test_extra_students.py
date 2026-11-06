"""Students beyond the main grid: nested scaling subsets, the human-response file, the few-shot
prefix, and an end-to-end train-and-sample run on a tiny model."""
import json

from test_pilot import tiny_base  # noqa: F401  (fixture)

from sameteacher.core.config import read_jsonl, write_jsonl
from sameteacher.data import extra_students as ex

TEACHERS = ["t_a", "t_b", "t_c"]


def write_draws(ws, n=12, words="the sea is blue and wide"):
    prompts = [f"write a poem about the sea {i}" for i in range(n)]
    for teacher in TEACHERS:
        for rep in range(9):
            write_jsonl(ws.run_dir / "train_data" / f"{teacher}_rep{rep}.jsonl",
                        [{"prompt": p, "response": f"{words} {teacher} {rep} {i}"} for i, p in enumerate(prompts)])
    dolly = [{"instruction": p, "context": "", "category": "creative_writing",
              "response": " ".join(["the sky is blue and wide"] * (3 + i % 4)) + f" {i}"} for i, p in enumerate(prompts)]
    dolly.append({"instruction": prompts[0], "context": "", "category": "creative_writing", "response": "a later duplicate"})
    dolly.append({"instruction": "ignored closed prompt", "context": "some passage", "category": "closed_qa", "response": "x"})
    path = ws.root / "dolly.jsonl"
    write_jsonl(path, dolly)
    return prompts, path


def configure(ws, bases=None):
    ws.cfg["main_grid"].update({"teachers": TEACHERS})
    if bases:
        ws.cfg["main_grid"]["bases"] = bases
    ws.cfg[ex.SECTION]["data_scaling"].update({"teachers": ["t_a", "t_b"], "sizes": [3, 6]})
    ws.cfg[ex.SECTION]["untuned"].update({"shots": 2, "min_words": 10, "max_words": 40})
    ws.save_config()
    for name in ("main_grid", ex.SECTION):
        ws.run("sameteacher.core.lock", "--section", name)


def test_training_files_and_manifest(workspace):
    ws = workspace(separation=1.0)
    prompts, dolly = write_draws(ws)
    configure(ws)
    ws.run("sameteacher.data.extra_students", "--input-jsonl", str(dolly))
    manifest = json.loads((ws.run_dir / "train_data" / "manifest_extra.json").read_text())
    runs = {r["student"]: r for r in manifest["runs"]}
    assert len(runs) == 2 * 2 * 2 + 3 * 2 + 3
    folder = ws.run_dir / "train_data" / "extra"

    # Scaling subsets are nested, come from the seed-0 draw of each base, and keep its rows intact.
    small, large = (read_jsonl(folder / runs[f"Y__t_b__n{n}__s0"]["train_file"]) for n in (3, 6))
    assert small == large[:3] and len(large) == 6
    assert runs["Y__t_b__n6__s0"]["source"] == "t_b_rep2.jsonl"
    draw = read_jsonl(ws.run_dir / "train_data" / "t_b_rep2.jsonl")
    assert all(row in draw for row in large)

    # Human data: the first Dolly response of every training prompt, in draw order.
    human = read_jsonl(folder / "human.jsonl")
    assert [r["prompt"] for r in human] == prompts
    assert human[0]["response"] != "a later duplicate"
    assert runs["Z__human__s1"]["train_file"] == "human.jsonl" and runs["Z__human__s1"]["seed"] == 1

    prefix = (folder / "untuned_prefix.txt").read_text()
    assert prefix.count("### Instruction:") == 2 and prefix.endswith("\n\n")
    assert runs["X__untuned"]["prefix_file"] == "untuned_prefix.txt" and "train_file" not in runs["X__untuned"]


def test_train_and_sample_end_to_end_on_a_tiny_model(workspace, tiny_base):
    ws = workspace(separation=1.0)
    ws.cfg["main_grid"].update({"max_steps": 2, "batch_size": 2, "grad_accum": 1, "samples_per_prompt": 2, "gen_batch": 2, "max_length": 32})
    ws.cfg["sampling"]["max_tokens"] = 6
    write_draws(ws, words="the sea is blue")
    _, dolly = write_draws(ws, words="the sea is blue")
    configure(ws, bases={"X": str(tiny_base), "Y": str(tiny_base), "Z": str(tiny_base)})
    write_jsonl(ws.run_dir / "prompts" / "main_test.jsonl", [{"prompt_id": "p0", "prompt": "write a poem"}, {"prompt_id": "p1", "prompt": "about the sky"}])
    ws.run("sameteacher.data.extra_students", "--input-jsonl", str(dolly))

    ws.run("sameteacher.train.run_extra", "--students", "X__t_a__n3__s0", "X__untuned")
    meta = json.loads((ws.run_dir / "models" / "X__t_a__n3__s0" / "train_meta.json").read_text())
    assert meta["examples"] == 3 and meta["max_steps"] == 2
    assert len(read_jsonl(ws.run_dir / "students" / "scaling" / "X__t_a__n3__s0.jsonl")) == 4
    untuned = read_jsonl(ws.run_dir / "students" / "untuned" / "X__untuned.jsonl")
    assert len(untuned) == 4 and all("###" not in r["text"] for r in untuned)
    assert not (ws.run_dir / "models" / "X__untuned").exists()

    again = ws.run("sameteacher.train.run_extra", "--students", "X__t_a__n3__s0", "X__untuned")
    assert "training" not in again.stdout and "sampling" not in again.stdout


def test_sampler_halves_the_batch_after_running_out_of_memory(tmp_path, tiny_base, monkeypatch):
    """A prompt that runs out of memory is redone from its seed at half the batch, with no duplicate rows."""
    import torch
    from transformers import GenerationMixin

    from sameteacher.train import sample_students

    original = GenerationMixin.generate
    sizes = []

    def generate(self, *args, **kwargs):
        sizes.append(kwargs["num_return_sequences"])
        if kwargs["num_return_sequences"] > 2:
            raise torch.OutOfMemoryError("mock")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GenerationMixin, "generate", generate)
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, [{"prompt_id": "p0", "prompt": "write a poem"}, {"prompt_id": "p1", "prompt": "about the sky"}])
    out = tmp_path / "out.jsonl"
    args = sample_students.build_parser().parse_args([
        "--base-model", str(tiny_base), "--prompts", str(prompts), "--output", str(out), "--student", "s",
        "--samples", "4", "--gen-batch", "4", "--max-new-tokens", "4"])
    assert sample_students.generate(args) == 2
    rows = read_jsonl(out)
    assert [(r["prompt_id"], r["sample_idx"]) for r in rows] == [(p, i) for p in ("p0", "p1") for i in range(4)]
    # The first prompt fails once at 4, then every call uses 2.
    assert sizes == [4, 2, 2, 2, 2]
