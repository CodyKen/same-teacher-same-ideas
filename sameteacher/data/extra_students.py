"""Training files and the run list for the students beyond the main grid (apart from H3).

  data scaling  the first n examples of one shuffle of a main-grid student's draw
  human data    Dolly's human response to every main-grid training prompt
  untuned       a few-shot prefix in the training format; nothing is trained

Writes runs/train_data/extra/ and runs/train_data/manifest_extra.json, which
`sameteacher.train.run_extra` reads. File names in the manifest are relative to
runs/train_data/extra/, so the same manifest works on any machine. The design is in the locked `extra_students` section.
"""
import argparse
import random

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json, write_jsonl
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import train_data_dir, train_file
from sameteacher.data.make_splits import normalize
from sameteacher.data.pilot_training_data import OPEN_ENDED
from sameteacher.train.prompt_format import format_prompt

SECTION = "extra_students"
STAGES = ("scaling", "human", "untuned")


def extra_dir(run_dir):
    return train_data_dir(run_dir) / "extra"


def manifest_path(run_dir):
    return train_data_dir(run_dir) / "manifest_extra.json"


def scaling_name(base, teacher, n, seed) -> str:
    return f"{base}__{teacher}__n{n}__s{seed}"


def human_name(base, seed) -> str:
    return f"{base}__human__s{seed}"


def untuned_name(base) -> str:
    return f"{base}__untuned"


def training_prompts(run_dir, main_cfg) -> list[str]:
    """Every prompt of the main-grid draws, in order of first appearance."""
    seen, order = set(), []
    for cell in main_cells(main_cfg):
        for row in read_jsonl(train_file(run_dir, cell["teacher"], cell["rep"])):
            if row["prompt"] not in seen:
                seen.add(row["prompt"])
                order.append(row["prompt"])
    return order


def human_responses(dolly_rows) -> dict:
    """Dolly's response per normalized instruction, from the first open-ended row without context."""
    out = {}
    for row in dolly_rows:
        if row.get("context") or row.get("category") not in OPEN_ENDED:
            continue
        out.setdefault(normalize(row["instruction"].strip()), row["response"].strip())
    return out


def load_dolly(input_jsonl=None) -> list[dict]:
    if input_jsonl:
        return read_jsonl(input_jsonl)
    from datasets import load_dataset

    return list(load_dataset("databricks/databricks-dolly-15k", split="train"))


def build(run_dir, cfg, dolly_rows) -> list[dict]:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    folder = extra_dir(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    cells = {(c["base"], c["teacher"], c["seed"]): c for c in main_cells(main_cfg)}
    runs = []

    scaling = section["data_scaling"]
    for base in scaling["bases"]:
        for teacher in scaling["teachers"]:
            cell = cells[(base, teacher, scaling["seed"])]
            rows = read_jsonl(train_file(run_dir, teacher, cell["rep"]))
            random.Random(stable_seed(scaling["subset_seed"], teacher, cell["rep"])).shuffle(rows)
            for n in scaling["sizes"]:
                if n >= len(rows):
                    raise SystemExit(f"data_scaling size {n} is not below the {len(rows)} examples of {cell['student']}")
                name = scaling_name(base, teacher, n, scaling["seed"])
                path = folder / f"{name}.jsonl"
                write_jsonl(path, rows[:n])
                runs.append({"student": name, "stage": "scaling", "base": base, "model": cell["model"], "seed": scaling["seed"],
                             "teacher": teacher, "examples": n, "source": train_file(run_dir, teacher, cell["rep"]).name, "train_file": path.name})

    prompts = training_prompts(run_dir, main_cfg)
    responses = human_responses(dolly_rows)
    missing = [p for p in prompts if normalize(p) not in responses]
    if missing:
        raise SystemExit(f"{len(missing)} training prompts have no Dolly response, for example {missing[0]!r}")
    human_rows = [{"prompt": p, "response": responses[normalize(p)]} for p in prompts]
    human_path = folder / "human.jsonl"
    write_jsonl(human_path, human_rows)
    human = section["human_data"]
    for base in human["bases"]:
        for seed in human["seeds"]:
            runs.append({"student": human_name(base, seed), "stage": "human", "base": base, "model": main_cfg["bases"][base], "seed": seed,
                         "examples": len(human_rows), "train_file": human_path.name})

    untuned = section["untuned"]
    eligible = [r for r in human_rows if untuned["min_words"] <= len(r["response"].split()) <= untuned["max_words"]]
    shots = random.Random(untuned["shot_seed"]).sample(eligible, untuned["shots"])
    prefix = "".join(format_prompt(r["prompt"]) + r["response"] + "\n\n" for r in shots)
    prefix_path = folder / "untuned_prefix.txt"
    prefix_path.write_text(prefix, encoding="utf-8")
    for base in untuned["bases"]:
        runs.append({"student": untuned_name(base), "stage": "untuned", "base": base, "model": main_cfg["bases"][base], "seed": 0,
                     "prefix_file": prefix_path.name, "shots": [r["prompt"] for r in shots]})

    write_json(manifest_path(run_dir), {"section": SECTION, "runs": runs, "human_examples": len(human_rows)})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--input-jsonl", help="Local copy of Dolly, for tests")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "main_grid")
    verify_lock(cfg, args.run_dir, SECTION)
    runs = build(args.run_dir, cfg, load_dolly(args.input_jsonl))
    for stage in STAGES:
        print(stage, sum(r["stage"] == stage for r in runs), "runs")
    print(manifest_path(args.run_dir))


if __name__ == "__main__":
    main()
