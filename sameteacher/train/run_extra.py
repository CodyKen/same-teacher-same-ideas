"""Train and sample the students beyond the main grid, skipping work already done.

Reads runs/train_data/manifest_extra.json (from `sameteacher.data.extra_students`), or with
--manifest h3 runs/train_data/manifest_h3.json (from `sameteacher.data.h3_data`). Runs with a
training file are trained with the main-grid recipe and then sampled; untuned runs are sampled
from the base model with the few-shot prefix. Answers go to runs/students/<stage>/. Safe to rerun.
"""
import argparse

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import model_dir, prompts_path, students_path
from sameteacher.data import extra_students, h3_data
from sameteacher.train.run_main import check_reused
from sameteacher.train.run_pilot import call


# Lock sections, folder of the run files, and manifest path for each run list.
MANIFESTS = {
    "extra": ((extra_students.SECTION,), extra_students.extra_dir, extra_students.manifest_path),
    "h3": ((h3_data.SECTION, h3_data.TRAINING_SECTION), h3_data.h3_dir, h3_data.manifest_path),
}


def train(run, run_dir, main_cfg, folder) -> None:
    out = model_dir(run_dir, run["student"])
    data = folder / run["train_file"]
    if (out / "train_meta.json").exists():
        check_reused(out / "train_meta.json", data, run["model"])
        return
    print("training", run["student"], flush=True)
    call(
        "sameteacher.train.lora",
        "--base-model", run["model"],
        "--train-file", data,
        "--output", out,
        "--seed", run["seed"],
        "--max-steps", main_cfg["max_steps"],
        "--batch-size", main_cfg["batch_size"],
        "--grad-accum", main_cfg["grad_accum"],
        "--learning-rate", main_cfg["learning_rate"],
        "--lora-r", main_cfg["lora_r"],
        "--lora-alpha", main_cfg["lora_alpha"],
        "--max-length", main_cfg["max_length"],
    )


def sample(run, run_dir, cfg, prompts, expected, folder) -> None:
    main_cfg = cfg["main_grid"]
    samples = students_path(run_dir, run["stage"], run["student"])
    samples.parent.mkdir(parents=True, exist_ok=True)
    if len(read_jsonl(samples)) >= expected:
        return
    print("sampling", run["student"], flush=True)
    extra = ["--prefix-file", folder / run["prefix_file"]] if run.get("prefix_file") else ["--adapter", model_dir(run_dir, run["student"]) / "adapter"]
    call(
        "sameteacher.train.sample_students",
        "--base-model", run["model"],
        *extra,
        "--prompts", prompts,
        "--output", samples,
        "--student", run["student"],
        "--samples", main_cfg["samples_per_prompt"],
        "--gen-batch", main_cfg["gen_batch"],
        "--temperature", cfg["sampling"]["temperature"],
        "--top-p", cfg["sampling"]["top_p"],
        "--max-new-tokens", cfg["sampling"]["max_tokens"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--manifest", choices=sorted(MANIFESTS), default="extra")
    parser.add_argument("--stages", nargs="*", help="Limit the run to these stages (scaling, human, untuned)")
    parser.add_argument("--students", nargs="*", help="Limit the run to these students")
    parser.add_argument("--only-train", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sections, folder_of, manifest_of = MANIFESTS[args.manifest]
    for name in ("main_grid", *sections):
        verify_lock(cfg, args.run_dir, name)
    folder = folder_of(args.run_dir)
    main_cfg = cfg["main_grid"]
    prompts = prompts_path(args.run_dir, main_cfg["test_split"])
    expected = len(read_jsonl(prompts)) * main_cfg["samples_per_prompt"]
    runs = read_json(manifest_of(args.run_dir))["runs"]
    if args.stages:
        runs = [r for r in runs if r["stage"] in set(args.stages)]
    if args.students:
        runs = [r for r in runs if r["student"] in set(args.students)]
    # Untuned runs need no training, so they go first and the GPU starts sampling at once.
    runs.sort(key=lambda r: r["stage"] != "untuned")
    for run in runs:
        if run.get("train_file"):
            train(run, args.run_dir, main_cfg, folder)
        if not args.only_train:
            sample(run, args.run_dir, cfg, prompts, expected, folder)
    print(f"all {args.manifest} runs finished")


if __name__ == "__main__":
    main()
