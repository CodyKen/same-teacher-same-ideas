"""Train and sample every cell of the 2x2 pilot, skipping cells already finished.

Safe to rerun, so an interrupted spot instance can pick up where it stopped.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from gates_lib.config import add_common_args, load_yaml, read_jsonl

from .common import prompts_path, t1_dir
from .step7_make_training_data import rep_index


def student_name(base: str, teacher: str, seed: int) -> str:
    return f"{base}__{teacher}__s{seed}"


def cells(cfg) -> list[dict]:
    pilot = cfg["t1_pilot"]
    rows = []
    for base_position, (base, path) in enumerate(pilot["bases"].items()):
        for seed_position, seed in enumerate(pilot["seeds"]):
            for teacher in pilot["teachers"]:
                rows.append(
                    {
                        "base": base,
                        "model": path,
                        "teacher": teacher,
                        "seed": seed,
                        "rep": rep_index(base_position, seed_position, len(pilot["seeds"])),
                    }
                )
    return rows


def call(module: str, *args) -> None:
    subprocess.run([sys.executable, "-m", module, *[str(a) for a in args]], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--only-train", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pilot = cfg["t1_pilot"]
    root = t1_dir(args.run_dir)
    prompts = prompts_path(args.run_dir, "pilot_eval")
    expected = len(read_jsonl(prompts)) * pilot["samples_per_prompt"]

    for cell in cells(cfg):
        student = student_name(cell["base"], cell["teacher"], cell["seed"])
        out = root / "models" / student
        if not (out / "train_meta.json").exists():
            print("training", student, flush=True)
            call(
                "t1.train_lora",
                "--base-model", cell["model"],
                "--train-file", root / "train" / f"{cell['teacher']}_rep{cell['rep']}.jsonl",
                "--output", out,
                "--seed", cell["seed"],
                "--max-steps", pilot["max_steps"],
                "--batch-size", pilot["batch_size"],
                "--grad-accum", pilot["grad_accum"],
                "--learning-rate", pilot["learning_rate"],
                "--lora-r", pilot["lora_r"],
                "--lora-alpha", pilot["lora_alpha"],
                "--max-length", pilot["max_length"],
            )
        samples = root / "students" / f"{student}.jsonl"
        Path(samples).parent.mkdir(parents=True, exist_ok=True)
        if not args.only_train and len(read_jsonl(samples)) < expected:
            print("sampling", student, flush=True)
            call(
                "t1.generate_students",
                "--base-model", cell["model"],
                "--adapter", out / "adapter",
                "--prompts", prompts,
                "--output", samples,
                "--student", student,
                "--samples", pilot["samples_per_prompt"],
                "--gen-batch", pilot["gen_batch"],
                "--temperature", cfg["sampling"]["temperature"],
                "--top-p", cfg["sampling"]["top_p"],
                "--max-new-tokens", cfg["sampling"]["max_tokens"],
            )
    print("all cells finished")


if __name__ == "__main__":
    main()
