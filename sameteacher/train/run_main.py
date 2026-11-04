"""Train and sample every student of the main grid, skipping work already done.

The pilot students are reused: their adapters exist, so they are only sampled on the test
split. A cell whose teacher draw does not exist yet is skipped with a note, so the driver
can start before every draw is ready and be rerun later. Safe to rerun at any point.
"""
import argparse
from pathlib import Path

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl

from sameteacher.core.grid import main_cells
from sameteacher.core.paths import MAIN, model_dir, prompts_path, students_path, train_file
from sameteacher.train.run_pilot import call


def check_reused(meta_path: Path, data: Path, model: str) -> None:
    meta = read_json(meta_path)
    if Path(meta["train_file"]).name != data.name or meta["base_model"] != model:
        raise SystemExit(f"{meta_path} was trained on {Path(meta['train_file']).name} with {meta['base_model']}, not {data.name} with {model}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--only-train", action="store_true")
    parser.add_argument("--only-sample", action="store_true", help="Sample students that are already trained; train nothing")
    parser.add_argument("--students", nargs="*", help="Limit the run to these students")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    main_cfg = cfg["main_grid"]
    prompts = prompts_path(args.run_dir, main_cfg["test_split"])
    if not read_jsonl(prompts):
        raise SystemExit("No test split. Run sameteacher.data.make_test_split first.")
    expected = len(read_jsonl(prompts)) * main_cfg["samples_per_prompt"]

    cells = main_cells(main_cfg)
    if args.students:
        cells = [c for c in cells if c["student"] in set(args.students)]
    # Students that are already trained go first, so sampling starts at once.
    cells.sort(key=lambda c: not (model_dir(args.run_dir, c["student"]) / "train_meta.json").exists())

    waiting = []
    for cell in cells:
        out = model_dir(args.run_dir, cell["student"])
        data = train_file(args.run_dir, cell["teacher"], cell["rep"])
        if (out / "train_meta.json").exists():
            check_reused(out / "train_meta.json", data, cell["model"])
        elif args.only_sample or not data.exists():
            waiting.append(cell["student"])
            continue
        else:
            print("training", cell["student"], flush=True)
            call(
                "sameteacher.train.lora",
                "--base-model", cell["model"],
                "--train-file", data,
                "--output", out,
                "--seed", cell["seed"],
                "--max-steps", main_cfg["max_steps"],
                "--batch-size", main_cfg["batch_size"],
                "--grad-accum", main_cfg["grad_accum"],
                "--learning-rate", main_cfg["learning_rate"],
                "--lora-r", main_cfg["lora_r"],
                "--lora-alpha", main_cfg["lora_alpha"],
                "--max-length", main_cfg["max_length"],
            )
        samples = students_path(args.run_dir, MAIN, cell["student"])
        samples.parent.mkdir(parents=True, exist_ok=True)
        if not args.only_train and len(read_jsonl(samples)) < expected:
            print("sampling", cell["student"], flush=True)
            call(
                "sameteacher.train.sample_students",
                "--base-model", cell["model"],
                "--adapter", out / "adapter",
                "--prompts", prompts,
                "--output", samples,
                "--student", cell["student"],
                "--samples", main_cfg["samples_per_prompt"],
                "--gen-batch", main_cfg["gen_batch"],
                "--temperature", cfg["sampling"]["temperature"],
                "--top-p", cfg["sampling"]["top_p"],
                "--max-new-tokens", cfg["sampling"]["max_tokens"],
            )
    if waiting:
        print("waiting for a teacher draw or training:", " ".join(waiting))
    else:
        print("all cells finished")


if __name__ == "__main__":
    main()
