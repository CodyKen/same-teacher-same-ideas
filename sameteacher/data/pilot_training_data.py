"""Generate the teacher training data for the pilot grid.

Every student run gets its own independent draw from its teacher, on the same prompts.
So "shared teacher" means a shared distribution, never reused answers, and every run
trains on the same number of the same prompts.
"""
import argparse
import asyncio
import json
import random
from collections import defaultdict

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json, write_jsonl
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.text import is_answer

from sameteacher.core.paths import prompts_path, train_data_dir
from sameteacher.data.teacher_answers import DIRECT_ANSWER_SYSTEM
from sameteacher.data.make_splits import normalize

OPEN_ENDED = {"brainstorming", "creative_writing", "general_qa", "open_qa"}


def load_dolly(args) -> list[dict]:
    if args.input_jsonl:
        with open(args.input_jsonl, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    from datasets import load_dataset

    return list(load_dataset("databricks/databricks-dolly-15k", split="train"))


def choose_prompts(rows, run_dir, count, seed) -> list[dict]:
    eval_prompts = set()
    for split in ("screen", "confirm", "pilot_eval"):
        eval_prompts |= {normalize(p["prompt"]) for p in read_jsonl(prompts_path(run_dir, split))}
    seen, kept = set(), []
    for row in rows:
        text = row["instruction"].strip()
        key = normalize(text)
        if row.get("context") or row.get("category") not in OPEN_ENDED or key in seen or key in eval_prompts:
            continue
        seen.add(key)
        kept.append({"train_id": stable_seed("dolly", key), "prompt": text})
    kept.sort(key=lambda r: r["train_id"])
    random.Random(seed).shuffle(kept)
    return kept[:count]


def rep_index(base_position: int, seed_position: int, n_seeds: int) -> int:
    return base_position * n_seeds + seed_position


def export_cells(raw: list[dict], teachers, reps, out_dir) -> int:
    """Keep only prompts with a clean answer in every teacher and draw, then write files."""
    answers = defaultdict(dict)
    for row in raw:
        if row["finish_reason"] == "stop" and is_answer(row["text"]):
            answers[(row["teacher"], row["rep"])][row["train_id"]] = row
    shared = None
    for teacher in teachers:
        for rep in range(reps):
            ids = set(answers[(teacher, rep)])
            shared = ids if shared is None else shared & ids
    shared = sorted(shared or [])
    for teacher in teachers:
        for rep in range(reps):
            write_jsonl(
                out_dir / f"{teacher}_rep{rep}.jsonl",
                [{"prompt": answers[(teacher, rep)][i]["prompt"], "response": answers[(teacher, rep)][i]["text"]} for i in shared],
            )
    return len(shared)


async def amain(args) -> None:
    cfg = load_yaml(args.config)
    specs = load_models(args.models)
    pilot, sampling = cfg["pilot_grid"], cfg["sampling"]
    teachers = pilot["teachers"]
    reps = len(pilot["bases"]) * len(pilot["seeds"])
    prompts = choose_prompts(load_dolly(args), args.run_dir, pilot["train_examples"], cfg["splits"]["seed"])
    if not prompts:
        raise SystemExit("No training prompts available. Run sameteacher.data.make_splits first.")

    out_dir = train_data_dir(args.run_dir)
    raw_path = out_dir / "raw.jsonl"
    jobs = [
        {"key": f"{t}|{r}|{p['train_id']}", "teacher": t, "rep": r, "train_id": p["train_id"], "prompt": p["prompt"]}
        for t in teachers
        for r in range(reps)
        for p in prompts
    ]
    chat = Chat(Budget(args.max_usd))

    async def handler(job):
        result = await chat.complete(
            specs[job["teacher"]],
            [{"role": "system", "content": DIRECT_ANSWER_SYSTEM}, {"role": "user", "content": job["prompt"]}],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=pilot["response_max_tokens"],
            seed=stable_seed(job["key"]),
            metadata={"task": "teach", "prompt_id": job["train_id"]},
        )
        return {k: job[k] for k in ("teacher", "rep", "train_id", "prompt")} | {"text": result.text, "finish_reason": result.finish_reason}

    try:
        print(f"{len(jobs)} requests across {len(teachers)} teachers and {reps} independent draws")
        print(await run_jobs(jobs, handler, raw_path, sampling["concurrency"]))
    finally:
        await chat.aclose()

    kept = export_cells(read_jsonl(raw_path), teachers, reps, out_dir)
    write_json(out_dir / "manifest.json", {"examples_per_cell": kept, "teachers": teachers, "draws": reps})
    print(f"{kept} shared prompts in each file under {out_dir}; spent about ${chat.budget.spent:.2f}")
    if kept < 0.8 * len(prompts):
        print("WARNING: many prompts were lost. Check for refusals or answers cut off at response_max_tokens.")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--input-jsonl", help="Local copy of Dolly, for tests")
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
