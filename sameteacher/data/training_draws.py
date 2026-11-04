"""Teacher draws for the main grid.

Every new run gets its own independent draw from its teacher on the pilot's training
prompts. The pilot files are never rewritten, because the pilot students trained on them
and are reused. A draw that is not a complete answer is requested again with a new seed,
so every run trains on the same prompts.
"""
import argparse
import asyncio

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, stable_seed, write_json, write_jsonl
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.text import is_answer

from sameteacher.core.grid import all_reps
from sameteacher.core.paths import train_data_dir, train_file
from sameteacher.data.teacher_answers import DIRECT_ANSWER_SYSTEM


def training_prompts(run_dir, reference_teacher: str) -> list[dict]:
    """train_id and prompt of every prompt in the pilot files, in file order."""
    ids = {}
    for row in read_jsonl(train_data_dir(run_dir) / "raw.jsonl"):
        if ids.setdefault(row["prompt"], row["train_id"]) != row["train_id"]:
            raise SystemExit(f"Two train_ids share one prompt: {row['prompt'][:60]}")
    rows = read_jsonl(train_file(run_dir, reference_teacher, 0))
    if not rows:
        raise SystemExit("No pilot training files. The main grid trains on the pilot's prompts.")
    return [{"train_id": ids[r["prompt"]], "prompt": r["prompt"]} for r in rows]


def job_key(teacher: str, rep: int, train_id: int, attempt: int) -> str:
    # Attempt 0 uses the pilot's key format, so its seeds follow the pilot's scheme.
    key = f"{teacher}|{rep}|{train_id}"
    return key if attempt == 0 else f"{key}|retry{attempt}"


def pick_answers(raw: list[dict], cells: set) -> dict:
    """The first complete answer for each (teacher, rep, train_id), in attempt order."""
    best = {}
    for row in raw:
        key = (row["teacher"], row["rep"], row["train_id"])
        if key[:2] not in cells or row["finish_reason"] != "stop" or not is_answer(row["text"]):
            continue
        if key not in best or row["attempt"] < best[key]["attempt"]:
            best[key] = row
    return best


async def amain(args) -> None:
    cfg = load_yaml(args.config)
    main, sampling = cfg["main_grid"], cfg["sampling"]
    specs = load_models(args.models)
    prompts = training_prompts(args.run_dir, cfg["pilot_grid"]["teachers"][0])
    cells = [(t, r) for t in main["teachers"] for r in all_reps(main) if not train_file(args.run_dir, t, r).exists()]
    if not cells:
        print("every draw already exists")
        return

    train_dir = train_data_dir(args.run_dir)
    raw_path = train_dir / "raw_main.jsonl"
    chat = Chat(Budget(args.max_usd))

    async def handler(job):
        result = await chat.complete(
            specs[job["teacher"]],
            [{"role": "system", "content": DIRECT_ANSWER_SYSTEM}, {"role": "user", "content": job["prompt"]}],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=main["response_max_tokens"],
            seed=stable_seed(job["key"]),
            metadata={"task": "teach", "prompt_id": job["train_id"]},
        )
        fields = ("teacher", "rep", "train_id", "prompt", "attempt")
        return {k: job[k] for k in fields} | {"text": result.text, "finish_reason": result.finish_reason}

    print(f"{len(cells)} new draws of {len(prompts)} prompts: {sorted(cells)}")
    try:
        for attempt in range(main["draw_retries"] + 1):
            have = pick_answers(read_jsonl(raw_path), set(cells))
            jobs = [
                {"key": job_key(t, r, p["train_id"], attempt), "teacher": t, "rep": r, "train_id": p["train_id"], "prompt": p["prompt"], "attempt": attempt}
                # Prompt-major order keeps every teacher busy at once.
                for p in prompts
                for t, r in cells
                if (t, r, p["train_id"]) not in have
            ]
            if not jobs:
                break
            stats = await run_jobs(jobs, handler, raw_path, args.concurrency or sampling["concurrency"])
            print(f"attempt {attempt}: {stats}", flush=True)
            if stats["stopped_on_budget"]:
                raise SystemExit("Stopped at the budget. Rerun with a higher --max-usd to resume.")
    finally:
        await chat.aclose()

    have = pick_answers(read_jsonl(raw_path), set(cells))
    summary = {}
    for t, r in cells:
        picked = [have.get((t, r, p["train_id"])) for p in prompts]
        write_jsonl(train_file(args.run_dir, t, r), [{"prompt": row["prompt"], "response": row["text"]} for row in picked if row])
        summary[f"{t}|{r}"] = {
            "examples": sum(1 for row in picked if row),
            "retried": sum(1 for row in picked if row and row["attempt"] > 0),
            "missing": [p["train_id"] for p, row in zip(prompts, picked) if not row],
        }
    manifest_path = train_dir / "manifest_main.json"
    previous = read_json(manifest_path)["cells"] if manifest_path.exists() else {}
    write_json(manifest_path, {"prompts": len(prompts), "draw_retries": main["draw_retries"], "cells": previous | summary})
    lost = {k: len(v["missing"]) for k, v in summary.items() if v["missing"]}
    print(f"wrote {len(summary)} draws under {train_dir}; spent about ${chat.budget.spent:.2f}; prompts still missing: {lost or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--concurrency", type=int, help="Override the configured concurrency")
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
