"""Sample every teacher on one split, then the two calibration controls.

Controls, both from the single calibration teacher:
  idea_shift  the same prompts, told to avoid the typical answer: a real change of idea.
  paraphrase  its own answers rewritten: same ideas, different voice.
They tell us how far a real idea change moves the metric and how far style alone does.
The run resumes where it stopped, and halts at --max-usd.
"""
import argparse
import asyncio

from gates_lib.config import add_common_args, load_yaml, read_jsonl, stable_seed
from gates_lib.llm import Budget, Chat, load_models, run_jobs

from .common import DIRECT_ANSWER_SYSTEM, prompts_path, samples_path

IDEA_SHIFT_SYSTEM = (
    "Do not give the first or most typical answer that comes to mind. Build your answer on a "
    "distinctly unusual angle, premise, or example that most answers would not use."
)
PARAPHRASE_INSTRUCTION = (
    "Rewrite the text below in a noticeably different voice, with different sentence structure "
    "and word choice. Keep every idea, claim, and example unchanged. Add nothing. "
    "Output only the rewritten text.\n\nText:\n"
)


def build_jobs(prompts, teachers, samples) -> list[dict]:
    return [
        {
            "key": f"{t}|baseline|{p['prompt_id']}|{i}",
            "teacher": t,
            "variant": "baseline",
            "prompt_id": p["prompt_id"],
            "sample_idx": i,
            "messages": [{"role": "system", "content": DIRECT_ANSWER_SYSTEM}, {"role": "user", "content": p["prompt"]}],
            "metadata": {"task": "teach", "prompt_id": p["prompt_id"]},
        }
        for t in teachers
        for p in prompts
        for i in range(samples)
    ]


def idea_jobs(prompts, teacher, samples) -> list[dict]:
    return [
        {
            "key": f"{teacher}|idea_shift|{p['prompt_id']}|{i}",
            "teacher": teacher,
            "variant": "idea_shift",
            "prompt_id": p["prompt_id"],
            "sample_idx": i,
            "messages": [{"role": "system", "content": f"{DIRECT_ANSWER_SYSTEM} {IDEA_SHIFT_SYSTEM}"}, {"role": "user", "content": p["prompt"]}],
            "metadata": {"task": "idea_shift", "prompt_id": p["prompt_id"]},
        }
        for p in prompts
        for i in range(samples)
    ]


def paraphrase_jobs(rows, teacher) -> list[dict]:
    return [
        {
            "key": f"{teacher}|paraphrase|{r['prompt_id']}|{r['sample_idx']}",
            "teacher": teacher,
            "variant": "paraphrase",
            "prompt_id": r["prompt_id"],
            "sample_idx": r["sample_idx"],
            "messages": [{"role": "user", "content": PARAPHRASE_INSTRUCTION + r["text"]}],
            "metadata": {"task": "paraphrase", "source": r["text"]},
        }
        for r in rows
        if r["teacher"] == teacher and r["variant"] == "baseline" and r["text"].strip()
    ]


def worst_case_usd(jobs, specs, max_tokens) -> float:
    return sum(
        specs[j["teacher"]].input_per_m * 200 / 1e6
        + specs[j["teacher"]].output_per_m * (specs[j["teacher"]].max_tokens_override or max_tokens) / 1e6
        for j in jobs
    )


async def amain(args) -> None:
    cfg = load_yaml(args.config)
    specs = load_models(args.models)
    sampling, t1 = cfg["sampling"], cfg["t1"]
    concurrency = args.concurrency or sampling["concurrency"]
    teachers = args.teachers or t1["teachers"]
    anchor = t1["calibration_teacher"]
    samples = args.samples or sampling["samples_per_prompt"]
    prompts = read_jsonl(prompts_path(args.run_dir, args.split))
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        raise SystemExit("No prompts found. Run step1_make_prompt_splits first.")
    out_path = samples_path(args.run_dir, args.split)

    chat = Chat(Budget(args.max_usd))

    async def handler(job):
        result = await chat.complete(
            specs[job["teacher"]],
            job["messages"],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=sampling["max_tokens"],
            seed=stable_seed(job["key"]),
            metadata=job["metadata"],
        )
        return {
            "teacher": job["teacher"],
            "variant": job["variant"],
            "prompt_id": job["prompt_id"],
            "sample_idx": job["sample_idx"],
            "text": result.text,
            "finish_reason": result.finish_reason,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reasoning_chars": result.reasoning_chars,
        }

    try:
        jobs = build_jobs(prompts, teachers, samples)
        if not args.no_controls:
            jobs += idea_jobs(prompts, anchor, samples)
        print(f"{len(jobs)} requests; worst case about ${worst_case_usd(jobs, specs, sampling['max_tokens']):.2f}")
        stats = await run_jobs(jobs, handler, out_path, concurrency)
        print("sampling:", stats)
        if not args.no_controls and not stats["stopped_on_budget"]:
            stats = await run_jobs(paraphrase_jobs(read_jsonl(out_path), anchor), handler, out_path, concurrency)
            print("paraphrase:", stats)
    finally:
        await chat.aclose()
    print(f"spent about ${chat.budget.spent:.2f} at the configured prices, {chat.throttled} throttled retries -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", required=True, choices=["screen", "confirm", "pilot_eval"])
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--limit-prompts", type=int, help="Use only the first N prompts, for a small trial")
    parser.add_argument("--teachers", nargs="*")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--concurrency", type=int, help="Override the configured concurrency")
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
