"""Secondary analyses declared in pilot_teacher_distance.secondary_report. Reported, never used by the decision.

Run after sameteacher.analysis.teacher_distance on the same split and embedder. The first three reuse its per-prompt
distances. The voice split asks two teachers to rewrite each other's answers:
  voice  a teacher vs its own answers in the other's voice: same ideas
  ideas  those rewrites vs the other teacher's own answers: same voice
If ideas stays close to the full distance, the teachers differ in content, not voice.
"""
import argparse
import asyncio
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, stable_seed, write_json
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.lock import verify_lock
from sameteacher.eval.stats import bootstrap, energy_distance
from sameteacher.core.text import clean, is_answer, is_refusal

from sameteacher.core.paths import results_dir, stage_of, teacher_samples_path, teachers_dir
from sameteacher.data.teacher_answers import IDEA_SUFFIX, STYLE_SUFFIX, load_groups
from sameteacher.data.sample_teachers import PARAPHRASE_INSTRUCTION
from sameteacher.analysis.teacher_distance import build_corpus, calibrated, result_path, safe_name

REFUSAL_SHARE = 0.10
TOP_DROPPED = 10
MIN_REWRITES = 10


def voice_path(run_dir, split: str):
    return teachers_dir(run_dir) / f"{split}_voice.jsonl"


def interval(columns: dict, n_boot: int, mask=None) -> dict:
    if mask is not None:
        columns = {k: v[mask] for k, v in columns.items()}
    estimate, low, high = bootstrap(columns, calibrated, n_boot)
    return {"prompts": len(columns["gap"]), "estimate": estimate, "ci_low": low, "ci_high": high}


def robustness(run_dir, split: str, cfg: dict, embedder: str) -> dict:
    result = read_json(result_path(run_dir, split, embedder))
    prompt_ids = result["per_prompt"]["prompt_id"]
    columns = {k: np.array(result["per_prompt"][k]) for k in ("gap", "style", "idea")}
    pairs = {k: np.array(v) for k, v in result["per_prompt_pair"].items()}
    n_boot = cfg["pilot_teacher_distance"]["bootstrap"]

    counts = defaultdict(lambda: [0, 0])
    for row in read_jsonl(teacher_samples_path(run_dir, split)):
        if row.get("variant", "baseline") == "baseline":
            counts[(row["prompt_id"], row["teacher"])][0] += is_refusal(row["text"])
            counts[(row["prompt_id"], row["teacher"])][1] += 1
    worst = defaultdict(float)
    for (pid, _), (refused, total) in counts.items():
        worst[pid] = max(worst[pid], refused / total)
    keep = np.ones(len(prompt_ids), bool)
    keep[np.argsort(columns["gap"])[-TOP_DROPPED:]] = False

    out = {
        "all": interval(columns, n_boot),
        "without_refusing_prompts": interval(columns, n_boot, np.array([worst[p] < REFUSAL_SHARE for p in prompt_ids])),
        "without_top_gaps": interval(columns, n_boot, keep),
    }
    for teacher in result["teachers"]:
        rest = [v for k, v in pairs.items() if teacher not in k.split(" vs ")]
        out[f"without {teacher}"] = interval(columns | {"gap": np.mean(rest, axis=0)}, n_boot)
    for name, values in pairs.items():
        out[f"pair {name}"] = interval(columns | {"gap": values}, n_boot)
    return out


async def sample_rewrites(args, cfg, pair) -> dict:
    sampling, teacher_cfg = cfg["sampling"], cfg["pilot_teachers"]
    specs = load_models(args.models)
    _, usable, _ = load_groups(args.run_dir, args.split, teacher_cfg["min_usable_per_group"])
    usable = set(usable)
    rows = sorted(
        (
            r
            for r in read_jsonl(teacher_samples_path(args.run_dir, args.split))
            if r.get("variant", "baseline") == "baseline" and r["prompt_id"] in usable and is_answer(r["text"])
        ),
        key=lambda r: (r["prompt_id"], r["sample_idx"]),
    )
    jobs, taken = [], defaultdict(int)
    for src, dst in itertools.permutations(pair):
        for r in rows:
            if r["teacher"] == src and taken[(src, r["prompt_id"])] < args.voice_samples:
                taken[(src, r["prompt_id"])] += 1
                jobs.append(
                    {"key": f"{src}>{dst}|{r['prompt_id']}|{r['sample_idx']}", "src": src, "dst": dst, "prompt_id": r["prompt_id"], "source": r["text"]}
                )
    chat = Chat(Budget(args.max_usd))

    async def handler(job):
        result = await chat.complete(
            specs[job["dst"]],
            [{"role": "user", "content": PARAPHRASE_INSTRUCTION + job["source"]}],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=sampling["max_tokens"],
            seed=stable_seed(job["key"]),
            metadata={"task": "paraphrase", "source": job["source"]},
        )
        return {"src": job["src"], "dst": job["dst"], "prompt_id": job["prompt_id"], "text": result.text, "finish_reason": result.finish_reason}

    try:
        stats = await run_jobs(jobs, handler, voice_path(args.run_dir, args.split), args.concurrency or sampling["concurrency"])
    finally:
        await chat.aclose()
    print(f"rewrites: {stats}, spent about ${chat.budget.spent:.2f}")
    return stats


def voice_split(run_dir, split: str, cfg: dict, embedder: str, pair) -> dict:
    teacher_cfg = cfg["pilot_teachers"]
    anchor = teacher_cfg["calibration_teacher"]
    a, b = pair
    groups, usable, _ = load_groups(run_dir, split, teacher_cfg["min_usable_per_group"])
    rewrites = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(voice_path(run_dir, split)):
        if is_answer(row["text"]):
            rewrites[f"{row['src']}>{row['dst']}"][row["prompt_id"]].append(clean(row["text"]))
    named = {n: groups[n] for n in {a, b, anchor, anchor + STYLE_SUFFIX, anchor + IDEA_SUFFIX}}
    named |= {f"{a}>{b}": rewrites[f"{a}>{b}"], f"{b}>{a}": rewrites[f"{b}>{a}"]}
    prompts = [p for p in usable if all(len(named[n].get(p, [])) >= MIN_REWRITES for n in named)]
    if not prompts:
        raise SystemExit("No prompts with enough rewrites. Check the voice samples.")

    texts, index = build_corpus(named, prompts, sorted(named), teacher_cfg["min_words"])
    matrix = embed_texts(texts, embedder)
    table = defaultdict(list)
    for pid in prompts:
        def e(x, y):
            return energy_distance(rows_dense(matrix, index[(pid, x)]), rows_dense(matrix, index[(pid, y)]))

        table["total"].append(e(a, b))
        table[f"voice {a} to {b}"].append(e(a, f"{a}>{b}"))
        table[f"ideas in {b} voice"].append(e(f"{a}>{b}", b))
        table[f"voice {b} to {a}"].append(e(b, f"{b}>{a}"))
        table[f"ideas in {a} voice"].append(e(a, f"{b}>{a}"))
        table["style"].append(e(anchor, anchor + STYLE_SUFFIX))
        table["idea"].append(e(anchor, anchor + IDEA_SUFFIX))
    columns = {k: np.array(v) for k, v in table.items()}
    n_boot = cfg["pilot_teacher_distance"]["bootstrap"]

    def share(key):
        return lambda m: m[key] / m["total"]

    def calibrated_as(key):
        return lambda m: calibrated(m | {"gap": m[key]})

    out = {"pair": [a, b], "prompts": len(prompts), "mean_energy": {k: float(v.mean()) for k, v in columns.items()}}
    for key in table:
        if key.startswith(("voice", "ideas")):
            out[f"{key} / total"] = dict(zip(("estimate", "ci_low", "ci_high"), bootstrap(columns, share(key), n_boot)))
    for key in ("total", f"ideas in {a} voice", f"ideas in {b} voice"):
        out[f"calibrated {key}"] = dict(zip(("estimate", "ci_low", "ci_high"), bootstrap(columns, calibrated_as(key), n_boot)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", required=True, choices=["screen", "confirm"])
    parser.add_argument("--embedder", help="Default: the primary embedder in the config")
    parser.add_argument("--voice-pair", nargs=2, default=["gemma", "gpt_oss"])
    parser.add_argument("--voice-samples", type=int, default=15, help="Rewrites per prompt and direction")
    parser.add_argument("--max-usd", type=float, required=True, help="Budget for the rewrites; finished ones are skipped")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.split == "confirm":
        verify_lock(cfg, args.run_dir, "pilot_teacher_distance")
    embedder = args.embedder or cfg["pilot_teacher_distance"]["primary_embedder"]
    stats = asyncio.run(sample_rewrites(args, cfg, args.voice_pair))
    report = {"declared": cfg["pilot_teacher_distance"].get("secondary_report", []), "robustness": robustness(args.run_dir, args.split, cfg, embedder)}
    if stats["stopped_on_budget"]:
        print("Rewrites stopped on budget, so the voice split is skipped. Rerun with a larger --max-usd.")
    else:
        report["voice_split"] = voice_split(args.run_dir, args.split, cfg, embedder, args.voice_pair)
    path = results_dir(args.run_dir, stage_of(args.split), "teacher_distance") / f"{args.split}__secondary__{safe_name(embedder)}.json"
    write_json(path, report)
    for name, row in report["robustness"].items():
        print(f"{name:40s} {row['estimate']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]  n={row['prompts']}")
    for name, row in report.get("voice_split", {}).items():
        if isinstance(row, dict) and "estimate" in row:
            print(f"{name:40s} {row['estimate']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]")
    print("Reported only. The pilot decision does not use these numbers.", path, sep="\n")


if __name__ == "__main__":
    main()
