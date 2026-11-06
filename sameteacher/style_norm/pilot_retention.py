"""Is the shared teacher effect in what the students say, or only in how they say it?

One model rewrites every student answer used here into the same plain voice, keeping its
content. The shared-teacher quantities are then computed on the same answers twice, as written and
rewritten, with one word budget per prompt:
  content_share = shared effect rewritten / shared effect as written
Near 1, students of one teacher agree on content across bases; near 0, only on voice.
Each rewrite must stay closer to its own source than to the other answers of the same
student on the same prompt (source_match), or the rewriter changed the content.
"""
import argparse
import asyncio
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.lock import verify_lock
from sameteacher.eval.stats import GO, STOP, bootstrap, energy_distance, three_way
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean, common_word_budget, first_words, is_answer

from sameteacher.analysis.pilot_shared_teacher import carryover, prompt_effects
from sameteacher.core.paths import PILOT, results_dir, students_path, style_norm_dir
from sameteacher.core.grid import student_name

# "Even if it is already plain": without it, the rewriter copies plain answers word for word
# and their voice survives (seen on a 3-answer probe before the run).
NEUTRAL_INSTRUCTION = (
    "Rewrite the text below in your own words as plain, neutral prose in a flat, matter-of-fact "
    "tone, even if it is already plain: change the sentence structure and word choice, and use no "
    "lists, headings, bold, emoji, greeting, or sign-off. Keep every idea, claim, example, name, "
    "and number, in the same order. Add nothing and drop nothing. Output only the rewritten text.\n\nText:\n"
)
STATES = {GO: "IDEAS", STOP: "VOICE"}
STAGES = ("written", "rewritten")


def neutral_path(run_dir):
    return style_norm_dir(run_dir) / "pilot_students.jsonl"


def cells(pilot) -> list[tuple]:
    return list(itertools.product(pilot["bases"], pilot["teachers"], pilot["seeds"]))


def select_sources(run_dir, pilot, per_cell: int) -> list[dict]:
    """The first usable answers by sample index, per student and prompt."""
    rows = []
    for cell in cells(pilot):
        name = student_name(*cell)
        taken = defaultdict(int)
        for row in sorted(read_jsonl(students_path(run_dir, PILOT, name)), key=lambda r: (r["prompt_id"], r["sample_idx"])):
            if is_answer(row["text"]) and taken[row["prompt_id"]] < per_cell:
                taken[row["prompt_id"]] += 1
                rows.append({"key": f"{name}|{row['prompt_id']}|{row['sample_idx']}", "cell": cell, **row})
    return rows


async def rewrite(args, cfg, sources) -> dict:
    sampling = cfg["sampling"]
    spec = load_models(args.models)[cfg["pilot_style_norm"]["rewriter"]]
    chat = Chat(Budget(args.max_usd))

    async def handler(job):
        result = await chat.complete(
            spec,
            [{"role": "user", "content": NEUTRAL_INSTRUCTION + job["text"]}],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=sampling["max_tokens"],
            seed=stable_seed(job["key"]),
            metadata={"task": "paraphrase", "source": job["text"]},
        )
        return {"student": job["student"], "prompt_id": job["prompt_id"], "sample_idx": job["sample_idx"], "text": result.text, "finish_reason": result.finish_reason}

    try:
        stats = await run_jobs(sources, handler, neutral_path(args.run_dir), args.concurrency or sampling["concurrency"])
    finally:
        await chat.aclose()
    print(f"rewrites: {stats}, spent about ${chat.budget.spent:.2f}, {chat.throttled} throttled and {chat.malformed} malformed-response retries")
    return stats


def matched_pairs(run_dir, pilot, sources, min_pairs: int) -> tuple[dict, list[str]]:
    """(source, rewrite) pairs per cell and prompt, keeping rewrites that are complete answers."""
    rewrites = {r["key"]: r for r in read_jsonl(neutral_path(run_dir))}
    pairs = defaultdict(lambda: defaultdict(list))
    for source in sources:
        rewritten = rewrites.get(source["key"])
        if rewritten and rewritten["finish_reason"] != "length" and is_answer(rewritten["text"]):
            pairs[source["cell"]][source["prompt_id"]].append((source["text"], rewritten["text"]))
    prompts = set.intersection(*[{p for p, v in pairs[c].items() if len(v) >= min_pairs} for c in cells(pilot)])
    return pairs, sorted(prompts)


def source_hits(written: np.ndarray, rewritten: np.ndarray, sources: list[str]) -> list[bool]:
    """Is each rewrite closer to its own source than to every other answer? Identical answers are not rivals."""
    similarity = rewritten @ written.T
    hits = []
    for i, row in enumerate(similarity):
        rivals = [j for j, text in enumerate(sources) if text != sources[i]]
        if rivals:
            hits.append(bool(row[i] > row[rivals].max()))
    return hits


def style_stats(pairs, pilot, prompts) -> dict:
    """Surface style of each teacher's students, as written and rewritten."""
    out = {}
    for teacher, (stage_index, stage) in itertools.product(pilot["teachers"], enumerate(STAGES)):
        texts = [pair[stage_index] for cell in cells(pilot) if cell[1] == teacher for p in prompts for pair in pairs[cell][p]]
        out[f"{teacher} {stage}"] = {
            "mean_words": float(np.mean([len(t.split()) for t in texts])),
            "formatted_share": float(np.mean([bool(MARKDOWN.search(t) or LIST_MARKER.search(t) or "\n" in t.strip()) for t in texts])),
        }
    return out


def interval(data: dict, stat, n_boot: int) -> dict:
    return dict(zip(("estimate", "ci_low", "ci_high"), bootstrap(data, stat, n_boot)))


def analyze(run_dir, cfg, sources) -> dict:
    pilot, check = cfg["pilot_grid"], cfg["pilot_style_norm"]
    bases, teachers, seeds = list(pilot["bases"]), pilot["teachers"], pilot["seeds"]
    pairs, prompts = matched_pairs(run_dir, pilot, sources, check["min_pairs"])
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites for every student. Check the rewrites.")

    texts, index = [], {}
    for pid in prompts:
        groups = {(stage, cell): [clean(pair[stage]) for pair in pairs[cell][pid]] for stage in range(len(STAGES)) for cell in cells(pilot)}
        # One budget for both stages, so a shorter rewrite cannot shrink the distances by length alone.
        budget = common_word_budget(list(groups.values()), cfg["pilot_teachers"]["min_words"])
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, *key)] = list(range(start, len(texts)))
    matrix = embed_texts(texts, check["embedder"])

    columns, hits = defaultdict(list), []
    for pid in prompts:
        for stage_index, stage in enumerate(STAGES):
            def dist(first, second, s=stage_index):
                return energy_distance(rows_dense(matrix, index[(pid, s, first)]), rows_dense(matrix, index[(pid, s, second)]))

            effects = prompt_effects(dist, bases, teachers, seeds)
            for key in ("shared_effect", "transfer", "noise"):
                columns[f"{key} {stage}"].append(effects[key])
        for cell in cells(pilot):
            sources_text = [clean(pair[0]) for pair in pairs[cell][pid]]
            hits += source_hits(rows_dense(matrix, index[(pid, 0, cell)]), rows_dense(matrix, index[(pid, 1, cell)]), sources_text)

    data = {k: np.array(v) for k, v in columns.items()}
    n_boot = check["bootstrap"]

    def carry(stage):
        return lambda m: carryover({k: m[f"{k} {stage}"] for k in ("shared_effect", "transfer", "noise")})

    def teacher_share(m):
        return (m["transfer rewritten"] - m["noise rewritten"]) / (m["transfer written"] - m["noise written"])

    content = interval(data, lambda m: m["shared_effect rewritten"] / m["shared_effect written"], n_boot)
    written = interval(data, lambda m: m["shared_effect written"], n_boot)
    match = float(np.mean(hits))
    if match < check["min_source_match"]:
        state = "INSTRUMENT_INVALID"
    elif written["ci_low"] <= 0:
        state = "NO_SHARED_EFFECT"
    else:
        state = STATES.get(three_way(content["ci_low"], content["ci_high"], check["ideas_low"], check["voice_high"]), "MIXED_OR_UNCLEAR")

    return {
        "state": state,
        "rewriter": check["rewriter"],
        "prompts_used": len(prompts),
        "answers_used": sum(len(pairs[c][p]) for c in cells(pilot) for p in prompts),
        "content_share": content,
        "shared_effect_written": written,
        "shared_effect_rewritten": interval(data, lambda m: m["shared_effect rewritten"], n_boot),
        "carryover_written": interval(data, carry("written"), n_boot),
        "carryover_rewritten": interval(data, carry("rewritten"), n_boot),
        "teacher_effect_share": interval(data, teacher_share, n_boot),
        "source_match": match,
        "style": style_stats(pairs, pilot, prompts),
        "mean_energy": {k: float(v.mean()) for k, v in data.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-usd", type=float, required=True, help="Budget for the rewrites; finished ones are skipped")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "pilot_style_norm")
    sources = select_sources(args.run_dir, cfg["pilot_grid"], cfg["pilot_style_norm"]["samples_per_cell"])
    if asyncio.run(rewrite(args, cfg, sources))["stopped_on_budget"]:
        raise SystemExit("Rewrites stopped on budget. Rerun with a larger --max-usd; finished rewrites are kept.")
    result = analyze(args.run_dir, cfg, sources)
    path = results_dir(args.run_dir, PILOT, "style_norm") / "retention.json"
    write_json(path, result)
    print(f"pilot style normalization: {result['state']}")
    for key in ("content_share", "shared_effect_written", "shared_effect_rewritten", "carryover_written", "carryover_rewritten", "teacher_effect_share"):
        row = result[key]
        print(f"{key:26s} {row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
    print(f"source_match {result['source_match']:.3f} over {result['answers_used']} answers on {result['prompts_used']} prompts")
    for name, row in result["style"].items():
        print(f"{name:22s} words {row['mean_words']:.1f}  formatted {row['formatted_share']:.2f}")
    print(path)


if __name__ == "__main__":
    main()
