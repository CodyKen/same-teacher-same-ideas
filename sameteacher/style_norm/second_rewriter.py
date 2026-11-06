"""H2 robustness to the rewriter: a second rewriter from another model family.

The second rewriter turns the same source answers as `main_style_norm` into plain prose, on a
prompt subset stratified by category, with all 27 students on every prompt. The H1 quantities
are computed in three stages on the same answers (as written, rewritten by the primary
rewriter, rewritten by the second one), with one word budget per prompt across the stages:
    R_second = S second / S written,   R_primary = S primary / S written
The decision rule is in the locked `main_style_norm_second` config section.
"""
import argparse
import asyncio
from collections import defaultdict

import numpy as np

from sameteacher.analysis.main_shared_teacher import prompt_table, quantities, student_columns
from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, prompts_path, results_dir, style_norm_dir
from sameteacher.core.text import LIST_MARKER, MARKDOWN, clean, common_word_budget, first_words, is_answer
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.eval.stats import GO, STOP, bootstrap_means, percentile_interval, three_way
from sameteacher.style_norm import main_retention as primary
from sameteacher.style_norm.pilot_retention import NEUTRAL_INSTRUCTION, source_hits

SECTION = "main_style_norm_second"
STAGES = ("written", "primary", "second")
REWRITERS = STAGES[1:]
STATES = {GO: "SUPPORTED", STOP: "VOICE"}


def second_path(run_dir, rewriter: str):
    return style_norm_dir(run_dir) / f"main_students__{rewriter}.jsonl"


def usable(row) -> bool:
    return bool(row) and row["finish_reason"] != "length" and is_answer(row["text"])


def prompt_subset(run_dir, cfg) -> list[str]:
    """Every prompt_step-th prompt of the H2 pool, sorted by category and a seeded hash."""
    main_cfg, first, section = cfg["main_grid"], cfg[primary.SECTION], cfg[SECTION]
    _, _, _, cells = primary.grid(main_cfg)
    sources = primary.select_sources(run_dir, main_cfg, first["samples_per_cell"])
    _, pool = primary.matched_pairs(run_dir, cells, sources, first["min_pairs"])
    path = prompts_path(run_dir, main_cfg["test_split"])
    category = {p["prompt_id"]: p.get("category", "") for p in read_jsonl(path)} if path.exists() else {}
    ordered = sorted(pool, key=lambda pid: (category.get(pid, ""), stable_seed(section["subset_seed"], pid)))
    return sorted(ordered[:: section["prompt_step"]])


def select_sources(run_dir, cfg, prompts) -> list[dict]:
    keep = set(prompts)
    rows = primary.select_sources(run_dir, cfg["main_grid"], cfg[SECTION]["samples_per_cell"])
    return [row for row in rows if row["prompt_id"] in keep]


async def rewrite(args, cfg, sources) -> dict:
    sampling, section = cfg["sampling"], cfg[SECTION]
    spec = load_models(args.models)[section["rewriter"]]
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
        stats = await run_jobs(sources, handler, second_path(args.run_dir, section["rewriter"]), args.concurrency or sampling["concurrency"])
    finally:
        await chat.aclose()
    print(f"rewrites: {stats}, spent about ${chat.budget.spent:.2f} at the configured prices, {chat.throttled} throttled and {chat.malformed} malformed-response retries")
    return stats


def matched_triples(run_dir, cells, sources, rewriter: str, min_pairs: int) -> tuple[dict, list[str]]:
    """(source, primary rewrite, second rewrite) per cell and prompt, when both rewrites are usable."""
    first = {f"{r['student']}|{r['prompt_id']}|{r['sample_idx']}": r for r in read_jsonl(primary.rewrites_path(run_dir))}
    path = second_path(run_dir, rewriter)
    second = {f"{r['student']}|{r['prompt_id']}|{r['sample_idx']}": r for r in read_jsonl(path)} if path.exists() else {}
    triples = defaultdict(lambda: defaultdict(list))
    for source in sources:
        one, two = first.get(source["key"]), second.get(source["key"])
        if usable(one) and usable(two):
            triples[source["cell"]][source["prompt_id"]].append((source["text"], one["text"], two["text"]))
    prompts = set.intersection(*[{p for p, v in triples[c].items() if len(v) >= min_pairs} for c in cells])
    return triples, sorted(prompts)


def retention(m: dict, bases, teachers) -> dict:
    """Per group: S and carryover in every stage, R and teacher-effect share per rewriter."""
    q = {stage: quantities(primary.stage_view(m, stage), bases, teachers) for stage in STAGES}
    out = {}
    for group, written in q["written"].items():
        row = {"S written": written["S"], "carryover written": written["carryover"]}
        for stage in REWRITERS:
            r = q[stage][group]
            row[f"R {stage}"] = r["S"] / written["S"] if written["S"] != 0 else float("nan")
            row[f"S {stage}"] = r["S"]
            row[f"carryover {stage}"] = r["carryover"]
            row[f"teacher_effect_share {stage}"] = r["net"] / written["net"] if written["net"] > 0 else float("nan")
        row["R second - R primary"] = row["R second"] - row["R primary"]
        out[group] = row
    return out


def style_stats(triples, cells, teachers, prompts) -> dict:
    out = {}
    for teacher in teachers:
        for stage_index, stage in enumerate(STAGES):
            texts = [t[stage_index] for cell in cells if cell[1] == teacher for p in prompts for t in triples[cell][p]]
            out[f"{teacher} {stage}"] = {
                "mean_words": float(np.mean([len(t.split()) for t in texts])),
                "formatted_share": float(np.mean([bool(MARKDOWN.search(t) or LIST_MARKER.search(t) or "\n" in t.strip()) for t in texts])),
            }
    return out


def decide(section, pooled, match_second) -> str:
    if match_second < section["min_source_match"]:
        return "INSTRUMENT_INVALID"
    if pooled["S written"]["ci_low"] <= 0:
        return "NO_SHARED_EFFECT"
    return STATES.get(three_way(pooled["R second"]["ci_low"], pooled["R second"]["ci_high"], section["most_low"], section["voice_high"]), "MIXED_OR_UNCLEAR")


def analyze(run_dir, cfg, sources) -> dict:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    bases, teachers, seeds, cells = primary.grid(main_cfg)
    triples, prompts = matched_triples(run_dir, cells, sources, section["rewriter"], section["min_pairs"])
    if not prompts:
        raise SystemExit("No prompt has enough usable rewrites from both rewriters for every student.")

    texts, index = [], {}
    for pid in prompts:
        groups = {(stage, cell): [clean(t[stage]) for t in triples[cell][pid]] for stage in range(len(STAGES)) for cell in cells}
        budget = common_word_budget(list(groups.values()), section["min_words"])
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, key)] = list(range(start, len(texts)))
    matrix = embed_texts(texts, section["embedder"])

    columns, hits = defaultdict(list), {stage: [] for stage in REWRITERS}
    for pid in prompts:
        for stage_index, stage in enumerate(STAGES):
            table = prompt_table(matrix, index, pid, [(stage_index, cell) for cell in cells])

            def dist(first, second, table=table, s=stage_index):
                return table[((s, first), (s, second))]

            for k, v in student_columns(dist, bases, teachers, seeds).items():
                columns[f"{stage} {k}"].append(v)
        for cell in cells:
            originals = [clean(t[0]) for t in triples[cell][pid]]
            for stage_index, stage in enumerate(STAGES[1:], start=1):
                hits[stage] += source_hits(rows_dense(matrix, index[(pid, (0, cell))]), rows_dense(matrix, index[(pid, (stage_index, cell))]), originals)

    point, draws = bootstrap_means({k: np.array(v) for k, v in columns.items()}, section["bootstrap"], section["bootstrap_seed"])
    estimate = retention(point, bases, teachers)
    sampled = [retention(d, bases, teachers) for d in draws]
    summary = {}
    for group, values in estimate.items():
        summary[group] = {}
        for name, value in values.items():
            low, high = percentile_interval([s[group][name] for s in sampled], section["ci_level"])
            summary[group][name] = {"estimate": value, "ci_low": low, "ci_high": high}

    match = {stage: float(np.mean(v)) for stage, v in hits.items()}
    return {
        "state": decide(section, summary["pooled"], match["second"]),
        "rewriters": {"primary": cfg[primary.SECTION]["rewriter"], "second": section["rewriter"]},
        "embedder": section["embedder"],
        "prompts_used": len(prompts),
        "answers_used": sum(len(triples[c][p]) for c in cells for p in prompts),
        "source_match": match,
        **summary,
        "style": style_stats(triples, cells, teachers, prompts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--max-usd", type=float, required=True, help="Budget for the rewrites at the configured prices; finished ones are skipped")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--skip-rewrite", action="store_true", help="Analyze the rewrites that exist")
    parser.add_argument("--subset-only", action="store_true", help="Print the prompt subset and the number of sources, then stop")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for name in ("main_grid", primary.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    prompts = prompt_subset(args.run_dir, cfg)
    sources = select_sources(args.run_dir, cfg, prompts)
    print(f"{len(prompts)} prompts, {len(sources)} source answers")
    if args.subset_only:
        return
    if not args.skip_rewrite and asyncio.run(rewrite(args, cfg, sources))["stopped_on_budget"]:
        raise SystemExit("Rewrites stopped on budget. Rerun with a larger --max-usd; finished rewrites are kept.")
    result = analyze(args.run_dir, cfg, sources)
    result["subset_prompts"] = prompts
    path = results_dir(args.run_dir, MAIN, "style_norm") / "h2_second_rewriter.json"
    write_json(path, result)
    print(f"second rewriter: {result['state']}")
    for group in [k for k in result if k == "pooled" or k.startswith(("teacher ", "base "))]:
        row = result[group]
        print(f"{group:22s} R second {row['R second']['estimate']:.3f} [{row['R second']['ci_low']:.3f}, {row['R second']['ci_high']:.3f}]"
              f"  R primary {row['R primary']['estimate']:.3f}  diff {row['R second - R primary']['estimate']:+.3f} [{row['R second - R primary']['ci_low']:+.3f}, {row['R second - R primary']['ci_high']:+.3f}]")
    print(f"source match {result['source_match']} over {result['answers_used']} answers on {result['prompts_used']} prompts")
    print(path)


if __name__ == "__main__":
    main()
