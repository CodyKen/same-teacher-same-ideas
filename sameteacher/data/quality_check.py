"""Check samples for problems that would fake a result, before measuring anything.

High exact-duplicate rates usually mean a provider is caching responses, which would
look like low diversity. Refusals are dropped because refusing is not an idea.
"""
import argparse
from collections import Counter, defaultdict

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, write_json

from sameteacher.core.paths import results_dir, stage_of, teacher_samples_path
from sameteacher.data.teacher_answers import group_name, load_groups


def duplicate_rate(texts: list[str]) -> float:
    return 1 - len(set(texts)) / len(texts) if texts else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    teacher_cfg = load_yaml(args.config)["pilot_teachers"]
    rows = read_jsonl(teacher_samples_path(args.run_dir, args.split))
    if not rows:
        raise SystemExit("No samples found. Run sameteacher.data.sample_teachers first.")

    by_group = defaultdict(lambda: defaultdict(list))
    finish = defaultdict(Counter)
    tokens = Counter()
    for row in rows:
        name = group_name(row)
        by_group[name][row["prompt_id"]].append(row["text"])
        finish[name][row["finish_reason"]] += 1
        tokens[name] += row.get("output_tokens", 0)

    groups, usable, counts = load_groups(args.run_dir, args.split, teacher_cfg["min_usable_per_group"])
    report, problems = {}, []
    for name in sorted(by_group):
        dups = [duplicate_rate(texts) for texts in by_group[name].values()]
        mean_dup = sum(dups) / len(dups)
        c = counts[name]
        truncated = finish[name]["length"] / max(1, c["total"])
        report[name] = {
            "responses": c["total"],
            "empty_rate": round(c["empty"] / c["total"], 3),
            "refusal_rate": round(c["refusal"] / c["total"], 3),
            "reasoning_rate": round(c["reasoning"] / c["total"], 3),
            "exact_duplicate_rate": round(mean_dup, 3),
            "truncated_rate": round(truncated, 3),
            "mean_output_tokens": round(tokens[name] / c["total"], 1),
        }
        if mean_dup > teacher_cfg["max_exact_duplicate_rate"]:
            problems.append(f"{name}: duplicates {mean_dup:.0%}, check for provider caching or ignored sampling settings")
        if truncated > 0.2:
            problems.append(f"{name}: {truncated:.0%} cut off at max_tokens, so answers are incomplete")
        if c["empty"] / c["total"] > 0.05:
            problems.append(f"{name}: {c['empty'] / c['total']:.0%} empty, hidden reasoning may be using up max_tokens")
        if c["reasoning"] / c["total"] > 0.02:
            problems.append(f"{name}: {c['reasoning'] / c['total']:.0%} reasoning without an answer, dropped from every measurement")

    prompts_total = len({r["prompt_id"] for r in rows})
    if len(usable) < 0.7 * prompts_total:
        problems.append(f"only {len(usable)} of {prompts_total} prompts usable across every group")

    summary = {"groups": report, "usable_prompts": len(usable), "total_prompts": prompts_total, "problems": problems}
    write_json(results_dir(args.run_dir, stage_of(args.split), "teacher_distance") / f"qc_{args.split}.json", summary)
    for name, row in report.items():
        print(name, row)
    print(f"usable prompts: {len(usable)}/{prompts_total}")
    print("PROBLEMS:" if problems else "no problems found", *problems, sep="\n  " if problems else " ")


if __name__ == "__main__":
    main()
