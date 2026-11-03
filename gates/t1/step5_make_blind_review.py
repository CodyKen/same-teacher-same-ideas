"""Build blind review sheets from the confirm split.

For each sampled prompt there are four pairs of answers:
  cross   two different teachers: the pair type under test
  within  the same teacher twice: the comparison point
  style   an answer and its own paraphrase: reviewers should say same idea
  idea    an ordinary answer and an idea-shifted one: reviewers should say different
Reviewers see no teacher names and no pair type. The key is stored separately.
Each reviewer fills `label` with S (same idea), D (different idea) or U (unsure).
"""
import argparse
import csv
import random
from pathlib import Path

from gates_lib.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json
from gates_lib.text import clean, common_word_budget, first_words, is_answer

from .common import IDEA_SUFFIX, STYLE_SUFFIX, group_name, load_groups, prompts_path, samples_path, t1_dir

MAX_WORDS = 120


def review_dir(run_dir) -> Path:
    return t1_dir(run_dir) / "review"


def build_pairs(run_dir, cfg, rng) -> list[dict]:
    t1 = cfg["t1"]
    anchor = t1["calibration_teacher"]
    groups, usable, _ = load_groups(run_dir, "confirm", t1["min_usable_per_group"])
    teachers = [n for n in groups if "::" not in n]
    prompts = {p["prompt_id"]: p["prompt"] for p in read_jsonl(prompts_path(run_dir, "confirm"))}

    originals = {}
    for row in read_jsonl(samples_path(run_dir, "confirm")):
        if is_answer(row["text"]):
            originals[(group_name(row), row["prompt_id"], row["sample_idx"])] = clean(row["text"])
    paraphrase_pairs = {}
    for (name, pid, idx), text in originals.items():
        if name == anchor + STYLE_SUFFIX and (anchor, pid, idx) in originals:
            paraphrase_pairs.setdefault(pid, []).append((originals[(anchor, pid, idx)], text))

    chosen = rng.sample(sorted(usable), min(cfg["t1_gate1"]["human"]["prompts"], len(usable)))
    pairs = []
    for pid in chosen:
        if pid not in paraphrase_pairs:
            continue
        budget = min(MAX_WORDS, common_word_budget([groups[n][pid] for n in sorted(groups)], t1["min_words"]))
        a, b = rng.sample(teachers, 2)
        same = rng.choice(teachers)
        first, second = rng.sample(groups[same][pid], 2)
        style_first, style_second = rng.choice(paraphrase_pairs[pid])
        candidates = {
            "cross": (rng.choice(groups[a][pid]), rng.choice(groups[b][pid])),
            "within": (first, second),
            "style": (style_first, style_second),
            "idea": (rng.choice(groups[anchor][pid]), rng.choice(groups[anchor + IDEA_SUFFIX][pid])),
        }
        for pair_type, (text_1, text_2) in candidates.items():
            pairs.append(
                {
                    "pair_type": pair_type,
                    "prompt_id": pid,
                    "prompt": prompts.get(pid, ""),
                    "text_1": first_words(text_1, budget),
                    "text_2": first_words(text_2, budget),
                }
            )
    # Number pairs after shuffling, or the id would give away the pair type.
    rng.shuffle(pairs)
    for i, pair in enumerate(pairs):
        pair["pair_id"] = f"{i:04d}"
    return pairs


def write_sheet(path, pairs, rng) -> None:
    order = pairs[:]
    rng.shuffle(order)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "prompt", "text_1", "text_2", "label"])
        for pair in order:
            one, two = pair["text_1"], pair["text_2"]
            if rng.random() < 0.5:
                one, two = two, one
            writer.writerow([pair["pair_id"], pair["prompt"], one, two, ""])


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--raters", type=int, default=3)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pairs = build_pairs(args.run_dir, cfg, random.Random(cfg["splits"]["seed"]))
    if not pairs:
        raise SystemExit("No pairs could be built. Sample the confirm split with controls first.")
    folder = review_dir(args.run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "key_DO_NOT_OPEN.json", {p["pair_id"]: {"pair_type": p["pair_type"], "prompt_id": p["prompt_id"]} for p in pairs})
    for rater in range(1, args.raters + 1):
        write_sheet(folder / f"review_sheet_r{rater}.csv", pairs, random.Random(stable_seed("rater", rater)))
    print(f"{len(pairs)} pairs for {args.raters} reviewers in {folder}")


if __name__ == "__main__":
    main()
