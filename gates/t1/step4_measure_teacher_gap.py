"""Measure how different teachers are on the same prompts, on a calibrated scale.

Per prompt, energy distance (0 = same distribution) is computed for:
  gap    mean over teacher pairs
  style  anchor teacher vs its paraphrases: same ideas, different voice
  idea   anchor teacher vs its idea-shifted answers: different ideas
  noise  two random halves of the anchor teacher: should sit near 0

calibrated_gap = (gap - style) / (idea - style)
0 means teachers differ no more than a rewrite does; 1 means as much as forcing a new idea.
If the idea control does not beat the style control, the instrument itself has failed,
and that says nothing about the teachers.
"""
import argparse
import itertools
import random
import re
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from gates_lib.config import add_common_args, load_yaml, read_jsonl, write_json
from gates_lib.embed import embed_texts, rows_dense
from gates_lib.lock import verify_lock
from gates_lib.stats import bootstrap, energy_distance, three_way
from gates_lib.text import common_word_budget, first_words

from .common import IDEA_SUFFIX, STYLE_SUFFIX, load_groups, prompts_path, t1_dir


def safe_name(embedder: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", embedder)


def result_path(run_dir, split: str, embedder: str):
    return t1_dir(run_dir) / "gate1" / f"{split}__{safe_name(embedder)}.json"


def calibrated(m: dict) -> float:
    span = m["idea"] - m["style"]
    return (m["gap"] - m["style"]) / span if span > 0 else float("nan")


def build_corpus(groups, prompt_ids, names, floor):
    texts, index = [], {}
    for pid in prompt_ids:
        budget = common_word_budget([groups[n][pid] for n in names], floor)
        for name in names:
            start = len(texts)
            texts.extend(first_words(t, budget) for t in groups[name][pid])
            index[(pid, name)] = list(range(start, len(texts)))
    return texts, index


def attribution_accuracy(matrix, labels, prompt_groups, max_rows=5000, seed=0) -> float:
    """Can a classifier name the teacher on prompts it has not seen? Style counts here."""
    rows = list(range(len(labels)))
    if len(rows) > max_rows:
        rows = random.Random(seed).sample(rows, max_rows)
    x = matrix[rows]
    y = np.array([labels[i] for i in rows])
    g = np.array([prompt_groups[i] for i in rows])
    correct = 0
    for train, test in GroupKFold(n_splits=min(5, len(set(g)))).split(x, y, g):
        model = LogisticRegression(max_iter=300)
        model.fit(x[train], y[train])
        correct += int((model.predict(x[test]) == y[test]).sum())
    return correct / len(y)


def measure(run_dir, split: str, cfg: dict, embedder: str, seed: int = 0) -> dict:
    t1, gate = cfg["t1"], cfg["t1_gate1"]
    anchor = t1["calibration_teacher"]
    groups, usable, _ = load_groups(run_dir, split, t1["min_usable_per_group"])
    teachers = [n for n in groups if "::" not in n]
    names = sorted(groups)
    if not usable:
        raise SystemExit("No usable prompts. Check step3_quality_check.")

    texts, index = build_corpus(groups, usable, names, t1["min_words"])
    matrix = embed_texts(texts, embedder)
    rng = random.Random(seed)
    table = defaultdict(list)
    per_pair = defaultdict(list)

    def vectors(pid, name):
        return rows_dense(matrix, index[(pid, name)])

    for pid in usable:
        by_teacher = {t: vectors(pid, t) for t in teachers}
        pairs = []
        for a, b in itertools.combinations(teachers, 2):
            distance = energy_distance(by_teacher[a], by_teacher[b])
            pairs.append(distance)
            per_pair[f"{a} vs {b}"].append(distance)
        base = vectors(pid, anchor)
        order = list(range(len(base)))
        rng.shuffle(order)
        half = len(order) // 2
        table["gap"].append(float(np.mean(pairs)))
        table["style"].append(energy_distance(base, vectors(pid, anchor + STYLE_SUFFIX)))
        table["idea"].append(energy_distance(base, vectors(pid, anchor + IDEA_SUFFIX)))
        table["noise"].append(energy_distance(base[order[:half]], base[order[half:]]))

    columns = {k: np.array(v) for k, v in table.items()}
    n_boot = gate["bootstrap"]
    estimate, low, high = bootstrap(columns, calibrated, n_boot)
    sep_est, sep_low, sep_high = bootstrap(columns, lambda m: m["idea"] - m["style"], n_boot)

    teacher_rows = [i for (pid, name), idx in index.items() if name in teachers for i in idx]
    labels_by_row = {i: name for (pid, name), idx in index.items() if name in teachers for i in idx}
    prompts_by_row = {i: pid for (pid, name), idx in index.items() if name in teachers for i in idx}
    labels = [labels_by_row[i] for i in teacher_rows]
    prompt_ids = [prompts_by_row[i] for i in teacher_rows]
    attribution = attribution_accuracy(matrix[teacher_rows], labels, prompt_ids)

    categories = {p["prompt_id"]: p.get("category", "unknown") for p in read_jsonl(prompts_path(run_dir, split))}
    by_category = defaultdict(list)
    for pid, gap in zip(usable, table["gap"]):
        by_category[categories.get(pid, "unknown")].append(gap)

    return {
        "split": split,
        "embedder": embedder,
        "prompts_used": len(usable),
        "teachers": teachers,
        "mean_energy": {k: float(v.mean()) for k, v in columns.items()},
        "calibrated_gap": {"estimate": estimate, "ci_low": low, "ci_high": high},
        "idea_minus_style": {"estimate": sep_est, "ci_low": sep_low, "ci_high": sep_high},
        "instrument_valid": sep_low > 0,
        "state": three_way(low, high, gate["go_low"], gate["stop_high"]) if sep_low > 0 else "INSTRUMENT_INVALID",
        "teacher_attribution_accuracy": attribution,
        "attribution_chance": 1 / len(teachers),
        "per_pair_energy": {k: float(np.mean(v)) for k, v in sorted(per_pair.items(), key=lambda kv: np.mean(kv[1]))},
        "gap_by_category": {k: {"mean_energy": float(np.mean(v)), "prompts": len(v)} for k, v in sorted(by_category.items())},
        "per_prompt": {"prompt_id": usable, **{k: [float(x) for x in v] for k, v in table.items()}},
        "per_prompt_pair": {k: [float(x) for x in v] for k, v in per_pair.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", required=True, choices=["screen", "confirm"])
    parser.add_argument("--embedder", help="Default: the primary embedder in the config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.split == "confirm":
        verify_lock(cfg, args.run_dir, "t1_gate1")
    embedder = args.embedder or cfg["t1_gate1"]["primary_embedder"]
    result = measure(args.run_dir, args.split, cfg, embedder)
    path = result_path(args.run_dir, args.split, embedder)
    write_json(path, result)
    short = {k: result[k] for k in ("prompts_used", "mean_energy", "calibrated_gap", "idea_minus_style", "state")}
    print(short, f"attribution {result['teacher_attribution_accuracy']:.2f} vs chance {result['attribution_chance']:.2f}", sep="\n")
    if args.split == "screen":
        print("screen is for tuning only. The gate decision comes from the confirm split.")
    print(path)


if __name__ == "__main__":
    main()
