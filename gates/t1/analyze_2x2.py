"""Does sharing a teacher pull two different base models closer together?

              teacher A   teacher B
  base X        X<-A        X<-B
  base Y        Y<-A        Y<-B

shared_effect = mean(D(X<-A, Y<-B), D(X<-B, Y<-A)) - mean(D(X<-A, Y<-A), D(X<-B, Y<-B))
transfer      = distance between teachers within one base
noise         = distance between seeds of the same cell
carryover     = shared_effect / (transfer - noise)

carryover asks how much of what a teacher does to one base also shows up in the other.
This is the total effect of the teacher's data, and these are few training runs.
A pilot result justifies more seeds; it does not establish a mechanism.
"""
import argparse
import itertools
from collections import defaultdict

import numpy as np

from gates_lib.config import add_common_args, load_yaml, read_jsonl, write_json
from gates_lib.embed import embed_texts, rows_dense
from gates_lib.lock import verify_lock
from gates_lib.stats import GO, INCONCLUSIVE, bootstrap, energy_distance, three_way
from gates_lib.text import clean, common_word_budget, first_words, is_answer

from .common import load_groups, t1_dir
from .run_2x2 import student_name


def carryover(m: dict) -> float:
    net = m["transfer"] - m["noise"]
    return m["shared_effect"] / net if net > 0 else float("nan")


def prompt_effects(dist, bases, teachers, seeds) -> dict:
    """Shared effect, transfer and noise on one prompt, from dist(cell, cell) on (base, teacher, seed)."""
    (x, y), (a, b) = bases, teachers
    seed_pairs = list(itertools.product(seeds, seeds))
    by_seed = {}
    for sx, sy in seed_pairs:
        shared = (dist((x, a, sx), (y, a, sy)) + dist((x, b, sx), (y, b, sy))) / 2
        different = (dist((x, a, sx), (y, b, sy)) + dist((x, b, sx), (y, a, sy))) / 2
        by_seed[(sx, sy)] = different - shared
    return {
        "shared_effect": float(np.mean(list(by_seed.values()))),
        "transfer": float(np.mean([dist((base, a, s1), (base, b, s2)) for base in bases for s1, s2 in seed_pairs])),
        "noise": float(np.mean([dist((base, t, s1), (base, t, s2)) for base in bases for t in teachers for s1, s2 in itertools.combinations(seeds, 2)])),
        "by_seed": by_seed,
    }


def load_students(run_dir, pilot) -> dict:
    students = {}
    for base, teacher, seed in itertools.product(pilot["bases"], pilot["teachers"], pilot["seeds"]):
        name = student_name(base, teacher, seed)
        by_prompt = defaultdict(list)
        for row in read_jsonl(t1_dir(run_dir) / "students" / f"{name}.jsonl"):
            if is_answer(row["text"]):
                by_prompt[row["prompt_id"]].append(clean(row["text"]))
        students[(base, teacher, seed)] = by_prompt
    return students


def analyze(run_dir, cfg) -> dict:
    pilot, gate, t1 = cfg["t1_pilot"], cfg["t1_gate2"], cfg["t1"]
    bases, teachers, seeds = list(pilot["bases"]), pilot["teachers"], pilot["seeds"]
    if len(bases) != 2 or len(teachers) != 2 or len(seeds) < 2:
        raise SystemExit("This analysis needs 2 bases, 2 teachers, and at least 2 seeds.")
    (x, y), (a, b) = bases, teachers

    students = load_students(run_dir, pilot)
    teacher_groups, _, _ = load_groups(run_dir, "pilot_eval", 2)
    has_teachers = all(t in teacher_groups for t in teachers)
    minimum = t1["min_usable_per_group"]
    prompts = sorted(set.intersection(*[{p for p, v in s.items() if len(v) >= minimum} for s in students.values()]))
    if has_teachers:
        prompts = [p for p in prompts if all(len(teacher_groups[t].get(p, [])) >= minimum for t in teachers)]
    if not prompts:
        raise SystemExit("No prompts are covered by every cell. Check that run_2x2 finished.")

    texts, index = [], {}
    for pid in prompts:
        groups = {key: s[pid] for key, s in students.items()}
        if has_teachers:
            groups |= {t: teacher_groups[t][pid] for t in teachers}
        budget = common_word_budget(list(groups.values()), t1["min_words"])
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, key)] = list(range(start, len(texts)))
    matrix = embed_texts(texts, gate["embedder"])

    columns = defaultdict(list)
    seed_effects = defaultdict(list)
    for pid in prompts:
        cache = {}

        def vec(key):
            return cache.setdefault(key, rows_dense(matrix, index[(pid, key)]))

        def dist(first, second):
            return energy_distance(vec(first), vec(second))

        effects = prompt_effects(dist, bases, teachers, seeds)
        for key in ("shared_effect", "transfer", "noise"):
            columns[key].append(effects[key])
        for pair, value in effects["by_seed"].items():
            seed_effects[pair].append(value)
        if has_teachers:
            gaps = []
            for base, seed in itertools.product(bases, seeds):
                gaps.append(dist((base, a, seed), b) - dist((base, a, seed), a))
                gaps.append(dist((base, b, seed), a) - dist((base, b, seed), b))
            columns["closer_to_own_teacher"].append(float(np.mean(gaps)))

    data = {k: np.array(v) for k, v in columns.items()}
    n_boot = gate["bootstrap"]
    core = {k: data[k] for k in ("shared_effect", "transfer", "noise")}
    carry, carry_low, carry_high = bootstrap(core, carryover, n_boot)
    net, net_low, net_high = bootstrap(core, lambda m: m["transfer"] - m["noise"], n_boot)
    effect, effect_low, effect_high = bootstrap(core, lambda m: m["shared_effect"], n_boot)
    per_seed = {f"{sx}-{sy}": float(np.mean(v)) for (sx, sy), v in seed_effects.items()}
    all_positive = all(v > 0 for v in per_seed.values())

    if net_low <= 0:
        state = "NO_TRANSFER"
    else:
        state = three_way(carry_low, carry_high, gate["go_low"], gate["stop_high"])
        if state == GO and not (effect_low > 0 and all_positive):
            state = INCONCLUSIVE

    result = {
        "state": state,
        "prompts_used": len(prompts),
        "seeds": seeds,
        "carryover": {"estimate": carry, "ci_low": carry_low, "ci_high": carry_high},
        "shared_effect": {"estimate": effect, "ci_low": effect_low, "ci_high": effect_high},
        "net_transfer": {"estimate": net, "ci_low": net_low, "ci_high": net_high},
        "shared_effect_by_seed_pair": per_seed,
        "mean_energy": {k: float(v.mean()) for k, v in data.items()},
    }
    if has_teachers:
        closer, closer_low, closer_high = bootstrap({"c": data["closer_to_own_teacher"]}, lambda m: m["c"], n_boot)
        result["closer_to_own_teacher"] = {"estimate": closer, "ci_low": closer_low, "ci_high": closer_high}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "t1_gate2")
    result = analyze(args.run_dir, cfg)
    write_json(t1_dir(args.run_dir) / "gate2" / "decision.json", result)
    print(f"T1 gate 2: {result['state']}")
    for key in ("carryover", "shared_effect", "net_transfer", "shared_effect_by_seed_pair", "closer_to_own_teacher"):
        print(key, result.get(key))
    print(f"based on {len(cfg['t1_pilot']['seeds'])} seeds per cell: exploratory, confirm with more seeds before claiming an effect")


if __name__ == "__main__":
    main()
