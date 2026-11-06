"""H1 on the main grid: are students on different bases closer when they share a teacher?

Three bases, three teachers and three seeds give 27 students. For one prompt, D is the
energy distance between two answer sets. For bases x != y and teachers a != b, one block is
the pilot's 2x2 shared effect,
    [D(x<-a, y<-b) + D(x<-b, y<-a)]/2 - [D(x<-a, y<-a) + D(x<-b, y<-b)]/2,
for one seed pair. S averages the blocks over seed pairs, base pairs and teacher pairs.
    T          distance between students of different teachers on the same base
    N          distance between seeds of the same base and teacher
    carryover  S / (T - N)
The same quantities are reported per teacher pair and per base pair, where T and N use only
that pair's teachers or bases. Every reported number is a function of per-prompt column
means, so one prompt bootstrap gives joint intervals for all of them. The decision rule is
in the locked `main_shared_teacher` config section.
"""
import argparse
import itertools
from collections import defaultdict

import numpy as np

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir, students_path
from sameteacher.core.text import clean, common_word_budget, first_words, is_answer
from sameteacher.data.teacher_answers import load_groups
from sameteacher.eval.embed import embed_texts
from sameteacher.eval.stats import bootstrap_means, percentile_interval

SECTION = "main_shared_teacher"


def joined(pair) -> str:
    return "-".join(pair)


def teacher_key(teacher: str) -> tuple:
    return ("teacher", teacher)


def student_columns(dist, bases, teachers, seeds) -> dict:
    """Per-prompt building blocks of S, T and N for the given seeds."""
    cols = {}
    seed_pairs = list(itertools.product(seeds, seeds))
    for (x, y), (a, b) in itertools.product(itertools.combinations(bases, 2), itertools.combinations(teachers, 2)):
        blocks = []
        for sx, sy in seed_pairs:
            different = (dist((x, a, sx), (y, b, sy)) + dist((x, b, sx), (y, a, sy))) / 2
            same = (dist((x, a, sx), (y, a, sy)) + dist((x, b, sx), (y, b, sy))) / 2
            blocks.append(different - same)
            cols[f"S {x}-{y} {a}-{b} s{sx}-{sy}"] = different - same
        cols[f"S {x}-{y} {a}-{b}"] = float(np.mean(blocks))
    for base, (a, b) in itertools.product(bases, itertools.combinations(teachers, 2)):
        cols[f"T {base} {a}-{b}"] = float(np.mean([dist((base, a, s1), (base, b, s2)) for s1, s2 in seed_pairs]))
        cols[f"Td {base} {a}-{b}"] = float(np.mean([dist((base, a, s1), (base, b, s2)) for s1, s2 in seed_pairs if s1 != s2]))
    for base, t in itertools.product(bases, teachers):
        cols[f"N {base} {t}"] = float(np.mean([dist((base, t, s1), (base, t, s2)) for s1, s2 in itertools.combinations(seeds, 2)]))
    return cols


def teacher_columns(dist, bases, teachers, seeds) -> dict:
    """Teacher reference distances, and how much closer each student is to its own teacher."""
    cols = {f"Ref {a}-{b}": dist(teacher_key(a), teacher_key(b)) for a, b in itertools.combinations(teachers, 2)}
    for base, t, s in itertools.product(bases, teachers, seeds):
        own = dist((base, t, s), teacher_key(t))
        cols[f"C {base} {t} s{s}"] = float(np.mean([dist((base, t, s), teacher_key(o)) - own for o in teachers if o != t]))
    return cols


def quantities(m: dict, bases, teachers, transfer: str = "T") -> dict:
    """S, T, N, T - N and carryover, pooled, per teacher pair and per base pair."""
    teacher_pairs = list(itertools.combinations(teachers, 2))
    base_pairs = list(itertools.combinations(bases, 2))

    def avg(keys):
        return float(np.mean([m[k] for k in keys]))

    def block(s_keys, t_keys, n_keys):
        S, T, N = avg(s_keys), avg(t_keys), avg(n_keys)
        net = T - N
        return {"S": S, "T": T, "N": N, "net": net, "carryover": S / net if net > 0 else float("nan")}

    out = {"pooled": block(
        [f"S {joined(bp)} {joined(tp)}" for bp in base_pairs for tp in teacher_pairs],
        [f"{transfer} {b} {joined(tp)}" for b in bases for tp in teacher_pairs],
        [f"N {b} {t}" for b in bases for t in teachers],
    )}
    for tp in teacher_pairs:
        out[f"teacher {joined(tp)}"] = block(
            [f"S {joined(bp)} {joined(tp)}" for bp in base_pairs],
            [f"{transfer} {b} {joined(tp)}" for b in bases],
            [f"N {b} {t}" for b in bases for t in tp],
        )
    for bp in base_pairs:
        out[f"base {joined(bp)}"] = block(
            [f"S {joined(bp)} {joined(tp)}" for tp in teacher_pairs],
            [f"{transfer} {b} {joined(tp)}" for b in bp for tp in teacher_pairs],
            [f"N {b} {t}" for b in bp for t in teachers],
        )
    return out


def summarize(columns: dict, bases, teachers, seeds, section: dict, transfer: str = "T") -> dict:
    """Point estimates and intervals for every quantity, from one set of bootstrap draws."""
    point, draws = bootstrap_means(columns, section["bootstrap"], section["bootstrap_seed"])
    estimate = quantities(point, bases, teachers, transfer)
    sampled = [quantities(d, bases, teachers, transfer) for d in draws]
    out = {}
    for group, values in estimate.items():
        out[group] = {}
        for name, value in values.items():
            series = [s[group][name] for s in sampled]
            low, high = percentile_interval(series, section["ci_level"])
            row = {"estimate": value, "ci_low": low, "ci_high": high}
            if group != "pooled" and name == "S":
                row["pair_ci_low"], row["pair_ci_high"] = percentile_interval(series, section["pair_ci_level"])
            out[group][name] = row
    return out | {"_draws": sampled, "_point": point, "_column_draws": draws}


def state(summary: dict, section: dict) -> str:
    pooled = summary["pooled"]
    if pooled["net"]["ci_low"] <= 0:
        return "NO_TRANSFER"
    if pooled["S"]["ci_low"] <= 0:
        return "NOT_SUPPORTED"
    pairs = [v for k, v in summary.items() if k.startswith(("teacher ", "base "))]
    if pooled["carryover"]["ci_low"] >= section["carryover_floor"] and all(p["S"]["pair_ci_low"] > 0 for p in pairs):
        return "SUPPORTED"
    return "PARTIAL"


def public(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def load_students(run_dir, main_cfg) -> dict:
    students = {}
    for cell in main_cells(main_cfg):
        by_prompt = defaultdict(list)
        for row in sorted(read_jsonl(students_path(run_dir, MAIN, cell["student"])), key=lambda r: (r["prompt_id"], r["sample_idx"])):
            if is_answer(row["text"]):
                by_prompt[row["prompt_id"]].append(clean(row["text"]))
        students[(cell["base"], cell["teacher"], cell["seed"])] = by_prompt
    return students


def build_texts(run_dir, cfg) -> tuple[list[str], dict, list[str], dict]:
    """Texts cut to one word budget per prompt, the index of each group, and the prompts used."""
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    minimum = section["min_usable_per_group"]
    students = load_students(run_dir, main_cfg)
    teacher_groups, _, _ = load_groups(run_dir, main_cfg["test_split"], 2)
    missing = [t for t in main_cfg["teachers"] if t not in teacher_groups]
    if missing:
        raise SystemExit(f"No teacher answers on {main_cfg['test_split']} for {missing}. Run sameteacher.data.sample_teachers first.")
    groups_by_prompt = {}
    candidates = set.intersection(*[{p for p, v in s.items() if len(v) >= minimum} for s in students.values()])
    for pid in sorted(candidates):
        if all(len(teacher_groups[t].get(pid, [])) >= minimum for t in main_cfg["teachers"]):
            groups_by_prompt[pid] = {key: s[pid] for key, s in students.items()} | {teacher_key(t): teacher_groups[t][pid] for t in main_cfg["teachers"]}
    if not groups_by_prompt:
        raise SystemExit("No prompt is covered by every student and teacher. Check that sameteacher.train.run_main finished.")

    texts, index, budgets = [], {}, {}
    for pid, groups in groups_by_prompt.items():
        budget = common_word_budget(list(groups.values()), section["min_words"])
        budgets[pid] = budget
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, key)] = list(range(start, len(texts)))
    return texts, index, list(groups_by_prompt), budgets


def prompt_table(matrix, index, pid, keys) -> dict:
    """Energy distance between every pair of groups on one prompt, from one Gram matrix.

    Gives the same values as `energy_distance` on each pair, without recomputing the
    within-group terms for every pair (and without densifying tf-idf rows).
    """
    rows = [i for k in keys for i in index[(pid, k)]]
    block = matrix[rows]
    gram = block @ block.T
    gram = gram.toarray() if hasattr(gram, "toarray") else np.asarray(gram)
    pairwise = np.sqrt(np.clip(2.0 - 2.0 * gram, 0.0, None))
    spans, start = {}, 0
    for k in keys:
        spans[k] = slice(start, start + len(index[(pid, k)]))
        start += len(index[(pid, k)])
    within = {}
    for k, s in spans.items():
        n = s.stop - s.start
        within[k] = float(pairwise[s, s].sum()) / (n * (n - 1))
    table = {}
    for first, second in itertools.combinations(keys, 2):
        cross = float(pairwise[spans[first], spans[second]].mean())
        table[(first, second)] = table[(second, first)] = 2 * cross - within[first] - within[second]
    return table


def distance_tables(matrix, index, prompts, keys) -> dict:
    """Energy distance between every pair of groups on every prompt."""
    return {pid: prompt_table(matrix, index, pid, keys) for pid in prompts}


def columns_for(tables, prompts, bases, teachers, seeds, with_teachers: bool = True) -> dict:
    columns = defaultdict(list)
    for pid in prompts:
        table = tables[pid]

        def dist(first, second, table=table):
            return table[(first, second)]

        row = student_columns(dist, bases, teachers, seeds)
        if with_teachers:
            row |= teacher_columns(dist, bases, teachers, seeds)
        for k, v in row.items():
            columns[k].append(v)
    return {k: np.array(v) for k, v in columns.items()}


def analyze_embedding(matrix, index, prompts, cfg, primary: bool) -> dict:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    bases, teachers, seeds = list(main_cfg["bases"]), list(main_cfg["teachers"]), list(main_cfg["seeds"])
    keys = [(b, t, s) for b, t, s in itertools.product(bases, teachers, seeds)] + [teacher_key(t) for t in teachers]
    tables = distance_tables(matrix, index, prompts, keys)
    columns = columns_for(tables, prompts, bases, teachers, seeds)
    summary = summarize(columns, bases, teachers, seeds, section)
    result = {"state": state(summary, section), "prompts_used": len(prompts)} | public(summary)

    # Prediction from the pilot's teacher distances: the order of S across teacher pairs.
    predicted = [f"teacher {joined(p)}" for p in section["predicted_order"]]
    observed = sorted((k for k in summary if k.startswith("teacher ")), key=lambda k: -summary[k]["S"]["estimate"])
    in_order = [all(d[predicted[i]]["S"] > d[predicted[i + 1]]["S"] for i in range(len(predicted) - 1)) for d in summary["_draws"]]
    result["order"] = {"predicted": predicted, "observed": observed, "share_of_draws_in_predicted_order": float(np.mean(in_order))}

    if not primary:
        return result

    # Seed-pair consistency: blocks of one base pair, teacher pair and seed pair, averaged over prompts.
    seed_blocks = {k[2:]: float(v.mean()) for k, v in columns.items() if k.startswith("S ") and k.count(" ") == 3}
    result["seed_pair_blocks"] = {"positive": sum(v > 0 for v in seed_blocks.values()), "total": len(seed_blocks), "min": min(seed_blocks.values()), "values": seed_blocks}

    point, draws = summary["_point"], summary["_column_draws"]

    def mean_of(prefix, d):
        return float(np.mean([v for k, v in d.items() if k.startswith(prefix)]))

    def interval_of(prefix):
        low, high = percentile_interval([mean_of(prefix, d) for d in draws], section["ci_level"])
        return {"estimate": mean_of(prefix, point), "ci_low": low, "ci_high": high}

    result["teacher_reference"] = {joined(p): interval_of(f"Ref {joined(p)}") for p in itertools.combinations(teachers, 2)}
    result["closer_to_own_teacher"] = {"pooled": interval_of("C ")} | {t: interval_of_teacher(t, point, draws, section) for t in teachers}

    if section.get("distinct_seed_transfer"):
        distinct = summarize(columns, bases, teachers, seeds, section, transfer="Td")
        result["distinct_seed_transfer"] = {k: {"T": v["T"], "net": v["net"], "carryover": v["carryover"]} for k, v in public(distinct).items()}

    if section.get("leave_one_seed_out"):
        result["leave_one_seed_out"] = {}
        for left_out in seeds:
            kept = [s for s in seeds if s != left_out]
            sub = summarize(columns_for(tables, prompts, bases, teachers, kept, with_teachers=False), bases, teachers, kept, section)
            result["leave_one_seed_out"][f"without seed {left_out}"] = {"state": state(sub, section)} | {k: {"S": v["S"], "carryover": v["carryover"]} for k, v in public(sub).items()}

    result["per_prompt"] = {
        pid: {"S": float(np.mean([columns[f"S {joined(bp)} {joined(tp)}"][i] for bp in itertools.combinations(bases, 2) for tp in itertools.combinations(teachers, 2)]))}
        | {f"S {joined(tp)}": float(np.mean([columns[f"S {joined(bp)} {joined(tp)}"][i] for bp in itertools.combinations(bases, 2)])) for tp in itertools.combinations(teachers, 2)}
        for i, pid in enumerate(prompts)
    }
    return result


def interval_of_teacher(teacher, point, draws, section) -> dict:
    def mean_of(d):
        return float(np.mean([v for k, v in d.items() if k.startswith("C ") and k.split()[2] == teacher]))

    low, high = percentile_interval([mean_of(d) for d in draws], section["ci_level"])
    return {"estimate": mean_of(point), "ci_low": low, "ci_high": high}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--only-primary", action="store_true", help="Skip the robustness embedders")
    parser.add_argument(
        "--embedders", nargs="+",
        help="Run only these configured embedders. One process per embedder keeps GPU memory from piling up across models.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "main_grid")
    verify_lock(cfg, args.run_dir, SECTION)
    section = cfg[SECTION]
    configured = [section["embedder"]] + list(section["robustness_embedders"])
    if args.embedders:
        unknown = [n for n in args.embedders if n not in configured]
        if unknown:
            raise SystemExit(f"{unknown} are not configured embedders: {configured}")
        names = args.embedders
    else:
        names = configured[:1] if args.only_primary else configured
    texts, index, prompts, budgets = build_texts(args.run_dir, cfg)
    out_dir = results_dir(args.run_dir, MAIN, "shared_teacher")
    print(f"{len(texts)} answers on {len(prompts)} prompts; word budget median {int(np.median(list(budgets.values())))}")

    for name in names:
        primary = name == section["embedder"]
        result = analyze_embedding(embed_texts(texts, name), index, prompts, cfg, primary)
        result["embedder"] = name
        if primary:
            result["word_budget"] = {"median": float(np.median(list(budgets.values()))), "min": min(budgets.values()), "max": max(budgets.values())}
            path = out_dir / "h1.json"
        else:
            result["note"] = "robustness embedder, reported and not part of the locked decision"
            path = out_dir / f"robustness__{name.split('/')[-1]}.json"
        write_json(path, result)
        pooled = result["pooled"]
        print(f"[{name}] state {result['state']}  S {pooled['S']['estimate']:.4f} [{pooled['S']['ci_low']:.4f}, {pooled['S']['ci_high']:.4f}]"
              f"  carryover {pooled['carryover']['estimate']:.3f} [{pooled['carryover']['ci_low']:.3f}, {pooled['carryover']['ci_high']:.3f}] -> {path}")


if __name__ == "__main__":
    main()
