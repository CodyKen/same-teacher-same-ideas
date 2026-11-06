"""Cross-base attribution: can a probe trained on two bases name the teacher of a third base?

On the answers that the H2 analysis used (as written and as rewritten into plain prose, cut to
one word budget per prompt), prompts are split in half. For each held-out base and each half, a
logistic regression on answer embeddings is trained on the other two bases' answers to the other
half and tested on the held-out base's answers to this half. Chance is 1/3. The settings are in
the locked `attribution_probe` config section.
"""
import argparse
import random
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from sameteacher.analysis.teacher_match import embed_cached
from sameteacher.core.config import add_common_args, load_yaml, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir
from sameteacher.core.text import clean, common_word_budget, first_words
from sameteacher.eval.stats import percentile_interval
from sameteacher.style_norm import main_retention as h2
from sameteacher.style_norm.pilot_retention import STAGES

SECTION = "attribution_probe"


def build_texts(run_dir, cfg) -> tuple[list[str], dict, list[str], list]:
    """The H2 texts: {(prompt, (stage, base, teacher, seed)): rows}, with one budget per prompt."""
    main_cfg, section = cfg["main_grid"], cfg[h2.SECTION]
    _, _, _, cells = h2.grid(main_cfg)
    sources = h2.select_sources(run_dir, main_cfg, section["samples_per_cell"])
    pairs, prompts = h2.matched_pairs(run_dir, cells, sources, section["min_pairs"])
    texts, index = [], {}
    for pid in prompts:
        groups = {(stage, *cell): [clean(pair[s]) for pair in pairs[cell][pid]] for s, stage in enumerate(STAGES) for cell in cells}
        budget = common_word_budget(list(groups.values()), section["min_words"])
        for key, values in groups.items():
            start = len(texts)
            texts.extend(first_words(t, budget) for t in values)
            index[(pid, key)] = list(range(start, len(texts)))
    return texts, index, prompts, cells


def halves(prompts, seed) -> list[list[str]]:
    order = sorted(prompts)
    random.Random(seed).shuffle(order)
    middle = len(order) // 2
    return [sorted(order[:middle]), sorted(order[middle:])]


def predictions(matrix, index, prompts, cells, stage, section) -> list[dict]:
    """One row per test answer: prompt, held-out base, true teacher and whether the probe was right."""
    bases = sorted({c[0] for c in cells})
    teachers = sorted({c[1] for c in cells})
    label = {t: i for i, t in enumerate(teachers)}
    split = halves(prompts, section["split_seed"])
    rows = []
    for held_out in bases:
        for test_half, train_half in ((split[0], split[1]), (split[1], split[0])):
            x_train, y_train = [], []
            for pid in train_half:
                for cell in cells:
                    if cell[0] != held_out:
                        ids = index[(pid, (stage, *cell))]
                        x_train.append(matrix[ids])
                        y_train += [label[cell[1]]] * len(ids)
            probe = LogisticRegression(C=section["C"], max_iter=section["max_iter"])
            probe.fit(np.vstack(x_train), np.array(y_train))
            for pid in test_half:
                for cell in cells:
                    if cell[0] == held_out:
                        guess = probe.predict(matrix[index[(pid, (stage, *cell))]])
                        rows += [{"prompt": pid, "base": held_out, "teacher": cell[1], "correct": bool(g == label[cell[1]])} for g in guess]
    return rows


def summarize(rows_by_stage: dict, prompts, section) -> dict:
    """Accuracy and recall with prompt-bootstrap intervals; one set of draws for every stage."""
    stages = list(rows_by_stage)
    groups = ["pooled"] + [f"base {b}" for b in sorted({r["base"] for r in rows_by_stage[stages[0]]})] + [f"teacher {t}" for t in sorted({r["teacher"] for r in rows_by_stage[stages[0]]})]

    # Per prompt: [correct, total] for each stage and group.
    counts = {pid: defaultdict(lambda: np.zeros(2)) for pid in prompts}
    for stage, rows in rows_by_stage.items():
        for r in rows:
            for group in ("pooled", f"base {r['base']}", f"teacher {r['teacher']}"):
                counts[r["prompt"]][(stage, group)] += (r["correct"], 1)

    def accuracy(sample) -> dict:
        totals = defaultdict(lambda: np.zeros(2))
        for pid in sample:
            for key, value in counts[pid].items():
                totals[key] += value
        out = {f"{stage} {group}": totals[(stage, group)][0] / totals[(stage, group)][1] for stage in stages for group in groups}
        for group in groups:
            out[f"difference {group}"] = out[f"{stages[1]} {group}"] - out[f"{stages[0]} {group}"]
        return out

    point = accuracy(prompts)
    rng = np.random.default_rng(section["bootstrap_seed"])
    draws = [accuracy([prompts[i] for i in rng.integers(0, len(prompts), len(prompts))]) for _ in range(section["bootstrap"])]
    result = {}
    for key, value in point.items():
        low, high = percentile_interval([d[key] for d in draws], section["ci_level"])
        result[key] = {"estimate": float(value), "ci_low": low, "ci_high": high}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h2.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    section = cfg[SECTION]
    texts, index, prompts, cells = build_texts(args.run_dir, cfg)
    matrix = embed_cached(args.run_dir, texts, index, section["embedder"], name="main_h2",
                          note="Rows are the main_style_norm texts (first usable answers and their rewrites), cut to one word budget per prompt. Keys: prompt|stage|base|teacher|seed.")
    rows = {stage: predictions(matrix, index, prompts, cells, stage, section) for stage in STAGES}
    result = {"prompts": len(prompts), "test_answers_per_stage": len(rows[STAGES[0]]), "chance": 1 / 3, "stages": list(STAGES)} | summarize(rows, prompts, section)
    path = results_dir(args.run_dir, MAIN, "attribution") / "probe.json"
    write_json(path, result)
    for key in [k for k in result if k.endswith((" pooled",)) or " base " in k]:
        v = result[key]
        print(f"{key:28s} {v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    print(path)


if __name__ == "__main__":
    main()
