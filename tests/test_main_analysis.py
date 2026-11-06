"""Main-grid H1 analysis: it reduces to the pilot 2x2, and it finds a planted teacher effect."""
import itertools
import json
import random

import numpy as np
import pytest

from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.analysis.pilot_shared_teacher import prompt_effects
from sameteacher.core.grid import main_cells
from sameteacher.core.paths import MAIN, students_path, teacher_samples_path
from sameteacher.eval.embed import embed_texts
from sameteacher.eval.embed import rows_dense
from sameteacher.eval.stats import bootstrap, bootstrap_means, energy_distance

TEACHERS = ["t_a", "t_b", "t_c"]


def random_dist(keys, seed=0):
    rng = np.random.default_rng(seed)
    table = {}
    for first, second in itertools.combinations(keys, 2):
        table[(first, second)] = table[(second, first)] = float(rng.uniform(0.1, 1.0))
    return lambda a, b: table[(a, b)]


def test_reduces_to_the_pilot_2x2():
    bases, teachers, seeds = ["X", "Y"], ["a", "b"], [0, 1]
    dist = random_dist(list(itertools.product(bases, teachers, seeds)))
    pilot = prompt_effects(dist, bases, teachers, seeds)
    ours = h1.quantities(h1.student_columns(dist, bases, teachers, seeds), bases, teachers)["pooled"]
    assert ours["S"] == pytest.approx(pilot["shared_effect"])
    assert ours["T"] == pytest.approx(pilot["transfer"])
    assert ours["N"] == pytest.approx(pilot["noise"])


def test_pooled_effect_is_the_difference_of_cross_base_means():
    bases, seeds = ["X", "Y", "Z"], [0, 1, 2]
    cells = list(itertools.product(bases, TEACHERS, seeds))
    dist = random_dist(cells, seed=3)
    cross = [(c, d) for c, d in itertools.combinations(cells, 2) if c[0] != d[0]]
    different = np.mean([dist(c, d) for c, d in cross if c[1] != d[1]])
    same = np.mean([dist(c, d) for c, d in cross if c[1] == d[1]])
    pooled = h1.quantities(h1.student_columns(dist, bases, TEACHERS, seeds), bases, TEACHERS)["pooled"]
    assert pooled["S"] == pytest.approx(different - same)


def test_shared_draws_match_the_single_statistic_bootstrap():
    rng = np.random.default_rng(1)
    columns = {"a": rng.normal(1, 1, 40), "b": rng.normal(2, 1, 40)}
    expected = bootstrap(columns, lambda m: m["a"] / m["b"], 300)
    point, draws = bootstrap_means(columns, 300)
    ratios = [d["a"] / d["b"] for d in draws]
    assert point["a"] / point["b"] == pytest.approx(expected[0])
    assert np.percentile(ratios, [2.5, 97.5]) == pytest.approx([expected[1], expected[2]], rel=1e-9)


@pytest.mark.parametrize("name", ["dense", "tfidf"])
def test_gram_table_matches_energy_distance(name):
    rng = np.random.default_rng(2)
    if name == "dense":
        matrix = rng.normal(size=(23, 8))
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    else:
        matrix = embed_texts([" ".join(f"w{rng.integers(30)}" for _ in range(12)) for _ in range(23)], "tfidf")
    index = {("p", "a"): list(range(0, 7)), ("p", "b"): list(range(7, 15)), ("p", "c"): list(range(15, 23))}
    table = h1.prompt_table(matrix, index, "p", ["a", "b", "c"])
    for first, second in itertools.combinations("abc", 2):
        expected = energy_distance(rows_dense(matrix, index[("p", first)]), rows_dense(matrix, index[("p", second)]))
        assert table[(first, second)] == pytest.approx(expected, abs=1e-9)


def config(n_boot=200):
    return {
        "main_grid": {
            "teachers": TEACHERS,
            "bases": {"X": "base-x", "Y": "base-y", "Z": "base-z"},
            "seeds": [0, 1, 2],
            "draws": {"X": [0, 1, 2], "Y": [3, 4, 5], "Z": [6, 7, 8]},
            "test_split": "main_test",
        },
        h1.SECTION: {
            "embedder": "tfidf",
            "bootstrap": n_boot,
            "bootstrap_seed": 0,
            "ci_level": 0.95,
            "pair_ci_level": 0.98333,
            "carryover_floor": 0.10,
            "min_usable_per_group": 15,
            "min_words": 30,
            "predicted_order": [["t_a", "t_b"], ["t_b", "t_c"], ["t_a", "t_c"]],
            "robustness_embedders": [],
            "leave_one_seed_out": True,
            "distinct_seed_transfer": True,
        },
    }


def write_grid(run_dir, teacher_weight: float, prompts: int = 12, answers: int = 20):
    """Answers mix words from the teacher's ideas for the prompt, the base's habits, and noise."""
    cfg = config()
    rng = random.Random(0)
    ideas = {(p, t): [f"idea{p}{t}{i}" for i in range(6)] for p in range(prompts) for t in TEACHERS}
    habits = {(p, b): [f"habit{p}{b}{i}" for i in range(6)] for p in range(prompts) for b in "XYZ"}
    filler = [f"word{i}" for i in range(400)]

    def answer(p, base, teacher):
        words = []
        for _ in range(40):
            u = rng.random()
            if u < teacher_weight:
                words.append(rng.choice(ideas[(p, teacher)]))
            elif u < teacher_weight + 0.3:
                words.append(rng.choice(habits[(p, base)]))
            else:
                words.append(rng.choice(filler))
        return " ".join(words)

    for cell in main_cells(cfg["main_grid"]):
        path = students_path(run_dir, MAIN, cell["student"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for p, i in itertools.product(range(prompts), range(answers)):
                f.write(json.dumps({"student": cell["student"], "prompt_id": f"p{p}", "sample_idx": i, "text": answer(p, cell["base"], cell["teacher"])}) + "\n")
    path = teacher_samples_path(run_dir, "main_test")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t, p, i in itertools.product(TEACHERS, range(prompts), range(answers)):
            text = " ".join(rng.choice(ideas[(p, t)]) if rng.random() < 0.6 else rng.choice(filler) for _ in range(40))
            f.write(json.dumps({"teacher": t, "variant": "baseline", "prompt_id": f"p{p}", "sample_idx": i, "text": text, "finish_reason": "stop"}) + "\n")
    return cfg


def run(run_dir, cfg):
    texts, index, prompts, _ = h1.build_texts(run_dir, cfg)
    return h1.analyze_embedding(embed_texts(texts, "tfidf"), index, prompts, cfg, primary=True)


def test_finds_a_planted_teacher_effect(tmp_path):
    cfg = write_grid(tmp_path, teacher_weight=0.4)
    result = run(tmp_path, cfg)
    assert result["prompts_used"] == 12
    assert result["state"] == "SUPPORTED"
    assert result["pooled"]["S"]["ci_low"] > 0
    assert result["closer_to_own_teacher"]["pooled"]["ci_low"] > 0
    assert result["seed_pair_blocks"]["positive"] == result["seed_pair_blocks"]["total"] == 81
    assert set(result["leave_one_seed_out"]) == {"without seed 0", "without seed 1", "without seed 2"}
    assert len(result["per_prompt"]) == 12


def test_no_teacher_effect_is_not_reported_as_one(tmp_path):
    cfg = write_grid(tmp_path, teacher_weight=0.0)
    result = run(tmp_path, cfg)
    assert result["state"] in {"NO_TRANSFER", "NOT_SUPPORTED"}
