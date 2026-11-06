"""Teacher-match diagnostic: the distances agree with H1, the direction distance separates
location from spread, and the cache and training-answer spread work."""
import itertools
import json

import numpy as np
import pytest

from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.analysis import teacher_match as tm
from sameteacher.core.paths import train_file
from tests.test_main_analysis import TEACHERS, write_grid


def section(cfg):
    cfg[tm.SECTION] = {"embedder": "tfidf", "bootstrap": 200, "bootstrap_seed": 0, "ci_level": 0.95, "train_prompts": 20, "train_seed": 0, "min_words": 5}
    return cfg


def test_energy_matches_the_h1_table(tmp_path):
    cfg = section(write_grid(tmp_path, teacher_weight=0.4, prompts=4))
    texts, index, prompts, _ = h1.build_texts(tmp_path, cfg)
    matrix = tm.embed_cached(tmp_path, texts, index, "tfidf")
    students = [("X", t, 0) for t in TEACHERS]
    teachers = [h1.teacher_key(t) for t in TEACHERS]
    stats = tm.prompt_stats(matrix, index, prompts[0], students, teachers)
    table = h1.prompt_table(matrix, index, prompts[0], students + teachers)
    for s, t in itertools.product(students, teachers):
        assert stats[("E", s, t)] == pytest.approx(table[(s, t)], abs=1e-5)  # float32 Gram in H1, float64 here
    # The cache returns the same matrix and records every group.
    again = tm.embed_cached(tmp_path, texts, index, "tfidf")
    assert np.array_equal(matrix, again)
    meta = json.loads(tm.cache_paths(tmp_path, "tfidf")[1].read_text())
    assert len(meta["index"]) == len(index)


def test_direction_separates_location_from_spread():
    """Students centered on teacher a but more varied than a's answers look closer to teacher b
    by energy and centroid distance; the direction distance still points to a."""
    rng = np.random.default_rng(0)
    d = 16
    mu_a, mu_b = np.zeros(d), np.zeros(d)
    mu_a[0], mu_b[0], mu_b[1] = 1.0, 0.97, 0.243

    def cloud(mu, sd):
        x = mu + sd * rng.normal(size=(30, d))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    groups = {("S", "a"): cloud(mu_a, 0.25), ("teacher", "a"): cloud(mu_a, 0.03), ("teacher", "b"): cloud(mu_b, 0.25)}
    rows, index = [], {}
    for k, v in groups.items():
        index[("p", k)] = list(range(len(rows), len(rows) + len(v)))
        rows += list(v)
    st = tm.prompt_stats(np.array(rows), index, "p", [("S", "a")], [("teacher", "a"), ("teacher", "b")])
    student, a, b = ("S", "a"), ("teacher", "a"), ("teacher", "b")
    assert st[("E", student, a)] > st[("E", student, b)]
    assert st[("M", student, a)] > st[("M", student, b)]
    assert st[("D", student, a)] < st[("D", student, b)]
    assert st[("W", a)] < st[("W", student)]


def test_planted_effect_and_training_spread(tmp_path):
    cfg = section(write_grid(tmp_path, teacher_weight=0.4, prompts=6))
    result = tm.analyze(tmp_path, cfg)
    for t in TEACHERS:
        assert result["energy own"][t]["estimate"] > 0
        assert result["direction own"][t]["estimate"] > 0
    assert set(result["energy X"]) == set(TEACHERS)

    # t_a gives the same training answer in every draw, the others vary.
    reps = tm.train_reps(cfg["main_grid"])
    for t, rep in itertools.product(TEACHERS, reps):
        path = train_file(tmp_path, t, rep)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for p in range(25):
                words = f"fixed answer {p} with steady words" if t == "t_a" else " ".join(f"w{(p * 7 + rep * 13 + k * 5) % 97}" for k in range(8))
                f.write(json.dumps({"prompt": f"dolly prompt {p}", "response": words}) + "\n")
    spread = tm.train_spread(tmp_path, cfg)
    assert spread["prompts"] == 20
    assert spread["spread"]["t_a"]["estimate"] == pytest.approx(0.0, abs=1e-6)
    assert spread["spread"]["t_b"]["estimate"] > 0.5
    style = tm.style_table(tmp_path, cfg["main_grid"])
    assert style["t_a"]["training answers"]["n"] == 25 * len(reps)
