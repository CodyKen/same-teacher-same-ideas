import numpy as np

from gates_lib.stats import GO, INCONCLUSIVE, STOP, bootstrap, cohen_kappa, energy_distance, three_way
from gates_lib.text import is_answer, split_reasoning


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_energy_distance_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    values = [energy_distance(unit(rng.normal(size=(40, 16))), unit(rng.normal(size=(40, 16)))) for _ in range(50)]
    assert abs(np.mean(values)) < 0.01


def test_energy_distance_grows_with_separation():
    rng = np.random.default_rng(1)
    centre = np.zeros(16)
    centre[0] = 1
    x = unit(rng.normal(size=(40, 16)) * 0.1 + centre)
    near = unit(rng.normal(size=(40, 16)) * 0.1 + centre + 0.1 * np.eye(16)[1])
    far = unit(rng.normal(size=(40, 16)) * 0.1 - centre)
    assert energy_distance(x, far) > energy_distance(x, near) > 0


def test_three_way_covers_all_outcomes():
    assert three_way(0.3, 0.6, 0.25, 0.1) == GO
    assert three_way(0.0, 0.08, 0.25, 0.1) == STOP
    assert three_way(0.05, 0.5, 0.25, 0.1) == INCONCLUSIVE


def test_wide_uncertain_interval_stays_inconclusive():
    rng = np.random.default_rng(2)
    columns = {"x": rng.normal(0.15, 1.0, size=8)}
    estimate, low, high = bootstrap(columns, lambda m: m["x"], 500)
    assert low < estimate < high
    assert three_way(low, high, 0.25, 0.1) == INCONCLUSIVE


def test_cohen_kappa_bounds():
    assert cohen_kappa(["S", "D", "S"], ["S", "D", "S"]) == 1.0
    assert abs(cohen_kappa(["S", "S", "D", "D"], ["S", "D", "S", "D"])) < 1e-9


def test_split_reasoning_keeps_only_the_answer():
    assert split_reasoning("The user wants a joke.</think>\n\nKnock knock.") == ("The user wants a joke.", "Knock knock.")
    assert split_reasoning("Knock knock.") == ("", "Knock knock.")


def test_is_answer_rejects_unfinished_reasoning_only():
    assert not is_answer("The user wants a satirical take. I should keep it under")
    assert not is_answer("The user's query is about pins.")
    assert not is_answer("<|channel|>analysis<|message|>Need to answer directly")
    assert is_answer("The user interface should stay simple and clear.")
    assert is_answer("Let me tell you about the sea.")
