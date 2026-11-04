"""Distribution distance, prompt-level bootstrap, and three-way decisions."""
from collections.abc import Callable

import numpy as np

GO, STOP, INCONCLUSIVE = "GO", "STOP", "INCONCLUSIVE"


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(2.0 - 2.0 * (a @ b.T), 0.0, None))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased energy distance between two sets of unit vectors.

    About 0 when both come from one distribution, and grows with the size of the shift.
    Sampling noise can make it slightly negative, so it is not clipped. It is deliberately
    not divided by the spread: answers on one prompt are often tightly clustered, and the
    normalised coefficient then saturates at 1 for any small consistent change in style.
    """
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError("energy_distance needs at least two points per set")
    cross = float(_pairwise(x, y).mean())
    within_x = float(_pairwise(x, x).sum()) / (n * (n - 1))
    within_y = float(_pairwise(y, y).sum()) / (m * (m - 1))
    return 2 * cross - within_x - within_y


def bootstrap(columns: dict[str, np.ndarray], stat: Callable[[dict], float], n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% interval, resampling prompts jointly across columns."""
    columns = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
    size = len(next(iter(columns.values())))
    rng = np.random.default_rng(seed)
    estimate = stat({k: v.mean() for k, v in columns.items()})
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, size, size)
        draws[i] = stat({k: v[idx].mean() for k, v in columns.items()})
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(estimate), float(low), float(high)


def three_way(low: float, high: float, go_low: float, stop_high: float) -> str:
    """GO needs the whole interval above go_low; STOP needs it below stop_high."""
    if low >= go_low:
        return GO
    if high <= stop_high:
        return STOP
    return INCONCLUSIVE


def cohen_kappa(a: list, b: list) -> float:
    if not a:
        return float("nan")
    labels = sorted(set(a) | set(b))
    a_arr, b_arr = np.array(a), np.array(b)
    observed = float((a_arr == b_arr).mean())
    expected = sum(float((a_arr == c).mean()) * float((b_arr == c).mean()) for c in labels)
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
