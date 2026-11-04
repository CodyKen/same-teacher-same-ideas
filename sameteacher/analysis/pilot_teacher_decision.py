"""Combine embedding and human evidence into the pilot decision on teacher distance.

GO needs both sources to clear their bar. Any clear STOP stops it. If a control fails,
the verdict is INSTRUMENT_INVALID: the measure is broken, not the teachers. Anything
else is INCONCLUSIVE, which means collect more, not treat "not significant" as "no gap".
"""
import argparse
import csv
import itertools
from collections import defaultdict
from statistics import mean

import numpy as np

from sameteacher.core.config import add_common_args, load_yaml, read_json, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.eval.stats import GO, INCONCLUSIVE, STOP, bootstrap, cohen_kappa, three_way

from sameteacher.core.paths import PILOT, results_dir
from sameteacher.analysis.teacher_distance import result_path
from sameteacher.eval.blind_review import review_dir


def read_reviews(run_dir, min_sheets: int = 2) -> dict[str, dict[str, str]]:
    reviews = {}
    for path in sorted(review_dir(run_dir).glob("review_sheet_r*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        labels = {r["pair_id"]: r["label"].strip().upper() for r in rows}
        bad = [k for k, v in labels.items() if v not in {"S", "D", "U"}]
        if bad:
            raise SystemExit(f"{path.name}: {len(bad)} rows have no valid label (S, D or U).")
        reviews[path.stem] = labels
    if len(reviews) < min_sheets:
        raise SystemExit(f"At least {min_sheets} completed review sheets are needed.")
    return reviews


def score_reviews(reviews: dict, key: dict, human: dict, n_boot: int) -> dict:
    by_prompt = defaultdict(lambda: defaultdict(list))
    for labels in reviews.values():
        for pair_id, label in labels.items():
            if label != "U":
                info = key[pair_id]
                by_prompt[info["prompt_id"]][info["pair_type"]].append(label == "D")

    complete = [p for p, t in by_prompt.items() if t["cross"] and t["within"]]
    if not complete:
        raise SystemExit("Reviews have no usable cross and within pairs on the same prompt.")
    columns = {
        "cross": np.array([mean(by_prompt[p]["cross"]) for p in complete], dtype=float),
        "within": np.array([mean(by_prompt[p]["within"]) for p in complete], dtype=float),
    }
    gap, low, high = bootstrap(columns, lambda m: m["cross"] - m["within"], n_boot)

    def rate(pair_type):
        values = [v for t in by_prompt.values() for v in t[pair_type]]
        return float(mean(values)) if values else float("nan")

    kappas = []
    for first, second in itertools.combinations(reviews.values(), 2):
        shared = [k for k in first if first[k] != "U" and second[k] != "U"]
        kappas.append(cohen_kappa([first[k] for k in shared], [second[k] for k in shared]))
    # A single LLM-labelled sheet has no second rater, so only the controls check it.
    kappa = float(np.mean(kappas)) if kappas else None

    rates = {t: rate(t) for t in ("cross", "within", "style", "idea")}
    valid = (
        (kappa is None or kappa >= human["min_kappa"])
        and rates["idea"] >= human["idea_control_min_different"]
        and rates["style"] <= human["style_control_max_different"]
    )
    state = three_way(low, high, human["go_gap_low"], human["stop_gap_high"]) if valid else "INSTRUMENT_INVALID"
    return {"different_rate": rates, "gap": gap, "ci_low": low, "ci_high": high, "kappa": kappa, "state": state, "prompts": len(complete)}


def combine(embedding_state: str, human_state: str, robustness_states: list[str]) -> str:
    states = {embedding_state, human_state}
    if "INSTRUMENT_INVALID" in states:
        return "INSTRUMENT_INVALID"
    if STOP in states:
        return STOP
    if states == {GO}:
        return INCONCLUSIVE if STOP in robustness_states else GO
    return INCONCLUSIVE


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Accept exactly one sheet labelled by an LLM instead of human reviewers. Kappa is not computed; recorded as a deviation.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "pilot_teacher_distance")
    reviews = read_reviews(args.run_dir, 1 if args.llm_judge else 2)
    if args.llm_judge and len(reviews) != 1:
        raise SystemExit("--llm-judge expects exactly one labelled sheet.")
    thresholds = cfg["pilot_teacher_distance"]

    primary = read_json(result_path(args.run_dir, "confirm", thresholds["primary_embedder"]))
    robustness = {}
    for name in thresholds["robustness_embedders"]:
        path = result_path(args.run_dir, "confirm", name)
        if path.exists():
            robustness[name] = read_json(path)["state"]

    human = score_reviews(reviews, read_json(review_dir(args.run_dir) / "key_DO_NOT_OPEN.json"), thresholds["human"], thresholds["bootstrap"])
    decision = combine(primary["state"], human["state"], list(robustness.values()))
    out = {
        "decision": decision,
        "review_mode": "llm_judge: one sheet labelled by an LLM, no inter-rater kappa" if args.llm_judge else "human",
        "embedding": {k: primary[k] for k in ("state", "calibrated_gap", "idea_minus_style", "mean_energy", "prompts_used")},
        "attribution": {"accuracy": primary["teacher_attribution_accuracy"], "chance": primary["attribution_chance"]},
        "robustness_states": robustness,
        "human": human,
    }
    write_json(results_dir(args.run_dir, PILOT, "teacher_distance") / "decision.json", out)
    print(f"pilot teacher-distance decision: {decision}")
    print("embedding", out["embedding"]["state"], out["embedding"]["calibrated_gap"])
    print("human", human)
    print("robustness", robustness)
    if decision == INCONCLUSIVE:
        print("Inconclusive means more data, not a negative finding. Add samples or prompts once, then decide.")


if __name__ == "__main__":
    main()
