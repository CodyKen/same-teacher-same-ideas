import copy
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def run_module(module: str, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )


def make_pool(path: Path, exclude: Path, n: int = 60) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "conversation_id": f"conv{i:03d}",
                "messages": [{"role": "user", "content": f"Suggest a creative name for project number {i} with a theme"}],
                "lm_judge_annotation": {
                    "meaningful_information": {"classification": True, "reasoning": "x"},
                    "greeting_model_inquiry": {"classification": False, "reasoning": "x"},
                },
                "categories": [{"category": "Writing" if i % 2 else "Ideas"}],
            }
            f.write(json.dumps(row) + "\n")
        for index, (meaningful, greeting) in enumerate([(False, False), (True, True)]):
            row = {
                "conversation_id": f"flagged{index}",
                "messages": [{"role": "user", "content": f"Flagged prompt {index} that the classifier rejected for a reason"}],
                "lm_judge_annotation": {
                    "meaningful_information": {"classification": meaningful, "reasoning": "x"},
                    "greeting_model_inquiry": {"classification": greeting, "reasoning": "x"},
                },
                "categories": [{"category": "Ideas"}],
            }
            f.write(json.dumps(row) + "\n")
        row = {
            "conversation_id": "neardup",
            "messages": [{"role": "user", "content": "Suggest a creative name for project number 5 with a theme and a tagline too"}],
            "lm_judge_annotation": {},
            "categories": [{"category": "Ideas"}],
        }
        f.write(json.dumps(row) + "\n")
    with open(exclude, "w", encoding="utf-8") as f:
        f.write(json.dumps({"query": "Suggest a creative name for project number 0 with a theme"}) + "\n")


class Workspace:
    def __init__(self, root: Path, separation: float, base_cfg: dict):
        self.root = root
        self.run_dir = root / "runs"
        self.config = root / "experiment.yaml"
        self.models = root / "models.yaml"
        self.cfg = copy.deepcopy(base_cfg)
        self.cfg["splits"].update({"screen": 12, "confirm": 12, "pilot_eval": 12})
        self.cfg["sampling"].update({"samples_per_prompt": 16, "concurrency": 4})
        self.cfg["pilot_teachers"].update({"teachers": ["t_a", "t_b", "t_c"], "calibration_teacher": "t_a", "min_usable_per_group": 8})
        self.cfg["pilot_teacher_distance"].update({"primary_embedder": "tfidf", "robustness_embedders": [], "bootstrap": 300})
        self.cfg["pilot_teacher_distance"]["human"]["prompts"] = 12
        self.cfg["pilot_grid"].update({"teachers": ["t_a", "t_b"]})
        self.cfg["pilot_shared_teacher"].update({"embedder": "tfidf", "bootstrap": 300})
        self.cfg["pilot_style_norm"].update({"rewriter": "t_c", "embedder": "tfidf", "bootstrap": 300})
        self.cfg["main_grid"].update({"teachers": ["t_a", "t_b", "t_c"], "test_prompts": 10, "draw_retries": 8})
        self.save_config()
        models = {
            name: {"kind": "mock", "mock": {"separation": separation, "style": "plain", "family": name if separation else "shared"}}
            for name in ("t_a", "t_b", "t_c")
        }
        self.models.write_text(yaml.safe_dump({"models": models}))
        self.pool = root / "pool.jsonl"
        self.exclude = root / "exclude.jsonl"
        make_pool(self.pool, self.exclude)

    def save_config(self) -> None:
        self.config.write_text(yaml.safe_dump(self.cfg))

    def args(self) -> list[str]:
        return ["--config", str(self.config), "--models", str(self.models), "--run-dir", str(self.run_dir)]

    def run(self, module: str, *extra: str):
        return run_module(module, *self.args(), *extra)


@pytest.fixture
def base_cfg():
    return yaml.safe_load((REPO / "configs" / "experiment.yaml").read_text())


@pytest.fixture
def workspace(tmp_path, base_cfg):
    def build(separation: float) -> Workspace:
        return Workspace(tmp_path, separation, base_cfg)

    return build


def fill_reviews(run_dir: Path, policy, raters: int = 3, seed: int = 0) -> None:
    """Label each sheet using policy(pair_type, rng) -> S, D or U."""
    folder = run_dir / "human_eval" / "teacher_pairs"
    key = json.loads((folder / "key_DO_NOT_OPEN.json").read_text())
    for rater in range(1, raters + 1):
        path = folder / f"review_sheet_r{rater}.csv"
        rng = random.Random(seed + rater)
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["label"] = policy(key[row["pair_id"]]["pair_type"], rng)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
