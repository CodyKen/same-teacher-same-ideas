"""Where each kind of run output lives under runs/.

  prompts/                 evaluation splits (screen, confirm, pilot_eval, main_test)
  teachers/                teacher answers per split, and teacher voice rewrites
  train_data/              teacher answers used for fine-tuning, one file per teacher draw
  models/                  LoRA adapters and training logs (GPU machine only)
  models_meta/             copies of each student's train_meta.json
  students/<stage>/        student answers on the test prompts (stage: pilot or main)
  style_norm/              neutral-style rewrites of student answers
  human_eval/              blinded annotation sheets and their keys
  results/<stage>/<name>/  result summaries
  locks/                   frozen config sections
"""
from pathlib import Path

PILOT, MAIN = "pilot", "main"


def stage_of(split: str) -> str:
    return MAIN if split.startswith("main") else PILOT


def prompts_dir(run_dir) -> Path:
    return Path(run_dir) / "prompts"


def prompts_path(run_dir, split: str) -> Path:
    return prompts_dir(run_dir) / f"{split}.jsonl"


def teachers_dir(run_dir) -> Path:
    return Path(run_dir) / "teachers"


def teacher_samples_path(run_dir, split: str) -> Path:
    return teachers_dir(run_dir) / f"{split}.jsonl"


def train_data_dir(run_dir) -> Path:
    return Path(run_dir) / "train_data"


def train_file(run_dir, teacher: str, rep: int) -> Path:
    return train_data_dir(run_dir) / f"{teacher}_rep{rep}.jsonl"


def model_dir(run_dir, student: str) -> Path:
    return Path(run_dir) / "models" / student


def students_path(run_dir, stage: str, student: str) -> Path:
    return Path(run_dir) / "students" / stage / f"{student}.jsonl"


def style_norm_dir(run_dir) -> Path:
    return Path(run_dir) / "style_norm"


def human_eval_dir(run_dir, name: str) -> Path:
    return Path(run_dir) / "human_eval" / name


def results_dir(run_dir, stage: str, name: str) -> Path:
    return Path(run_dir) / "results" / stage / name
