import json
import subprocess

import pytest
import yaml
from conftest import fill_reviews

from gates_lib.config import read_jsonl


def prepare(ws) -> None:
    ws.run("t1.step1_make_prompt_splits", "--input-jsonl", str(ws.pool), "--exclude-jsonl", str(ws.exclude))
    ws.run("gates_lib.lock", "--section", "t1_gate1")
    for split in ("screen", "confirm"):
        ws.run("t1.step2_sample_teachers", "--split", split, "--max-usd", "5")
    ws.run("t1.step3_quality_check", "--split", "confirm")
    ws.run("t1.step4_measure_teacher_gap", "--split", "screen")
    ws.run("t1.step4_measure_teacher_gap", "--split", "confirm")
    ws.run("t1.step5_make_blind_review")


def decision(ws) -> dict:
    ws.run("t1.step6_gate1_decision")
    return json.loads((ws.run_dir / "t1" / "gate1" / "decision.json").read_text())


def human_policy(cross_rate: float):
    def policy(pair_type, rng):
        rate = {"cross": cross_rate, "within": 0.2, "style": 0.05, "idea": 0.95}[pair_type]
        return "D" if rng.random() < rate else "S"

    return policy


def test_planted_teacher_difference_passes(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    fill_reviews(ws.run_dir, human_policy(0.95))
    out = decision(ws)
    assert out["embedding"]["state"] == "GO", out
    assert out["decision"] == "GO", out


def test_secondary_report_matches_primary_and_splits_voice(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    ws.run("t1.step4b_secondary", "--split", "screen", "--voice-pair", "t_a", "t_b", "--max-usd", "5")
    gate1 = ws.run_dir / "t1" / "gate1"
    primary = json.loads((gate1 / "screen__tfidf.json").read_text())
    report = json.loads((gate1 / "screen__secondary__tfidf.json").read_text())
    assert report["robustness"]["all"]["estimate"] == pytest.approx(primary["calibrated_gap"]["estimate"])
    assert {"without t_a", "without_top_gaps", "pair t_a vs t_b"} <= set(report["robustness"])
    voice = report["voice_split"]
    # Mock rewrites keep every idea word, so voice must explain less than ideas do.
    assert voice["ideas in t_b voice / total"]["estimate"] > voice["voice t_a to t_b / total"]["estimate"]


def test_pair_ids_do_not_reveal_pair_type(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    key = json.loads((ws.run_dir / "t1" / "review" / "key_DO_NOT_OPEN.json").read_text())
    for pair_type in ("cross", "within", "style", "idea"):
        assert len({int(i) % 4 for i, v in key.items() if v["pair_type"] == pair_type}) > 1


def test_single_sheet_needs_llm_judge_flag(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    for extra in (2, 3):
        (ws.run_dir / "t1" / "review" / f"review_sheet_r{extra}.csv").unlink()
    # One sheet has too few control pairs for a random policy, so label by pair type.
    fill_reviews(ws.run_dir, lambda pair_type, rng: "D" if pair_type in ("cross", "idea") else "S", raters=1)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("t1.step6_gate1_decision")
    assert "At least 2" in caught.value.stderr
    ws.run("t1.step6_gate1_decision", "--llm-judge")
    out = json.loads((ws.run_dir / "t1" / "gate1" / "decision.json").read_text())
    assert out["human"]["kappa"] is None and out["review_mode"].startswith("llm_judge")
    assert out["decision"] == "GO", out


def test_identical_teachers_stop(workspace):
    ws = workspace(separation=0.0)
    prepare(ws)
    fill_reviews(ws.run_dir, human_policy(0.2))
    out = decision(ws)
    assert out["embedding"]["state"] == "STOP", out
    assert out["decision"] == "STOP", out


def test_failed_human_control_is_invalid_not_negative(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    fill_reviews(ws.run_dir, lambda pair_type, rng: "D" if rng.random() < 0.5 else "S")
    assert decision(ws)["decision"] == "INSTRUMENT_INVALID"


def test_style_dominated_metric_never_goes(workspace):
    ws = workspace(separation=0.0)
    models = yaml.safe_load(ws.models.read_text())
    for model in models["models"].values():
        model["mock"]["style_heavy"] = True
    ws.models.write_text(yaml.safe_dump(models))
    prepare(ws)
    fill_reviews(ws.run_dir, human_policy(0.95))
    assert decision(ws)["embedding"]["state"] != "GO"


def test_confirm_refuses_without_lock(workspace):
    ws = workspace(separation=1.0)
    ws.run("t1.step1_make_prompt_splits", "--input-jsonl", str(ws.pool), "--exclude-jsonl", str(ws.exclude))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("t1.step4_measure_teacher_gap", "--split", "confirm")
    assert "not locked" in caught.value.stderr


def test_changed_threshold_after_lock_is_refused(workspace):
    ws = workspace(separation=1.0)
    prepare(ws)
    fill_reviews(ws.run_dir, human_policy(0.95))
    ws.cfg["t1_gate1"]["go_low"] = 0.01
    ws.save_config()
    with pytest.raises(subprocess.CalledProcessError) as caught:
        ws.run("t1.step6_gate1_decision")
    assert "changed after locking" in caught.value.stderr


def test_splits_drop_flagged_excluded_and_near_duplicate_prompts(workspace):
    ws = workspace(separation=1.0)
    ws.cfg["splits"].update({"screen": 20, "confirm": 20, "pilot_eval": 19})
    ws.save_config()
    ws.run("t1.step1_make_prompt_splits", "--input-jsonl", str(ws.pool), "--exclude-jsonl", str(ws.exclude))
    ids = []
    for split in ("screen", "confirm", "pilot_eval"):
        ids += [r["prompt_id"] for r in read_jsonl(ws.run_dir / "t1" / "prompts" / f"{split}.jsonl")]
    assert len(ids) == len(set(ids)) == 59
    assert not {"flagged0", "flagged1", "conv000"} & set(ids)
    assert len({"neardup", "conv005"} & set(ids)) == 1


def test_sampling_resumes_without_repeating_work(workspace):
    ws = workspace(separation=1.0)
    ws.run("t1.step1_make_prompt_splits", "--input-jsonl", str(ws.pool), "--exclude-jsonl", str(ws.exclude))
    ws.run("t1.step2_sample_teachers", "--split", "screen", "--max-usd", "5")
    path = ws.run_dir / "t1" / "samples" / "screen.jsonl"
    before = path.read_text().count("\n")
    ws.run("t1.step2_sample_teachers", "--split", "screen", "--max-usd", "5")
    assert path.read_text().count("\n") == before


def test_budget_guard_stops_sampling(workspace):
    ws = workspace(separation=1.0)
    ws.run("t1.step1_make_prompt_splits", "--input-jsonl", str(ws.pool), "--exclude-jsonl", str(ws.exclude))
    models = yaml.safe_load(ws.models.read_text())
    for model in models["models"].values():
        model["input_per_m"] = 1_000_000.0
    ws.models.write_text(yaml.safe_dump(models))
    result = ws.run("t1.step2_sample_teachers", "--split", "screen", "--max-usd", "1")
    assert "budget stop" in result.stdout
