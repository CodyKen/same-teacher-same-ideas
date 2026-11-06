"""Human evaluation of H1: pairs are cross-base, balanced over cells and kinds, blind, on distinct
H1 prompts, and identical across sheets in different orders; scoring recovers planted labels."""
import csv
import json

from test_main_analysis import write_grid

from sameteacher.core.config import write_jsonl
from sameteacher.eval import human_eval as he


def setup(tmp_path, per_kind=9, prompts=20):
    cfg = write_grid(tmp_path, teacher_weight=0.4, prompts=prompts)
    cfg[he.SECTION] = {"pairs_per_kind": per_kind, "annotators": 3, "seed": 7, "embedder": "tfidf", "bootstrap": 300, "bootstrap_seed": 0, "ci_level": 0.95}
    write_jsonl(tmp_path / "prompts" / "main_test.jsonl", [{"prompt_id": f"p{p}", "prompt": f"request number {p}"} for p in range(prompts)])
    return cfg


def sheets(folder):
    out = {}
    for a in (1, 2, 3):
        with open(folder / f"sheet_a{a}.csv", encoding="utf-8", newline="") as f:
            out[a] = list(csv.DictReader(f))
    return out


def test_pairs_are_balanced_blind_and_shared_by_every_sheet(tmp_path):
    cfg = setup(tmp_path)
    summary = he.build(tmp_path, cfg)
    assert summary["kinds"] == {"same": 9, "different": 9}
    folder = he.check_dir(tmp_path)
    key = json.loads((folder / "key_DO_NOT_OPEN.json").read_text())
    assert len({v["prompt_id"] for v in key.values()}) == 18
    for v in key.values():
        assert v["a"]["base"] != v["b"]["base"]
        assert (v["a"]["teacher"] == v["b"]["teacher"]) == (v["kind"] == "same")
    # Nine same pairs cover the nine base pair x teacher cells once each, and likewise for different pairs.
    same = {(frozenset((v["a"]["base"], v["b"]["base"])), v["a"]["teacher"]) for v in key.values() if v["kind"] == "same"}
    different = {(frozenset((v["a"]["base"], v["b"]["base"])), frozenset((v["a"]["teacher"], v["b"]["teacher"]))) for v in key.values() if v["kind"] == "different"}
    assert len(same) == 9 and len(different) == 9

    by_sheet = sheets(folder)
    assert list(by_sheet[1][0]) == ["item_id", "prompt", "answer_a", "answer_b", "same_idea", "notes"]
    assert all({r["item_id"] for r in rows} == set(key) for rows in by_sheet.values())
    assert [r["item_id"] for r in by_sheet[1]] != [r["item_id"] for r in by_sheet[2]]
    # The shown answer is the keyed student's answer to that prompt.
    row = by_sheet[1][0]
    side = key[row["item_id"]]["a"]
    student = f"{side['base']}__{side['teacher']}__s{side['seed']}"
    rows = [json.loads(line) for line in (tmp_path / "students" / "main" / f"{student}.jsonl").read_text().splitlines()]
    assert next(r["text"] for r in rows if r["prompt_id"] == key[row["item_id"]]["prompt_id"] and r["sample_idx"] == side["sample_idx"]) .strip() == row["answer_a"]

    page = (folder / "sheet_a2.html").read_text()
    assert "Answer Pairs" in page and "Answer A" in page and '"store": "main-pairs"' in page
    payload = json.loads(page.split('id="data">')[1].split("</script>")[0])
    assert all(set(item) == {"item_id", "prompt", "answer_a", "answer_b"} for item in payload["items"])


def test_scoring_recovers_planted_labels(tmp_path):
    cfg = setup(tmp_path)
    he.build(tmp_path, cfg)
    folder = he.check_dir(tmp_path)
    key = json.loads((folder / "key_DO_NOT_OPEN.json").read_text())
    for a, rows in sheets(folder).items():
        for r in rows:
            r["same_idea"] = "Y" if key[r["item_id"]]["kind"] == "same" else "N"
        # Annotator 3 disagrees on one pair; the majority keeps the planted label.
        if a == 3:
            rows[0]["same_idea"] = "N" if rows[0]["same_idea"] == "Y" else "Y"
        with open(folder / f"sheet_a{a}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    out = he.score(tmp_path, cfg)
    assert out["same"]["rate"] == 1.0 and out["different"]["rate"] == 0.0
    assert out["same minus different"]["ci_low"] == 1.0
    assert 0.8 < out["kappa"] < 1.0
    assert out["pairs_scored"] == 18
    assert out["embedding check"]["mean cosine, labeled Y"] > out["embedding check"]["mean cosine, labeled N"]
    assert {"same base X-Y", "different teachers t_a-t_c", "same teacher t_b", "annotator 3 same"} <= set(out)
