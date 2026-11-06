"""Rewrite hand check: sheets are blind and balanced, and scoring separates real items from decoys."""
import csv
import json
import re

from sameteacher.eval import rewrite_check as rc
from tests.test_main_retention import prepare, write_students


def build(ws):
    prepare(ws)
    ws.cfg[rc.SECTION].update({"per_cell": 4, "shared_per_cell": 1, "decoys": 6, "shared_decoys": 2})
    ws.save_config()
    write_students(ws, lambda base, teacher, prompt, rng: f"{teacher}{rng.randrange(3)}")
    for name in ("main_grid", "main_style_norm", rc.SECTION):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.style_norm.main_retention", "--max-usd", "5")
    ws.run("sameteacher.eval.rewrite_check", "build")
    folder = ws.run_dir / "human_eval" / "rewrite_fidelity"
    return folder, json.loads((folder / "key_DO_NOT_OPEN.json").read_text())


def test_sheets_are_blind_and_shared_items_go_to_everyone(workspace):
    ws = workspace(separation=1.0)
    folder, key = build(ws)
    assert len(key) == 9 * 4 + 6
    sheets = {}
    for a in (1, 2, 3):
        text = (folder / f"sheet_a{a}.csv").read_text()
        assert "t_a" not in text and "__" not in text
        with open(folder / f"sheet_a{a}.csv", newline="") as f:
            sheets[a] = {row["item_id"] for row in csv.DictReader(f)}
    shared = {i for i, k in key.items() if k["shared"]}
    assert len(shared) == 9 + 2
    assert all(shared <= ids for ids in sheets.values())
    assert set.union(*sheets.values()) == set(key)
    assert sorted(len(ids - shared) for ids in sheets.values()) == [10, 10, 11]

    # One HTML page per sheet, with the same items in the same order and no key fields.
    assert not (folder / "RUBRIC.md").exists()
    for a in (1, 2, 3):
        page = (folder / f"sheet_a{a}.html").read_text()
        blob = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S).group(1)
        data = json.loads(blob)
        with open(folder / f"sheet_a{a}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [i["item_id"] for i in data["items"]] == [r["item_id"] for r in rows]
        assert data["items"][0]["original"] == rows[0]["original"]
        assert "<" not in blob and "decoy" not in page and "t_a" not in page
        assert "Kept" in page and "Export CSV" in page


def test_scoring_separates_decoys(workspace):
    ws = workspace(separation=1.0)
    folder, key = build(ws)
    for a in (1, 2, 3):
        path = folder / f"sheet_a{a}.csv"
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            real = key[row["item_id"]]["kind"] == "real"
            row.update({"kept": "Y" if real else "N", "added": "N" if real else "Y", "plain": "y"})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    ws.run("sameteacher.eval.rewrite_check", "score")
    out = json.loads((ws.run_dir / "results" / "main" / "style_norm" / "rewrite_check.json").read_text())
    assert out["real"]["pass"]["rate"] == 1.0 and out["real"]["pass"]["n"] == 36
    assert out["decoy"]["pass"]["rate"] == 0.0
    assert out["kappa_shared"]["kept"] == 1.0
    assert set(k for k in out if k.startswith("teacher ")) == {"teacher t_a", "teacher t_b", "teacher t_c"}
