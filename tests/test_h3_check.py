"""H3 hand check: items come from the training files, the key records each real item's
source-match outcome, decoys pair a source with another draw's rewrite, sheets are blind and share
the common items, and scoring counts kept and not added."""
import csv
import json

from test_h3_data import setup

from sameteacher.core.config import read_jsonl
from sameteacher.data import h3_data


def test_build_sheets_and_score(workspace):
    ws = workspace(separation=1.0)
    ws.cfg[h3_data.SECTION].update({"hand_check_per_condition": 4})
    ws.cfg["h3_rewrite_check"].update({"shared_per_condition": 1, "decoys": 4, "shared_decoys": 1})
    # t_b's answers to prompts 0-4 lose their content when rewritten, so those rewrites fail the source-match test.
    setup(ws, lambda teacher, p: "idea" if teacher == "t_b" and p < 5 else "concept")
    ws.run("sameteacher.core.lock", "--section", "h3_rewrite_check")
    ws.run("sameteacher.eval.h3_check", "build")

    folder = ws.run_dir / "human_eval" / "h3_rewrites"
    key = json.loads((folder / "key_DO_NOT_OPEN.json").read_text())
    assert len(key) == 4 * 4 + 4
    assert {v["condition_name"] for v in key.values()} == {"t_a in t_b style", "t_b in t_a style", "t_a in t_a style", "t_b in t_b style"}
    decoys = [v for v in key.values() if v["kind"] == "decoy"]
    assert all(v["rewrite_of"] != v["student"] for v in decoys)

    sheets = {}
    for a in (1, 2, 3):
        with open(folder / f"sheet_a{a}.csv", encoding="utf-8", newline="") as f:
            sheets[a] = list(csv.DictReader(f))
        assert list(sheets[a][0]) == ["item_id", "prompt", "original", "rewrite", "kept", "added", "notes"]
        page = (folder / f"sheet_a{a}.html").read_text()
        assert "H3 rewrite check, Annotator" in page and "own words and its own style" in page and "Plain" not in page
    shared = [i for i, v in key.items() if v["shared"]]
    assert len(shared) == 4 + 1 and all(i in {r["item_id"] for r in sheets[a]} for a in sheets for i in shared)
    assert sum(len(s) for s in sheets.values()) == len(key) + 2 * len(shared)

    # Every real item was trained on: its rewrite is a row of its student's training file.
    real = next(i for i, v in key.items() if v["kind"] == "real")
    row = next(r for r in sheets[1] + sheets[2] + sheets[3] if r["item_id"] == real)
    trained = [r["response"] for r in read_jsonl(h3_data.h3_dir(ws.run_dir) / f"{key[real]['student']}.jsonl")]
    assert row["rewrite"] in trained

    # The key's source-match outcome matches the rewrites that lost their content.
    prompts = {r["item_id"]: r["prompt"] for a in sheets for r in sheets[a]}
    for i, v in key.items():
        if v["kind"] == "real":
            lost = v["student"].split("__")[1].startswith("t_b_in") and int(prompts[i].split()[-1]) < 5
            assert v["source_match"] is (not lost)

    # Annotators pass real items and fail decoys.
    for a, rows in sheets.items():
        for r in rows:
            r["kept"], r["added"] = ("Y", "N") if key[r["item_id"]]["kind"] == "real" else ("N", "N")
        with open(folder / f"sheet_a{a}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    ws.run("sameteacher.eval.h3_check", "score")
    out = json.loads((ws.run_dir / "results" / "main" / "h3" / "rewrite_check.json").read_text())
    assert out["real"]["rate"] == 1.0 and out["decoy"]["rate"] == 0.0
    assert out["cross"]["n"] == 8 and out["t_a in t_b style"]["n"] == 4
    assert out["items_scored"] == len(key)
    assert out["real, source match passes"]["n"] + out["real, source match fails"]["n"] == 16
