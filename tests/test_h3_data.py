"""H3 data: rewrites that keep their source's content pass the source-match test and rewrites that
lose it fail; the test is reported, and students train on every prompt whose rewrites are complete
answers in all 16 files. The mock rewriter keeps only words that start with "concept"."""
import json

from sameteacher.core.config import read_jsonl, write_jsonl
from sameteacher.data import h3_data as h3


def setup(ws, word_of):
    ws.cfg["main_grid"].update({"teachers": ["t_a", "t_b", "t_c"]})
    ws.cfg[h3.SECTION].update({"teachers": ["t_a", "t_b"], "embedder": "tfidf"})
    ws.save_config()
    for teacher in ("t_a", "t_b", "t_c"):
        for rep in range(9):
            rows = []
            for p in range(10):
                words = " ".join(f"{word_of(teacher, p)}{p}x{teacher.replace('_', '')}{rep}w{k}" for k in range(12))
                rows.append({"prompt": f"prompt number {p}", "response": f"shared opening words {words}"})
            write_jsonl(ws.run_dir / "train_data" / f"{teacher}_rep{rep}.jsonl", rows)
    for name in ("main_grid", h3.SECTION, h3.TRAINING_SECTION):
        ws.run("sameteacher.core.lock", "--section", name)
    ws.run("sameteacher.data.h3_data", "rewrite", "--max-usd", "5")
    ws.run("sameteacher.data.h3_data", "filter")
    ws.run("sameteacher.data.h3_data", "build")
    return json.loads((ws.run_dir / "results" / "main" / "h3" / "source_match.json").read_text())


def training(ws):
    return json.loads((ws.run_dir / "results" / "main" / "h3" / "training_data.json").read_text())


def test_faithful_rewrites_pass_and_build_sixteen_students(workspace):
    ws = workspace(separation=1.0)
    summary = setup(ws, lambda teacher, p: "concept")
    assert len(read_jsonl(h3.rewrites_path(ws.run_dir))) == 16 * 10
    assert all(rate == 1.0 for rate in summary["pass_rate"].values()), summary["pass_rate"]
    assert set(summary["pass_rate"]) == {"t_a in t_b style", "t_b in t_a style", "t_a in t_a style", "t_b in t_b style"}
    assert summary["prompts_passing_in_all_16"] == 10 and training(ws)["prompts_kept"] == 10
    manifest = json.loads(h3.manifest_path(ws.run_dir).read_text())
    runs = {r["student"]: r for r in manifest["runs"]}
    assert len(runs) == 16
    cross = runs["Y__t_a_in_t_b__s1"]
    assert cross["condition"] == "cross" and cross["source"] == "t_a_rep3.jsonl" and cross["examples"] == 10
    assert runs["X__t_b_in_t_b__s0"]["condition"] == "self"
    rows = read_jsonl(h3.h3_dir(ws.run_dir) / cross["train_file"])
    assert all("ta3" in r["response"] for r in rows)


def test_rewrites_that_lose_content_fail_the_test_and_are_still_trained_on(workspace):
    ws = workspace(separation=1.0)
    # t_b's answers to prompts 0-4 have no "concept" words, so every rewrite of them is empty of content.
    summary = setup(ws, lambda teacher, p: "idea" if teacher == "t_b" and p < 5 else "concept")
    assert summary["pass_rate"]["t_a in t_b style"] == 1.0
    assert summary["pass_rate"]["t_b in t_a style"] == 0.5
    assert summary["prompts_passing_in_all_16"] == 5
    assert training(ws)["prompts_kept"] == 10


def test_a_cut_or_empty_rewrite_drops_its_prompt_from_every_file(workspace):
    ws = workspace(separation=1.0)
    setup(ws, lambda teacher, p: "concept")
    rows = read_jsonl(h3.rewrites_path(ws.run_dir))
    cut = next(r for r in rows if r["file"].startswith("t_a|") and r["prompt"] == "prompt number 2")
    empty = next(r for r in rows if r["file"].startswith("t_b|") and r["prompt"] == "prompt number 7")
    cut["finish_reason"], empty["text"] = "length", "   "
    write_jsonl(h3.rewrites_path(ws.run_dir), rows)
    ws.run("sameteacher.data.h3_data", "build")
    out = training(ws)
    assert out["prompts_kept"] == 8
    assert sum(f["cut_at_max_tokens"] for f in out["per_file"].values()) == 1
    manifest = json.loads(h3.manifest_path(ws.run_dir).read_text())
    assert manifest["training_section"] == h3.TRAINING_SECTION
    for run in manifest["runs"]:
        prompts = {r["prompt"] for r in read_jsonl(h3.h3_dir(ws.run_dir) / run["train_file"])}
        assert run["examples"] == 8 and not prompts & {"prompt number 2", "prompt number 7"}


def test_diagnostic_reproduces_the_locked_rule_and_names_near_duplicate_rivals(workspace):
    ws = workspace(separation=1.0)
    summary = setup(ws, lambda teacher, p: "idea" if teacher == "t_b" and p < 5 else "concept")
    ws.run("sameteacher.data.h3_data", "diagnose")
    out = json.loads((ws.run_dir / "results" / "main" / "h3" / "source_match_diagnostic.json").read_text())
    locked = out["rules"]["locked: every other answer of both teachers"]
    assert locked["pass_rate"] == summary["pass_rate"] and locked["prompts_kept_in_all_16"] == summary["prompts_passing_in_all_16"]
    narrow = out["rules"]["other teacher's answers only"]
    assert all(narrow["pass_rate"][c] >= locked["pass_rate"][c] for c in locked["pass_rate"])
    assert out["usable rewrites"] == 16 * 10
