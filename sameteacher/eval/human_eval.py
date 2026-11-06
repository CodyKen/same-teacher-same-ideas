"""Human evaluation of H1: do cross-base answer pairs from students of one teacher share their
main idea more often than pairs from students of two teachers?

`build` draws the pairs from the main-grid students and writes one sheet per annotator (every
annotator gets every pair, in an order of their own) plus a key in runs/human_eval/main_pairs/.
Each sheet is a CSV and an HTML page with the instructions; the page exports the CSV. `html`
rewrites the pages from the CSV sheets. `score` reads the filled sheets. The design is in the
locked `main_human_eval` config section.

Annotators answer one question with Y or N:
  same_idea  the two answers give the same main idea
"""
import argparse
import csv
import itertools
import random

import numpy as np

from sameteacher.analysis import main_shared_teacher as h1
from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.grid import main_cells
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, human_eval_dir, prompts_path, results_dir, students_path
from sameteacher.core.text import clean, is_answer
from sameteacher.eval.embed import embed_texts, rows_dense
from sameteacher.eval.rewrite_check import fleiss_kappa, majority, wilson
from sameteacher.eval.rewrite_check_page import render
from sameteacher.eval.stats import percentile_interval

SECTION = "main_human_eval"
LABELS = ("same_idea",)
LABEL_QUESTIONS = [("same_idea", "Same idea", "Do the two answers give the same main idea?")]
BOXES = [("answer_a", "Answer A"), ("answer_b", "Answer B")]
STORE = "main-pairs"
TITLE = "Answer Pairs"
RUBRIC_HTML = """
<p>Each item shows a request and two answers to it, written by two different models. Decide
whether the two answers give the same <b>main idea</b>: what each answer chiefly says or does
with the request. That can be its central claim or recommendation, its reading of the request,
or the example, story or approach it builds on. Judge only the main idea. Wording, length, tone,
formatting and the quality of the writing do not count, and neither do small details that one
answer adds or leaves out. Use the notes box for anything unclear.</p>
<dl>
  <dt>Same idea</dt>
  <dd><b>Y</b> if a reader would say that both answers make the same main point or take the same
  approach, even in different words. <b>N</b> if their main points or approaches differ, or if
  the main point of one answer is missing from the other. When an answer makes several points,
  compare the ones it leads with or spends the most words on.</dd>
</dl>
<p>Work alone, and do not discuss pairs until every sheet is in. Do not open
<code>key_DO_NOT_OPEN.json</code>.</p>
<p>Labels are saved in this browser as you go. When you finish, click <b>Export CSV</b> and put
the downloaded file in <code>runs/human_eval/main_pairs/</code>, replacing the blank sheet of the
same name. Export also works as a backup at any time; <b>Import CSV</b> loads it back.</p>
<p>The repository is public and every sheet has the same pairs, so keep your filled sheet to
yourself until all three sheets are done. Then commit it, or send it to the person who runs the
scoring.</p>
"""


def check_dir(run_dir):
    return human_eval_dir(run_dir, "main_pairs")


def usable_answers(run_dir, main_cfg) -> dict:
    """{(base, teacher, seed): {prompt_id: [(sample_idx, text)]}} for every usable answer, as written."""
    out = {}
    for cell in main_cells(main_cfg):
        by_prompt = {}
        for row in sorted(read_jsonl(students_path(run_dir, MAIN, cell["student"])), key=lambda r: (r["prompt_id"], r["sample_idx"])):
            if is_answer(row["text"]):
                by_prompt.setdefault(row["prompt_id"], []).append((row["sample_idx"], row["text"].strip()))
        out[(cell["base"], cell["teacher"], cell["seed"])] = by_prompt
    return out


def cells(main_cfg) -> dict:
    base_pairs = list(itertools.combinations(sorted(main_cfg["bases"]), 2))
    teachers = list(main_cfg["teachers"])
    return {
        "same": [(bp, (t, t)) for bp in base_pairs for t in teachers],
        "different": [(bp, tp) for bp in base_pairs for tp in itertools.combinations(teachers, 2)],
    }


def build_items(run_dir, cfg) -> list[dict]:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    rng = random.Random(section["seed"])
    n = section["pairs_per_kind"]
    prompts = sorted(h1.build_texts(run_dir, cfg)[2])
    if len(prompts) < 2 * n:
        raise SystemExit(f"Only {len(prompts)} H1 prompts for {2 * n} pairs.")
    text_of = {r["prompt_id"]: r["prompt"] for r in read_jsonl(prompts_path(run_dir, main_cfg["test_split"]))}
    answers = usable_answers(run_dir, main_cfg)

    plan = []
    for kind, options in cells(main_cfg).items():
        order = options[:]
        rng.shuffle(order)
        plan += [(kind, *order[i % len(order)]) for i in range(n)]
    rng.shuffle(plan)
    rng.shuffle(prompts)

    items = []
    for i, ((kind, (x, y), (a, b)), pid) in enumerate(zip(plan, prompts)):
        if rng.random() < 0.5:
            a, b = b, a
        sides = []
        for base, teacher in ((x, a), (y, b)):
            seed = rng.choice(main_cfg["seeds"])
            idx, text = rng.choice(answers[(base, teacher, seed)][pid])
            sides.append({"base": base, "teacher": teacher, "seed": seed, "sample_idx": idx, "text": text})
        if rng.random() < 0.5:
            sides.reverse()
        items.append({"item_id": f"P{i + 1:03d}", "kind": kind, "prompt_id": pid, "prompt": text_of[pid],
                      "answer_a": sides[0]["text"], "answer_b": sides[1]["text"],
                      "a": {k: v for k, v in sides[0].items() if k != "text"}, "b": {k: v for k, v in sides[1].items() if k != "text"}})
    return items


def build(run_dir, cfg) -> dict:
    section = cfg[SECTION]
    items = build_items(run_dir, cfg)
    folder = check_dir(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for annotator in range(1, section["annotators"] + 1):
        order = items[:]
        random.Random(section["seed"] + annotator).shuffle(order)
        with open(folder / f"sheet_a{annotator}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "prompt", "answer_a", "answer_b", *LABELS, "notes"])
            for item in order:
                writer.writerow([item["item_id"], item["prompt"], item["answer_a"], item["answer_b"], "", ""])
    write_json(folder / "key_DO_NOT_OPEN.json", {i["item_id"]: {k: i[k] for k in ("kind", "prompt_id", "a", "b")} for i in items})
    write_pages(folder, section["annotators"])
    return {"pairs": len(items), "per_sheet": len(items), "kinds": {k: sum(i["kind"] == k for i in items) for k in ("same", "different")}}


def write_pages(folder, annotators: int) -> list:
    paths = []
    for annotator in range(1, annotators + 1):
        with open(folder / f"sheet_a{annotator}.csv", encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
        path = folder / f"sheet_a{annotator}.html"
        path.write_text(render(f"a{annotator}", items, labels=LABEL_QUESTIONS, rubric=RUBRIC_HTML, title=TITLE, boxes=BOXES, store=STORE), encoding="utf-8")
        paths.append(path)
    return paths


def read_sheets(folder, annotators: int) -> tuple[dict, dict]:
    """Labels {item_id: {annotator: "Y" or "N"}} and the texts of each pair, from the filled sheets."""
    labels, texts = {}, {}
    for annotator in range(1, annotators + 1):
        with open(folder / f"sheet_a{annotator}.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                value = row["same_idea"].strip().upper()[:1]
                if value not in ("Y", "N"):
                    raise SystemExit(f"sheet_a{annotator}.csv item {row['item_id']}: same_idea must be Y or N")
                labels.setdefault(row["item_id"], {})[annotator] = value
                texts[row["item_id"]] = (row["answer_a"], row["answer_b"])
    missing = [i for i, by in labels.items() if len(by) != annotators]
    if missing:
        raise SystemExit(f"Pairs not labeled by every annotator: {missing[:5]}")
    return labels, texts


def auc(scores_yes, scores_no) -> float:
    """Probability that a Y pair has a higher score than an N pair (ties count half)."""
    if not scores_yes or not scores_no:
        return float("nan")
    yes, no = np.asarray(scores_yes)[:, None], np.asarray(scores_no)[None, :]
    return float((yes > no).mean() + 0.5 * (yes == no).mean())


def score(run_dir, cfg) -> dict:
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    folder = check_dir(run_dir)
    key = read_json(folder / "key_DO_NOT_OPEN.json")
    labels, texts = read_sheets(folder, section["annotators"])
    final = {i: majority(list(by.values())) for i, by in labels.items()}
    yes = lambda ids: sum(final[i] == "Y" for i in ids)

    def rate(ids):
        return wilson(yes(ids), len(ids), section["ci_level"])

    kinds = {k: [i for i in final if key[i]["kind"] == k] for k in ("same", "different")}
    out = {"same": rate(kinds["same"]), "different": rate(kinds["different"])}
    rng = np.random.default_rng(section["bootstrap_seed"])
    hits = {k: np.array([final[i] == "Y" for i in ids], dtype=float) for k, ids in kinds.items()}
    draws = [rng.choice(hits["same"], hits["same"].size).mean() - rng.choice(hits["different"], hits["different"].size).mean() for _ in range(section["bootstrap"])]
    low, high = percentile_interval(draws, section["ci_level"])
    out["same minus different"] = {"estimate": float(hits["same"].mean() - hits["different"].mean()), "ci_low": low, "ci_high": high}

    for (x, y) in itertools.combinations(sorted(main_cfg["bases"]), 2):
        for kind, ids in kinds.items():
            out[f"{kind} base {x}-{y}"] = rate([i for i in ids if {key[i]["a"]["base"], key[i]["b"]["base"]} == {x, y}])
    for t in main_cfg["teachers"]:
        out[f"same teacher {t}"] = rate([i for i in kinds["same"] if key[i]["a"]["teacher"] == t])
    for a, b in itertools.combinations(main_cfg["teachers"], 2):
        out[f"different teachers {a}-{b}"] = rate([i for i in kinds["different"] if {key[i]["a"]["teacher"], key[i]["b"]["teacher"]} == {a, b}])
    for annotator in range(1, section["annotators"] + 1):
        for kind, ids in kinds.items():
            out[f"annotator {annotator} {kind}"] = wilson(sum(labels[i][annotator] == "Y" for i in ids), len(ids), section["ci_level"])
    out["kappa"] = fleiss_kappa([[labels[i][a] for a in sorted(labels[i])] for i in sorted(labels)])

    ids = sorted(final)
    # Embedded after the same cleaning as H1 (formatting removed), on the full answers.
    matrix = embed_texts([clean(t) for i in ids for t in texts[i]], section["embedder"])
    vectors = rows_dense(matrix, list(range(2 * len(ids))))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = {i: float(vectors[2 * n] @ vectors[2 * n + 1]) for n, i in enumerate(ids)}
    by_label = {v: [cosine[i] for i in ids if final[i] == v] for v in ("Y", "N")}
    out["embedding check"] = {
        "embedder": section["embedder"],
        "mean cosine, labeled Y": float(np.mean(by_label["Y"])) if by_label["Y"] else float("nan"),
        "mean cosine, labeled N": float(np.mean(by_label["N"])) if by_label["N"] else float("nan"),
        "auc": auc(by_label["Y"], by_label["N"]),
    }
    out["pairs_scored"] = len(final)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("action", choices=["build", "html", "score"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h1.SECTION, SECTION):
        verify_lock(cfg, args.run_dir, name)
    if args.action == "build":
        if (check_dir(args.run_dir) / "key_DO_NOT_OPEN.json").exists():
            raise SystemExit("Sheets already exist. They are built once; remove the folder by hand to rebuild.")
        print(build(args.run_dir, cfg), check_dir(args.run_dir))
    elif args.action == "html":
        for path in write_pages(check_dir(args.run_dir), cfg[SECTION]["annotators"]):
            print(path)
    else:
        result = score(args.run_dir, cfg)
        path = results_dir(args.run_dir, MAIN, "human_eval") / "pairs.json"
        write_json(path, result)
        print(result["same"], result["different"], result["same minus different"], result["kappa"], path)


if __name__ == "__main__":
    main()
