"""Hand check of the H3 training rewrites: did the rewriting teacher keep the source's content?

`build` draws rewrites per condition from the H3 training files (so every item was trained on),
adds decoys (a source with the rewrite of another draw's answer to the same prompt by the same
rewriter), and writes one blind sheet per annotator plus a key in runs/human_eval/h3_rewrites/.
The key records whether each real item passes the locked source-match test. Each sheet is a CSV
and an HTML page with the instructions; the page exports the CSV. `html` rewrites the pages from
the CSV sheets. `score` reads the filled sheets. The design is in the locked `h3_rewrite_check`
and `h3_training_data` config sections.

Annotators answer two questions with Y or N:
  kept   every idea, claim, example and recommendation of the source is in the rewrite
  added  the rewrite states an idea, claim or example that is not in the source
"""
import argparse
import csv
import random
from collections import Counter, defaultdict

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, human_eval_dir, results_dir, train_file
from sameteacher.data import h3_data
from sameteacher.eval.rewrite_check import assign, fleiss_kappa, majority, wilson
from sameteacher.eval.rewrite_check_page import render

SECTION = "h3_rewrite_check"
LABELS = ("kept", "added")
LABEL_QUESTIONS = [
    ("kept", "Kept", "Is every idea, claim, example and recommendation of the original in the rewrite?"),
    ("added", "Added", "Does the rewrite state an idea, claim or example that is not in the original?"),
]
RUBRIC_HTML = """
<p>Each item shows a request, an <b>original</b> answer and a <b>rewrite</b>. The rewrite was
written by another model, asked to restate the original in its own words and its own style,
keeping every idea, claim, example and recommendation and adding none. A change of voice, length
or formatting is expected and is not judged: judge only the content. Answer two questions with Y
or N, and use the notes box for anything unclear.</p>
<dl>
  <dt>Kept</dt>
  <dd><b>Y</b> if every idea, claim, example and recommendation of the original appears in the
  rewrite. Rewording, merging or splitting sentences, a new tone and new formatting are fine.
  <b>N</b> if anything of substance is missing or its meaning changed.</dd>
  <dt>Added</dt>
  <dd><b>Y</b> if the rewrite states an idea, claim or example that is not in the original.
  Connecting words, a short restatement of the request and a change of tone do not count.
  Otherwise <b>N</b>.</dd>
</dl>
<p>Work alone, and do not discuss items until every sheet is in. Do not open
<code>key_DO_NOT_OPEN.json</code>.</p>
<p>Labels are saved in this browser as you go. When you finish, click <b>Export CSV</b> and put
the downloaded file in <code>runs/human_eval/h3_rewrites/</code>, replacing the blank sheet of
the same name. Export also works as a backup at any time; <b>Import CSV</b> loads it back.</p>
<p>The repository is public and some items are on every sheet, so keep your filled sheet to
yourself until all three sheets are done. Then commit it, or send it to the person who runs the
scoring.</p>
"""
TITLE = "H3 Rewrite Check"


def check_dir(run_dir):
    return human_eval_dir(run_dir, "h3_rewrites")


def condition_name(content, style) -> str:
    return f"{content} in {style} style"


def training_pairs(run_dir, cfg) -> dict:
    """{condition: [item]} for every row of the H3 training files, with its source answer."""
    by_condition = defaultdict(list)
    for f in h3_data.files(cfg):
        sources = {r["prompt"]: r["response"] for r in read_jsonl(train_file(run_dir, f["content"], f["rep"]))}
        for row in read_jsonl(h3_data.h3_dir(run_dir) / f"{f['student']}.jsonl"):
            by_condition[condition_name(f["content"], f["style"])].append({
                "student": f["student"], "condition": f["condition"], "base": f["base"], "seed": f["seed"], "file": h3_data.file_key(f),
                "prompt": row["prompt"], "original": sources[row["prompt"]], "rewrite": row["response"],
            })
    return by_condition


def build_items(run_dir, cfg) -> list[dict]:
    section, design = cfg[SECTION], cfg[h3_data.SECTION]
    rng = random.Random(design["hand_check_seed"])
    pools = training_pairs(run_dir, cfg)
    items = []
    sims = h3_data.similarities(run_dir, cfg)
    for condition, pool in sorted(pools.items()):
        for i, pair in enumerate(rng.sample(pool, design["hand_check_per_condition"])):
            v = sims[(pair["file"], pair["prompt"])]
            items.append({"kind": "real", "condition_name": condition, "shared": i < section["shared_per_condition"], **pair,
                          "source_match": bool(v["own"] > max(v["same_teacher"], v["other_teacher"]))})

    # Decoys: a source with the same rewriter's rewrite of another draw's answer to the same prompt.
    decoy_rng = random.Random(section["seed"])
    conditions = sorted(pools)
    for i in range(section["decoys"]):
        condition = conditions[i % len(conditions)]
        pair = decoy_rng.choice(pools[condition])
        other = decoy_rng.choice([p for p in pools[condition] if p["prompt"] == pair["prompt"] and p["student"] != pair["student"]])
        items.append({"kind": "decoy", "condition_name": condition, "shared": i < section["shared_decoys"], **pair,
                      "rewrite": other["rewrite"], "rewrite_of": other["student"]})
    decoy_rng.shuffle(items)
    for i, item in enumerate(items):
        item["item_id"] = f"H{i:03d}"
    return items


def build(run_dir, cfg) -> dict:
    section = cfg[SECTION]
    items = build_items(run_dir, cfg)
    sheets = assign(items, section["annotators"], random.Random(section["seed"] + 1))
    folder = check_dir(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for annotator, sheet in sheets.items():
        with open(folder / f"sheet_a{annotator}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "prompt", "original", "rewrite", *LABELS, "notes"])
            for item in sheet:
                writer.writerow([item["item_id"], item["prompt"], item["original"], item["rewrite"], "", "", ""])
    fields = ("kind", "condition_name", "condition", "student", "base", "seed", "shared", "rewrite_of", "source_match")
    write_json(folder / "key_DO_NOT_OPEN.json", {i["item_id"]: {k: i[k] for k in fields if k in i} for i in items})
    write_pages(folder, section["annotators"])
    counts = Counter((i["kind"], i["shared"]) for i in items)
    return {"items": len(items), "per_sheet": {a: len(s) for a, s in sheets.items()}, "counts": {f"{k} {'shared' if s else 'single'}": v for (k, s), v in sorted(counts.items())}}


def write_pages(folder, annotators: int) -> list:
    paths = []
    for annotator in range(1, annotators + 1):
        with open(folder / f"sheet_a{annotator}.csv", encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
        path = folder / f"sheet_a{annotator}.html"
        path.write_text(render(f"a{annotator}", items, labels=LABEL_QUESTIONS, rubric=RUBRIC_HTML, title=TITLE), encoding="utf-8")
        paths.append(path)
    return paths


def read_sheets(folder, annotators: int) -> dict:
    labels = defaultdict(dict)
    for annotator in range(1, annotators + 1):
        with open(folder / f"sheet_a{annotator}.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                values = {k: row[k].strip().upper()[:1] for k in LABELS}
                if any(v not in ("Y", "N") for v in values.values()):
                    raise SystemExit(f"sheet_a{annotator}.csv item {row['item_id']}: every label must be Y or N")
                labels[row["item_id"]][annotator] = values
    return labels


def score(run_dir, cfg) -> dict:
    section = cfg[SECTION]
    folder = check_dir(run_dir)
    key = read_json(folder / "key_DO_NOT_OPEN.json")
    labels = read_sheets(folder, section["annotators"])
    final = {i: {k: majority([v[k] for v in by.values()]) for k in LABELS} for i, by in labels.items()}

    def rate(ids):
        return wilson(sum(final[i]["kept"] == "Y" and final[i]["added"] == "N" for i in ids), len(ids), section["ci_level"])

    real = [i for i in final if key[i]["kind"] == "real"]
    out = {"real": rate(real), "decoy": rate([i for i in final if key[i]["kind"] == "decoy"])}
    for condition in ("cross", "self"):
        out[condition] = rate([i for i in real if key[i]["condition"] == condition])
    for name in sorted({key[i]["condition_name"] for i in real}):
        out[name] = rate([i for i in real if key[i]["condition_name"] == name])
    for outcome, label in ((True, "passes"), (False, "fails")):
        out[f"real, source match {label}"] = rate([i for i in real if key[i]["source_match"] == outcome])
    shared = [i for i in final if key[i]["shared"]]
    out["kappa_shared"] = {k: fleiss_kappa([[labels[i][a][k] for a in sorted(labels[i])] for i in shared]) for k in LABELS}
    out["items_scored"] = len(final)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("action", choices=["build", "html", "score"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", h3_data.SECTION, h3_data.TRAINING_SECTION, SECTION):
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
        path = results_dir(args.run_dir, MAIN, "h3") / "rewrite_check.json"
        write_json(path, result)
        print(result["real"], result["decoy"], result["kappa_shared"], path)


if __name__ == "__main__":
    main()
