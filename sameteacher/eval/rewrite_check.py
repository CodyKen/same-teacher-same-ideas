"""Hand check of the primary rewriter's content fidelity on the main grid.

`build` draws (original, rewrite) pairs from the pairs that main_style_norm analyzed, stratified
by base and teacher, adds decoys (an original with the rewrite of another answer by the same
student to the same prompt), and writes one blind sheet per annotator plus a key. Each sheet is
a CSV and an HTML page with the instructions (`rewrite_check_page`); the page exports the CSV.
`html` rewrites the pages from the existing CSV sheets. `score` reads the filled CSV sheets. The
design is in the locked `main_rewrite_check` config section.

Annotators answer three questions with Y or N:
  kept   every claim, example and recommendation of the original is in the rewrite
  added  the rewrite states a claim or example that is not in the original
  plain  the rewrite is plain prose, with no lists, headings or marked voice
"""
import argparse
import csv
import math
import random
from collections import Counter, defaultdict

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_json
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, human_eval_dir, prompts_path, results_dir
from sameteacher.eval.rewrite_check_page import render
from sameteacher.style_norm import main_retention as primary

SECTION = "main_rewrite_check"
LABELS = ("kept", "added", "plain")


def check_dir(run_dir):
    return human_eval_dir(run_dir, "rewrite_fidelity")


def analyzed_pairs(run_dir, cfg) -> tuple[dict, list]:
    main_cfg, h2 = cfg["main_grid"], cfg[primary.SECTION]
    _, _, _, cells = primary.grid(main_cfg)
    sources = primary.select_sources(run_dir, main_cfg, h2["samples_per_cell"])
    rewrites = {f"{r['student']}|{r['prompt_id']}|{r['sample_idx']}": r for r in read_jsonl(primary.rewrites_path(run_dir))}
    pairs, prompts = primary.matched_pairs(run_dir, cells, sources, h2["min_pairs"])
    keep = set(prompts)
    usable = defaultdict(list)
    for source in sources:
        rewritten = rewrites.get(source["key"])
        if source["prompt_id"] in keep and rewritten and rewritten["finish_reason"] != "length" and primary.is_answer(rewritten["text"]):
            usable[source["cell"]].append({"key": source["key"], "prompt_id": source["prompt_id"], "original": source["text"], "rewrite": rewritten["text"]})
    return usable, cells


def build_items(run_dir, cfg) -> list[dict]:
    section = cfg[SECTION]
    rng = random.Random(section["seed"])
    usable, cells = analyzed_pairs(run_dir, cfg)
    path = prompts_path(run_dir, cfg["main_grid"]["test_split"])
    prompts = {p["prompt_id"]: p["prompt"] for p in read_jsonl(path)} if path.exists() else {}
    by_stratum = defaultdict(list)
    for cell in cells:
        by_stratum[(cell[0], cell[1])].extend(dict(p, seed=cell[2]) for p in usable[cell])

    items = []
    for (base, teacher), pool in sorted(by_stratum.items()):
        chosen = rng.sample(pool, section["per_cell"])
        for i, pair in enumerate(chosen):
            items.append({"kind": "real", "base": base, "teacher": teacher, "shared": i < section["shared_per_cell"], **pair})

    # Decoys: the rewrite of a different answer by the same student to the same prompt.
    same_student = defaultdict(list)
    for pool in by_stratum.values():
        for pair in pool:
            same_student[pair["key"].rsplit("|", 1)[0]].append(pair)
    strata = sorted(by_stratum)
    for i in range(section["decoys"]):
        base, teacher = strata[i % len(strata)]
        while True:
            pair = rng.choice(by_stratum[(base, teacher)])
            others = [p for p in same_student[pair["key"].rsplit("|", 1)[0]] if p["key"] != pair["key"]]
            if others:
                break
        other = rng.choice(others)
        items.append({
            "kind": "decoy", "base": base, "teacher": teacher, "shared": i < section["shared_decoys"],
            "key": pair["key"], "rewrite_of": other["key"], "prompt_id": pair["prompt_id"], "seed": pair["seed"],
            "original": pair["original"], "rewrite": other["rewrite"],
        })
    rng.shuffle(items)
    for i, item in enumerate(items):
        item["item_id"] = f"F{i:03d}"
        item["prompt"] = prompts.get(item["prompt_id"], "")
    return items


def assign(items, annotators: int, rng) -> dict:
    """Shared items go to everyone; the rest are dealt out in turn."""
    sheets = {a: [i for i in items if i["shared"]] for a in range(1, annotators + 1)}
    rest = [i for i in items if not i["shared"]]
    for n, item in enumerate(rest):
        sheets[n % annotators + 1].append(item)
    for sheet in sheets.values():
        rng.shuffle(sheet)
    return sheets


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
                writer.writerow([item["item_id"], item["prompt"], item["original"], item["rewrite"], "", "", "", ""])
    key = {i["item_id"]: {k: i[k] for k in ("kind", "base", "teacher", "seed", "shared", "key", "prompt_id") if k in i} | ({"rewrite_of": i["rewrite_of"]} if "rewrite_of" in i else {}) for i in items}
    write_json(folder / "key_DO_NOT_OPEN.json", key)
    write_pages(folder, section["annotators"])
    counts = Counter((i["kind"], i["shared"]) for i in items)
    return {"items": len(items), "per_sheet": {a: len(s) for a, s in sheets.items()}, "counts": {f"{k} {'shared' if s else 'single'}": v for (k, s), v in sorted(counts.items())}}


def write_pages(folder, annotators: int) -> list:
    """One HTML page per CSV sheet, with the items in the sheet's order."""
    paths = []
    for annotator in range(1, annotators + 1):
        with open(folder / f"sheet_a{annotator}.csv", encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
        path = folder / f"sheet_a{annotator}.html"
        path.write_text(render(f"a{annotator}", items), encoding="utf-8")
        paths.append(path)
    return paths


def wilson(successes: int, n: int, level: float) -> dict:
    if n == 0:
        return {"rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    z = {0.95: 1.959964}.get(level, 1.959964)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"rate": p, "ci_low": centre - half, "ci_high": centre + half, "n": n}


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Fleiss' kappa for items rated by the same number of raters."""
    categories = sorted({r for item in ratings for r in item})
    n = len(ratings[0])
    counts = [[item.count(c) for c in categories] for item in ratings]
    p_items = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    p_bar = sum(p_items) / len(p_items)
    shares = [sum(row[j] for row in counts) / (len(counts) * n) for j in range(len(categories))]
    p_e = sum(s * s for s in shares)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


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


def majority(votes: list[str]) -> str:
    return Counter(votes).most_common(1)[0][0]


def score(run_dir, cfg) -> dict:
    section = cfg[SECTION]
    folder = check_dir(run_dir)
    key = read_json(folder / "key_DO_NOT_OPEN.json")
    labels = read_sheets(folder, section["annotators"])
    final = {item_id: {k: majority([v[k] for v in by.values()]) for k in LABELS} for item_id, by in labels.items()}

    def rates(ids):
        passed = sum(final[i]["kept"] == "Y" and final[i]["added"] == "N" for i in ids)
        plain = sum(final[i]["plain"] == "Y" for i in ids)
        return {"pass": wilson(passed, len(ids), section["ci_level"]), "plain": wilson(plain, len(ids), section["ci_level"])}

    real = [i for i in final if key[i]["kind"] == "real"]
    decoys = [i for i in final if key[i]["kind"] == "decoy"]
    out = {"real": rates(real), "decoy": rates(decoys)}
    for field in ("teacher", "base"):
        for value in sorted({key[i][field] for i in real}):
            out[f"{field} {value}"] = rates([i for i in real if key[i][field] == value])
    shared = [i for i in final if key[i]["shared"]]
    out["kappa_shared"] = {k: fleiss_kappa([[labels[i][a][k] for a in sorted(labels[i])] for i in shared]) for k in ("kept", "added")}
    out["items_scored"] = len(final)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("action", choices=["build", "html", "score"])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for name in ("main_grid", primary.SECTION, SECTION):
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
        path = results_dir(args.run_dir, MAIN, "style_norm") / "rewrite_check.json"
        write_json(path, result)
        print(result["real"], result["decoy"], result["kappa_shared"], path)


if __name__ == "__main__":
    main()
