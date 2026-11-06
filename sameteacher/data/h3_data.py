"""H3 training data: cross-style and self rewrites of the main-grid teacher draws.

  rewrite  every answer of the source draws, rewritten by the other teacher (cross) and by its
           own teacher (self)
  filter   the locked source-match test, reported per file and condition
  diagnose similarities behind the source-match test: the locked rival set against narrower ones
  build    the training files and runs/train_data/manifest_h3.json for
           `sameteacher.train.run_extra --manifest h3`

A rewrite passes the source-match test when it is a complete answer that did not hit max_tokens
and its embedding is closer to its source than to every other answer either teacher gave to that
prompt in any draw (locked `h3_cross_style` section). Students train on the prompts whose
rewrites are complete answers in all 16 files (locked `h3_training_data` section).
"""
import argparse
import asyncio
from collections import defaultdict

import numpy as np

from sameteacher.core.config import add_common_args, load_yaml, read_jsonl, stable_seed, write_json, write_jsonl
from sameteacher.core.grid import main_cells
from sameteacher.core.llm import Budget, Chat, load_models, run_jobs
from sameteacher.core.lock import verify_lock
from sameteacher.core.paths import MAIN, results_dir, train_data_dir, train_file
from sameteacher.core.text import is_answer
from sameteacher.data.teacher_answers import DIRECT_ANSWER_SYSTEM
from sameteacher.eval.embed import embed_texts

SECTION = "h3_cross_style"
TRAINING_SECTION = "h3_training_data"
KEEP_RULE = "complete_in_every_file"
REWRITE_INSTRUCTION = (
    "Here is a request and an answer to it that someone else wrote.\n\nRequest:\n{prompt}\n\nAnswer:\n{answer}\n\n"
    "Rewrite the answer in your own words and your own style, the way you would normally answer this request. "
    "Keep every idea, claim, example and recommendation, and add none of your own. Output only the rewritten answer."
)


def h3_dir(run_dir):
    return train_data_dir(run_dir) / "h3"


def rewrites_path(run_dir):
    return h3_dir(run_dir) / "rewrites.jsonl"


def manifest_path(run_dir):
    return train_data_dir(run_dir) / "manifest_h3.json"


def usable(row) -> bool:
    """A rewrite that did not hit max_tokens and is a usable answer."""
    return bool(row) and row["finish_reason"] != "length" and is_answer(row["text"])


def student_name(base, content, style, seed) -> str:
    return f"{base}__{content}_in_{style}__s{seed}"


def files(cfg) -> list[dict]:
    """The 16 training files: base, seed, the draw (rep), content teacher and rewriting teacher."""
    main_cfg, section = cfg["main_grid"], cfg[SECTION]
    teachers = list(section["teachers"])
    cells = {(c["base"], c["teacher"], c["seed"]): c for c in main_cells(main_cfg)}
    out = []
    for base in section["bases"]:
        for seed in section["seeds"]:
            for content in teachers:
                cell = cells[(base, content, seed)]
                for style in [t for t in teachers if t != content] + [content]:
                    out.append({"base": base, "seed": seed, "rep": cell["rep"], "model": cell["model"], "content": content, "style": style,
                                "condition": "self" if style == content else "cross", "student": student_name(base, content, style, seed)})
    return out


def file_key(f) -> str:
    return f"{f['content']}|{f['rep']}|{f['style']}"


def rewrite_jobs(run_dir, cfg) -> list[dict]:
    """One job per answer and file, interleaved across files so both rewriters work at once."""
    per_file = []
    for f in files(cfg):
        per_file.append([
            {"key": f"{file_key(f)}|{stable_seed(row['prompt'])}", "file": file_key(f), "style": f["style"], "prompt": row["prompt"], "source": row["response"]}
            for row in read_jsonl(train_file(run_dir, f["content"], f["rep"]))
        ])
    width = max(len(jobs) for jobs in per_file)
    return [jobs[i] for i in range(width) for jobs in per_file if i < len(jobs)]


async def rewrite(args, cfg) -> dict:
    sampling, section = cfg["sampling"], cfg[SECTION]
    specs = load_models(args.models)
    chat = Chat(Budget(args.max_usd))
    jobs = rewrite_jobs(args.run_dir, cfg)

    async def handler(job):
        result = await chat.complete(
            specs[job["style"]],
            [{"role": "system", "content": DIRECT_ANSWER_SYSTEM},
             {"role": "user", "content": REWRITE_INSTRUCTION.format(prompt=job["prompt"], answer=job["source"])}],
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            max_tokens=section["max_tokens"],
            seed=stable_seed(job["key"]),
            metadata={"task": "paraphrase", "source": job["source"]},
        )
        return {"file": job["file"], "prompt": job["prompt"], "text": result.text, "finish_reason": result.finish_reason}

    try:
        stats = await run_jobs(jobs, handler, rewrites_path(args.run_dir), args.concurrency or sampling["concurrency"])
    finally:
        await chat.aclose()
    print(f"rewrites: {stats}, spent about ${chat.budget.spent:.2f} at the configured prices, {chat.throttled} throttled and {chat.malformed} malformed-response retries")
    return stats


def all_answers(run_dir, cfg) -> dict:
    """Every answer each teacher gave to each prompt in any draw: {prompt: {(teacher, rep): text}}."""
    reps = sorted({r for draws in cfg["main_grid"]["draws"].values() for r in draws})
    out = defaultdict(dict)
    for teacher in cfg[SECTION]["teachers"]:
        for rep in reps:
            path = train_file(run_dir, teacher, rep)
            for row in read_jsonl(path):
                out[row["prompt"]][(teacher, rep)] = row["response"]
    return out


def source_match(run_dir, cfg) -> tuple[dict, dict]:
    """Pass or fail for every expected rewrite, and the similarity margin of each."""
    section = cfg[SECTION]
    answers = all_answers(run_dir, cfg)
    rewrites = {(r["file"], r["prompt"]): r for r in read_jsonl(rewrites_path(run_dir))}
    expected = []
    for f in files(cfg):
        for row in read_jsonl(train_file(run_dir, f["content"], f["rep"])):
            expected.append((file_key(f), f["content"], f["rep"], row["prompt"]))

    texts, index = [], {}
    for prompt, by in answers.items():
        for owner, text in by.items():
            index[("answer", prompt, owner)] = len(texts)
            texts.append(text)
    complete = {}
    for key, content, rep, prompt in expected:
        row = rewrites.get((key, prompt))
        complete[(key, prompt)] = usable(row)
        if complete[(key, prompt)]:
            index[("rewrite", key, prompt)] = len(texts)
            texts.append(row["text"])
    matrix = embed_texts(texts, section["embedder"])
    matrix = np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float32)

    passed, margin = {}, {}
    for key, content, rep, prompt in expected:
        if not complete[(key, prompt)]:
            passed[(key, prompt)] = False
            continue
        vector = matrix[index[("rewrite", key, prompt)]]
        owners = list(answers[prompt])
        sims = matrix[[index[("answer", prompt, o)] for o in owners]] @ vector
        own = sims[owners.index((content, rep))]
        others = np.delete(sims, owners.index((content, rep)))
        margin[(key, prompt)] = float(own - others.max())
        passed[(key, prompt)] = bool(own > others.max())
    return passed, margin


def similarity_texts(run_dir, cfg) -> tuple[list[str], dict, list, dict]:
    """Every teacher answer and usable rewrite, indexed (prompt, ("answer", teacher, rep)) or (prompt, ("rewrite", file))."""
    answers = all_answers(run_dir, cfg)
    rewrites = {(r["file"], r["prompt"]): r for r in read_jsonl(rewrites_path(run_dir))}
    texts, index = [], {}
    for prompt, by in answers.items():
        for owner, text in by.items():
            index[(prompt, ("answer", *owner))] = [len(texts)]
            texts.append(text)
    expected = []
    for f in files(cfg):
        for row in read_jsonl(train_file(run_dir, f["content"], f["rep"])):
            key, prompt = file_key(f), row["prompt"]
            expected.append((f, key, prompt))
            r = rewrites.get((key, prompt))
            if usable(r):
                index[(prompt, ("rewrite", key))] = [len(texts)]
                texts.append(r["text"])
    return texts, index, expected, answers


SIMILARITY_NOTE = ("Rows are every answer of the H3 teachers in all draws and every usable H3 rewrite (full texts). "
                   "Keys: prompt|answer|teacher|rep or prompt|rewrite|file.")


def similarities(run_dir, cfg) -> dict:
    """Per expected rewrite: cosine to its source, best other draw of the content teacher, best
    answer of the other teacher (embeddings cached). None when the rewrite is not usable."""
    from sameteacher.analysis.teacher_match import embed_cached

    texts, index, expected, answers = similarity_texts(run_dir, cfg)
    matrix = embed_cached(run_dir, texts, index, cfg[SECTION]["embedder"], name="h3_source_match", note=SIMILARITY_NOTE)
    out = {}
    for f, key, prompt in expected:
        if (prompt, ("rewrite", key)) not in index:
            out[(key, prompt)] = None
            continue
        vector = matrix[index[(prompt, ("rewrite", key))][0]]
        sims = {owner: float(matrix[index[(prompt, ("answer", *owner))][0]] @ vector) for owner in answers[prompt]}
        own = sims.pop((f["content"], f["rep"]))
        same = max(v for (t, _), v in sims.items() if t == f["content"])
        other = max(v for (t, _), v in sims.items() if t != f["content"])
        out[(key, prompt)] = {"own": own, "same_teacher": same, "other_teacher": other}
    return out


def diagnose(run_dir, cfg) -> dict:
    """Pass rates and prompts kept under the locked rival set and under narrower ones."""
    sims = similarities(run_dir, cfg)
    table = files(cfg)
    rules = {
        "locked: every other answer of both teachers": lambda v: v["own"] > max(v["same_teacher"], v["other_teacher"]),
        "other teacher's answers only": lambda v: v["own"] > v["other_teacher"],
        "closer to the content teacher's draws than to the other teacher's": lambda v: max(v["own"], v["same_teacher"]) > v["other_teacher"],
    }
    out = {"rules": {}}
    prompts_of = {file_key(f): [p for (k, p) in sims if k == file_key(f)] for f in table}
    for name, rule in rules.items():
        passed = {k: v is not None and rule(v) for k, v in sims.items()}
        per_condition = defaultdict(list)
        for f in table:
            per_condition[f"{f['content']} in {f['style']} style"] += [passed[(file_key(f), p)] for p in prompts_of[file_key(f)]]
        kept = set.intersection(*[{p for p in ps if passed[(k, p)]} for k, ps in prompts_of.items()])
        out["rules"][name] = {
            "pass_rate": {c: float(np.mean(v)) for c, v in per_condition.items()},
            "prompts_kept_in_all_16": len(kept),
            "smallest_file_pass_count": int(min(sum(passed[(k, p)] for p in ps) for k, ps in prompts_of.items())),
        }
    scored = [v for v in sims.values() if v is not None]
    failing = [v for v in scored if not rules["locked: every other answer of both teachers"](v)]
    out["locked failures beaten by a same-teacher draw"] = float(np.mean([v["same_teacher"] >= v["other_teacher"] for v in failing]))
    out["median own minus best same-teacher draw"] = float(np.median([v["own"] - v["same_teacher"] for v in scored]))
    out["median own minus best other-teacher answer"] = float(np.median([v["own"] - v["other_teacher"] for v in scored]))
    out["usable rewrites"] = len(scored)
    return out


def prompts_by_file(run_dir, cfg) -> dict:
    return {file_key(f): [r["prompt"] for r in read_jsonl(train_file(run_dir, f["content"], f["rep"]))] for f in files(cfg)}


def report(run_dir, cfg, passed, margin) -> dict:
    """The locked source-match test per condition and file; no rule is attached to it."""
    table, prompts_of = files(cfg), prompts_by_file(run_dir, cfg)
    per_file, conditions = {}, defaultdict(list)
    for f in table:
        key = file_key(f)
        values = [passed[(key, p)] for p in prompts_of[key]]
        margins = [margin[(key, p)] for p in prompts_of[key] if (key, p) in margin]
        per_file[f["student"]] = {"pass_rate": float(np.mean(values)), "rewrites": len(values), "median_margin": float(np.median(margins)) if margins else float("nan")}
        conditions[f"{f['content']} in {f['style']} style"].extend(values)
    kept = set.intersection(*[{p for p in prompts if passed[(key, p)]} for key, prompts in prompts_of.items()])
    return {
        "pass_rate": {k: float(np.mean(v)) for k, v in conditions.items()},
        "per_file": per_file,
        "prompts_passing_in_all_16": len(kept),
        "prompts_total": len(set().union(*prompts_of.values())),
    }


def build(run_dir, cfg) -> dict:
    """Training files of the prompts whose rewrites are complete answers in every file, and the manifest."""
    if cfg[TRAINING_SECTION]["keep"] != KEEP_RULE:
        raise SystemExit(f"Unknown {TRAINING_SECTION}.keep rule: {cfg[TRAINING_SECTION]['keep']}")
    folder = h3_dir(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    rewrites = {(r["file"], r["prompt"]): r for r in read_jsonl(rewrites_path(run_dir))}
    table, prompts_of = files(cfg), prompts_by_file(run_dir, cfg)
    kept = set.intersection(*[{p for p in prompts if usable(rewrites.get((key, p)))} for key, prompts in prompts_of.items()])

    runs, per_file = [], {}
    for f in table:
        key = file_key(f)
        rows = [{"prompt": p, "response": rewrites[(key, p)]["text"]} for p in prompts_of[key] if p in kept]
        write_jsonl(folder / f"{f['student']}.jsonl", rows)
        per_file[f["student"]] = {
            "examples": len(rows),
            "rewrites": len(prompts_of[key]),
            "complete": int(sum(usable(rewrites.get((key, p))) for p in prompts_of[key])),
            "cut_at_max_tokens": int(sum(bool(rewrites.get((key, p))) and rewrites[(key, p)]["finish_reason"] == "length" for p in prompts_of[key])),
        }
        runs.append({"student": f["student"], "stage": "h3", "base": f["base"], "model": f["model"], "seed": f["seed"], "content": f["content"],
                     "style": f["style"], "condition": f["condition"], "source": train_file(run_dir, f["content"], f["rep"]).name,
                     "examples": len(rows), "train_file": f"{f['student']}.jsonl"})
    summary = {"rule": KEEP_RULE, "prompts_kept": len(kept), "prompts_total": len(set().union(*prompts_of.values())), "per_file": per_file}
    write_json(manifest_path(run_dir), {"section": SECTION, "training_section": TRAINING_SECTION, "runs": runs, "prompts_kept": len(kept)})
    write_json(results_dir(run_dir, MAIN, "h3") / "training_data.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("step", choices=["rewrite", "filter", "diagnose", "build"])
    parser.add_argument("--max-usd", type=float, help="Budget for the rewrites at the configured prices (rewrite step)")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    verify_lock(cfg, args.run_dir, "main_grid")
    verify_lock(cfg, args.run_dir, SECTION)
    if args.step == "rewrite":
        if args.max_usd is None:
            raise SystemExit("--max-usd is required for the rewrite step")
        if asyncio.run(rewrite(args, cfg))["stopped_on_budget"]:
            raise SystemExit("Rewrites stopped on budget. Rerun with a larger --max-usd; finished rewrites are kept.")
        return
    if args.step == "diagnose":
        result = diagnose(args.run_dir, cfg)
        write_json(results_dir(args.run_dir, MAIN, "h3") / "source_match_diagnostic.json", result)
        print(result)
        return
    if args.step == "build":
        verify_lock(cfg, args.run_dir, TRAINING_SECTION)
        summary = build(args.run_dir, cfg)
        print(f"prompts kept {summary['prompts_kept']} of {summary['prompts_total']}")
        print(manifest_path(args.run_dir))
        return
    passed, margin = source_match(args.run_dir, cfg)
    summary = report(args.run_dir, cfg, passed, margin)
    write_json(results_dir(args.run_dir, MAIN, "h3") / "source_match.json", summary)
    print({k: round(v, 3) for k, v in summary["pass_rate"].items()}, f"prompts passing in all 16 files {summary['prompts_passing_in_all_16']} of {summary['prompts_total']}")

if __name__ == "__main__":
    main()
