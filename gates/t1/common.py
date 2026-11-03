"""Paths and loading shared by the T1 steps."""
from collections import defaultdict
from pathlib import Path

from gates_lib.config import read_jsonl
from gates_lib.text import clean, is_reasoning, is_refusal

IDEA_SUFFIX = "::idea_shift"
STYLE_SUFFIX = "::paraphrase"
# Long answers open with near-identical preamble and run past the token limit, so we
# would compare openers and formatting. One short complete answer holds one idea.
DIRECT_ANSWER_SYSTEM = (
    "Give one direct answer to the request, with no preamble, no list of alternatives, "
    "and no follow-up questions. Use at most 100 words."
)


def t1_dir(run_dir) -> Path:
    return Path(run_dir) / "t1"


def prompts_path(run_dir, split: str) -> Path:
    return t1_dir(run_dir) / "prompts" / f"{split}.jsonl"


def samples_path(run_dir, split: str) -> Path:
    return t1_dir(run_dir) / "samples" / f"{split}.jsonl"


def group_name(row: dict) -> str:
    variant = row.get("variant", "baseline")
    if variant == "idea_shift":
        return row["teacher"] + IDEA_SUFFIX
    if variant == "paraphrase":
        return row["teacher"] + STYLE_SUFFIX
    return row["teacher"]


def load_groups(run_dir, split: str, min_usable: int) -> tuple[dict, list[str], dict]:
    """Return usable text per group and prompt, plus prompts that every group covers."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    counts: dict = defaultdict(lambda: {"total": 0, "empty": 0, "refusal": 0, "reasoning": 0})
    for row in read_jsonl(samples_path(run_dir, split)):
        name = group_name(row)
        counts[name]["total"] += 1
        if not clean(row["text"]):
            counts[name]["empty"] += 1
        elif is_refusal(row["text"]):
            counts[name]["refusal"] += 1
        elif is_reasoning(row["text"]):
            counts[name]["reasoning"] += 1
        else:
            groups[name][row["prompt_id"]].append(clean(row["text"]))
    names = sorted(groups)
    prompt_ids = sorted({p for g in groups.values() for p in g})
    usable = [p for p in prompt_ids if all(len(groups[n].get(p, [])) >= min_usable for n in names)]
    return {n: dict(groups[n]) for n in names}, usable, dict(counts)
