"""Cut the main-study test split from the untouched pool.

The pool is rebuilt with the pilot's filters and seed, and the rebuild must reproduce the
three pilot splits and the reserved list exactly. The test split is the next prompts in
that fixed random order, so no pilot prompt can enter it.
"""
import argparse

from sameteacher.core.config import add_common_args, load_yaml, read_json, read_jsonl, write_jsonl

from sameteacher.core.paths import prompts_dir, prompts_path
from sameteacher.data.make_splits import add_pool_args, load_pool, shuffled_pool

PILOT_SPLITS = ("screen", "confirm", "pilot_eval")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_pool_args(parser)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    main_cfg = cfg["main_grid"]
    out = prompts_path(args.run_dir, main_cfg["test_split"])
    if out.exists():
        raise SystemExit(f"{out} already exists. A test split is drawn once.")

    pool, excluded = load_pool(args)
    rows = shuffled_pool(pool, excluded, args.min_chars, args.max_chars, cfg["splits"]["seed"])
    start = 0
    for split in PILOT_SPLITS:
        drawn = [r["prompt_id"] for r in read_jsonl(prompts_path(args.run_dir, split))]
        if [r["prompt_id"] for r in rows[start : start + len(drawn)]] != drawn:
            raise SystemExit(f"The rebuilt pool does not reproduce the {split} split; refusing to draw.")
        start += len(drawn)
    reserved = read_json(prompts_dir(args.run_dir) / "reserved_ids.json")
    if [r["prompt_id"] for r in rows[start:]] != reserved:
        raise SystemExit("The rebuilt pool does not reproduce reserved_ids.json; refusing to draw.")

    test = rows[start : start + main_cfg["test_prompts"]]
    if len(test) < main_cfg["test_prompts"]:
        raise SystemExit(f"Only {len(test)} untouched prompts are left.")
    write_jsonl(out, test)
    print(f"{len(test)} test prompts -> {out}; {len(reserved) - len(test)} prompts remain untouched")


if __name__ == "__main__":
    main()
