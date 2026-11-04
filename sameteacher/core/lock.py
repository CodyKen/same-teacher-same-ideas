"""Freeze a config section before results are seen, and refuse later edits.

Usage: python -m sameteacher.core.lock --section pilot_teacher_distance
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

from .config import add_common_args, load_yaml, read_json, write_json


class LockError(RuntimeError):
    pass


def section_hash(section: dict) -> str:
    return hashlib.sha256(json.dumps(section, sort_keys=True).encode()).hexdigest()


def lock_path(run_dir, name: str) -> Path:
    return Path(run_dir) / "locks" / f"{name}.json"


def write_lock(config: dict, run_dir, name: str) -> Path:
    path = lock_path(run_dir, name)
    if path.exists():
        raise LockError(f"{name} is already locked at {path}. Locks are not rewritten.")
    write_json(
        path,
        {
            "section": name,
            "sha256": section_hash(config[name]),
            "locked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "values": config[name],
        },
    )
    return path


def verify_lock(config: dict, run_dir, name: str) -> None:
    path = lock_path(run_dir, name)
    if not path.exists():
        raise LockError(f"{name} is not locked. Run: python -m sameteacher.core.lock --section {name}")
    if read_json(path)["sha256"] != section_hash(config[name]):
        raise LockError(f"{name} changed after locking. Restore the locked values; see {path}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--section", required=True)
    args = parser.parse_args()
    path = write_lock(load_yaml(args.config), args.run_dir, args.section)
    print(f"locked {args.section}: {path}")


if __name__ == "__main__":
    main()
