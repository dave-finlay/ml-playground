"""List, promote and delete training runs saved by train.py in
cartpole/runs/.

Run from the repo root:

    uv run python cartpole/runs.py list
    uv run python cartpole/runs.py promote normalized-b1000
    uv run python cartpole/runs.py promote normalized-b1000 normalized-seed3-long
    uv run python cartpole/runs.py delete normalized-seed3-again
    uv run python cartpole/runs.py delete cartpole/runs/default-2026*
    uv run python cartpole/runs.py delete --yes slow-lr fast-lr

promote copies a run's whole directory into cartpole/models/ (under the
run's name unless given another), where git tracks it, so configs can
rely on it. It never overwrites a model, and it leaves the run in place.

delete accepts run names or paths (so the shell can expand patterns of
paths), shows what it will delete and asks first unless given --yes. It
only deletes directories directly inside cartpole/runs/ that hold a
run's config.json, so it can't remove anything else, such as a kept
model in cartpole/models/.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from train import RUNS_DIR

MODELS_DIR = Path(__file__).parent / "models"


def is_run(path):
    """Whether path is a run directory: directly inside RUNS_DIR, holding
    a run's config.json."""
    path = Path(path).resolve()
    return (
        path.parent == RUNS_DIR.resolve()
        and path.is_dir()
        and (path / "config.json").is_file()
    )


def find_run(name_or_path):
    """Return the run directory a name or path refers to, or raise
    ValueError if it isn't one. A bare name is looked up in RUNS_DIR."""
    path = Path(name_or_path) if "/" in name_or_path else RUNS_DIR / name_or_path
    if not is_run(path):
        raise ValueError(f"not a run in {RUNS_DIR}: {name_or_path}")
    return path.resolve()


def read_json(path):
    """Return the JSON in path, or {} if it's missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def run_summary(path):
    """Return (name, started, test result, description) for a run, with
    blanks for anything its files don't say."""
    config = read_json(path / "config.json")
    results = read_json(path / "results.json")
    test = results.get("test_lengths", {})
    test_text = (
        f"{test['mean']:.0f}/{test['limit']}" if {"mean", "limit"} <= set(test) else ""
    )
    started = results.get("started", "")[:16].replace("T", " ")
    return path.name, started, test_text, config.get("description", "")


def list_runs():
    """Print every run, oldest first."""
    runs = sorted(
        (run_summary(p) for p in RUNS_DIR.iterdir() if is_run(p))
        if RUNS_DIR.is_dir()
        else [],
        key=lambda summary: (summary[1], summary[0]),
    )
    if not runs:
        print(f"No runs in {RUNS_DIR}")
        return
    width = max(len(name) for name, *_ in runs)
    print(f"{'run':{width}}  {'started':16}  {'test':>8}  description")
    for name, started, test, description in runs:
        print(f"{name:{width}}  {started:16}  {test:>8}  {description}")


def promote_run(run_path, model_name):
    """Copy the run at run_path into MODELS_DIR / model_name and return
    the model's path. Raise ValueError if model_name isn't a plain name or
    that model already exists."""
    if not model_name or "/" in model_name or model_name.startswith("."):
        raise ValueError(f"not a usable model name: {model_name!r}")
    model_path = MODELS_DIR / model_name
    if model_path.exists():
        raise ValueError(f"model already exists: {model_path}")
    shutil.copytree(run_path, model_path)
    return model_path


def delete_runs(paths, confirmed):
    """Delete run directories, asking first unless confirmed. Returns
    whether they were deleted."""
    print("Delete:")
    for path in paths:
        print(f"  {path}")
    if not confirmed:
        answer = input(f"Delete {len(paths)} run(s)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return False
    for path in paths:
        shutil.rmtree(path)
    print(f"Deleted {len(paths)} run(s).")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the runs, oldest first")
    promote = commands.add_parser(
        "promote", help="copy a run into cartpole/models/ to keep it"
    )
    promote.add_argument("run", help="run name or path")
    promote.add_argument(
        "name", nargs="?", help="name for the model (default: the run's name)"
    )
    delete = commands.add_parser("delete", help="delete runs")
    delete.add_argument("runs", nargs="+", help="run names or paths")
    delete.add_argument("--yes", action="store_true", help="don't ask before deleting")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "list":
            list_runs()
        elif args.command == "promote":
            run_path = find_run(args.run)
            model_path = promote_run(run_path, args.name or run_path.name)
            print(f"Copied {run_path.name} to {model_path}")
        else:
            paths = list(dict.fromkeys(find_run(r) for r in args.runs))
            delete_runs(paths, args.yes)
    except ValueError as e:
        sys.exit(f"runs.py: error: {e}")


if __name__ == "__main__":
    main()
