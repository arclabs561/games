# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=1.0"]
# ///
"""Run one experiment from this repository at a pinned commit, and store what it wrote.

Hugging Face Jobs runs this script (`hf jobs uv run`, or the Jobs tool). It clones the
repository at COMMIT, installs the dependency group GROUP from that commit's uv.lock
(`uv sync --locked`, which fails if the lock is stale instead of resolving again), runs the
module with its arguments, writes its manifest with `manifest.py`, and uploads the output folder to the private dataset
arclabs561/games-runs, under runs/<EXPERIMENT>/<RUN>/.

    environment: COMMIT (full SHA), GROUP, EXPERIMENT, RUN, HF_TOKEN (secret)
    arguments:   the module and its arguments, e.g. codenames.study --seed 3 --out OUT

The literal argument OUT is replaced by the output folder, and the manifest records the
arguments as they ran. The script's own dependency is locked in hfrun.py.lock.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "https://github.com/arclabs561/games"
DATASET = "arclabs561/games-runs"


def main(argv):
    commit, group = os.environ["COMMIT"], os.environ["GROUP"]
    experiment, run = os.environ["EXPERIMENT"], os.environ["RUN"]
    work, out = Path("/tmp/games"), Path("/tmp/out")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", REPO, str(work)], check=True)
    subprocess.run(["git", "checkout", "-q", commit], cwd=work, check=True)
    uv = shutil.which("uv") or "uv"
    subprocess.run([uv, "sync", "-q", "--locked", "--group", group], cwd=work, check=True)
    args = [str(out) if a == "OUT" else a for a in argv]
    start = time.time()
    done = subprocess.run([uv, "run", "--group", group, "python", "-m", *args], cwd=work)
    seconds = time.time() - start
    subprocess.run(
        [
            sys.executable,
            "experiments/manifest.py",
            str(out),
            "--experiment",
            experiment,
            "--run",
            run,
            "--commit",
            commit,
            "--platform",
            "hf-jobs",
            "--returncode",
            str(done.returncode),
            "--seconds",
            str(seconds),
            "--",
            *args,
        ],
        cwd=work,
        check=True,
    )
    HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
        folder_path=str(out),
        path_in_repo=f"runs/{experiment}/{run}",
        repo_id=DATASET,
        repo_type="dataset",
        commit_message=f"{experiment}/{run} at {commit[:7]}",
    )
    return done.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
