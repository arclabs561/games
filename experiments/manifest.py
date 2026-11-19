"""Write the manifest that every experiment run stores beside its outputs.

Both launchers call this, so a run's record has the same fields whether it ran on
Hugging Face Jobs (`hfrun.py`) or GitHub Actions (`.github/workflows/experiment.yml`).
It needs only the standard library.

    python3 experiments/manifest.py OUT --experiment C3 --run seed-1 --commit SHA \
        --platform actions --returncode 0 --seconds 12.5 -- codenames.rl --seed 1
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def gpu_name():
    """The first GPU's name, or None when there is no NVIDIA GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.splitlines()[0].strip() if out.stdout.strip() else None


def manifest(experiment, run, commit, platform_name, returncode, seconds, command):
    """The record for one run, as a dict."""
    return {
        "experiment": experiment,
        "run": run,
        "commit": commit,
        "platform": platform_name,
        "command": ["python", "-m", *command],
        "returncode": returncode,
        "seconds": seconds,
        "job_id": os.environ.get("JOB_ID") or os.environ.get("GITHUB_RUN_ID"),
        "hardware": os.environ.get("ACCELERATOR") or os.environ.get("RUNNER_NAME"),
        "gpu": gpu_name(),
        "python": platform.python_version(),
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    options, command = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--platform", required=True, choices=["hf-jobs", "actions", "local"])
    parser.add_argument("--returncode", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args(options)
    record = manifest(
        args.experiment,
        args.run,
        args.commit,
        args.platform,
        args.returncode,
        round(args.seconds, 1),
        command,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
