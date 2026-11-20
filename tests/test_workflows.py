"""The GitHub Actions workflows, and the inert copies of workflows kept in game folders.

Artifacts of a public repository can be downloaded by anyone signed in to GitHub, so no
workflow may upload a `.cache` folder (third-party downloads) or, in a workflow with a
private store (`store: hf`), make an artifact in that mode. Every job has a timeout and
its own permissions; only a job that pushes may write, and a job that only reads keeps no
credentials in its checkout. Packages added with `--with` or `uvx --from` are pinned.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent


def workflow_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.yml", "*.yaml"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    paths = [ROOT / line for line in out.stdout.split()]
    return [p for p in paths if "jobs" in (yaml.safe_load(p.read_text()) or {})]


def load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    data["on"] = data.pop(True, data.get("on"))  # YAML 1.1 reads the key `on` as true
    return data


def steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def uses(step: dict, action: str) -> bool:
    return step.get("uses", "").split("@")[0] == action


def runs(job: dict) -> str:
    return "\n".join(s.get("run", "") for s in steps(job))


WORKFLOWS = workflow_files()
JOBS = [(p, name, job) for p in WORKFLOWS for name, job in load(p)["jobs"].items()]
IDS = [f"{p.relative_to(ROOT)}:{name}" for p, name, _ in JOBS]


def test_the_workflows_are_found():
    names = {p.relative_to(ROOT).as_posix() for p in WORKFLOWS}
    assert {".github/workflows/ci.yml", ".github/workflows/experiment.yml"} <= names
    assert "codenames/compute/codenames-c3.yml" in names


@pytest.mark.parametrize(("path", "name", "job"), JOBS, ids=IDS)
def test_no_artifact_holds_a_cache_folder(path, name, job):
    for s in steps(job):
        if uses(s, "actions/upload-artifact"):
            assert ".cache" not in str(s.get("with", {}).get("path", "")), (path, name)


@pytest.mark.parametrize(("path", "name", "job"), JOBS, ids=IDS)
def test_a_private_store_makes_no_artifact(path, name, job):
    inputs = (load(path)["on"] or {}).get("workflow_dispatch", {}) or {}
    store = (inputs.get("inputs") or {}).get("store")
    if not store or "hf" not in store.get("options", []):
        return
    for s in steps(job):
        if uses(s, "actions/upload-artifact"):
            assert "inputs.store == 'branch'" in s.get("if", ""), (path, name)
    if "secrets.HF_TOKEN" in yaml.safe_dump(job):
        assert "inputs.store == 'hf'" in job.get("if", ""), (path, name)
        assert not any(uses(s, "actions/checkout") for s in steps(job)), (path, name)


@pytest.mark.parametrize(("path", "name", "job"), JOBS, ids=IDS)
def test_every_job_has_a_timeout_and_its_own_permissions(path, name, job):
    assert "timeout-minutes" in job, (path, name)
    perms = job.get("permissions")
    assert isinstance(perms, dict), (path, name)
    pushes = "git push" in runs(job)
    assert (perms.get("contents") == "write") == pushes, (path, name, perms)
    assert set(perms.values()) <= {"read", "write", "none"}
    assert all(v != "write" for k, v in perms.items() if k != "contents"), (path, name)
    if not pushes:
        for s in steps(job):
            if uses(s, "actions/checkout"):
                assert s.get("with", {}).get("persist-credentials") is False, (path, name)


@pytest.mark.parametrize(("path", "name", "job"), JOBS, ids=IDS)
def test_packages_added_on_the_command_line_are_pinned(path, name, job):
    script = runs(job)
    for flag in ("--with", "--from"):
        for spec in re.findall(rf"{flag}[ =]+[\"']?([^\s\"']+)", script):
            assert "==" in spec, (path, name, spec)


@pytest.mark.parametrize("path", WORKFLOWS, ids=[str(p.relative_to(ROOT)) for p in WORKFLOWS])
def test_manifests_record_the_arguments_that_ran(path):
    for line in path.read_text().splitlines():
        if "--seconds" in line and " -- " in line:
            assert '"${argv[@]}"' in line and "raw[@]" not in line, (path, line)
