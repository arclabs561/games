"""Every third-party import is declared: in the game's dependency group, or inline (PEP 723)."""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent
# The reference regex from PEP 723.
BLOCK = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")


def tracked_python_files():
    out = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / line for line in out.stdout.split()]


def script_metadata(path):
    blocks = [m for m in BLOCK.finditer(path.read_text()) if m["type"] == "script"]
    assert len(blocks) <= 1, f"{path}: more than one script block"
    if not blocks:
        return None
    content = "".join(
        line[2:] if line.startswith("# ") else line[1:]
        for line in blocks[0]["content"].splitlines(keepends=True)
    )
    return tomllib.loads(content)


def first_party_names():
    names = {p.stem for p in tracked_python_files()}
    names |= {p.name for p in ROOT.iterdir() if p.is_dir()}
    return names


def third_party_imports(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found - set(sys.stdlib_module_names) - first_party_names() - {"pytest"}


def test_every_script_block_parses():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for path in tracked_python_files():
        meta = script_metadata(path)
        if meta is not None:
            assert meta["requires-python"] == project["requires-python"], path
            assert isinstance(meta.get("dependencies", []), list), path


# Import names that differ from the distribution names in pyproject.toml.
DISTRIBUTION = {"sklearn": "scikit-learn", "z3": "z3-solver"}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def group_packages(group):
    """Distribution names in one dependency group of pyproject.toml."""
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]
    specs = {normalize(k): v for k, v in groups.items()}.get(group, [])
    return {normalize(re.split(r"[<>=!~;\[ ]", spec)[0]) for spec in specs}


def test_third_party_imports_are_declared():
    missing = {}
    for path in tracked_python_files():
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "tests":
            continue
        imports = {normalize(DISTRIBUTION.get(m, m)) for m in third_party_imports(path)}
        meta = script_metadata(path)
        if meta is not None:
            declared = {normalize(re.split(r"[<>=!~;\[ ]", d)[0]) for d in meta["dependencies"]}
        else:
            declared = group_packages(normalize(rel.parts[0]))
        if imports - declared:
            missing[str(rel)] = sorted(imports - declared)
    assert not missing, f"declare these in the folder's dependency group or inline: {missing}"


def test_scripts_with_a_shebang_are_executable_and_only_they():
    out = subprocess.run(
        ["git", "ls-files", "-s", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    for line in out.stdout.splitlines():
        mode, path = line.split()[0], line.split()[-1]
        has_shebang = (ROOT / path).read_text().startswith("#!")
        assert has_shebang == (mode == "100755"), path
