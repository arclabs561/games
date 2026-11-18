"""Scripts that import third-party packages declare them inline (PEP 723)."""

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


def test_third_party_imports_are_declared():
    missing = {}
    for path in tracked_python_files():
        imports = third_party_imports(path)
        if imports and script_metadata(path) is None:
            missing[str(path.relative_to(ROOT))] = sorted(imports)
    assert not missing, f"add a PEP 723 block to: {missing}"
