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


def dynamic_import(node):
    """The module named by importlib.import_module("x") or __import__("x"), else None."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    f, arg = node.func, node.args[0]
    named = (isinstance(f, ast.Name) and f.id in ("__import__", "import_module")) or (
        isinstance(f, ast.Attribute) and f.attr == "import_module"
    )
    if named and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def third_party_imports(path, dynamic=False):
    """Top-level names of the import statements in a file, or with dynamic=True, of the
    modules it imports by name at run time."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if dynamic:
            name = dynamic_import(node)
            if name is not None:
                found.add(name.split(".")[0])
        elif isinstance(node, ast.Import):
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


def dependency_groups():
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]
    return {normalize(k): v for k, v in groups.items()}


def group_packages(group):
    """Distribution names in one dependency group of pyproject.toml."""
    specs = dependency_groups().get(group, [])
    return {normalize(re.split(r"[<>=!~;\[ ]", s)[0]) for s in specs if isinstance(s, str)}


def including_groups(group):
    """The group and every group that includes it, such as codenames-gpu for codenames."""
    groups = dependency_groups()

    def includes(name, seen=()):
        subs = [normalize(s["include-group"]) for s in groups.get(name, []) if isinstance(s, dict)]
        return group in subs or any(includes(s, (*seen, name)) for s in subs if s not in seen)

    return {group} | {name for name in groups if includes(name)}


# Imported by name at run time and installed by hand, not by a dependency group, with the
# place that says how. torch and maia3 (AGPL-3.0, from git) are installed inside the chess
# job only; nltk is added with `--with nltk` for one step of the Codenames bench.
INSTALLED_BY_HAND = {
    "chess_ratings/maia.py": {"torch", "maia3"},  # chess_ratings/README.md, Usage
    "codenames/bench_data.py": {"nltk"},  # codenames/results/bench.md
}


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


def test_imports_by_name_are_declared():
    """importlib.import_module("x") and __import__("x") need a declaration too: in the
    folder's group or in a group that includes it (a GPU group), or in INSTALLED_BY_HAND."""
    missing = {}
    for path in tracked_python_files():
        rel = path.relative_to(ROOT)
        if rel.parts[0] == "tests":
            continue
        imports = {normalize(DISTRIBUTION.get(m, m)) for m in third_party_imports(path, True)}
        meta = script_metadata(path)
        if meta is not None:
            declared = {normalize(re.split(r"[<>=!~;\[ ]", d)[0]) for d in meta["dependencies"]}
        else:
            groups = including_groups(normalize(rel.parts[0]))
            declared = set().union(*(group_packages(g) for g in groups))
        declared |= INSTALLED_BY_HAND.get(rel.as_posix(), set())
        if imports - declared:
            missing[str(rel)] = sorted(imports - declared)
    assert not missing, f"declare these in the folder's dependency group: {missing}"


def test_imports_by_name_are_found():
    tree = ast.parse(
        "import importlib\nfrom importlib import import_module\n"
        "a = importlib.import_module('torch.nn')\nb = __import__('huggingface_hub')\n"
        "c = import_module('peft')\nd = importlib.import_module(name)\n"
    )
    found = {dynamic_import(n) for n in ast.walk(tree)} - {None}
    assert found == {"torch.nn", "huggingface_hub", "peft"}


def test_scripts_with_a_shebang_are_executable_and_only_they():
    out = subprocess.run(
        ["git", "ls-files", "-s", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    for line in out.stdout.splitlines():
        mode, path = line.split()[0], line.split()[-1]
        has_shebang = (ROOT / path).read_text().startswith("#!")
        assert has_shebang == (mode == "100755"), path
