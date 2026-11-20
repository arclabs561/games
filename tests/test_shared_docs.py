"""The shared sections of each README are what tools/sync_docs.py writes from the game's
docs/*.md fragments and results.json, and every section a paper includes exists."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("sync_docs", ROOT / "tools" / "sync_docs.py")
assert spec and spec.loader
sync_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_docs)


@pytest.mark.parametrize("game", sync_docs.games())
def test_readme_shared_sections_are_current(game):
    assert (ROOT / game / "README.md").read_text() == sync_docs.synced(game), (
        f"run uv run tools/sync_docs.py {game}"
    )


def test_games_with_shared_sections():
    assert {"cannonballs", "erase", "peggle", "tenzi"} <= set(sync_docs.games())


def test_placeholder_formats():
    data = {"a": {"b c": [1, 24], "n": 12362536, "x": 0.71, "m": 14716329, "k": 7}}
    f = lambda t: sync_docs.fill(t, data, "test")  # noqa: E731
    assert f("{{a.n}} of {{a.b c:and}}") == "12,362,536 of 1 and 24"
    assert f("{{a.b c:or}}; {{a.b c:list}}; {{a.b c:cards}}") == "1 or 24; 1, 24; 1 24"
    assert (
        f("{{a.x:.4f}} {{a.m:million}} {{a.k:words}} {{a.b c.1}}") == "0.7100 14.7 million seven 24"
    )
    with pytest.raises(SystemExit):
        f("{{a.missing}}")


def test_papers_include_only_sections_the_readme_has():
    for typ in [*ROOT.glob("*/*.typ"), *ROOT.glob("*/paper/*.typ")]:
        if typ.parent.name == "tools":
            continue
        text = typ.read_text()
        names = re.findall(r'shared\(readme, "([\w-]+)"', text)
        if not names:
            continue
        m = re.search(r'#let readme = read\("([^"]+)"\)', text)
        assert m, typ
        readme = (typ.parent / m.group(1)).read_text()
        for name in names:
            assert f"<!-- shared:{name} -->" in readme, (typ, name)
