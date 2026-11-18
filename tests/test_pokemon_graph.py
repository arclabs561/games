"""The type chart, the graph built from it, and the copy in examples/."""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "pokemon"))

from graph import DISPLAY, chart_graph  # noqa: E402
from pokemon_types import gen6  # noqa: E402


def test_the_eight_immunities():
    immune = {(a, d) for a, row in gen6.items() for d, m in row.items() if m == 0}
    assert immune == {
        ("normal", "ghost"),
        ("fighting", "ghost"),
        ("poison", "steel"),
        ("ground", "flying"),
        ("ghost", "normal"),
        ("electric", "ground"),
        ("psychic", "dark"),
        ("dragon", "fairy"),
    }


def test_graph_keeps_every_direction():
    graph = chart_graph(gen6)
    assert len(graph["nodes"]) == 18
    assert len(graph["links"]) == sum(len(row) for row in gen6.values()) == 120
    multipliers = {(e["source"], e["target"]): e["multiplier"] for e in graph["links"]}
    assert multipliers["fire", "rock"] == 0.5
    assert multipliers["rock", "fire"] == 2.0
    assert multipliers["psychic", "dark"] == 0.0
    assert multipliers["dark", "psychic"] == 2.0


def test_committed_json_is_current():
    assert json.loads((DISPLAY / "graph.json").read_text()) == chart_graph(gen6)


def test_examples_copy_matches():
    tree = ast.parse((ROOT / "examples" / "pokemon_graph_analysis.py").read_text())
    (copy,) = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "GEN6"
        and node.value is not None
    ]
    assert copy == gen6
