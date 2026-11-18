"""The type chart and the graph built from it."""

import json

from pokemon.graph import DISPLAY, chart_graph
from pokemon.pokemon_types import gen6


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
