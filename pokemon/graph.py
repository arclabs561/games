#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flask>=3.1",
#     "networkx>=3.7",
# ]
#
# # graph.py imports pokemon_types from its own folder.
# [tool.ty.environment]
# root = ["."]
# ///
import json
from pathlib import Path

import flask
import networkx as nx
import pokemon_types

graph = nx.Graph()
for attack, defenders in pokemon_types.gen6.items():
    for defend, effect in defenders.items():
        graph.add_node(attack)
        graph.add_edge(attack, defend, weight=effect)

# display/graph.js wants D3 force-layout links: "links", with nodes given by index.
index = {node: i for i, node in enumerate(graph.nodes)}
data = {
    "directed": False,
    "multigraph": False,
    "graph": {},
    "nodes": [{"id": node} for node in graph.nodes],
    "links": [
        {"source": index[u], "target": index[v], "weight": w}
        for u, v, w in graph.edges(data="weight")
    ],
}
DISPLAY = Path(__file__).resolve().parent / "display"
with open(DISPLAY / "graph.json", "w") as f:
    json.dump(data, f)


app = flask.Flask(__name__, static_folder=DISPLAY)


@app.route("/<path:path>")
def static_proxy(path):
    return app.send_static_file(path)


print("http://localhost:8000/index.html")
app.run(port=8000)
