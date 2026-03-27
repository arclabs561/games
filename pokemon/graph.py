import json

import flask
import networkx as nx
import pokemon_types
from networkx.readwrite import json_graph

graph = nx.Graph()
for attack, defenders in pokemon_types.gen6.items():
    for defend, effect in defenders.items():
        graph.add_node(attack)
        graph.add_edge(attack, defend, weight=effect)

data = json_graph.node_link_data(graph)
with open("display/graph.json", "w") as f:
    json.dump(data, f)


app = flask.Flask(__name__, static_folder="display")


@app.route("/<path:path>")
def static_proxy(path):
    return app.send_static_file(path)


print("http://localhost:8000/index.html")
app.run(port=8000)
