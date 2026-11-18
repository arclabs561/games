"""Draw the Gen 6 type chart as a force-directed graph in the browser.

Each arrow runs from an attacking type to a defending type and is colored by the damage
multiplier: 2 (super effective), 1/2 (not very effective) or 0 (no effect). Pairs at the
usual 1x are not drawn.

    uv run python -m pokemon.graph              # writes display/graph.json, serves http://localhost:8000
    uv run python -m pokemon.graph --no-serve   # only writes the JSON
"""

import argparse
import functools
import http.server
import json
from pathlib import Path

from pokemon.pokemon_types import gen6

DISPLAY = Path(__file__).resolve().parent / "display"


def chart_graph(chart: dict[str, dict[str, float]]) -> dict:
    """Nodes and links for D3 from a chart {attacker: {defender: multiplier}}."""
    types = list(chart)
    types += sorted({d for row in chart.values() for d in row} - set(types))
    links = [
        {"source": attacker, "target": defender, "multiplier": multiplier}
        for attacker, row in chart.items()
        for defender, multiplier in row.items()
    ]
    return {"nodes": [{"id": t} for t in types], "links": links}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-serve", action="store_true", help="only write display/graph.json")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    out = DISPLAY / "graph.json"
    out.write_text(json.dumps(chart_graph(gen6), indent=1) + "\n")
    print(f"wrote {out}")
    if args.no_serve:
        return
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DISPLAY)
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"http://localhost:{args.port}/index.html")
        server.serve_forever()


if __name__ == "__main__":
    main()
