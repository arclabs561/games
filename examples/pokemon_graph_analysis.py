# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "networkx>=3.2",
#     "matplotlib>=3.8",
#     "scipy>=1.12",
# ]
# ///
"""Pokemon type effectiveness as a directed graph.

Builds the Gen 6 type chart as a weighted digraph and computes graph-theoretic
properties: PageRank (offensive dominance), betweenness centrality (bridging
types), and strongly connected components.

graphops equivalents:
  - Graph construction: graphops::DiGraph with typed edge weights
  - PageRank: graphops::pagerank (power iteration on sparse adjacency)
  - Betweenness centrality: graphops::betweenness_centrality (Brandes' algorithm)
  - SCCs: graphops::tarjan_scc (Tarjan's algorithm, O(V+E))
  - Edge filtering by weight: graphops edge predicates / subgraph views
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import networkx as nx

# Gen 6 type chart. Keys are attacking types, values map defending types to
# effectiveness multipliers. Omitted pairs have 1.0x (normal) effectiveness.
GEN6: dict[str, dict[str, float]] = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "dragon": 0.5,
        "steel": 2.0,
    },
    "water": {
        "fire": 2.0,
        "water": 0.5,
        "grass": 0.5,
        "ground": 2.0,
        "rock": 2.0,
        "dragon": 0.5,
    },
    "electric": {
        "water": 2.0,
        "electric": 0.5,
        "grass": 0.5,
        "ground": 0.0,
        "flying": 2.0,
        "dragon": 0.5,
    },
    "grass": {
        "fire": 0.5,
        "water": 2.0,
        "grass": 0.5,
        "poison": 0.5,
        "ground": 2.0,
        "flying": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "dragon": 0.5,
        "steel": 0.5,
    },
    "ice": {
        "fire": 0.5,
        "water": 0.5,
        "grass": 2.0,
        "ice": 0.5,
        "ground": 2.0,
        "flying": 2.0,
        "dragon": 2.0,
        "steel": 0.5,
    },
    "fighting": {
        "normal": 2.0,
        "ice": 2.0,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 0.5,
        "bug": 0.5,
        "rock": 2.0,
        "ghost": 0.0,
        "dark": 2.0,
        "steel": 2.0,
        "fairy": 0.5,
    },
    "poison": {
        "grass": 2.0,
        "poison": 0.5,
        "ground": 0.5,
        "rock": 0.5,
        "ghost": 0.5,
        "steel": 0.0,
        "fairy": 2.0,
    },
    "ground": {
        "fire": 2.0,
        "electric": 2.0,
        "grass": 0.5,
        "poison": 2.0,
        "flying": 0.0,
        "bug": 0.5,
        "rock": 2.0,
        "steel": 2.0,
    },
    "flying": {
        "electric": 0.5,
        "grass": 2.0,
        "fighting": 2.0,
        "bug": 2.0,
        "rock": 0.5,
        "steel": 0.5,
    },
    "psychic": {
        "fighting": 2.0,
        "poison": 2.0,
        "psychic": 0.5,
        "dark": 0.0,
        "steel": 0.5,
    },
    "bug": {
        "fire": 0.5,
        "grass": 2.0,
        "fighting": 0.5,
        "poison": 0.5,
        "flying": 0.5,
        "psychic": 2.0,
        "ghost": 0.5,
        "dark": 2.0,
        "steel": 0.5,
        "fairy": 0.5,
    },
    "rock": {
        "fire": 2.0,
        "ice": 2.0,
        "fighting": 0.5,
        "ground": 0.5,
        "flying": 2.0,
        "bug": 2.0,
        "steel": 0.5,
    },
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {
        "fighting": 0.5,
        "psychic": 2.0,
        "ghost": 2.0,
        "dark": 0.5,
        "fairy": 0.5,
    },
    "steel": {
        "fire": 0.5,
        "water": 0.5,
        "electric": 0.5,
        "ice": 2.0,
        "rock": 2.0,
        "steel": 0.5,
        "fairy": 2.0,
    },
    "fairy": {
        "fire": 0.5,
        "fighting": 2.0,
        "poison": 0.5,
        "dragon": 2.0,
        "dark": 2.0,
        "steel": 0.5,
    },
}

ALL_TYPES = sorted(GEN6.keys())


def build_type_graph() -> nx.DiGraph:
    """Build a directed graph where edge (A -> B) has weight = effectiveness of
    A attacking B. Only non-1.0x edges are stored (super-effective, not-very,
    immune)."""
    g = nx.DiGraph()
    g.add_nodes_from(ALL_TYPES)
    for attacker, matchups in GEN6.items():
        for defender, mult in matchups.items():
            g.add_edge(attacker, defender, weight=mult)
    return g


def super_effective_subgraph(g: nx.DiGraph) -> nx.DiGraph:
    """Subgraph containing only super-effective (2x) edges."""
    edges = [(u, v) for u, v, d in g.edges(data=True) if d["weight"] == 2.0]
    return g.edge_subgraph(edges).copy()


def immunity_pairs(g: nx.DiGraph) -> list[tuple[str, str]]:
    """Pairs where the attacker deals 0x damage."""
    return [(u, v) for u, v, d in g.edges(data=True) if d["weight"] == 0.0]


def print_analysis(g: nx.DiGraph) -> None:
    se = super_effective_subgraph(g)

    # PageRank on super-effective graph: types that are targeted by many
    # super-effective attacks rank higher (they're "central" defensively).
    pr = nx.pagerank(se, alpha=0.85)
    pr_sorted = sorted(pr.items(), key=lambda x: -x[1])

    print("=== PageRank on super-effective subgraph ===")
    print("(Higher = more central as a target of super-effective attacks)")
    for typ, score in pr_sorted:
        bar = "#" * int(score * 200)
        print(f"  {typ:<10s} {score:.4f}  {bar}")

    # Betweenness centrality: types that sit on many shortest paths between
    # other types in the super-effective graph. These are "bridge" types.
    bc = nx.betweenness_centrality(se)
    bc_sorted = sorted(bc.items(), key=lambda x: -x[1])

    print("\n=== Betweenness centrality (super-effective edges) ===")
    print("(Higher = more bridging role in type matchup chains)")
    for typ, score in bc_sorted[:8]:
        print(f"  {typ:<10s} {score:.4f}")

    # Strongly connected components: groups of types where each type can
    # reach every other type via chains of super-effective hits.
    sccs = list(nx.strongly_connected_components(se))
    sccs_sorted = sorted(sccs, key=len, reverse=True)

    print(f"\n=== Strongly connected components ({len(sccs_sorted)} total) ===")
    for i, scc in enumerate(sccs_sorted[:5]):
        print(f"  SCC {i}: {sorted(scc)}")

    # Offensive and defensive summaries.
    print("\n=== Offensive power (super-effective hit count) ===")
    out_deg = sorted(se.out_degree(), key=lambda x: -x[1])
    for typ, deg in out_deg[:6]:
        targets = [v for _, v in se.out_edges(typ)]
        print(f"  {typ:<10s} hits {deg} types SE: {', '.join(sorted(targets))}")

    print("\n=== Defensive vulnerability (super-effective weakness count) ===")
    in_deg = sorted(se.in_degree(), key=lambda x: -x[1])
    for typ, deg in in_deg[:6]:
        sources = [u for u, _ in se.in_edges(typ)]
        print(f"  {typ:<10s} weak to {deg} types: {', '.join(sorted(sources))}")

    print("\n=== Immunities (0x damage) ===")
    for attacker, defender in immunity_pairs(g):
        print(f"  {attacker:<10s} -> {defender} (immune)")


def plot_type_graph(g: nx.DiGraph) -> None:
    se = super_effective_subgraph(g)
    pr = nx.pagerank(se, alpha=0.85)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    pos = nx.spring_layout(se, seed=42, k=2.0)

    node_sizes = [pr.get(n, 0.01) * 8000 for n in se.nodes()]
    nx.draw_networkx_nodes(se, pos, ax=ax, node_size=node_sizes, node_color="lightblue")
    nx.draw_networkx_labels(se, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        se,
        pos,
        ax=ax,
        edge_color="gray",
        alpha=0.4,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title("Pokemon type super-effective graph (node size = PageRank)")
    ax.axis("off")
    fig.tight_layout()
    plt.savefig("pokemon_type_graph.png", dpi=150)
    print("\nSaved pokemon_type_graph.png")
    plt.close()


def main() -> None:
    g = build_type_graph()
    print(f"Type graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")
    print_analysis(g)

    if "--plot" in sys.argv:
        plot_type_graph(g)


if __name__ == "__main__":
    main()
