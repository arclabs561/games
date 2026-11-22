"""Find EV Nova's best trade routes from a ConText export.

    uv run python -m ev_nova.study --evnova NovaConText.txt

Every rate is exact (a fraction of credits per ton), and every best route comes with potentials
that are checked to prove that no route does better.
"""

import argparse
import os
import time
from fractions import Fraction
from pathlib import Path

from . import naev
from .guides import GUIDES
from .routes import (
    Galaxy,
    articulation_points,
    best_round_trips,
    best_route,
    best_trade,
    betweenness,
    capital_trade,
    check_certificate,
    components,
    eccentricities,
    fuel_trade_graph,
    jump_table,
    karp_rate,
    landing_envelope,
    route_parts,
    route_rate,
    trade_graph,
    undirected,
)

LANDINGS = (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4), Fraction(8))


def show(x: Fraction) -> str:
    """A fraction as itself and as a decimal."""
    return str(x) if x.denominator == 1 else f"{x} = {float(x):.2f}"


def intra_system_spread(g: Galaxy) -> tuple[int, str, str, str | None]:
    """The largest gain per ton between two stops of one system, needing no jump."""
    at: dict[str, list[str]] = {}
    for stop, sys in g.stop_system.items():
        at.setdefault(sys, []).append(stop)
    best = (0, "", "", None)
    for stops in at.values():
        for a in sorted(stops):
            for b in sorted(stops):
                gain, c = best_trade(g.prices[a], g.prices[b])
                if gain > best[0]:
                    best = (gain, a, b, c)
    return best


def describe(g: Galaxy, route: tuple[str, ...]) -> str:
    """A route as its stops, with what each leg carries and its jumps."""
    table = jump_table(g.links)
    parts = []
    for a, b in zip(route, route[1:] + route[:1], strict=True):
        gain, c = best_trade(g.prices[a], g.prices[b])
        d = table[g.stop_system[a]][g.stop_system[b]]
        parts.append(f"{a} -> {b}: {c or 'nothing'} +{gain} ({d} jumps)")
    return "; ".join(parts)


def solve(edges, label: str) -> None:
    best = best_route(edges)
    assert best is not None and check_certificate(edges, best), label
    print(f"  {label}: {show(best.rate)} per ton, route {' -> '.join(best.route)}")


def structure(g: Galaxy) -> None:
    """Size, diameter, chokepoints and betweenness of the jump graph, jumps taken both ways."""
    adj = undirected(g.links)
    parts = components(adj)
    main = parts[0]
    sub = {s: {t for t in adj[s] if t in main} for s in main}
    ecc = eccentricities(sub)
    cut = articulation_points(sub)
    leaves = sum(1 for s in sub if len(sub[s]) == 1)
    between = betweenness(sub)
    pairs = len(main) * (len(main) - 1) // 2
    links = sum(len(v) for v in sub.values()) // 2
    print(f"  {len(adj)} systems, {sum(len(v) for v in adj.values()) // 2} links; ", end="")
    print(f"{len(parts)} connected parts, the largest with {len(main)} systems and {links} links")
    print(f"  diameter {max(ecc.values())} jumps, radius {min(ecc.values())}; centers: ", end="")
    print(", ".join(sorted(s for s in ecc if ecc[s] == min(ecc.values()))))
    print(f"  {len(cut)} chokepoints (systems whose loss splits the map); {leaves} dead ends")
    top = sorted(between, key=lambda s: (-between[s], s))[:5]
    print("  highest betweenness (share of all shortest paths between pairs of systems):")
    for s in top:
        print(f"    {s}: {float(between[s] / pairs):.3f}{'  (chokepoint)' if s in cut else ''}")


def routes_report(g: Galaxy) -> None:
    """The best round trip and the best route under each cost of landing."""
    trips = best_round_trips(g, 3)
    print("  best round trips between two systems, gain per ton per jump:")
    for rate, a, b in trips:
        print(f"    {show(rate)}: {describe(g, (a, b))}")
    spread, a, b, c = intra_system_spread(g)
    print(f"  largest gain within one system: {c} {a} -> {b}, +{spread} per ton, no jump;")
    print("  so if landing costs nothing, the gain per jump has no upper bound.")
    t0 = time.time()
    single = trade_graph(g, 0)
    best = best_route(single)
    assert best is not None and check_certificate(single, best)
    karp0 = karp_rate(g, landing_step=False)
    print(f"  one landing per visit to a system, landing free: {show(best.rate)} per ton per")
    print(
        f"    jump (Howard, certified; Karp on (system, cargo) states agrees: {karp0 == best.rate})"
    )
    print(f"    {describe(g, best.route)}")
    for landing in LANDINGS:
        edges = trade_graph(g, landing)
        best = best_route(edges)
        assert best is not None and check_certificate(edges, best)
        extra = ""
        if landing == 1:
            extra = f" (Karp agrees: {karp_rate(g, landing_step=True) == best.rate})"
        print(f"  landing costs {landing} jump: {show(best.rate)} per ton per unit{extra}")
        print(f"    {describe(g, best.route)}")
    print("  the best route as the cost of a landing goes from 1/10 to 10 jumps (exact):")
    for start, route in landing_envelope(g, Fraction(1, 10), Fraction(10)):
        gain, jumps, stops = route_parts(g, route)
        print(f"    from {show(start)}: {' -> '.join(route)}, rate {gain} / ({jumps} + {stops} x)")
    print(f"  ({time.time() - t0:.0f} s)")


def fuel_report(g: Galaxy, landing: Fraction, tanks: tuple[int, ...]) -> None:
    """The best route when the tank holds `tank` jumps and refueling costs a landing."""
    for tank in tanks:
        edges = fuel_trade_graph(g, tank, landing)
        best = best_route(edges)
        if best is None:
            print(f"  tank of {tank} jumps: no repeatable trade route")
            continue
        assert check_certificate(edges, best)
        print(f"  tank of {tank} jumps: {show(best.rate)}, route {' -> '.join(best.route)}")


def capital_report(g: Galaxy, landing: Fraction, cases) -> None:
    """The best route when one load may cost at most a given number of credits."""
    for cargo, credits in cases:
        edges = trade_graph(
            g, landing, gain=lambda a, b, c=cargo, k=credits: capital_trade(a, b, c, k)[0]
        )
        best = best_route(edges)
        assert best is not None and check_certificate(edges, best)
        legs = [
            capital_trade(g.prices[a], g.prices[b], cargo, credits)[1]
            for a, b in zip(best.route, best.route[1:] + best.route[:1], strict=True)
        ]
        print(
            f"  {cargo} tons, {credits:,} credits: {show(best.rate)} credits per unit, "
            f"route {' -> '.join(best.route)} carrying {', '.join(c or 'nothing' for c in legs)}"
        )


def guide_report(g: Galaxy, levels: dict[str, dict[str, int]]) -> None:
    """Each published route against the data: price levels, jumps, and its rate."""
    names = {1: "low", 2: "medium", 4: "high"}
    table = jump_table(g.links)
    best1 = best_route(trade_graph(g, 1))
    assert best1 is not None
    print(f"  (best route when landing costs 1 jump: {show(best1.rate)})")
    edges = trade_graph(g, 1)
    for guide in GUIDES:
        problems = []
        legs = list(zip(guide.stops, guide.stops[1:] + guide.stops[:1], strict=True))
        for (a, b), claims in zip(legs, guide.legs, strict=True):
            for c, at_a, at_b in claims:
                got = (
                    names.get(levels[a].get(c, 0), "none"),
                    names.get(levels[b].get(c, 0), "none"),
                )
                if got != (at_a, at_b):
                    problems.append(f"{c} {a} -> {b} is {got[0]} -> {got[1]}, not {at_a} -> {at_b}")
        hops = [table[g.stop_system[a]][g.stop_system[b]] for a, b in legs]
        jumps = sum(hops)
        if tuple(hops) != guide.jumps:
            problems.append(f"jumps per leg are {hops}, the source says {list(guide.jumps)}")
        rate = route_rate(edges, guide.stops)
        gain = sum(best_trade(g.prices[a], g.prices[b])[0] for a, b in legs)
        per_jump = "no jumps" if jumps == 0 else f"{show(Fraction(gain, jumps))} per jump"
        print(
            f"  {guide.name} [{guide.source}]: +{gain} per ton in {jumps} jumps, {per_jump}; "
            f"{show(rate)} with landing 1, {float(rate / best1.rate):.0%} of the best"
        )
        for p in problems:
            print(f"      differs: {p}")


def naev_drift(
    data: dict, route: tuple[str, ...], step: Fraction = Fraction(1, 2), span: int = 20_000
) -> tuple[int, int]:
    """Least and greatest gain per ton of one pass, with prices at one moment, over a time grid.

    Sampled: times 0, `step`, ..., `span` periods. Ignores the time spent flying the route.
    """
    gains = []
    for k in range(int(span / step) + 1):
        t = float(k * step)
        prices = {
            s: {
                c: naev.price_at(naev.Price(**dict(zip(naev.PRICE_FIELDS, p, strict=True))), t)
                for c, p in data["spobs"][s]["prices"].items()
            }
            for s in route
        }
        gains.append(
            sum(
                best_trade(prices[a], prices[b])[0]
                for a, b in zip(route, route[1:] + route[:1], strict=True)
            )
        )
    return min(gains), max(gains)


def archived_study() -> None:
    data = naev.load()
    print(f"Naev, data from commit {data['source']['commit'][:12]}")
    g = naev.galaxy(data, hidden=False)
    print(f"  {len(g.stop_system)} stops with a commodity exchange; hidden jumps left out")
    structure(g)
    routes_report(g)
    print("  with hidden jumps too:")
    solve(trade_graph(naev.galaxy(data, hidden=True), 1), "landing costs 1 jump")
    print("  fuel, landing costs 1 jump, refueling costs another landing:")
    fuel_report(g, Fraction(1), (1, 2))
    print("  credits limit the load, landing costs 1 jump:")
    capital_report(g, Fraction(1), [(50, 2_000), (50, 10_000), (50, 100_000)])
    best = best_route(trade_graph(g, 1))
    assert best is not None
    low, high = naev_drift(data, best.route)
    print(f"  prices drift: one pass of the best route gains {low} to {high} per ton at different")
    print(
        "  times (sampled every half period over 20,000 periods); at mean prices it gains", end=""
    )
    print(
        f" {sum(best_trade(g.prices[a], g.prices[b])[0] for a, b in zip(best.route, best.route[1:] + best.route[:1], strict=True))}"
    )


def nova_study(path: Path) -> None:
    from . import evnova

    tables = evnova.read_context(path)
    g, levels = evnova.galaxy(tables)
    print(f"\nEV Nova, from {path.name}, at the start of a game")
    print(f"  {len(g.stop_system)} stops with a commodity exchange")
    structure(g)
    routes_report(g)
    print("  fuel, landing costs 1 jump, refueling costs another landing:")
    fuel_report(g, Fraction(1), (1, 2))
    print("  credits limit the load (a new player has a 10-ton Shuttle and 25,000 credits):")
    capital_report(g, Fraction(1), [(10, 25_000), (125, 25_000), (125, 100_000)])
    junk = evnova.junk_markets(tables, g)
    gj = evnova.with_junk(g, junk)
    print("  with special commodities, priced 80% of base where sold and 125% where bought:")
    solve(trade_graph(gj, 1), "landing costs 1 jump")
    solve(trade_graph(gj, 0), "one landing per visit, landing free")
    print("  published routes, checked:")
    guide_report(g, levels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evnova",
        type=Path,
        default=os.getenv("EV_NOVA_CONTEXT"),
        help="ConText export (defaults to EV_NOVA_CONTEXT)",
    )
    parser.add_argument("--naev", action="store_true", help="also run the archived comparison")
    args = parser.parse_args()
    if not args.evnova and not args.naev:
        parser.error("supply --evnova PATH or set EV_NOVA_CONTEXT to your ConText export")
    if args.evnova:
        nova_study(args.evnova)
    if args.naev:
        archived_study()


if __name__ == "__main__":
    main()
