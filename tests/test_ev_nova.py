"""Tests for ev_nova: small hand-checked cases, and fast methods against brute force."""

import itertools
import math
import os
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from ev_nova import evnova, naev
from ev_nova.guides import GUIDES
from ev_nova.routes import (
    Best,
    Galaxy,
    articulation_points,
    best_round_trips,
    best_route,
    best_trade,
    betweenness,
    brute_force_betweenness,
    brute_force_mean,
    brute_force_route,
    capital_trade,
    check_certificate,
    components,
    eccentricities,
    fuel_jumps,
    jumps_from,
    karp,
    karp_rate,
    landing_envelope,
    return_jumps,
    route_rate,
    strongly_connected,
    trade_graph,
    undirected,
)

# A hand-sized galaxy. A - B - C in a line, and D off B. Stops:
#   a1, a2 in A; b in B; c in C. D has no stop.
TOY = Galaxy(
    links={"A": ("B",), "B": ("A", "C", "D"), "C": ("B",), "D": ("B",)},
    stop_system={"a1": "A", "a2": "A", "b": "B", "c": "C"},
    prices={
        "a1": {"food": 10, "ore": 50},
        "a2": {"food": 12, "ore": 44},
        "b": {"food": 30, "ore": 20},
        "c": {"food": 70, "ore": 15, "gold": 100},
    },
    refuel_systems=frozenset({"A", "C"}),
)


def test_jumps():
    assert jumps_from(TOY.links, "A") == {"A": 0, "B": 1, "C": 2, "D": 2}
    assert return_jumps(TOY.links, "A") == 2  # A -> B -> A
    assert return_jumps({"X": ()}, "X") is None


def test_best_trade():
    assert best_trade(TOY.prices["a1"], TOY.prices["b"]) == (20, "food")  # 30 - 10
    assert best_trade(TOY.prices["b"], TOY.prices["a1"]) == (30, "ore")  # 50 - 20
    assert best_trade(TOY.prices["c"], TOY.prices["b"]) == (5, "ore")  # gold is not sold at b
    assert best_trade({"x": 5, "y": 3}, {"x": 8, "y": 6}) == (3, "x")  # a tie goes to x
    assert best_trade({"x": 5}, {"x": 5}) == (0, None)


def test_capital_trade():
    # 100 credits buy 10 tons of food at 10 (gain 20 each = 200) or 2 tons of ore at 50.
    assert capital_trade({"food": 10, "ore": 50}, {"food": 30, "ore": 200}, 10, 100) == (300, "ore")
    assert capital_trade({"food": 10, "ore": 50}, {"food": 30, "ore": 120}, 10, 100) == (
        200,
        "food",
    )
    assert capital_trade({"food": 10}, {"food": 30}, 10, 10**9) == (200, "food")


def test_toy_routes_by_hand():
    # a1 <-> b: food +20 out, ore +30 back, one jump each way: 50 per 2 jumps.
    # a1 <-> c: food +60 out, ore +35 back, two jumps each way: 95 per 4 jumps.
    # a1 -> c -> b -> a1: food +60 (2 jumps), ore +5 (1), ore +30 (1): 95 per 4 jumps.
    edges = trade_graph(TOY, 0)
    assert route_rate(edges, ("a1", "b")) == 25
    assert route_rate(edges, ("a1", "c")) == Fraction(95, 4)
    # a2 -> a1 within system A: ore +6, but leaving and coming back costs two jumps.
    assert edges["a2", "a1"] == (6, 2)
    best = best_route(edges)
    assert best is not None and best.rate == 25 and best.route == ("a1", "b")
    assert check_certificate(edges, best)
    # Landing costs one jump: a1 <-> b is 50 / 4; a1 <-> c is 95 / 6; a1 -> a2 -> a1 is
    # ore +6 and food +2 for two landings, 8 / 2 = 4.
    edges1 = trade_graph(TOY, 1)
    assert route_rate(edges1, ("a1", "a2")) == 4
    best1 = best_route(edges1)
    assert best1 is not None and best1.rate == Fraction(95, 6) and best1.route == ("a1", "c")
    # Landing costs 1/10 jump: a1 <-> a2 is 8 / (2/10) = 40, beating every route with jumps.
    best_cheap = best_route(trade_graph(TOY, Fraction(1, 10)))
    assert best_cheap is not None and best_cheap.rate == 40


def test_certificate_rejects_wrong_answers():
    edges = trade_graph(TOY, 1)
    best = best_route(edges)
    assert best is not None
    low = Best(Fraction(10), ("a1", "b"), best.rates, best.potentials)
    assert not check_certificate(edges, low)  # the route does not attain 10
    worse = Best(
        Fraction(25, 2), ("a1", "b"), dict.fromkeys(best.rates, Fraction(25, 2)), best.potentials
    )
    assert not check_certificate(edges, worse)  # a1 <-> c beats 25/2
    flat = Best(best.rate, best.route, best.rates, dict.fromkeys(best.potentials, Fraction(0)))
    assert not check_certificate(edges, flat)


def random_galaxy(rng: random.Random, systems: int, stops: int, commodities: int) -> Galaxy:
    """A random connected galaxy, with a few one-way links and random prices."""
    names = [f"S{i}" for i in range(systems)]
    links: dict[str, set[str]] = {s: set() for s in names}
    for i in range(1, systems):
        j = rng.randrange(i)
        links[names[i]].add(names[j])
        links[names[j]].add(names[i])
    for _ in range(rng.randrange(3)):
        a, b = rng.sample(names, 2)
        links[a].add(b)
    where = {f"p{k}": rng.choice(names) for k in range(stops)}
    goods = [f"c{k}" for k in range(commodities)]
    prices = {p: {c: rng.randrange(1, 30) for c in goods if rng.random() < 0.7} for p in where}
    return Galaxy({s: tuple(sorted(v)) for s, v in links.items()}, where, prices)


@pytest.mark.parametrize("seed", range(4))
def test_howard_karp_and_brute_force_agree(seed):
    """Howard's method, its certificate, brute force and Karp on 400 random galaxies."""
    rng = random.Random(seed)
    for _ in range(100):
        g = random_galaxy(rng, rng.randrange(2, 7), rng.randrange(2, 7), rng.randrange(1, 4))
        for landing in (0, Fraction(1, 3), 1, 2):
            edges = trade_graph(g, landing)
            best = best_route(edges)
            brute = brute_force_route(edges)
            assert (best is None) == (brute is None)
            if best is None or brute is None:
                continue
            assert best.rate == brute[0]
            assert check_certificate(edges, best)
            if landing == 0:
                assert karp_rate(g, landing_step=False) == best.rate
            if landing == 1:
                assert karp_rate(g, landing_step=True) == best.rate


def test_landing_envelope():
    # a1 <-> a2 gains 8 with no jump (rate 8 / 2x); a1 <-> c gains 95 in 4 jumps (95 / (4 + 2x)).
    # They tie where 8 (4 + 2x) = 190 x, at x = 16/87.
    assert landing_envelope(TOY, Fraction(1, 100), Fraction(8)) == [
        (Fraction(1, 100), ("a1", "a2")),
        (Fraction(16, 87), ("a1", "c")),
    ]
    rng = random.Random(5)
    for _ in range(100):
        g = random_galaxy(rng, rng.randrange(2, 6), rng.randrange(2, 6), rng.randrange(1, 4))
        if best_route(trade_graph(g, 1)) is None:
            continue
        envelope = landing_envelope(g, Fraction(1, 10), Fraction(10))
        for _ in range(10):
            x = Fraction(rng.randrange(10, 1001), 100)
            route = [r for start, r in envelope if start <= x][-1]
            edges = trade_graph(g, x)
            brute = brute_force_route(edges)
            assert brute is not None and route_rate(edges, route) == brute[0]


def test_karp_against_brute_force():
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randrange(1, 7)
        edges = [
            (a, b, rng.randrange(-9, 10)) for a in range(n) for b in range(n) if rng.random() < 0.4
        ]
        if not edges:
            continue
        arr = np.array(edges, dtype=np.int64)
        assert karp(n, arr[:, 0], arr[:, 1], arr[:, 2]) == brute_force_mean(edges)


def test_karp_by_hand():
    # Cycle 0 -> 1 -> 0 with gains 3 and 5 (mean 4); self-loop at 2 with gain 7 (mean 7).
    src, dst, gain = (np.array(x) for x in ([0, 1, 1, 2], [1, 0, 2, 2], [3, 5, -100, 7]))
    assert karp(3, src, dst, gain) == 7
    assert karp(2, np.array([0]), np.array([1]), np.array([5])) is None  # no cycle


def test_strongly_connected():
    parts = strongly_connected("abcd", [("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"), ("d", "c")])
    assert sorted(parts) == [["a", "b"], ["c", "d"]]


def test_fuel_jumps():
    # A line X0 - X1 - X2 - X3 - X4 with fuel only at X0 and X2.
    line = {f"X{i}": tuple(f"X{j}" for j in (i - 1, i + 1) if 0 <= j <= 4) for i in range(5)}
    refuel = frozenset({"X0", "X2"})
    two = fuel_jumps(line, refuel, 2, Fraction(1), "X0")
    assert two["X2"] == (2, 0)
    assert two["X4"] == (5, 1)  # 4 jumps and one refueling stop at X2
    one = fuel_jumps(line, refuel, 1, Fraction(1), "X0")
    assert "X2" not in one  # X1 has no fuel, so a one-jump tank never gets past it
    big = fuel_jumps(line, refuel, 9, Fraction(1), "X0")
    assert {s: c for s, (c, _) in big.items()} == jumps_from(line, "X0")


def small_graphs(rng: random.Random, count: int):
    for _ in range(count):
        n = rng.randrange(1, 8)
        adj = {i: set() for i in range(n)}
        for a, b in itertools.combinations(range(n), 2):
            if rng.random() < 0.35:
                adj[a].add(b)
                adj[b].add(a)
        yield {str(k): {str(x) for x in v} for k, v in adj.items()}


def test_betweenness_against_brute_force():
    for adj in small_graphs(random.Random(3), 200):
        assert betweenness(adj) == brute_force_betweenness(adj)


def test_articulation_points_against_brute_force():
    for adj in small_graphs(random.Random(4), 200):
        before = len(components(adj))
        brute = set()
        for v in adj:
            rest = {s: {t for t in adj[s] if t != v} for s in adj if s != v}
            if len(components(rest)) > before:
                brute.add(v)
        assert articulation_points(adj) == brute


def test_structure_by_hand():
    adj = undirected(TOY.links)
    assert eccentricities(adj) == {"A": 2, "B": 1, "C": 2, "D": 2}
    assert articulation_points(adj) == {"B"}
    # B lies on the A-C, A-D and C-D paths: 3 of the 6 pairs.
    assert betweenness(adj) == {"A": 0, "B": 3, "C": 0, "D": 0}


# Naev


def test_naev_spob_price_by_hand():
    com = naev.Commodity("Food", 200, 0.5, {"M": 0.5}, {"Empire": 0.8})
    spob = naev.Spob("X", "M", 1e8, "Empire", 3, "AB", "e" * 21, frozenset({"commodity"}), False)
    p = naev.spob_price(spob, com)
    # 200 * 0.5 (class M) * (1 + tanh(0) * 0.5) * 0.8 (Empire) * (1 - 3/30) = 72.
    assert math.isclose(p.price, 72)
    # Period: 32 * (ord("A") % 32) + ord("B") % 32 + 100 = 32 + 2 + 100 = 134, times
    # 1 + (21 - 19)/100 for the exterior name, times 1 for population, over 1 - 3/30.
    assert math.isclose(p.spob_period, 134 * 1.02 / 0.9)
    assert math.isclose(p.spob_variation, 0.5 * 0.5)
    # A short exterior name wraps around in unsigned arithmetic: the period is about 2**64 / 100.
    short = naev.Spob("Y", "M", 1e8, "Empire", 0, "AB", "lava", frozenset({"commodity"}), False)
    assert naev.spob_price(short, com).spob_period > 1e18


def test_naev_mean_prices_by_hand():
    com = naev.Commodity("Food", 100, 0.0, {}, {})
    spobs = {
        n: naev.Spob(n, "M", 1e8, "F", 0, "AA", "x" * 19, frozenset({"commodity"}), False)
        for n in ("p", "q", "r")
    }
    systems = {
        "S": naev.System("S", 0.0, 0.0, 0.0, (naev.Jump("T", False, False),), ("p", "q")),
        "T": naev.System("T", 20e3, 0.0, 0.0, (naev.Jump("S", False, False),), ("r",)),
    }
    prices = naev.mean_prices({"Food": com}, systems, spobs)
    # p and q start at 100; r at 100 * (1 + 20e3/200e3) = 110. System means: S 100, T 110.
    # After smoothing with the neighbor: S (100 + 110) / 2 = 105, T (110 + 100) / 2 = 105.
    # Each spob: 0.25 * own + 0.75 * 105, so p and q 103.75 and r 0.25 * 110 + 78.75 = 106.25.
    assert math.isclose(prices["p"]["Food"].price, 103.75)
    assert math.isclose(prices["r"]["Food"].price, 106.25)
    assert prices["r"]["Food"].sys_period == 1000  # 2000 / (1 jump + 1)


def test_naev_committed_data():
    data = naev.load()
    assert data["source"]["license"].startswith("CC BY-SA 4.0")
    g = naev.galaxy(data, hidden=False)
    assert len(g.links) == 554 and len(g.stop_system) == 331
    assert all(len(p) == 8 for p in g.prices.values())
    rate, a, b = best_round_trips(g, 1)[0]
    assert (rate, a, b) == (394, "Burnan", "Lora-Mor")


def test_naev_price_at():
    p = naev.Price(price=100, spob_variation=0, spob_period=1, sys_variation=10, sys_period=8)
    assert naev.price_at(p, 0) == 100
    assert naev.price_at(p, 2) == 110  # a quarter of the period: the peak
    assert naev.price_at(p, 6) == 90


# EV Nova


def test_levels_and_prices():
    # Earth's flags in the data: food, industrial and metal medium, medical low, luxury high.
    assert evnova.levels(0x2214204F) == {
        "Food": 2,
        "Industrial": 2,
        "Medical Supplies": 1,
        "Luxury Goods": 4,
        "Metal": 2,
    }
    # The wiki's table: food 60 / 75 / 93, medical 600 / 750 / 937, luxury 720 / 900 / 1125.
    assert [evnova.price(75, k) for k in (1, 2, 4)] == [60, 75, 93]
    assert [evnova.price(750, k) for k in (1, 2, 4)] == [600, 750, 937]
    assert [evnova.price(900, k) for k in (1, 2, 4)] == [720, 900, 1125]


@pytest.mark.parametrize(
    ("test", "bits", "value"),
    [
        ("", set(), True),
        ("b5", set(), False),
        ("!b5", set(), True),
        ("!(b147 | b305)", {305}, False),
        ("!(b147 | b305) | !b148", {147}, True),
        ("b1 | b2 & b3", {1}, True),  # & binds tighter than |
        ("(b1 | b2) & b3", {1}, False),
    ],
)
def test_control_bit_tests(test, bits, value):
    assert evnova.parse_test(test)(frozenset(bits)) is value


def write_context(path: Path) -> None:
    """A tiny ConText file: three systems, one of them in two versions, and three stellars."""
    t = "\t"

    def row(*fields):
        return t.join(str(f) if isinstance(f, int) else f'"{f}"' for f in fields)

    syst_head = ["Type", "ID", "Name", "File"] + [f"con{i}" for i in range(16)]
    syst_head += [f"nav{i}" for i in range(16)] + ["Visiblility", "EOR"]

    def syst(sid, name, cons, navs, vis):
        cons = list(cons) + [-1] * (16 - len(cons))
        navs = list(navs) + [-1] * (16 - len(navs))
        return "sÿst" + t + row(sid, name, "f", *cons, *navs, vis, "•")

    spob_head = ["Type", "ID", "Name", "File", "Flags", "EOR"]
    lines = [
        "• Begin Output",
        "• Begin spöb",
        row(*spob_head),
        "spöb" + t + t.join(["128", '"Alpha"', '"f"', "0x10004003", '"•"']),  # food low, metal high
        "spöb"
        + t
        + t.join(["129", '"Beta"', '"f"', "0x41001003", '"•"']),  # food high, ind low, metal low
        "spöb" + t + t.join(["130", '"Gamma"', '"f"', "0x00000021", '"•"']),  # no exchange, no fuel
        "• End",
        "• Begin sÿst",
        row(*syst_head),
        syst(128, "One", [129], [128], ""),
        syst(129, "Two", [128, 130, 131], [129], ""),
        syst(130, "Three", [129], [130], "!b7"),
        syst(131, "Three", [129], [130], "b7"),
        "• End",
        "• Begin STR#",
        row("Type", "ID", "Name", "File", "n", "Strings", "EOR"),
        "STR#" + t + row(4004, "Base Prices", "f", 6, "75", "350", "750", "900", "200", "550", "•"),
        "• End",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_evnova_reader(tmp_path):
    path = tmp_path / "Tiny.txt"
    write_context(path)
    tables = evnova.read_context(path)
    g, levels = evnova.galaxy(tables)
    assert g.links == {"One": ("Two",), "Two": ("One", "Three"), "Three": ("Two",)}
    assert g.prices == {
        "Alpha (One)": {"Food": 60, "Metal": 250},
        "Beta (Two)": {"Food": 93, "Industrial": 280, "Metal": 160},
    }
    assert g.refuel_systems == {"One", "Two"}
    # Food +33 out and metal +90 back, one jump each way.
    assert best_round_trips(g, 1)[0] == (Fraction(123, 2), "Alpha (One)", "Beta (Two)")
    # With bit 7 set the other version of Three is visible, and it is the same here.
    assert evnova.galaxy(tables, frozenset({7}))[0].links == g.links


CONTEXT = os.environ.get("EV_NOVA_CONTEXT")


@pytest.mark.skipif(not CONTEXT, reason="set EV_NOVA_CONTEXT to a ConText export of EV Nova")
def test_evnova_game_data():
    tables = evnova.read_context(Path(CONTEXT or ""))
    for r in tables["sÿst"]:
        test = r["Visiblility"]
        assert evnova.parse_test(test)(frozenset()) == evnova.left_to_right(test, frozenset())
    g, levels = evnova.galaxy(tables)
    assert len(g.links) == 398 and len(g.stop_system) == 141
    assert best_round_trips(g, 1)[0][0] == 217
    for guide in GUIDES:
        assert all(stop in g.stop_system for stop in guide.stops), guide.name


def test_guides_are_well_formed():
    for guide in GUIDES:
        assert len(guide.legs) == len(guide.stops) == len(guide.jumps), guide.name
        for leg in guide.legs:
            for commodity, a, b in leg:
                assert commodity in evnova.COMMODITIES
                assert {a, b} <= {"low", "medium", "high"}
