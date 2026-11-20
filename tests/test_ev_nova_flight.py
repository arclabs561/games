"""Naev routes per period: the flight model, the timed legs and the ratio cycle, checked."""

import json
import math
import random
from pathlib import Path

import pytest

from ev_nova import flight as fl
from ev_nova import naev
from ev_nova.routes import Galaxy, best_route, brute_force_route, check_certificate

LLAMA = fl.Ship("Llama", 295.0, 299.0, 181.0)
SLOW = fl.Ship("slow", 80.0, 40.0, 60.0, jump_delay=1.3, land_delay=0.75)
RESULTS = Path(fl.__file__).parent / "flight.json"


@pytest.mark.parametrize("ship", [LLAMA, SLOW], ids=lambda s: s.name)
@pytest.mark.parametrize("d", [5.0, 100.0, 1000.0, 5000.0, 20000.0])
def test_closed_form_matches_small_steps(ship, d):
    # Steps of 1 ms: the two agree to within a few steps.
    assert fl.segment_seconds(d, ship) == pytest.approx(fl.simulate_seconds(d, ship), abs=0.01)


@pytest.mark.parametrize("ship", [LLAMA, SLOW], ids=lambda s: s.name)
def test_long_flights_cruise_at_top_speed(ship):
    # Past the acceleration, each extra unit of distance costs 1 / (speed + accel / 4).
    d1, d2 = 200_000.0, 300_000.0
    extra = fl.segment_seconds(d2, ship) - fl.segment_seconds(d1, ship)
    assert extra == pytest.approx((d2 - d1) / ship.top_speed, rel=1e-9)


def test_short_flights_never_reach_cruise():
    # Thrust to a peak p, coast half a turn, brake: d = p^2 / a + p T. With p = 50 for the
    # Llama, d = 2500 / 299 + 50 * 180 / 181 and the time is 2 p / a + T.
    turn = 180 / 181
    d = 2500 / 299 + 50 * turn
    assert fl.segment_seconds(d, LLAMA) == pytest.approx(100 / 299 + turn, rel=1e-12)


def random_world(seed: int, systems: int = 5, stops: int = 5):
    """A small galaxy with positions: systems on a ring of links plus chords, jump points
    and spobs at random places, prices of two commodities."""
    rng = random.Random(seed)
    names = [f"S{i}" for i in range(systems)]
    links = {s: set() for s in names}
    for i in range(systems):
        a, b = names[i], names[(i + 1) % systems]
        links[a].add(b), links[b].add(a)
    for _ in range(systems // 2):
        a, b = rng.sample(names, 2)
        links[a].add(b), links[b].add(a)
    jump_pos = {
        s: {t: (rng.uniform(-9000, 9000), rng.uniform(-9000, 9000)) for t in links[s]}
        for s in names
    }
    stop_names = [f"P{i}" for i in range(stops)]
    stop_system = {p: rng.choice(names) for p in stop_names}
    spob_pos = {p: (rng.uniform(-6000, 6000), rng.uniform(-6000, 6000)) for p in stop_names}
    prices = {p: {"Ore": rng.randint(50, 150), "Food": rng.randint(50, 150)} for p in stop_names}
    galaxy = Galaxy({s: tuple(sorted(v)) for s, v in links.items()}, stop_system, prices)
    return galaxy, fl.Geometry(jump_pos, spob_pos)


def brute_leg(galaxy, geo, ship, a, b, max_jumps=7):
    """The least time from stop a to stop b over every walk of up to `max_jumps` jumps."""
    sec = fl.INCREMENTS_PER_SECOND / fl.INCREMENTS_PER_PERIOD
    jump = ship.jump_delay + fl.JUMP_SECONDS * sec
    land = ship.land_delay
    sa, sb = galaxy.stop_system[a], galaxy.stop_system[b]
    pa, pb = geo.spob_pos[a], geo.spob_pos[b]

    def fly(p, q):
        return fl.segment_seconds(fl.distance(p, q), ship) * sec

    best = fly(pa, pb) + land if sa == sb else math.inf

    def walk(here, came, t, k):
        nonlocal best
        if t >= best:
            return
        if here == sb:
            best = min(best, t + fly(geo.jump_pos[here][came], pb) + land)
        if k == max_jumps:
            return
        start = geo.jump_pos[here][came]
        for n in galaxy.links[here]:
            walk(n, here, t + fly(start, geo.jump_pos[here][n]) + jump, k + 1)

    for n in galaxy.links[sa]:
        walk(n, sa, fly(pa, geo.jump_pos[sa][n]) + jump, 1)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_dijkstra_legs_match_every_walk(seed):
    galaxy, geo = random_world(seed)
    ship = [LLAMA, SLOW][seed % 2]
    legs = fl.leg_periods(galaxy, geo, ship)
    for a in galaxy.stops():
        for b in galaxy.stops():
            if a != b:
                assert legs[a, b][0] == pytest.approx(brute_leg(galaxy, geo, ship, a, b), rel=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_howard_matches_brute_force_on_timed_legs(seed):
    galaxy, geo = random_world(seed, systems=4 + seed % 3, stops=4 + seed % 3)
    edges, _ = fl.time_graph(galaxy, geo, [LLAMA, SLOW][seed % 2])
    best = best_route(edges)
    brute = brute_force_route(edges)
    if brute is None:
        assert best is None
        return
    assert best is not None and best.rate == brute[0]
    assert check_certificate(edges, best)


def test_the_data_file_and_the_llama():
    data = json.loads(fl.DATA.read_text())
    assert data["source"]["commit"] == naev.load()["source"]["commit"]
    assert fl.LICENSE in data["source"]["license"]
    llama = data["ships"]["Llama"]
    # The Hawk 160's primary values, and a full hold under its engine limit (no slowdown).
    assert (llama["speed"], llama["accel"], llama["turn"]) == (295.0, 299.0, 181.0)
    assert llama["full mass"] == 128.0 and llama["engine_limit"] == 160.0
    assert llama["cargo"] == 30
    # Scarlet's jump to Druss is an autopos jump: at the radius, toward Druss.
    sys = naev.load()["systems"]
    x, y = data["jumps"]["Scarlet"]["Druss"]
    assert math.hypot(x, y) == pytest.approx(data["radius"]["Scarlet"], rel=1e-9)
    dx = sys["Druss"]["pos"][0] - sys["Scarlet"]["pos"][0]
    dy = sys["Druss"]["pos"][1] - sys["Scarlet"]["pos"][1]
    assert math.atan2(y, x) == pytest.approx(math.atan2(dy, dx), abs=1e-6)
    # Every jump has a jump point back, where a ship arrives.
    for s, links in data["jumps"].items():
        for t in links:
            assert s in data["jumps"][t], (s, t)


def test_committed_results_match_a_recomputation():
    committed = json.loads(RESULTS.read_text())
    galaxy = naev.galaxy(hidden=False)
    geo, ships = fl.load()
    assert fl.report(ships["Llama"], galaxy, geo) == committed["llama"]
    grid = committed["speed sweep"]["grid"]
    assert [g["scale"] for g in grid][:-1] == list(fl.SPEEDS[:-1])


def test_readme_numbers_are_the_committed_results():
    readme = (Path(fl.__file__).parent / "README.md").read_text()
    r = json.loads(RESULTS.read_text())
    trips = r["llama"]["round trips"]
    for t in trips[:2]:
        assert f"| {t['gain']:,} | {t['periods']:.4f} | {t['rate']:.2f} |" in readme
    timed = r["best per jump, timed"]
    assert f"| 788 | {timed['periods per pass']:.4f} | {timed['rate per period']:.2f} |" in readme
    assert f"{r['llama']['best']['rate per period']:.2f} per period" in readme
    grid = r["speed sweep"]["grid"]
    assert f"{grid[-1]['rate per period']:.2f} per period when flight takes no time" in readme
    lo, hi = (c["between"] for c in r["speed sweep"]["changes"])
    assert round(lo[0], 2) == round(lo[1], 2) == 0.36 and 16.7 < hi[0] < 16.8 < hi[1] < 16.9
