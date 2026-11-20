"""Naev trade routes in credits per period: flight time inside systems, jumps and landings.

The study in `routes.py` counts jumps. Here the cost of a leg is the game time it takes, in
**periods** (Naev's unit: 10,000 time increments). The problem is then the classical maximum
cost-to-time ratio cycle (Dantzig, Blattner and Rao, 1967): gain per pass over time per
pass, solved exactly by the same Howard iteration and certificate as before.

The time model, read from Naev's source at the commit of `naev.json` (file names below):

- Game time runs at 30 increments per second of game time (`dat/constants.lua`,
  TIMEDATE_INCREMENTS_PER_SECOND; `ntime_update` in `src/core/src/ntime.rs` is called with
  the game's time step in `src/naev.c`). A ship's time_mod speeds up real time, not this.
- A jump adds 10,000 increments times the ship's jump_delay (`pilot_hyperspaceDelay` in
  `src/pilot.c`, TIMEDATE_HYPERSPACE_INCREMENTS), after 3 seconds of engine start and 5 of
  acceleration (HYPERSPACE_ENGINE_DELAY and HYPERSPACE_FLY_DELAY). A ship must slow to 25
  units per second before it jumps (MAX_HYPERSPACE_VEL in `src/space.h`).
- A landing adds 10,000 increments times land_delay (`src/land.c`, TIMEDATE_LAND_INCREMENTS).
- A jump point marked `autopos` sits at the system's radius, toward the target system
  (`system_reconstructJumps` in `src/space.c`); others have a position in the system file.
  A ship arrives at the jump point of its new system that leads back.
- Flight: thrust `accel` until the ship's speed reaches `speed`; above it a drag of
  4 × (v − speed) per second holds it near speed + accel / 4 (`solid_update_rk4` in
  `src/physics.c`, PHYSICS_SPEED_DAMP = 4). To stop, the ship turns half a circle at its turn
  rate with the engine off, then thrusts against its motion (`pilot_brake` in `src/pilot.c`).

Simplifications, stated as such: every flight inside a system starts and ends at rest at the
point's center (the jump-in offset and the jump radius are left out), the ship flies
straight, the first turn out of a spob or a jump point is free, and the time on the landing
and take-off animations is left out. `simulate_seconds` integrates the same motion in small
steps, as a check on the closed form of `segment_seconds`.

    uv run python -m ev_nova.flight --checkout PATH   # writes ev_nova/naev_flight.json
    uv run python -m ev_nova.flight                    # the results, from the committed data
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

from . import naev
from .routes import Galaxy, best_route, check_certificate, route_rate, trade_graph

HERE = Path(__file__).parent
DATA = HERE / "naev_flight.json"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"

DAMP = 4.0  # PHYSICS_SPEED_DAMP
INCREMENTS_PER_SECOND = 30.0
INCREMENTS_PER_PERIOD = 10_000.0
HYPERSPACE_INCREMENTS = 10_000.0
LAND_INCREMENTS = 10_000.0
JUMP_SECONDS = 3.0 + 5.0  # HYPERSPACE_ENGINE_DELAY + HYPERSPACE_FLY_DELAY

type Point = tuple[float, float]


@dataclass(frozen=True)
class Ship:
    """What the time model needs of a ship, with its default outfits and a full hold."""

    name: str
    speed: float  # units per second, the speed thrust alone reaches
    accel: float  # units per second squared
    turn: float  # degrees per second
    jump_delay: float = 1.0
    land_delay: float = 1.0
    cargo: int = 0

    @property
    def top_speed(self) -> float:
        """The speed at which thrust and drag balance (`solid_maxspeed`)."""
        return self.speed + self.accel / DAMP


# Flight inside a system


def _thrust(tau: float, v0: float, s: float, a: float) -> tuple[float, float]:
    """Thrusting for tau seconds from speed v0 >= s: (distance, final speed)."""
    b = a / DAMP
    vm = s + b
    e = math.exp(-DAMP * tau)
    return vm * tau - (vm - v0) * (1 - e) / DAMP, vm - (vm - v0) * e


def _coast(t: float, v0: float, s: float) -> tuple[float, float]:
    """Engine off for t seconds: drag acts only above s."""
    if v0 <= s:
        return v0 * t, v0
    e = math.exp(-DAMP * t)
    return s * t + (v0 - s) * (1 - e) / DAMP, s + (v0 - s) * e


def _brake(v0: float, s: float, a: float) -> tuple[float, float]:
    """Thrust against the motion from v0 to rest: (distance, seconds)."""
    dist = secs = 0.0
    if v0 > s:  # thrust and drag together, down to s
        b = a / DAMP
        u0 = v0 - s
        t = math.log((u0 + b) / b) / DAMP
        dist += s * t + (u0 + b) * (1 - math.exp(-DAMP * t)) / DAMP - b * t
        secs += t
        v0 = s
    return dist + v0 * v0 / (2 * a), secs + v0 / a


def segment_seconds(d: float, ship: Ship) -> float:
    """Seconds to fly a distance d from rest to rest: thrust, turn half a circle with the
    engine off, then brake. The thrust time is the one that stops the ship at d."""
    if d <= 0 or math.isinf(ship.speed):
        return 0.0
    s, a = ship.speed, ship.accel
    turn = 180.0 / ship.turn

    def flown(tau: float) -> tuple[float, float]:
        x1, t1 = s * s / (2 * a), s / a
        x2, v2 = _thrust(tau, s, s, a)
        x3, v3 = _coast(turn, v2, s)
        x4, t4 = _brake(v3, s, a)
        return x1 + x2 + x3 + x4, t1 + tau + turn + t4

    if d <= flown(0.0)[0]:
        # The ship never reaches s: thrust to a peak p, coast at p, brake from p.
        p = a * (-turn + math.sqrt(turn * turn + 4 * d / a)) / 2
        return 2 * p / a + turn
    lo, hi = 0.0, d / s + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if flown(mid)[0] < d:
            lo = mid
        else:
            hi = mid
    return flown((lo + hi) / 2)[1]


def simulate_seconds(d: float, ship: Ship, dt: float = 1e-3) -> float:
    """The same flight by small time steps: thrust while the distance left exceeds what the
    turn and the brake would cover, then turn, then brake. A check on `segment_seconds`."""
    s, a, turn = ship.speed, ship.accel, 180.0 / ship.turn
    x = v = t = 0.0
    while True:  # thrust
        coast_x, v3 = _coast(turn, v, s)
        if x + coast_x + _brake(v3, s, a)[0] >= d:
            break
        drag = DAMP * (v - s) if v > s else 0.0
        v += (a - drag) * dt
        x += v * dt
        t += dt
    end = t + turn
    while t < end:  # turn, engine off
        drag = DAMP * (v - s) if v > s else 0.0
        v -= drag * dt
        x += v * dt
        t += dt
    while v > 0:  # brake
        drag = DAMP * (v - s) if v > s else 0.0
        v -= (a + drag) * dt
        x += max(v, 0.0) * dt
        t += dt
    return t


# The galaxy's geometry


@dataclass(frozen=True)
class Geometry:
    """Jump point and spob positions inside each system."""

    jump_pos: Mapping[str, Mapping[str, Point]]  # system -> target -> position
    spob_pos: Mapping[str, Point]


def distance(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def leg_periods(
    galaxy: Galaxy, geo: Geometry, ship: Ship
) -> dict[tuple[str, str], tuple[float, int]]:
    """For every ordered pair of stops, the least time in periods to fly from one, land at
    the other and take off again, and the jumps on that path. Dijkstra's method over states
    (system, the system it was entered from)."""
    per_second = INCREMENTS_PER_SECOND / INCREMENTS_PER_PERIOD
    jump = ship.jump_delay * HYPERSPACE_INCREMENTS / INCREMENTS_PER_PERIOD
    jump += JUMP_SECONDS * per_second
    land = ship.land_delay * LAND_INCREMENTS / INCREMENTS_PER_PERIOD

    @cache
    def fly(d: float) -> float:
        return segment_seconds(d, ship) * per_second

    by_system: dict[str, list[str]] = {}
    for stop, sys in galaxy.stop_system.items():
        by_system.setdefault(sys, []).append(stop)
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for a in galaxy.stops():
        sa, pa = galaxy.stop_system[a], geo.spob_pos[a]
        for b in by_system[sa]:
            if b != a:
                out[a, b] = (fly(distance(pa, geo.spob_pos[b])) + land, 0)
        dist: dict[tuple[str, str], tuple[float, int]] = {}
        heap: list[tuple[float, int, str, str]] = []
        for n in galaxy.links[sa]:
            t = fly(distance(pa, geo.jump_pos[sa][n])) + jump
            heapq.heappush(heap, (t, 1, n, sa))
        while heap:
            t, k, here, came = heapq.heappop(heap)
            if (here, came) in dist:
                continue
            dist[here, came] = (t, k)
            entry = geo.jump_pos[here][came]
            for n in galaxy.links[here]:
                if (n, here) not in dist:
                    step = fly(distance(entry, geo.jump_pos[here][n])) + jump
                    heapq.heappush(heap, (t + step, k + 1, n, here))
        for (here, came), (t, k) in dist.items():
            entry = geo.jump_pos[here][came]
            for b in by_system.get(here, ()):
                total = t + fly(distance(entry, geo.spob_pos[b])) + land
                if (a, b) not in out or total < out[a, b][0]:
                    out[a, b] = (total, k)
    return out


def time_graph(galaxy: Galaxy, geo: Geometry, ship: Ship):
    """The trade graph with each leg's time in periods as its cost, as exact fractions of
    the floating-point times, so that the certificate is checked exactly for those times."""
    legs = leg_periods(galaxy, geo, ship)

    def cost(a: str, b: str) -> Fraction | None:
        return Fraction(legs[a, b][0]) if (a, b) in legs else None

    return trade_graph(galaxy, cost=cost), legs


# Reading Naev's data


def _multicore(text: str) -> dict[str, float]:
    """Primary-slot values of an outfit's `multicore.init` list, or its plain elements."""
    out = {}
    for name, first in re.findall(r"\{\s*'(\w+)'\s*,\s*(-?[\d.]+)", text):
        out[name] = float(first)
    return out


def read_ship(naev_dir: Path, name: str) -> tuple[Ship, dict]:
    """A ship with its default outfits, for ships with one engine and no stat that changes
    speed or thrust. Returns the ship and the numbers it was built from."""
    dat = naev_dir / "dat"
    path = next(
        p for p in (dat / "ships").rglob("*.xml") if ET.parse(p).getroot().get("name") == name
    )
    root = ET.parse(path).getroot()
    mass = float(root.findtext("characteristics/mass", "0"))
    cargo = float(root.findtext("characteristics/cargo", "0"))
    found = root.find("stats")
    stats = {e.tag: float(e.text or 0) for e in (found if found is not None else [])}
    assert not {"speed_mod", "accel_mod", "turn_mod", "speed", "accel"} & set(stats), stats
    outfits = {ET.parse(p).getroot().get("name"): p for p in (dat / "outfits").rglob("*.xml")}
    engines = [s for s in root.find("slots") if s.get("prop", "").startswith("engines") and s.text]
    assert len(engines) == 1, "only ships with one engine are read"
    speed = accel = turn = limit = 0.0
    used = {"ship": str(path.relative_to(naev_dir))}
    for slot in root.find("slots"):
        if not slot.text:
            continue
        o = outfits[slot.text.strip()]
        vals = _multicore(o.read_text())
        used[slot.text.strip()] = str(o.relative_to(naev_dir))
        mass += vals.get("mass", 0.0)
        cargo += vals.get("cargo", 0.0)
        if slot in engines:
            speed, accel = vals["speed"], vals["accel"]
            turn, limit = vals["turn"], vals["engine_limit"]
    inertia = 1 + stats.get("cargo_inertia", 0.0) / 100
    full = mass + inertia * cargo
    factor = 1.0
    if full - limit > 0.1:  # pilot_massFactor
        f = (full - limit) / limit
        factor = 1 / (1 + 2 * f + 4 * f**3)
    ship = Ship(
        name,
        speed * factor,
        accel * factor,
        turn,
        1 + stats.get("jump_delay", 0.0) / 100,
        1 + stats.get("land_delay", 0.0) / 100,
        int(cargo),
    )
    facts = {"mass": mass, "full mass": full, "engine_limit": limit}
    return ship, {"files": used, **facts}


def export(naev_dir: Path, ships=("Llama",)) -> dict:
    """Jump point and spob positions of every system with a trading stop or a jump, and the
    ships' flight numbers, derived from a Naev checkout."""
    data = naev.load()
    stops = set(data["spobs"])
    jumps: dict[str, dict[str, list[float]]] = {}
    radius = {}
    pos = {n: tuple(s["pos"]) for n, s in data["systems"].items()}
    for path in sorted((naev_dir / "dat" / "ssys").glob("*.xml")):
        root = ET.parse(path).getroot()
        name = root.get("name")
        r = float(root.findtext("general/radius", "0"))
        radius[name] = r
        jumps[name] = {}
        for j in root.iter("jump"):
            target = j.get("target")
            p = j.find("pos")
            if p is not None:
                xy = (float(p.get("x")), float(p.get("y")))
            else:
                assert j.find("autopos") is not None
                dx, dy = pos[target][0] - pos[name][0], pos[target][1] - pos[name][1]
                a = math.atan2(dy, dx)
                xy = (r * math.cos(a), r * math.sin(a))
            jumps[name][target] = [round(xy[0], 6), round(xy[1], 6)]
    spobs = {}
    for path in sorted((naev_dir / "dat" / "spob").glob("*.xml")):
        root = ET.parse(path).getroot()
        if root.get("name") in stops:
            p = root.find("pos")
            spobs[root.get("name")] = [round(float(p.get("x")), 6), round(float(p.get("y")), 6)]
    commit = subprocess.run(
        ["git", "-C", str(naev_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    ship_data = {}
    for s in ships:
        ship, facts = read_ship(naev_dir, s)
        ship_data[s] = {
            "speed": ship.speed,
            "accel": ship.accel,
            "turn": ship.turn,
            "jump_delay": ship.jump_delay,
            "land_delay": ship.land_delay,
            "cargo": ship.cargo,
            **facts,
        }
    return {
        "source": {
            "repository": "https://github.com/naev/naev",
            "commit": commit,
            "license": f"CC BY-SA 4.0 ({LICENSE}): derived from Naev's XML data files, by the"
            " Naev developers; positions resolved and ship numbers combined from outfits",
        },
        "radius": dict(sorted(radius.items())),
        "jumps": dict(sorted(jumps.items())),
        "spobs": spobs,
        "ships": ship_data,
    }


def load(path: Path = DATA) -> tuple[Geometry, dict[str, Ship]]:
    data = json.loads(path.read_text())
    geo = Geometry(
        {s: {t: tuple(p) for t, p in j.items()} for s, j in data["jumps"].items()},
        {n: tuple(p) for n, p in data["spobs"].items()},
    )
    ships = {
        n: Ship(n, d["speed"], d["accel"], d["turn"], d["jump_delay"], d["land_delay"], d["cargo"])
        for n, d in data["ships"].items()
    }
    return geo, ships


# The study


def describe(galaxy: Galaxy, route: tuple[str, ...]) -> str:
    return " -> ".join(f"{s} ({galaxy.stop_system[s]})" for s in route)


def report(ship: Ship, galaxy: Galaxy, geo: Geometry, top: int = 5) -> dict:
    """The best route per period for a ship, its certificate, and the best round trips."""
    edges, legs = time_graph(galaxy, geo, ship)
    best = best_route(edges)
    assert best is not None and check_certificate(edges, best)
    route = best.route
    pairs = list(zip(route, route[1:] + route[:1], strict=True))
    gain = sum(edges[p][0] for p in pairs)
    periods = sum(legs[p][0] for p in pairs)
    jumps = sum(legs[p][1] for p in pairs)
    trips = []
    stops = galaxy.stops()
    for i, a in enumerate(stops):
        for b in stops[i + 1 :]:
            if (a, b) in edges and (b, a) in edges:
                g = edges[a, b][0] + edges[b, a][0]
                if g > 0:
                    t = legs[a, b][0] + legs[b, a][0]
                    trips.append((g / t, a, b, g, t, legs[a, b][1] + legs[b, a][1]))
    trips.sort(reverse=True)
    return {
        "ship": ship.name,
        "speed": ship.speed,
        "accel": ship.accel,
        "best": {
            "route": list(route),
            "rate per period": float(best.rate),
            "gain per pass": gain,
            "periods per pass": round(periods, 6),
            "jumps per pass": jumps,
            "certificate checked": True,
        },
        "round trips": [
            {"stops": [a, b], "rate": round(r, 4), "gain": g, "periods": round(t, 6), "jumps": k}
            for r, a, b, g, t, k in trips[:top]
        ],
    }


SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)


def scaled(base: Ship, k: float) -> Ship:
    """The ship with speed and thrust scaled by k (k = inf: no time inside systems)."""
    return Ship(f"{base.name} x{k}", base.speed * k, base.accel * k, base.turn)


def sweep(base: Ship, galaxy: Galaxy, geo: Geometry, steps: int = 6) -> dict:
    """The best route as speed and thrust are scaled together by each factor in SPEEDS, and
    where it changes between two factors, located by bisection on the log of the factor to
    within 2^(-steps) of a doubling."""

    @cache
    def best(k: float) -> tuple[tuple[str, ...], dict]:
        r = report(scaled(base, k), galaxy, geo, top=0)["best"]
        return tuple(r["route"]), r

    grid = [{"scale": "no flight time" if math.isinf(k) else k, **best(k)[1]} for k in SPEEDS]
    changes = []
    for lo, hi in zip(SPEEDS, SPEEDS[1:], strict=False):
        if best(lo)[0] == best(hi)[0]:
            continue
        if math.isinf(hi):
            changes.append(
                {"between": [lo, "no flight time"], "from": best(lo)[0], "to": best(hi)[0]}
            )
            continue
        a, b = math.log2(lo), math.log2(hi)
        for _ in range(steps):
            m = (a + b) / 2
            if best(2**m)[0] == best(2**a)[0]:
                a = m
            else:
                b = m
        changes.append(
            {"between": [round(2**a, 4), round(2**b, 4)], "from": best(lo)[0], "to": best(hi)[0]}
        )
    return {"grid": grid, "changes": changes}


def results() -> dict:
    galaxy = naev.galaxy(hidden=False)
    geo, ships = load()
    llama = ships["Llama"]
    edges, legs = time_graph(galaxy, geo, llama)
    burnan = ("Burnan", "Lora-Mor")
    per_jump = {
        "route": list(burnan),
        "rate per period": float(route_rate(edges, burnan)),
        "periods per pass": round(legs[burnan][0] + legs[burnan[::-1]][0], 6),
    }
    return {
        "llama": report(llama, galaxy, geo),
        "best per jump, timed": per_jump,
        "speed sweep": sweep(llama, galaxy, geo),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkout", type=Path, help="a clone of naev/naev at naev.json's commit")
    parser.add_argument("--out", type=Path, help="write the results as JSON here")
    args = parser.parse_args()
    if args.checkout:
        DATA.write_text(naev.dumps(export(args.checkout)) + "\n")
        print(f"wrote {DATA}")
        return
    r = results()
    text = json.dumps(r, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
