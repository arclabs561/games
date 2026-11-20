"""Naev's galaxy and its mean commodity prices, read from a checkout of github.com/naev/naev.

The price model follows `economy_initialiseCommodityPrices` in Naev's `src/economy.c` (GPL-3):
a price for each spob from the commodity's base price and modifiers, then averaging within each
system, then averaging with neighboring systems. Over time each price swings around its mean as a
sum of two sine waves; `price_at` gives the price at a time. This module is a re-implementation
of that arithmetic for this study, not a copy of Naev's code.

    uv run python -m ev_nova.naev --checkout PATH   # rewrites ev_nova/naev.json
"""

import argparse
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "naev.json"
PRICE_FIELDS = ("price", "sys_variation", "sys_period", "spob_variation", "spob_period")


@dataclass(frozen=True)
class Jump:
    """A jump from one system to another, with Naev's two restrictions on its use."""

    target: str
    hidden: bool  # must be discovered before it can be used
    exitonly: bool  # ships arrive by it but cannot leave by it


@dataclass(frozen=True)
class System:
    """A star system: where it is, what limits prices there, its jumps and its spobs."""

    name: str
    radius: float
    interference: float
    volatility: float
    jumps: tuple[Jump, ...]
    spobs: tuple[str, ...]
    pos: tuple[float, float] = (0.0, 0.0)  # map position, as in the system's XML file


@dataclass(frozen=True)
class Spob:
    """A space object (planet or station) and the fields the price model reads."""

    name: str
    spob_class: str
    population: float
    faction: str | None
    presence_range: int
    space_gfx: str | None
    exterior_gfx: str | None
    services: frozenset[str]
    nocommodities: bool


@dataclass(frozen=True)
class Commodity:
    """A standard commodity: base price and the modifiers the price model applies."""

    name: str
    price: int
    population_modifier: float
    spob_modifiers: dict[str, float]
    faction_modifiers: dict[str, float]


@dataclass
class Price:
    """One commodity at one spob: mean price, and the two sine waves around it."""

    price: float
    spob_variation: float = 0.5
    spob_period: float = 0.0
    sys_variation: float = 0.0
    sys_period: float = 0.0


def _float(node: ET.Element | None, default: float = 0.0) -> float:
    return default if node is None or node.text is None else float(node.text)


def read_commodity(path: Path) -> Commodity | None:
    """The commodity in one XML file, or None if it is not a standard commodity."""
    root = ET.parse(path).getroot()
    if root.find("standard") is None:
        return None
    return Commodity(
        name=root.findtext("name", ""),
        price=int(root.findtext("price", "0")),
        population_modifier=_float(root.find("population_modifier")),
        spob_modifiers={m.get("type", ""): float(m.text or 1) for m in root.iter("spob_modifier")},
        faction_modifiers={
            m.get("type", ""): float(m.text or 1) for m in root.iter("faction_modifier")
        },
    )


def read_spob(path: Path) -> Spob:
    """The spob in one XML file."""
    root = ET.parse(path).getroot()
    general = root.find("general")
    assert general is not None, path
    services = general.find("services")
    presence = root.find("presence")
    return Spob(
        name=root.get("name", ""),
        spob_class=general.findtext("class", ""),
        population=_float(general.find("population")),
        faction=None if presence is None else presence.findtext("faction"),
        presence_range=0 if presence is None else int(presence.findtext("range", "0")),
        space_gfx=root.findtext("GFX/space"),
        exterior_gfx=root.findtext("GFX/exterior"),
        services=frozenset() if services is None else frozenset(s.tag for s in services),
        nocommodities=services is not None and services.find("nocommodities") is not None,
    )


def read_system(path: Path) -> System:
    """The system in one XML file."""
    root = ET.parse(path).getroot()
    general = root.find("general")
    assert general is not None, path
    nebula = general.find("nebula")
    pos = root.find("pos")
    jumps = tuple(
        Jump(j.get("target", ""), j.find("hidden") is not None, j.find("exitonly") is not None)
        for j in root.iter("jump")
    )
    return System(
        name=root.get("name", ""),
        radius=_float(general.find("radius")),
        interference=_float(general.find("interference")),
        volatility=0.0 if nebula is None else float(nebula.get("volatility", "0")),
        jumps=jumps,
        spobs=tuple(s.text or "" for s in root.findall("spobs/spob")),
        pos=(float(pos.get("x", "0")), float(pos.get("y", "0"))) if pos is not None else (0.0, 0.0),
    )


def read_checkout(naev: Path):
    """Commodities, systems and spobs from the `dat/` folder of a Naev checkout."""
    dat = naev / "dat"
    commodities = [read_commodity(p) for p in sorted((dat / "commodities").glob("*.xml"))]
    systems = [read_system(p) for p in sorted((dat / "ssys").glob("*.xml"))]
    spobs = [read_spob(p) for p in sorted((dat / "spob").glob("*.xml"))]
    return (
        {c.name: c for c in commodities if c is not None},
        {s.name: s for s in systems},
        {s.name: s for s in spobs},
    )


def trades(spob: Spob) -> bool:
    """Whether the spob has a commodity exchange with the standard commodities."""
    return "commodity" in spob.services and not spob.nocommodities


def spob_price(spob: Spob, com: Commodity) -> Price:
    """The price of one commodity at one spob before any averaging (economy_calcPrice)."""
    assert spob.faction is not None, spob.name
    p = Price(price=round(com.price) * com.spob_modifiers.get(spob.spob_class, 1.0))
    # The period comes from the first two letters of the spob's space graphic.
    assert spob.space_gfx is not None and spob.exterior_gfx is not None, spob.name
    first, second = (ord(ch) for ch in spob.space_gfx[:2])
    p.spob_period = 32 * (first % 32) + second % 32 + 100
    # Naev computes this length difference in unsigned arithmetic, so a name shorter than 19
    # characters (every name in the data) wraps to nearly 2**64 and the period becomes so long
    # that this wave never moves.
    p.spob_period *= 1 + ((len(spob.exterior_gfx) - 19) % 2**64) / 100
    factor = -1.0
    if spob.population > 0:
        factor = math.tanh((math.log(spob.population) - math.log(1e8)) * 0.5)
    p.price *= 1 + factor * com.population_modifier
    p.spob_variation *= 0.5 - factor * 0.25
    p.spob_period *= 1 + factor * 0.5
    p.price *= com.faction_modifiers.get(spob.faction, 1.0)
    p.price *= 1 - spob.presence_range / 30
    p.spob_period *= 1 / (1 - spob.presence_range / 30)
    p.price = max(p.price, 1)
    return p


def mean_prices(commodities, systems, spobs) -> dict[str, dict[str, Price]]:
    """Every trading spob's prices after the averaging within and between systems."""
    standard = sorted(commodities.values(), key=lambda c: c.name)
    prices: dict[str, dict[str, Price]] = {}
    for sys in systems.values():
        for name in sys.spobs:
            if trades(spobs[name]):
                prices[name] = {c.name: spob_price(spobs[name], c) for c in standard}
    # Within each system (economy_modifySystemCommodityPrice).
    averages: dict[str, dict[str, Price]] = {}
    for sys in systems.values():
        here = [prices[n] for n in sys.spobs if n in prices]
        for table in here:
            for p in table.values():
                p.price *= 1 + sys.radius / 200e3
                p.spob_period *= 1 / (1 - sys.radius / 200e3)
                p.spob_variation *= 1 / (1 - sys.radius / 300e3)
                p.price *= 1 + sys.volatility / 600
                p.price *= 1 + sys.interference / 10e3
                p.sys_period = 2000 / (len(sys.jumps) + 1)
        if not here:
            continue
        avg = {}
        for c in here[0]:
            k = len(here)
            avg[c] = Price(
                price=sum(t[c].price for t in here) / k,
                spob_variation=sum(t[c].spob_variation for t in here) / k,
                spob_period=sum(t[c].spob_period for t in here) / k,
                sys_period=sum(t[c].sys_period for t in here) / k,
            )
        for table in here:
            for c, p in table.items():
                p.price = 0.25 * p.price + 0.75 * avg[c].price
                p.sys_variation = 0.2 * avg[c].spob_variation
        averages[sys.name] = avg
    # Between neighboring systems (economy_smoothCommodityPrice and ..._calcUpdated...).
    neighbor_mean = {}
    for name, avg in averages.items():
        near = [averages[j.target] for j in systems[name].jumps if j.target in averages]
        neighbor_mean[name] = {
            c: sum(a[c].price for a in near) / len(near) if near else avg[c].price for c in avg
        }
    for name, avg in averages.items():
        for c in avg:
            avg[c].price = 0.5 * (avg[c].price + neighbor_mean[name][c])
        for n in systems[name].spobs:
            for c, p in prices.get(n, {}).items():
                p.price = 0.25 * p.price + 0.75 * avg[c].price
                p.spob_variation = 0.1 * (0.5 * avg[c].spob_variation + 0.5 * p.spob_variation)
                p.spob_variation *= p.price
                p.sys_variation *= p.price
    return prices


def price_at(p: Price, periods: float) -> int:
    """The price shown in game at a time, in periods (economy_getPriceAtTime)."""
    value = (
        p.price
        + p.sys_variation * math.sin(2 * math.pi * periods / p.sys_period)
        + p.spob_variation * math.sin(2 * math.pi * periods / p.spob_period)
    )
    return max(1, round(value))


def export(naev: Path) -> dict:
    """The data this study uses, derived from a Naev checkout, as a JSON-ready dict."""
    commodities, systems, spobs = read_checkout(naev)
    prices = mean_prices(commodities, systems, spobs)
    commit = subprocess.run(
        ["git", "-C", str(naev), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    where = {n: s.name for s in systems.values() for n in s.spobs}
    return {
        "source": {
            "repository": "https://github.com/naev/naev",
            "commit": commit,
            "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), from"
            " Naev's XML data files by the Naev developers; this file is derived from them",
        },
        "commodities": {
            c.name: c.price for c in sorted(commodities.values(), key=lambda c: c.name)
        },
        "systems": {
            s.name: {
                "jumps": {
                    j.target: " ".join(k for k in ("hidden", "exitonly") if getattr(j, k))
                    for j in s.jumps
                },
                "refuel": any("refuel" in spobs[n].services for n in s.spobs),
                "pos": [round(s.pos[0], 6), round(s.pos[1], 6)],
            }
            for s in sorted(systems.values(), key=lambda s: s.name)
        },
        "price_fields": list(PRICE_FIELDS),
        "spobs": {
            name: {
                "system": where[name],
                "faction": spobs[name].faction,
                "prices": {
                    c: [round(getattr(p, f), 6) for f in PRICE_FIELDS]
                    for c, p in sorted(table.items())
                },
            }
            for name, table in sorted(prices.items())
        },
    }


def dumps(data: dict) -> str:
    """JSON with one line per list, so that the file reads and diffs well."""
    text = json.dumps(data, indent=1, ensure_ascii=False)
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + " ".join(m[1].split()) + "]", text)


def load(path: Path = DATA) -> dict:
    """The committed data file."""
    return json.loads(path.read_text())


def whole(x: float) -> int:
    """Round half up, as C's round() does for positive numbers."""
    return math.floor(x + 0.5)


def galaxy(data: dict | None = None, *, hidden: bool = True):
    """Naev's galaxy with mean prices rounded to whole credits.

    A jump marked exit-only cannot be left by; with `hidden` False, hidden jumps are left out too.
    """
    from .routes import Galaxy

    data = load() if data is None else data
    links = {
        name: tuple(
            sorted(
                target
                for target, flags in s["jumps"].items()
                if "exitonly" not in flags and (hidden or "hidden" not in flags)
            )
        )
        for name, s in data["systems"].items()
    }
    stop_system = {n: s["system"] for n, s in data["spobs"].items()}
    prices = {n: {c: whole(p[0]) for c, p in s["prices"].items()} for n, s in data["spobs"].items()}
    refuel = frozenset(n for n, s in data["systems"].items() if s["refuel"])
    return Galaxy(links, stop_system, prices, refuel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkout", type=Path, required=True, help="a clone of naev/naev")
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    args.out.write_text(dumps(export(args.checkout)) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
