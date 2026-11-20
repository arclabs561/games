"""Trade routes that EV Nova players recommend, as published, for checking against the data.

Sources (read 2026-11-18):

- WIKI: "Trade Routes", EV Nova wiki, https://evnova.miraheze.org/wiki/Trade_Routes
  ("the sequel to The Nova Trade Network Guide"). Routes that need a hypergate, or that carry
  only special commodities, are left out.
- MONEY: "Nova:Making Money", https://evnova.miraheze.org/wiki/Nova:Making_Money, the list of
  "short, profitable runs used by veteran pilots", standard commodities only.

The route data below are adapted from those pages, which are under CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/), with credit to the wiki's editors; this
file's data are under the same license.

Each route is its stops in order, the jumps the source gives for each leg, and for each leg the
commodities with the price levels the source names. The sources give one jump count for a
two-stop route; it is read as the jumps each way.
"""

from dataclasses import dataclass

type Claim = tuple[str, str, str]  # commodity, level where bought, level where sold


@dataclass(frozen=True)
class Guide:
    """A published route: its name and source, stops, claimed jumps, and claimed trades."""

    name: str
    source: str
    stops: tuple[str, ...]
    jumps: tuple[int, ...]
    legs: tuple[tuple[Claim, ...], ...]


def _g(name, source, stops, jumps, *legs) -> Guide:
    if isinstance(jumps, int):
        jumps = (jumps,) * len(stops)
    return Guide(name, source, tuple(stops), tuple(jumps), tuple(tuple(leg) for leg in legs))


L, M, H = "low", "medium", "high"
FOOD, IND, MED, LUX, METAL, EQ = (
    "Food",
    "Industrial",
    "Medical Supplies",
    "Luxury Goods",
    "Metal",
    "Equipment",
)

GUIDES = [
    # Single trades: nothing worth carrying back.
    _g("Sol", "WIKI", ["Europa (Sol)", "Earth (Sol)"], 0, [(LUX, M, H)], []),
    _g(
        "Sanddown",
        "WIKI",
        ["Kolan (Sanddown)", "Kiniké (Sanddown)"],
        0,
        [(FOOD, M, H), (MED, M, H)],
        [],
    ),
    _g("Fomalhaut", "WIKI", ["Snowmelt (Fomalhaut)", "Gem (Fomalhaut)"], 0, [(EQ, M, H)], []),
    _g(
        "Tre'pirana",
        "WIKI",
        ["Tre'a Ro (Tre'pirana)", "Tre'ar Vano (Tre'pirana)"],
        0,
        [(EQ, L, H)],
        [],
    ),
    _g(
        "Tri'lipra",
        "WIKI",
        ["Tre'ar Aeberl (Tri'lipra)", "Tre'a He (Tri'lipra)"],
        0,
        [(EQ, M, H)],
        [],
    ),
    # Two-stop round trips with cargo both ways.
    _g(
        "Kania/Altair",
        "WIKI",
        ["Port Kane (Kania)", "Altia (Altair)"],
        1,
        [(EQ, L, H)],
        [(MED, M, H)],
    ),
    _g(
        "Fire/Vega",
        "WIKI",
        ["Las Vegas (Vega)", "Lodestone (Fire)"],
        1,
        [(FOOD, L, H)],
        [(IND, L, M)],
    ),
    _g(
        "Lalande 21185/South Manchester",
        "WIKI",
        ["Diva (Lalande 21185)", "Cornwall (South Manchester)"],
        1,
        [(METAL, L, M), (IND, L, M)],
        [(EQ, L, M), (FOOD, M, H)],
    ),
    _g(
        "Kania/Fomalhaut",
        "WIKI",
        ["Port Kane (Kania)", "Gem (Fomalhaut)"],
        2,
        [(EQ, L, H)],
        [(METAL, L, M)],
    ),
    _g(
        "Sol/Sirius", "WIKI", ["Earth (Sol)", "Sirius II (Sirius)"], 2, [(MED, L, H)], [(LUX, M, H)]
    ),
    _g(
        "Sol/Tuatha",
        "WIKI",
        ["Earth (Sol)", "New Ireland (Tuatha)"],
        3,
        [(MED, L, H)],
        [(LUX, L, H)],
    ),
    _g(
        "Porto Rillia/Arcturus",
        "WIKI",
        ["Ryll (Porto Rillia)", "Boral I (Arcturus)"],
        3,
        [(FOOD, L, H)],
        [(METAL, L, H)],
    ),
    _g(
        "Aldebaran/Lesten",
        "WIKI",
        ["Menin (Lesten)", "Haven (Aldebaran)"],
        3,
        [(EQ, L, H)],
        [(FOOD, L, H)],
    ),
    _g(
        "Enlightenment/Nesre Primus",
        "WIKI",
        [
            "Xyon (Enlightenment)",
            "New Babylon (Nesre Primus)",
            "Georgia (Nesre Primus)",
            "Kismet (Enlightenment)",
        ],
        (3, 0, 3, 0),
        [(LUX, L, H)],
        [],
        [(IND, L, H)],
        [],
    ),
    _g("Kania/Aria", "WIKI", ["Port Kane (Kania)", "Aria (Aria)"], 3, [(EQ, L, H)], [(FOOD, L, H)]),
    _g(
        "Dani/Dani Fort",
        "WIKI",
        ["Forticus (Dani Fort)", "Dani (Dani)"],
        1,
        [(EQ, L, H)],
        [(MED, M, H)],
    ),
    _g(
        "Moash/Moash Teckra",
        "WIKI",
        ["Moash (Moash)", "Feret (Moash Teckra)"],
        1,
        [(LUX, M, H)],
        [(MED, M, H)],
    ),
    _g(
        "Vella Jocci/Vella Hevel",
        "WIKI",
        ["Joccika (Vella Jocci)", "Retsim (Vella Hevel)"],
        1,
        [(MED, L, M)],
        [(IND, M, H)],
    ),
    _g(
        "Vella Jocci/Vella",
        "WIKI",
        ["Joccika (Vella Jocci)", "Vella (Vella)"],
        1,
        [(MED, L, H)],
        [(IND, M, H)],
    ),
    _g(
        "Vella Jocci/Vella Cak",
        "WIKI",
        ["Joccika (Vella Jocci)", "Firebrand (Vella Cak)"],
        2,
        [(MED, L, M)],
        [(EQ, L, H)],
    ),
    _g(
        "Tekel Kymon/Eguit",
        "WIKI",
        ["Kymonthia (Tekel Kymon)", "Eguit (Eguit)"],
        2,
        [(MED, L, H)],
        [(IND, L, H)],
    ),
    _g("Dani/Pimen", "WIKI", ["Rimerta (Pimen)", "Dani (Dani)"], 3, [(LUX, L, H)], [(MED, M, H)]),
    _g("Pimen/Gary", "WIKI", ["Rimerta (Pimen)", "Schmoo (Gary)"], 4, [(LUX, L, H)], [(MED, M, H)]),
    _g(
        "Heraan Malo/Futurend",
        "WIKI",
        ["Tranquility (Heraan Malo)", "Habilis (Futurend)"],
        4,
        [(MED, L, H)],
        [(EQ, L, H)],
    ),
    _g(
        "Aria/Dani Fort",
        "WIKI",
        ["Aria (Aria)", "Forticus (Dani Fort)"],
        5,
        [(FOOD, L, H)],
        [(EQ, L, H)],
    ),
    _g(
        "Atar/Dani Fort",
        "WIKI",
        ["Atar (Pilot)", "Forticus (Dani Fort)"],
        5,
        [(METAL, L, H)],
        [(EQ, L, H)],
    ),
    _g(
        "Heraan Malo/Olein",
        "WIKI",
        ["Tranquility (Heraan Malo)", "Olos (Olein)"],
        6,
        [(MED, L, H)],
        [(EQ, L, H)],
    ),
    _g(
        "Ver'ashan",
        "WIKI",
        ["Ver'ar Shan (Ver'ashan)", "Ver'ar Sho (Ver'ashan)"],
        0,
        [(EQ, L, H)],
        [(IND, M, H)],
    ),
    _g(
        "Til'hapin/Tre'minor",
        "WIKI",
        ["Tre'a Se (Til'hapin)", "Tre'a Ca (Tre'minor)"],
        2,
        [(EQ, L, H)],
        [(METAL, L, H)],
    ),
    _g(
        "Til'hapin/Tre'vasar",
        "WIKI",
        ["Tre'a Se (Til'hapin)", "Tre'ar Illini (Tre'vasar)"],
        3,
        [(EQ, L, H)],
        [(METAL, L, H)],
    ),
    _g(
        "Ver'ashan/P'aedt",
        "WIKI",
        ["Ver'ar Shan (Ver'ashan)", "P'ar Aed (P'aedt)"],
        4,
        [(EQ, L, H)],
        [(IND, L, H)],
    ),
    # Triangle trade.
    _g(
        "Kania/Sol",
        "WIKI",
        ["Port Kane (Kania)", "Europa (Sol)", "Earth (Sol)"],
        (2, 0, 2),
        [(EQ, L, H)],
        [(LUX, M, H)],
        [(MED, L, H)],
    ),
    # From "Making Money" (the others there repeat routes above or carry special goods).
    _g("Xyon to Earth", "MONEY", ["Xyon (Enlightenment)", "Earth (Sol)"], 2, [(LUX, L, H)], []),
]
