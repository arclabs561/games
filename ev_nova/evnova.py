"""EV Nova's galaxy and prices, read from a ConText export of the game's data files.

EV Nova's data are Ambrosia Software's copyright and are not in this repository. ConText (by
Ralph and Roger Sutherland) exports the resources of the game files as one tab-separated text
file; `NovaConText.txt` in github.com/vasi/evnova-utils is such an export of EV Nova 1.1.1.

The file is a series of sections. Each starts with a line `• Begin <type>` (for example
`• Begin sÿst`), then a header row of quoted field names, then one row per resource, with
fields separated by tabs. Three sections are read:

- `sÿst` (systems): `ID`, `Name`, `con0`-`con15` (linked system IDs, -1 for none),
  `nav0`-`nav15` (the stellar objects in the system) and `Visiblility` (sic), a control-bit
  test that says when this version of the system exists.
- `spöb` (stellar objects): `ID`, `Name` and `Flags`. Bit 0x2 is a commodity exchange and 0x20
  an uninhabited stellar (no refueling). Four bits per commodity give its price level: food in
  bits 28-31, industrial 24-27, medical 20-23, luxury 16-19, metal 12-15, equipment 8-11, with
  1 for low, 2 for medium, 4 for high and 0 for not traded (Nova Bible, spöb resource).
- `STR#` resource 4004, "Base Prices": the medium price of each of the six commodities.
- `jünk` (special commodities): `SoldAt1`-`8`, `BoughtAt1`-`8`, `BasePrice`, `BuyOn`, `SellOn`.

    uv run python -m ev_nova.study --evnova PATH/NovaConText.txt
"""

import csv
import re
from collections.abc import Callable
from pathlib import Path

from .routes import Galaxy

COMMODITIES = ("Food", "Industrial", "Medical Supplies", "Luxury Goods", "Metal", "Equipment")
SHIFTS = (28, 24, 20, 16, 12, 8)  # where each commodity's price bits sit in spöb Flags
LEVELS = {1: "low", 2: "medium", 4: "high"}


def read_context(path: Path) -> dict[str, list[dict]]:
    """Every section of a ConText export, as a list of rows keyed by field name."""
    sections: dict[str, list[str]] = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("• Begin "):
                current = line.removeprefix("• Begin ").strip()
                sections[current] = []
            elif line.startswith("•"):
                current = None
            elif current is not None and line:
                sections[current].append(line)
    tables = {}
    for name, lines in sections.items():
        rows = list(csv.reader(lines, delimiter="\t"))
        if not rows:
            continue
        head = rows[0]
        # A row can be longer than its header: STR# lists all its strings after "Strings".
        tables[name] = [dict(zip(head, r, strict=False)) | {"_row": r} for r in rows[1:]]
    return tables


def price(base: int, level: int) -> int:
    """The price at a level: base for medium, 80% for low, 125% for high, rounded down.

    The rule is the one the EV Nova wiki's commodity table states; the base prices come from
    the data (STR# 4004).
    """
    return {1: base * 4 // 5, 2: base, 4: base * 5 // 4}[level]


def levels(flags: int) -> dict[str, int]:
    """The price level (1, 2 or 4) of each commodity a stellar trades, from its flags."""
    found = {}
    for name, shift in zip(COMMODITIES, SHIFTS, strict=True):
        level = (flags >> shift) & 0xF
        if level:
            assert level in LEVELS, (flags, name)
            found[name] = level
    return found


# Control-bit tests, such as "!(b147 | b305)". At the start of a game every bit is clear.

TOKEN = re.compile(r"\s*(b\d+|[!&|()])")


def parse_test(text: str) -> Callable[[frozenset[int]], bool]:
    """A function from the set bits to the test's value; an empty test is always true.

    Precedence: ! binds tightest, then &, then |. `test_order_free` checks that the data does
    not depend on this choice.
    """
    tokens = TOKEN.findall(text)
    assert "".join(tokens) == re.sub(r"\s", "", text), text
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def atom():
        tok = take()
        if tok == "!":
            inner = atom()
            return lambda bits: not inner(bits)
        if tok == "(":
            inner = either()
            assert take() == ")", text
            return inner
        n = int(tok[1:])
        return lambda bits: n in bits

    def both():
        left = atom()
        while peek() == "&":
            take()
            right, prev = atom(), left
            left = lambda bits, a=prev, b=right: a(bits) and b(bits)  # noqa: E731
        return left

    def either():
        left = both()
        while peek() == "|":
            take()
            right, prev = both(), left
            left = lambda bits, a=prev, b=right: a(bits) or b(bits)  # noqa: E731
        return left

    if not tokens:
        return lambda bits: True
    result = either()
    assert pos == len(tokens), text
    return result


def left_to_right(text: str, bits: frozenset[int]) -> bool:
    """The same test with & and | at equal precedence, read left to right."""
    tokens = TOKEN.findall(text)
    pos = 0

    def atom():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "!":
            return not atom()
        if tok == "(":
            v = chain()
            pos += 1
            return v
        return int(tok[1:]) in bits

    def chain():
        nonlocal pos
        v = atom()
        while pos < len(tokens) and tokens[pos] in "&|":
            op = tokens[pos]
            pos += 1
            w = atom()
            v = (v and w) if op == "&" else (v or w)
        return v

    return True if not tokens else chain()


def galaxy(
    tables: dict[str, list[dict]], bits: frozenset[int] = frozenset()
) -> tuple[Galaxy, dict[str, dict[str, int]]]:
    """The galaxy when the control bits in `bits` are set (none at the start of a game).

    Returns the galaxy with standard commodities only, and each stop's price levels. Stops are
    named "Stellar (System)"; systems keep their names, which are unique among visible versions.
    """
    base = next(r for r in tables["STR#"] if r["ID"] == "4004")
    first = list(base).index("Strings")
    strings = base["_row"][first : first + len(COMMODITIES)]
    bases = dict(zip(COMMODITIES, (int(s) for s in strings), strict=True))
    visible = [s for s in tables["sÿst"] if parse_test(s["Visiblility"])(bits)]
    names = [s["Name"] for s in visible]
    assert len(names) == len(set(names)), "two visible versions of one system"
    by_id = {s["ID"]: s["Name"] for s in visible}
    links = {
        s["Name"]: tuple(sorted({by_id[s[f"con{i}"]] for i in range(16) if s[f"con{i}"] in by_id}))
        for s in visible
    }
    spobs = {r["ID"]: r for r in tables["spöb"]}
    stop_system, prices, lv, refuel = {}, {}, {}, set()
    for s in visible:
        for i in range(16):
            sid = s[f"nav{i}"]
            if sid not in spobs:
                continue
            flags = int(spobs[sid]["Flags"], 16)
            if flags & 0x1 and not flags & 0x20:  # can land, and inhabited
                refuel.add(s["Name"])
            if not flags & 0x2:
                continue
            stop = f"{spobs[sid]['Name']} ({s['Name']})"
            stop_system[stop] = s["Name"]
            lv[stop] = levels(flags)
            prices[stop] = {c: price(bases[c], k) for c, k in lv[stop].items()}
    return Galaxy(links, stop_system, prices, frozenset(refuel)), lv


def junk_markets(
    tables: dict[str, list[dict]], g: Galaxy, bits: frozenset[int] = frozenset()
) -> dict[str, dict[str, tuple[str, int]]]:
    """For each special commodity whose tests pass: stop -> ("buy" or "sell", price).

    Assumption, not from the data: a special commodity costs 80% of its base price where it is
    sold to the player and fetches 125% where the player sells it, as the wiki's route tables
    label them "(low)" and "(high)".
    """
    spobs = {r["ID"]: r["Name"] for r in tables["spöb"]}
    where = {}
    for stop, sys in g.stop_system.items():
        where.setdefault(stop.removesuffix(f" ({sys})"), []).append(stop)
    found = {}
    for r in tables["jünk"]:
        if not (parse_test(r["BuyOn"])(bits) and parse_test(r["SellOn"])(bits)):
            continue
        base = int(r["BasePrice"])
        market = {}
        for i in range(1, 9):
            for key, side, value in (
                ("SoldAt", "buy", base * 4 // 5),
                ("BoughtAt", "sell", base * 5 // 4),
            ):
                sid = r[f"{key}{i}"]
                for stop in where.get(spobs.get(sid, ""), []):
                    market[stop] = (side, value)
        found[r["Name"]] = market
    return found


def with_junk(g: Galaxy, junk: dict[str, dict[str, tuple[str, int]]]) -> Galaxy:
    """The galaxy with special commodities added as commodities only bought, or only sold.

    A stop that only buys a commodity from the player gets a price at which nobody can buy
    there, and the reverse: a sale-only stop has a high price and a buy-only stop a low one, so
    `best_trade` can only move the goods the intended way.
    """
    prices = {s: dict(p) for s, p in g.prices.items()}
    for name, market in junk.items():
        sells = [v for side, v in market.values() if side == "sell"]
        buys = [v for side, v in market.values() if side == "buy"]
        if not sells or not buys:
            continue
        for stop, (_, value) in market.items():
            prices[stop][name] = value
    return Galaxy(g.links, g.stop_system, prices, g.refuel_systems)
