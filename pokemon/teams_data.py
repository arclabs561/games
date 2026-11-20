"""Moves, learnsets and abilities for the team study, from PokeAPI, checked against Showdown.

The study (``teams.py``) needs, for each Pokémon in its pool, the damaging moves it can learn
in Scarlet and Violet and the abilities it can have. This script downloads PokeAPI's CSV
files at the commit ``sources.py`` pins, and Pokémon Showdown's ``moves.ts``, ``learnsets.ts``
and ``pokedex.ts`` (MIT license, the commit ``sources.py`` pins), checks every file's SHA-256,
and writes two small files:

- ``teams_moves.csv``: each damaging move some pool Pokémon learns, with PokeAPI's type,
  category, power, accuracy, priority, hits and drain (negative for recoil), the stat
  changes it makes to its user (from Showdown), and whether the study's rule keeps it;
- ``teams_pokemon.csv``: each pool Pokémon's abilities in slot order and its damaging moves,
  and whether the study leaves it out: a forme that Showdown keeps only while it holds an
  item (the Ogerpon masks) cannot be played without items, so it is left out.

**The rule for moves.** A move is kept when its power is fixed: Showdown gives it no handler
that changes power, type, category, timing or success (``basePowerCallback``, ``onBasePower``,
``onTry``, ``onModifyType`` and the others in ``EXCLUDED_HANDLERS``), and none of the
properties in ``EXCLUDED_KEYS`` (charging or recharging turns, fixed damage, one-hit KOs,
fainting the user, switching out, guaranteed critical hits, attacks with another stat, and
so on). Six moves whose only handler changes accuracy in weather are kept. Secondary effects
are ignored. Knock Off, Sucker Punch, U-turn, Freeze-Dry and Body Press are excluded by this
rule; the README says so.

**Gaps in PokeAPI.** PokeAPI's ``move_meta.csv`` has no row for moves added since Legends:
Arceus, so for those the hits and drain come from Showdown (the ``meta`` column says which).
PokeAPI gives Tachyon Cutter accuracy 0; it never misses, and ``ACCURACY_FIXES`` says so.

**Checks.** For every kept move, PokeAPI and Showdown agree on type, category, power,
accuracy, priority, hits and drain. For every pool Pokémon, they agree on typing, base stats
and abilities, and every kept move it learns in PokeAPI's Scarlet and Violet data is also in
Showdown's Generation 9 learnset. The build stops on any disagreement.

    uv run --group pokemon python -m pokemon.teams_data            # download, check, write
    uv run --group pokemon python -m pokemon.teams_data --check    # also fail if files differ
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from pokemon import pokedex, usage
from pokemon.sources import (
    POKEAPI_COMMIT,
    POKEAPI_URL,
    SHOWDOWN_COMMIT,
    SHOWDOWN_URL,
    fetch,
    read_table,
    sha256,
    to_csv,
)

HERE = Path(__file__).resolve().parent
MOVES_CSV = HERE / "teams_moves.csv"
POKEMON_CSV = HERE / "teams_pokemon.csv"
USAGE_JSON = HERE / "results" / "usage-pokemon.json"
SCARLET_VIOLET = "25"  # PokeAPI's version group

POKEAPI_FILES = {
    "data/v2/csv/pokemon.csv": "eb8a7f59134bf3763a0f3072a7794cb097413e6f6027c380a92dd8d63082ab80",
    "data/v2/csv/moves.csv": "bda125c910e66d1117e7c9547f981aa137d9fb70fe1cba417d8c85590618d7eb",
    "data/v2/csv/move_meta.csv": (
        "93b92d5bddf4fc1536ca0a647c20453bf92e6389010f33ca6fc4c5e9c54363be"
    ),
    "data/v2/csv/pokemon_moves.csv": (
        "24e0ace4ae8841e8879864b11247a5ad7058f34898b53f89df541284fc0847df"
    ),
    "data/v2/csv/pokemon_abilities.csv": (
        "4a79ee53d386a8ad3a89657483ddfd31594e05a5d52400fe131e5e4dfd7ec04a"
    ),
    "data/v2/csv/abilities.csv": (
        "74c3588ad48e08e54ff01f432899a4028c3e91508bbd53201df85e1f6b9e9ef3"
    ),
    "data/v2/csv/types.csv": "37f039c8d722f47d51ba1c5c5ecf9b7007235b1a9a1af2827645c777b70307c8",
}
SHOWDOWN_FILES = {
    "data/moves.ts": "19d96eac13d7d89f13b42a975672ee75b8ad4dac011c66898ea811f2e3ef7681",
    "data/learnsets.ts": "cdab542e58b6d276122cce4ee2ff5597123f773cd1d053ecbfd4b81a54d1bafb",
    "data/pokedex.ts": usage.SHOWDOWN_POKEDEX["data/pokedex.ts"],
}

CATEGORIES = {"2": "physical", "3": "special"}  # PokeAPI's damage classes; 1 is status
EXCLUDED_HANDLERS = frozenset(
    {
        "basePowerCallback",
        "damageCallback",
        "beforeTurnCallback",
        "beforeMoveCallback",
        "priorityChargeCallback",
        "onBasePower",
        "onModifyType",
        "onModifyMove",
        "onModifyPriority",
        "onEffectiveness",
        "onTry",
        "onTryMove",
        "onTryImmunity",
        "onPrepareHit",
        "onDisableMove",
        "onMoveFail",
        "onDamage",
        "onModifyTarget",
    }
)
EXCLUDED_KEYS = frozenset(
    {
        "isZ",
        "isMax",
        "ohko",
        "damage",
        "selfdestruct",
        "willCrit",
        "selfSwitch",
        "forceSwitch",
        "sleepUsable",
        "overrideOffensiveStat",
        "overrideDefensiveStat",
        "overrideOffensivePokemon",
        "mindBlownRecoil",
        "chloroblastRecoil",
        "struggleRecoil",
        "multiaccuracy",
        "stealsBoosts",
        "callsMove",
    }
)
EXCLUDED_FLAGS = frozenset({"charge", "recharge"})
# Their only excluded handler is onModifyMove, which changes accuracy in weather.
WEATHER_ACCURACY = frozenset(
    {"blizzard", "thunder", "hurricane", "bleakwindstorm", "wildboltstorm", "sandsearstorm"}
)
# PokeAPI gives Tachyon Cutter accuracy 0; the move never misses (Showdown: accuracy true).
ACCURACY_FIXES = {"tachyon-cutter": ""}
BOOST_ORDER = ("atk", "def", "spa", "spd", "spe")
# Expected hits of a move that hits 2 to 5 times: 2 and 3 hits with chance 7/20 each,
# 4 and 5 with chance 3/20 each (Generation 5 on).
TWO_TO_FIVE = Fraction(31, 10)

Table = list[dict[str, str]]


# Downloading.


def download(cache: Path) -> tuple[dict[str, Table], dict[str, str]]:
    """PokeAPI's tables by name and Showdown's files as text, from ``cache`` if there."""
    got: list[dict[str, bytes]] = []
    for url, commit, files in (
        (POKEAPI_URL, POKEAPI_COMMIT, POKEAPI_FILES),
        (SHOWDOWN_URL, SHOWDOWN_COMMIT, SHOWDOWN_FILES),
    ):
        out = {}
        for path, digest in files.items():
            local = cache / commit / path
            if local.exists() and sha256(local.read_bytes()) == digest:
                out[path] = local.read_bytes()
                continue
            out[path] = fetch(url.format(commit=commit, path=path), digest)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(out[path])
        got.append(out)
    tables = {Path(p).stem: read_table(d) for p, d in got[0].items()}
    return tables, {p: d.decode("utf-8") for p, d in got[1].items()}


# Showdown's files.

ENTRY = re.compile(r"^\t(\w+): \{\n(.*?)^\t\},?$", re.S | re.M)


def parse_moves(text: str) -> dict[str, dict]:
    """{move id: properties} from Showdown's moves.ts, for the fields the study uses."""
    out = {}
    for key, body in ENTRY.findall(text):

        def field(name: str, body: str = body) -> str | None:
            m = re.search(rf"^\t\t{name}: (.*?),$", body, re.M)
            return m.group(1) if m else None

        flags = re.search(r"^\t\tflags: \{(.*?)\},$", body, re.M)
        self_boosts = re.search(r"^\t\tself: \{\n\t\t\tboosts: \{(.*?)\}", body, re.S | re.M)
        out[key] = {
            "name": (field("name") or '""').strip('"'),
            "type": (field("type") or '""').strip('"').lower(),
            "category": (field("category") or '""').strip('"').lower(),
            "power": int(field("basePower") or 0),
            "accuracy": None if field("accuracy") == "true" else int(field("accuracy") or 0),
            "priority": int(field("priority") or 0),
            "multihit": field("multihit"),
            "recoil": field("recoil"),
            "drain": field("drain"),
            "nonstandard": field("isNonstandard"),
            "flags": set(re.findall(r"(\w+): 1", flags.group(1))) if flags else set(),
            "keys": set(re.findall(r"^\t\t(\w+):", body, re.M)),
            "handlers": set(re.findall(r"^\t\t(\w+)\(", body, re.M)),
            "boosts": {k: int(v) for k, v in re.findall(r"(\w+): (-?\d+)", self_boosts.group(1))}
            if self_boosts
            else {},
        }
    return out


def parse_learnsets(text: str) -> dict[str, set[str]]:
    """{species id: moves learnable in Generation 9} from Showdown's learnsets.ts."""
    out = {}
    for key, body in ENTRY.findall(text):
        moves = set()
        for move, codes in re.findall(r"^\t\t\t(\w+): \[(.*?)\]", body, re.M):
            if any(c.startswith('"9') for c in codes.split(", ")):
                moves.add(move)
        out[key] = moves
    return out


def parse_species(text: str) -> dict[str, dict]:
    """{species id: typing, base stats, abilities, and the species it changes from}."""
    out = {}
    for key, body in ENTRY.findall(text):
        types = re.search(r"^\t\ttypes: \[(.*?)\]", body, re.M)
        stats = re.search(r"^\t\tbaseStats: \{(.*?)\}", body, re.M)
        abilities = re.search(r"^\t\tabilities: \{(.*?)\}", body, re.M)
        required = re.search(r"^\t\trequiredItems?: (.*?),$", body, re.M)
        battle_only = re.search(r"^\t\tbattleOnly: ", body, re.M)
        parents = re.findall(r'^\t\t(?:changesFrom|baseSpecies): "(.*?)"', body, re.M)
        out[key] = {
            "types": tuple(t.lower() for t in re.findall(r'"(\w+)"', types.group(1)))
            if types
            else (),
            "stats": {k: int(v) for k, v in re.findall(r"(\w+): (\d+)", stats.group(1))}
            if stats
            else {},
            # Slot S is a special ability from an event (Greninja's Battle Bond); PokeAPI
            # gives that one to a form of its own.
            "abilities": [
                usage.to_id(a)
                for k, a in re.findall(r'(\w+): "(.*?)"', abilities.group(1))
                if k != "S"
            ]
            if abilities
            else [],
            "parents": [usage.to_id(p) for p in parents],
            "excluded": "needs an item" if required else "battle only" if battle_only else "",
        }
    return out


# The rule.


def showdown_id(identifier: str) -> str:
    """A PokeAPI identifier as a Showdown ID: 'close-combat' -> 'closecombat'."""
    return usage.to_id(identifier)


def reason_excluded(m: dict) -> str:
    """Why the study's rule leaves a Showdown move out, or '' if it keeps it."""
    if m["nonstandard"]:
        return "nonstandard"
    flags = sorted(m["flags"] & EXCLUDED_FLAGS)
    if flags:
        return flags[0]
    keys = sorted(m["keys"] & EXCLUDED_KEYS)
    if keys:
        return keys[0]
    handlers = m["handlers"] & EXCLUDED_HANDLERS
    if m["handlers"] == {"onModifyMove"} and m["id"] in WEATHER_ACCURACY:
        return ""
    if handlers:
        return sorted(handlers)[0]
    return ""


def showdown_hits(m: dict) -> Fraction:
    """Expected hits from Showdown's ``multihit``: 3 -> 3, [2, 5] -> 31/10."""
    h = m["multihit"]
    if h is None:
        return Fraction(1)
    if h == "[2, 5]":
        return TWO_TO_FIVE
    return Fraction(int(h))


def showdown_drain(m: dict) -> Fraction:
    """Drain as a share of damage dealt; recoil is negative. [1, 2] -> 1/2, [33, 100] -> -33/100."""
    for field, sign in (("drain", 1), ("recoil", -1)):
        if m[field]:
            n, d = map(int, re.findall(r"\d+", m[field]))
            return sign * Fraction(n, d)
    return Fraction(0)


def pokeapi_hits(meta: dict[str, str]) -> Fraction:
    """Expected hits from PokeAPI's min_hits and max_hits."""
    lo, hi = meta.get("min_hits") or "1", meta.get("max_hits") or "1"
    if (lo, hi) == ("2", "5"):
        return TWO_TO_FIVE
    assert lo == hi, meta
    return Fraction(int(lo))


def boosts_label(b: dict[str, int]) -> str:
    """{'def': -1, 'spd': -1} -> 'def:-1 spd:-1', in a fixed order."""
    return " ".join(f"{k}:{b[k]}" for k in BOOST_ORDER if b.get(k))


# The build.


def pool() -> list[dict]:
    """The Pokémon of ``results/usage-pokemon.json``, most used first."""
    return json.loads(USAGE_JSON.read_text(encoding="utf-8"))["pokemon"]


def build(t: dict[str, Table], sd: dict[str, str]) -> tuple[Table, Table]:
    """The rows of ``teams_moves.csv`` and ``teams_pokemon.csv``, after every check."""
    moves_sd = parse_moves(sd["data/moves.ts"])
    for k, m in moves_sd.items():
        m["id"] = k
    learn_sd = parse_learnsets(sd["data/learnsets.ts"])
    species_sd = parse_species(sd["data/pokedex.ts"])
    type_name = {r["id"]: r["identifier"] for r in t["types"]}
    ability_name = {r["id"]: r["identifier"] for r in t["abilities"]}
    moves = {r["id"]: r for r in t["moves"]}
    meta = {r["move_id"]: r for r in t["move_meta"]}
    pid = {r["identifier"]: r["id"] for r in t["pokemon"]}
    learn: dict[str, set[str]] = defaultdict(set)
    for r in t["pokemon_moves"]:
        if r["version_group_id"] == SCARLET_VIOLET:
            learn[r["pokemon_id"]].add(r["move_id"])
    abilities: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in t["pokemon_abilities"]:
        abilities[r["pokemon_id"]].append((int(r["slot"]), ability_name[r["ability_id"]]))

    problems: list[str] = []
    move_rows: dict[str, dict[str, str]] = {}
    pokemon_rows = []
    for p in pool():
        ident, sid = p["identifier"], usage.to_id(p["showdown"])
        form = usage.form(ident)
        spec = species_sd[sid]
        if pokedex.typing(form, 9) != pokedex.canonical(spec["types"]):
            problems.append(f"{ident}: typing differs")
        ours = pokedex.base_stats(form, 9)
        theirs = dict(zip(pokedex.STATS, (spec["stats"][k] for k in SD_STATS), strict=True))
        if ours != theirs:
            problems.append(f"{ident}: base stats {ours} here, {theirs} in Showdown")
        abil = [a for _, a in sorted(abilities[pid[ident]])]
        if [showdown_id(a) for a in abil] != spec["abilities"]:
            problems.append(f"{ident}: abilities {abil} here, {spec['abilities']} in Showdown")
        sd_learn = set(learn_sd.get(sid, set()))
        for parent in spec["parents"]:
            sd_learn |= learn_sd.get(parent, set())
        damaging = []
        for mid in sorted(learn[pid[ident]], key=lambda m: moves[m]["identifier"]):
            m = moves[mid]
            if m["damage_class_id"] not in CATEGORIES:
                continue
            name = m["identifier"]
            msd = moves_sd[showdown_id(name)]
            reason = reason_excluded(msd)
            if not m["power"]:
                reason = reason or "no power"
            damaging.append(name)
            if not reason and showdown_id(name) not in sd_learn:
                problems.append(f"{ident}: {name} not in Showdown's Generation 9 learnset")
            if name in move_rows:
                continue
            has_meta = mid in meta
            row = {
                "move": name,
                "type": type_name[m["type_id"]],
                "category": CATEGORIES[m["damage_class_id"]],
                "power": m["power"],
                "accuracy": ACCURACY_FIXES.get(name, m["accuracy"]),
                "priority": m["priority"],
                "hits": str(pokeapi_hits(meta[mid]) if has_meta else showdown_hits(msd)),
                "drain": str(
                    Fraction(int(meta[mid]["drain"]), 100) if has_meta else showdown_drain(msd)
                ),
                "boosts": boosts_label(msd["boosts"]),
                "meta": "pokeapi" if has_meta else "showdown",
                "kept": "0" if reason else "1",
                "reason": reason,
            }
            if not reason:
                problems += check_move(row, msd)
            move_rows[name] = row
        pokemon_rows.append(
            {
                "identifier": ident,
                "showdown": p["showdown"],
                "abilities": " ".join(abil),
                "moves": " ".join(damaging),
                "excluded": spec["excluded"],
            }
        )
    if problems:
        raise ValueError("\n".join(problems))
    return sorted(move_rows.values(), key=lambda r: r["move"]), pokemon_rows


SD_STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def check_move(row: dict[str, str], m: dict) -> list[str]:
    """Disagreements between PokeAPI's row and Showdown's entry for a kept move."""
    sd = {
        "type": m["type"],
        "category": m["category"],
        "power": str(m["power"]),
        "accuracy": "" if m["accuracy"] is None else str(m["accuracy"]),
        "priority": str(m["priority"]),
        "hits": str(showdown_hits(m)),
        "drain": str(showdown_drain(m)),
    }
    return [
        f"{row['move']}: {k} {row[k]!r} here, {v!r} in Showdown"
        for k, v in sd.items()
        if row[k] != v
    ]


MOVE_COLUMNS = (
    "move",
    "type",
    "category",
    "power",
    "accuracy",
    "priority",
    "hits",
    "drain",
    "boosts",
    "meta",
    "kept",
    "reason",
)
POKEMON_COLUMNS = ("identifier", "showdown", "abilities", "moves", "excluded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, help="keep the downloads here")
    parser.add_argument("--check", action="store_true", help="fail if the files would change")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        t, sd = download(args.cache or Path(tmp))
    move_rows, pokemon_rows = build(t, sd)
    for path, rows, columns in (
        (MOVES_CSV, move_rows, MOVE_COLUMNS),
        (POKEMON_CSV, pokemon_rows, POKEMON_COLUMNS),
    ):
        text = to_csv(rows, columns)
        if args.check:
            assert path.read_text(encoding="utf-8") == text, f"{path.name} is out of date"
        else:
            path.write_text(text, encoding="utf-8")
        print(f"{path.name}: {len(rows)} rows")
    kept = sum(r["kept"] == "1" for r in move_rows)
    print(f"{kept} of {len(move_rows)} damaging moves kept; checks against Showdown passed")


if __name__ == "__main__":
    main()
