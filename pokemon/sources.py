"""Build the Pokémon data files from PokeAPI, and check the type charts against two sources.

PokeAPI (https://github.com/PokeAPI/pokeapi) keeps its data as CSV files. This script downloads
the few it needs at a pinned commit, checks their SHA-256, and writes two small files:

- ``pokeapi_forms.csv``: one row per Pokémon or form, with its species, National Pokédex
  number, the generation it was introduced in, legendary, mythical and Mega flags, its typing
  in each generation from 1 to 9 (blank before it existed), and its current base stats.
- ``pokeapi_stats_past.csv``: earlier base stats, as (form, last generation, stat, value).

Gigantamax, totem and the Let's Go partner (``-starter``) rows are left out: they are
special battle states of a species, not forms with their own entry. No names, flavor text or
other prose are copied, only identifiers and numbers.

It also rebuilds every type chart from PokeAPI (``type_efficacy.csv`` and
``type_efficacy_past.csv``) and from Pokémon Showdown's per-generation chart files (MIT
license, pinned commit), and checks both against ``pokemon_types.chart``.

    uv run --group pokemon python -m pokemon.sources            # download, check, rebuild
    uv run --group pokemon python -m pokemon.sources --check    # also fail if the files differ
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import tempfile
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pokemon.pokemon_types import chart

HERE = Path(__file__).resolve().parent
FORMS_CSV = HERE / "pokeapi_forms.csv"
STATS_PAST_CSV = HERE / "pokeapi_stats_past.csv"

POKEAPI_COMMIT = "e98f2ad82fc61c46036689ecb10c3aa89d562d6b"
POKEAPI_URL = "https://raw.githubusercontent.com/PokeAPI/pokeapi/{commit}/{path}"
POKEAPI_FILES = {
    "LICENSE.md": "1c04595dc662981c20956ae4251dfa7b62b475c345eddf98bc9b904cb2373f2c",
    "data/v2/csv/pokemon.csv": "eb8a7f59134bf3763a0f3072a7794cb097413e6f6027c380a92dd8d63082ab80",
    "data/v2/csv/pokemon_forms.csv": (
        "99bf8f7ad4dc1f2e291357a090cef6a575623ec3cbf9030d0e33656e6e608ae2"
    ),
    "data/v2/csv/pokemon_species.csv": (
        "e66e2eeb25fd3836b0ebab6bf87bbf01960aa3c0555e2bac495fa8393c5e0c45"
    ),
    "data/v2/csv/pokemon_stats.csv": (
        "fa2c44263a3706468682fefc3e0b3c4f5487fe9febed7d4ebcd6939c34b16aa8"
    ),
    "data/v2/csv/pokemon_stats_past.csv": (
        "0a074d9581369c1df13d8d4f4015fd7215b896c5dfb58107f3b529f7156950d2"
    ),
    "data/v2/csv/pokemon_types.csv": (
        "f1fc4bfd657a034ea3bf6972423b10276424aa068577b304a78a08996425ba05"
    ),
    "data/v2/csv/pokemon_types_past.csv": (
        "02553c38e3871f99c7ed809b944066fea3f42ef6bccd4daba19274aca01fbff0"
    ),
    "data/v2/csv/stats.csv": "ee8c541b45b64f40f0eb2545c767294b5801ce8a1410e19d8c33a54e6f1c3b9c",
    "data/v2/csv/type_efficacy.csv": (
        "cdd7de4066680414d0a2433af805f139f425d6350cbf998ecab559a0fe8ff587"
    ),
    "data/v2/csv/type_efficacy_past.csv": (
        "0d8cb1251c7f987c7978d77d6c1b9ecfc54e04474b7584b8ecf96fef498fb9d4"
    ),
    "data/v2/csv/types.csv": "37f039c8d722f47d51ba1c5c5ecf9b7007235b1a9a1af2827645c777b70307c8",
    "data/v2/csv/version_groups.csv": (
        "28da8d89d8eb4966941f81a9e62b3990510ed4d76dd774158246551a8e7707a7"
    ),
}

SHOWDOWN_COMMIT = "5b88c476113d51a939bddbcd00b5cd182a5d8058"
SHOWDOWN_URL = "https://raw.githubusercontent.com/smogon/pokemon-showdown/{commit}/{path}"
# Generation 9 is data/typechart.ts. Each older generation's mod inherits from the next newer
# one and overrides some types; generations 3, 4 and 7 override none, so they have no file.
SHOWDOWN_FILES = {
    "data/typechart.ts": "7b0eae126bdcf98edfd71cbe75af00b763f5f0d5db36024b6f3e7c5e4aea95ac",
    "data/mods/gen8/typechart.ts": (
        "cce8b487a56b13a05440e070bb7fadba6d7b625a9a78721f127b5139b03c739a"
    ),
    "data/mods/gen6/typechart.ts": (
        "725556aae95658f988b5690167e281d37c3a5d9eaf2afab0a10739a095c4a172"
    ),
    "data/mods/gen5/typechart.ts": (
        "e97ecc4391101613d626498dd485ef663ed1e7a93c4061bd7053c6eec4f2d138"
    ),
    "data/mods/gen2/typechart.ts": (
        "fa9f85693fee8edfccc4bbab14134b73fe4f4a6e3a7a02262cba52ab03aea17b"
    ),
    "data/mods/gen1/typechart.ts": (
        "dfacb37a1737d449c66a6b037b996862ade659cc6c140055c4ac6750f264e8d2"
    ),
}

GENERATIONS = range(1, 10)
STATS = ("hp", "attack", "defense", "sp_attack", "sp_defense", "speed")
# PokeAPI's stat identifiers, in the same order, then Generation 1's single Special stat.
POKEAPI_STATS = {
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "special-attack": "sp_attack",
    "special-defense": "sp_defense",
    "speed": "speed",
    "special": "special",
}
LEFT_OUT = ("-gmax", "-totem", "-starter")  # identifier endings, or infixes for totems
FORM_COLUMNS = (
    "identifier",
    "species",
    "ndex",
    "generation",
    "legendary",
    "mythical",
    "mega",
    *(f"types_{g}" for g in GENERATIONS),
    *STATS,
)

Table = list[dict[str, str]]


# Downloading.


def sha256(data: bytes) -> str:
    """Hex SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, expected: str) -> bytes:
    """Download a file and check its SHA-256."""
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if sha256(data) != expected:
        raise ValueError(f"{url}: SHA-256 {sha256(data)}, expected {expected}")
    return data


def download(cache: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The PokeAPI and Showdown files, from ``cache`` if present there, checked by hash."""
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
    return got[0], got[1]


def read_table(data: bytes) -> Table:
    """A CSV file as a list of rows."""
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def pokeapi_tables(files: dict[str, bytes]) -> dict[str, Table]:
    """{'pokemon': rows, 'types': rows, ...} from the downloaded CSV files."""
    return {
        Path(path).stem: read_table(data) for path, data in files.items() if path.endswith(".csv")
    }


# The extract.


def is_left_out(identifier: str) -> bool:
    """Gigantamax, totem and Let's Go partner rows."""
    return any(x in identifier for x in LEFT_OUT)


def typing_label(types: Iterable[str]) -> str:
    """'electric/steel': the types in slot order, joined by '/'."""
    return "/".join(types)


def build_forms(t: dict[str, Table]) -> tuple[Table, Table]:
    """The rows of ``pokeapi_forms.csv`` and ``pokeapi_stats_past.csv``."""
    type_name = {r["id"]: r["identifier"] for r in t["types"]}
    stat_name = {
        r["id"]: POKEAPI_STATS[r["identifier"]]
        for r in t["stats"]
        if r["identifier"] in POKEAPI_STATS
    }
    vg_generation = {r["id"]: int(r["generation_id"]) for r in t["version_groups"]}
    species = {r["id"]: r for r in t["pokemon_species"]}

    # A Pokémon's own form is its default form. Eight ride forms of Koraidon and Miraidon
    # have one form each, none marked default; that form is theirs.
    own: dict[str, list[dict[str, str]]] = defaultdict(list)
    for f in t["pokemon_forms"]:
        own[f["pokemon_id"]].append(f)
    introduced: dict[str, int] = {}
    mega: dict[str, bool] = {}
    for pid, fs in own.items():
        defaults = [f for f in fs if f["is_default"] == "1"] or fs
        assert len(defaults) == 1, fs
        introduced[pid] = vg_generation[defaults[0]["introduced_in_version_group_id"]]
        mega[pid] = defaults[0]["is_mega"] == "1"

    current: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for r in t["pokemon_types"]:
        current[r["pokemon_id"]].append((int(r["slot"]), type_name[r["type_id"]]))
    past: dict[str, dict[int, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in t["pokemon_types_past"]:
        past[r["pokemon_id"]][int(r["generation_id"])].append(
            (int(r["slot"]), type_name[r["type_id"]])
        )
    stats: dict[str, dict[str, int]] = defaultdict(dict)
    for r in t["pokemon_stats"]:
        if r["stat_id"] in stat_name:
            stats[r["pokemon_id"]][stat_name[r["stat_id"]]] = int(r["base_stat"])

    def typing_in(pid: str, generation: int) -> str:
        # A past typing holds up to and including its generation; the earliest such wins.
        later = sorted(g for g in past[pid] if g >= generation)
        slots = past[pid][later[0]] if later else current[pid]
        return typing_label(t for _, t in sorted(slots))

    forms, kept = [], {}
    for p in t["pokemon"]:
        if is_left_out(p["identifier"]):
            continue
        pid, s = p["id"], species[p["species_id"]]
        kept[pid] = p["identifier"]
        row = {
            "identifier": p["identifier"],
            "species": s["identifier"],
            "ndex": s["id"],
            "generation": str(introduced[pid]),
            "legendary": s["is_legendary"],
            "mythical": s["is_mythical"],
            "mega": str(int(mega[pid])),
        }
        for g in GENERATIONS:
            row[f"types_{g}"] = typing_in(pid, g) if g >= introduced[pid] else ""
        row.update({k: str(stats[pid][k]) for k in STATS})
        forms.append(row)

    stats_past = [
        {
            "identifier": kept[r["pokemon_id"]],
            "through_generation": r["generation_id"],
            "stat": stat_name[r["stat_id"]],
            "base_stat": r["base_stat"],
        }
        for r in t["pokemon_stats_past"]
        if r["pokemon_id"] in kept
    ]
    stats_past.sort(key=lambda r: (int(r["through_generation"]), r["identifier"], r["stat"]))
    return forms, stats_past


def to_csv(rows: Table, columns: Iterable[str]) -> str:
    """CSV text with Unix line endings."""
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(columns), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


# Type charts from the two sources.

Chart = dict[str, dict[str, float]]


def full(c: Chart) -> Chart:
    """A chart with every pair written, 1.0 included."""
    return {a: {d: c[a].get(d, 1.0) for d in c} for a in c}


def pokeapi_chart(t: dict[str, Table], generation: int) -> Chart:
    """PokeAPI's chart for a generation, over the types that exist then."""
    name = {r["id"]: r["identifier"] for r in t["types"]}
    exists = {
        r["id"] for r in t["types"] if int(r["generation_id"]) <= generation and int(r["id"]) <= 18
    }
    c: Chart = {name[i]: {} for i in sorted(exists, key=int)}
    for r in t["type_efficacy"]:
        if r["damage_type_id"] in exists and r["target_type_id"] in exists:
            c[name[r["damage_type_id"]]][name[r["target_type_id"]]] = int(r["damage_factor"]) / 100
    # A past row holds through its generation; where several apply, the earliest wins.
    for r in sorted(t["type_efficacy_past"], key=lambda r: -int(r["generation_id"])):
        a, d = r["damage_type_id"], r["target_type_id"]
        if generation <= int(r["generation_id"]) and a in exists and d in exists:
            c[name[a]][name[d]] = int(r["damage_factor"]) / 100
    return c


SHOWDOWN_CODE = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}  # damageTaken: neutral, weak, resists, immune
ENTRY = re.compile(r"^\t(\w+): \{\n(.*?)^\t\},?$", re.S | re.M)
TAKEN = re.compile(r"damageTaken: \{(.*?)\}", re.S)
PAIR = re.compile(r"\b([A-Z]\w*): (\d)")


def parse_showdown(text: str) -> dict[str, dict | None]:
    """{defending type: {attacking type: multiplier}} from a typechart.ts.

    A type marked ``isNonstandard: 'Future'`` maps to None (it does not exist yet). Keys that
    are not capitalized type names, such as ``sandstorm`` or ``prankster``, are skipped.
    """
    out: dict[str, dict | None] = {}
    for name, body in ENTRY.findall(text):
        if "isNonstandard: 'Future'" in body:
            out[name] = None
            continue
        m = TAKEN.search(body)
        if m:
            out[name] = {a.lower(): SHOWDOWN_CODE[int(c)] for a, c in PAIR.findall(m.group(1))}
    return out


def showdown_charts(files: dict[str, bytes]) -> dict[int, Chart]:
    """Showdown's chart for each generation 1 to 9, following the mods' inheritance."""
    taken: dict[str, dict | None] = {}
    charts = {}
    for g in reversed(GENERATIONS):
        path = "data/typechart.ts" if g == 9 else f"data/mods/gen{g}/typechart.ts"
        if path in files:
            taken.update(parse_showdown(files[path].decode("utf-8")))
        rows = {d: row for d, row in taken.items() if row is not None and d != "stellar"}
        charts[g] = {a: {d: rows[d][a] for d in rows} for a in rows}
    return charts


def same_chart(a: Chart, b: Chart) -> bool:
    """Same types and the same multiplier on every pair."""
    fa, fb = full(a), full(b)
    return set(fa) == set(fb) and all(fa[x][y] == fb[x][y] for x in fa for y in fa)


def check_charts(t: dict[str, Table], showdown: dict[str, bytes]) -> None:
    """Fail unless all three sources agree on every generation's chart."""
    sd = showdown_charts(showdown)
    for g in GENERATIONS:
        ours = chart(g)
        assert same_chart(ours, pokeapi_chart(t, g)), f"PokeAPI differs in generation {g}"
        assert same_chart(ours, sd[g]), f"Showdown differs in generation {g}"
        print(f"generation {g}: {len(ours)} types, {len(ours) ** 2} pairs agree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, help="keep the downloads here")
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed files would change"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        pokeapi, showdown = download(args.cache or Path(tmp))
    t = pokeapi_tables(pokeapi)
    check_charts(t, showdown)
    forms, stats_past = build_forms(t)
    for path, rows, columns in (
        (FORMS_CSV, forms, FORM_COLUMNS),
        (STATS_PAST_CSV, stats_past, ("identifier", "through_generation", "stat", "base_stat")),
    ):
        text = to_csv(rows, columns)
        if args.check:
            assert path.read_text(encoding="utf-8") == text, f"{path.name} is out of date"
        else:
            path.write_text(text, encoding="utf-8")
        print(f"{path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
