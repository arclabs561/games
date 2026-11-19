"""Usage shares by typing on the Pokémon Showdown ladder, from Smogon's monthly statistics.

Smogon publishes, for each format, month and rating cutoff, a "chaos" JSON file
(https://www.smogon.com/stats/<month>/chaos/<format>-<cutoff>.json). Its ``data`` maps each
Pokémon's Showdown name to, among other things, ``usage``: the weighted share of teams that
include it. A team has six Pokémon, so the shares sum to about 6. Pokémon below a small
usage threshold are not listed (0.01% in Smogon's public counting scripts).

This module turns those files into usage shares by typing (and by single type), for plan P1
(``experiments/plans/P1.md``). The raw files are not kept: their license is unknown. Only the
derived shares and each file's SHA-256 are written.

**Names.** A Showdown name is mapped to an identifier of ``pokeapi_forms.csv`` by
``identifier``: cosmetic formes become their base species, names are lower-cased with spaces as
hyphens and punctuation dropped (``Mr. Mime`` is ``mr-mime``), Gigantamax and totem names lose
their suffix, a species name with no row of its own becomes its default form (``Landorus`` is
``landorus-incarnate``), and ``ALIASES`` covers the rest. Every name in the files must map,
and its typing in the format's generation must equal Showdown's own (``pokedex.ts`` and its
per-generation mods, MIT license, pinned commit), or the build stops.

    uv run python -m pokemon.usage build --formats gen9ou gen1ou --months 2025-11..2026-10 \\
        --cutoffs 0 1500 1695 1825 --out pokemon/results/usage-typings.json
    uv run python -m pokemon.usage check      # every Showdown species maps, with its typing

The build downloads from smogon.com, which the machine that wrote this cannot reach, so it
runs on GitHub Actions (``experiments/plans/P1.md``).
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path

from pokemon import pokedex
from pokemon.sources import SHOWDOWN_COMMIT, SHOWDOWN_URL, fetch, sha256

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CHAOS_URL = "https://www.smogon.com/stats/{month}/chaos/{format}-{cutoff}.json"
USER_AGENT = "usage-stats-download/1.0"
TEAM_SIZE = 6

# Showdown's species data: Generation 9 in data/pokedex.ts, older generations as mods that
# inherit from the next newer one. Generation 3 has no pokedex mod.
SHOWDOWN_POKEDEX = {
    "data/pokedex.ts": "5e572601f4700f99835845eee5cd3e53362e43d885e2753fcc32418f636b83ed",
    "data/mods/gen8/pokedex.ts": (
        "ef0511011bb28806053e9993f6f466d011933054e7a2ed46c0c8ee2f0771c3a6"
    ),
    "data/mods/gen7/pokedex.ts": (
        "243644f150123187c6941ae89a92f1be3d0ea7e128f22e759d5924b334ce529e"
    ),
    "data/mods/gen6/pokedex.ts": (
        "cf692e61ba60a9affd69e97f12b04b8b6a2d74103bd0ef0df6194bf2c0a19ce5"
    ),
    "data/mods/gen5/pokedex.ts": (
        "ab262749c0ee90268aee92343a186597765daadb12373796aaebe0ab98d48b6f"
    ),
    "data/mods/gen4/pokedex.ts": (
        "8ee17d63fed6f5255e7ec531032e237a2a539782e013dbe013fc845a86d3d2d6"
    ),
    "data/mods/gen2/pokedex.ts": (
        "93b42665a92f2e2916cd99661cc0343b7d54388d141b41969e194c18f7e10717"
    ),
    "data/mods/gen1/pokedex.ts": (
        "ac05fc45df4674aaf4e4fdf70dbc468f6448680c47de85934feba85640548853"
    ),
}

# Showdown names whose PokeAPI identifier the rules in ``identifier`` do not find.
ALIASES = {
    "Darmanitan-Galar": "darmanitan-galar-standard",
    "Tauros-Paldea-Combat": "tauros-paldea-combat-breed",
    "Tauros-Paldea-Blaze": "tauros-paldea-blaze-breed",
    "Tauros-Paldea-Aqua": "tauros-paldea-aqua-breed",
    "Meowstic-F": "meowstic-female",
    "Indeedee-F": "indeedee-female",
    "Basculegion-F": "basculegion-female",
    "Oinkologne-F": "oinkologne-female",
    "Maushold": "maushold-family-of-three",
    "Maushold-Four": "maushold-family-of-four",
    "Squawkabilly-Blue": "squawkabilly-blue-plumage",
    "Squawkabilly-Yellow": "squawkabilly-yellow-plumage",
    "Squawkabilly-White": "squawkabilly-white-plumage",
    "Ogerpon-Wellspring": "ogerpon-wellspring-mask",
    "Ogerpon-Hearthflame": "ogerpon-hearthflame-mask",
    "Ogerpon-Cornerstone": "ogerpon-cornerstone-mask",
    "Ogerpon-Teal-Tera": "ogerpon",
    "Ogerpon-Wellspring-Tera": "ogerpon-wellspring-mask",
    "Ogerpon-Hearthflame-Tera": "ogerpon-hearthflame-mask",
    "Ogerpon-Cornerstone-Tera": "ogerpon-cornerstone-mask",
    "Necrozma-Dusk-Mane": "necrozma-dusk",
    "Necrozma-Dawn-Wings": "necrozma-dawn",
    "Greninja-Bond": "greninja-battle-bond",
    "Rockruff-Dusk": "rockruff-own-tempo",
    "Minior-Meteor": "minior-red-meteor",
    "Xerneas-Neutral": "xerneas",
    "Meowstic-M-Mega": "meowstic-male-mega",
    "Meowstic-F-Mega": "meowstic-female-mega",
    # Formes that differ only in looks or in battle, which PokeAPI's file has no row for.
    "Cherrim-Sunshine": "cherrim",
    "Genesect-Burn": "genesect",
    "Genesect-Chill": "genesect",
    "Genesect-Douse": "genesect",
    "Genesect-Shock": "genesect",
    "Pichu-Spiky-eared": "pichu",
    "Poltchageist-Artisan": "poltchageist",
    "Polteageist-Antique": "polteageist",
    "Sinistcha-Masterpiece": "sinistcha",
    "Sinistea-Antique": "sinistea",
    "Vivillon-Fancy": "vivillon",
    "Vivillon-Pokeball": "vivillon",
    **{
        f"Pikachu-{cap}": f"pikachu-{cap.lower()}-cap"
        for cap in ("Original", "Hoenn", "Sinnoh", "Unova", "Kalos", "Alola", "Partner", "World")
    },
}
# Not mapped: the Arceus and Silvally formes, whose typing PokeAPI's file does not give.
# Neither is legal in the formats of plan P1; a file that names one stops the build.
# Battle states of a species that PokeAPI's file leaves out (``sources.LEFT_OUT``).
SUFFIXES = ("-gmax", "-totem", "-starter")

Typing = tuple[str, ...]
Chaos = dict
Species = dict[str, dict]


# Showdown's species data.


ENTRY = re.compile(r"^\t(\w+): \{\n(.*?)^\t\},?$", re.S | re.M)


def unescape(s: str) -> str:
    """A TypeScript string literal's text: ``\\u2019`` becomes ’."""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)


def parse_pokedex(text: str) -> Species:
    """{key: {name, types, base, cosmetic}} from a pokedex.ts; a mod's entries may lack fields.

    ``types`` is a tuple of lower-case types or None if the entry does not set them; ``base``
    is the base species' name for a cosmetic forme, else None; ``cosmetic`` lists the names
    of cosmetic formes the entry declares.
    """
    out: Species = {}
    for key, body in ENTRY.findall(text):
        name = re.search(r'^\t\tname: "(.*?)"', body, re.M)
        types = re.search(r"^\t\ttypes: \[(.*?)\]", body, re.M)
        cosmetic = re.search(r"^\t\tcosmeticFormes: \[(.*?)\]", body, re.M)
        base = re.search(r'^\t\tbaseSpecies: "(.*?)"', body, re.M)
        out[key] = {
            "name": unescape(name.group(1)) if name else None,
            "types": tuple(t.lower() for t in re.findall(r'"(\w+)"', types.group(1)))
            if types
            else None,
            "base": unescape(base.group(1)) if base and "isCosmeticForme: true" in body else None,
            "cosmetic": [unescape(c) for c in re.findall(r'"(.*?)"', cosmetic.group(1))]
            if cosmetic
            else [],
            "num": int(m.group(1)) if (m := re.search(r"^\t\tnum: (-?\d+)", body, re.M)) else None,
        }
    return out


def showdown_typings(files: dict[str, str], generation: int) -> tuple[dict[str, Typing], dict]:
    """Showdown's typing of every species name as of a generation, and each cosmetic forme's base.

    Starts from Generation 9's pokedex.ts and applies each mod's types from Generation 8 down
    to the one asked for. Cosmetic formes take their base species' typing. Species with a
    non-positive number (Create-A-Pokémon and test entries) are left out.
    """
    dex = parse_pokedex(files["data/pokedex.ts"])
    types = {k: e["types"] for k, e in dex.items()}
    for g in range(8, generation - 1, -1):
        path = f"data/mods/gen{g}/pokedex.ts"
        for k, e in parse_pokedex(files.get(path, "")).items():
            if e["types"] is not None and k in types:
                types[k] = e["types"]
    typings: dict[str, Typing] = {}
    base_of: dict[str, str] = {}
    for k, e in dex.items():
        if e["name"] is None:
            continue
        if e["base"] is not None:
            base_of[e["name"]] = e["base"]
            continue
        if e["num"] is None or e["num"] <= 0:
            continue
        typings[e["name"]] = pokedex.canonical(types[k])
        for c in e["cosmetic"]:
            base_of[c] = e["name"]
    for c, b in base_of.items():
        if b in typings:
            typings.setdefault(c, typings[b])
    return typings, base_of


def download_showdown(cache: Path) -> dict[str, str]:
    """Showdown's pokedex files at the pinned commit, checked by SHA-256."""
    out = {}
    for path, digest in SHOWDOWN_POKEDEX.items():
        local = cache / SHOWDOWN_COMMIT / path
        if local.exists() and sha256(local.read_bytes()) == digest:
            out[path] = local.read_text(encoding="utf-8")
            continue
        data = fetch(SHOWDOWN_URL.format(commit=SHOWDOWN_COMMIT, path=path), digest)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        out[path] = data.decode("utf-8")
    return out


# Names.


def slug(name: str) -> str:
    """'Mr. Mime' -> 'mr-mime', 'Farfetch’d' -> 'farfetchd', 'Flabébé' -> 'flabebe'."""
    s = name.replace("♀", "-f").replace("♂", "-m")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(" ", "-")
    return re.sub(r"[.'’:%]", "", s)


@cache
def default_forms() -> dict[str, str]:
    """Each species' first row in PokeAPI's order, which is its default form."""
    first: dict[str, str] = {}
    for f in pokedex.forms():
        first.setdefault(f["species"], f["identifier"])
    return first


def identifier(name: str, base_of: dict[str, str] | None = None) -> str | None:
    """The ``pokeapi_forms.csv`` identifier of a Showdown name, or None if no rule finds one."""
    if base_of and name in base_of:
        name = base_of[name]
    if name in ALIASES:
        return ALIASES[name]
    ids = by_identifier()
    first = default_forms()
    s = slug(name)
    for suffix in SUFFIXES:
        if s.endswith(suffix):
            s = s.removesuffix(suffix)
    if s in ids:
        return s
    return first.get(s)


@cache
def by_identifier() -> dict[str, pokedex.Form]:
    """Every row of ``pokeapi_forms.csv``, by identifier."""
    return {f["identifier"]: f for f in pokedex.forms()}


def form(ident: str) -> pokedex.Form:
    """The row of ``pokeapi_forms.csv`` with this identifier."""
    return by_identifier()[ident]


def total(f: pokedex.Form, generation: int) -> int:
    """Base-stat total as of a generation; in Generation 1, with its single Special stat."""
    if generation >= 2:
        return pokedex.total(f, generation)
    stats = {k: int(f[k]) for k in ("hp", "attack", "defense", "speed")}
    rows = [
        r
        for r in pokedex.stats_past()
        if r["identifier"] == f["identifier"] and r["stat"] in (*stats, "special")
    ]
    for r in sorted(rows, key=lambda r: -int(r["through_generation"])):
        stats[r["stat"]] = int(r["base_stat"])
    return sum(stats.values())


# Shares.


def chaos_url(fmt: str, month: str, cutoff: int) -> str:
    """Where Smogon publishes one chaos file."""
    return CHAOS_URL.format(month=month, format=fmt, cutoff=cutoff)


def generation_of(fmt: str) -> int:
    """'gen9ou' -> 9."""
    m = re.match(r"gen(\d)", fmt)
    assert m, fmt
    return int(m.group(1))


def usage_by_form(chaos: Chaos, resolve: Callable[[str], str | None]) -> dict[str, float]:
    """{identifier: weighted team share}, summing names that map to one form.

    Raises ValueError listing every name that does not map.
    """
    out: dict[str, float] = defaultdict(float)
    missing = []
    for name, entry in chaos["data"].items():
        ident = resolve(name)
        if ident is None:
            missing.append(name)
            continue
        out[ident] += float(entry["usage"])
    if missing:
        raise ValueError(f"names with no identifier: {sorted(missing)}")
    return dict(out)


def typing_shares(
    chaos: Chaos, generation: int, resolve: Callable[[str], str | None] = identifier
) -> dict[Typing, float]:
    """Usage share of each typing in one file: the sum of its Pokémon's team shares, over 6."""
    shares: dict[Typing, float] = defaultdict(float)
    for ident, u in usage_by_form(chaos, resolve).items():
        t = pokedex.typing(form(ident), generation)
        assert t is not None, (ident, generation)
        shares[t] += u / TEAM_SIZE
    return dict(shares)


def type_shares(typings: dict[Typing, float]) -> dict[str, float]:
    """Usage share of each single type: a Pokémon's share is split equally among its types."""
    out: dict[str, float] = defaultdict(float)
    for t, s in typings.items():
        for x in t:
            out[x] += s / len(t)
    return dict(out)


def label(t: Iterable[str]) -> str:
    return pokedex.label(tuple(t))


def summarize(
    files: list[tuple[str, Chaos]], generation: int, resolve: Callable[[str], str | None]
) -> tuple[list[dict], list[dict]]:
    """Rows by typing and by type for each month, and pooled with equal weight per month.

    ``files`` holds (month, chaos) pairs of one format and cutoff. A Pokémon is available
    if it is listed in any of the files; ``pokemon`` counts the available forms with that
    typing (or type), and ``max_total`` is the highest base-stat total among them.
    """
    forms_seen: set[str] = set()
    by_month: dict[str, dict[Typing, float]] = {}
    for month, chaos in files:
        by_month[month] = typing_shares(chaos, generation, resolve)
        forms_seen |= set(usage_by_form(chaos, resolve))
    groups: dict[Typing, list[str]] = defaultdict(list)
    for ident in forms_seen:
        t = pokedex.typing(form(ident), generation)
        assert t is not None
        groups[t].append(ident)
    type_groups: dict[str, list[str]] = defaultdict(list)
    for t, idents in groups.items():
        for x in t:
            type_groups[x].extend(idents)

    def stats(idents: list[str]) -> dict:
        return {
            "pokemon": len(idents),
            "max_total": max(total(form(i), generation) for i in idents),
        }

    months = sorted(by_month)
    pooled = {t: sum(by_month[m].get(t, 0.0) for m in months) / len(months) for t in groups}
    typing_rows, type_rows = [], []
    for month, shares in [("pooled", pooled)] + [(m, by_month[m]) for m in months]:
        for t in sorted(groups, key=label):
            typing_rows.append(
                {
                    "month": month,
                    "typing": label(t),
                    "share": shares.get(t, 0.0),
                    **stats(groups[t]),
                }
            )
        singles = type_shares({t: shares.get(t, 0.0) for t in groups})
        for x in sorted(type_groups):
            type_rows.append(
                {"month": month, "type": x, "share": singles.get(x, 0.0), **stats(type_groups[x])}
            )
    return typing_rows, type_rows


def check_names(
    names: Iterable[str], generation: int, sd: dict[str, Typing], base_of: dict[str, str]
) -> list[str]:
    """Problems with a set of Showdown names: not Showdown species, no identifier, or a typing
    that differs from Showdown's in that generation."""
    problems = []
    for name in sorted(set(names)):
        if name not in sd:
            problems.append(f"{name}: not a species in Showdown's pokedex.ts")
            continue
        ident = identifier(name, base_of)
        if ident is None:
            problems.append(f"{name}: no identifier")
            continue
        ours = pokedex.typing(form(ident), generation)
        if ours != sd[name]:
            problems.append(f"{name} -> {ident}: {ours} here, {sd[name]} in Showdown")
    return problems


# Downloading and building.


def months_between(spec: str) -> list[str]:
    """'2025-11..2026-10' -> ['2025-11', ..., '2026-10']; a single month stands for itself."""
    first, _, last = spec.partition("..")
    last = last or first
    y, m = map(int, first.split("-"))
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def download_chaos(url: str, cache: Path) -> bytes | None:
    """One chaos file, kept in ``cache``; None if Smogon has no such file (HTTP 404)."""
    local = cache / url.removeprefix("https://www.smogon.com/")
    if local.exists():
        return local.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(5 * (attempt + 1))
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return data


def build(
    formats: dict[str, list[int]], months: list[str], cache: Path, required: set[tuple[str, int]]
) -> dict:
    """Download every file, check every name, and return the derived table.

    ``required`` lists (format, cutoff) pairs whose files must all exist; for other pairs a
    missing file is recorded and skipped.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sd_files = download_showdown(Path(tmp))
    out: dict = {
        "source": "Smogon usage statistics, https://www.smogon.com/stats/ (chaos JSON)",
        "license": "unknown; the raw files are not redistributed",
        "showdown_commit": SHOWDOWN_COMMIT,
        "team_size": TEAM_SIZE,
        "files": [],
        "missing": [],
        "names": {},
        "typings": [],
        "types": [],
    }
    problems: list[str] = []
    loaded: dict[tuple[str, int], list[tuple[str, Chaos]]] = defaultdict(list)
    for fmt, cutoffs in formats.items():
        g = generation_of(fmt)
        sd, base_of = showdown_typings(sd_files, g)
        for cutoff in cutoffs:
            for month in months:
                url = chaos_url(fmt, month, cutoff)
                data = download_chaos(url, cache)
                if data is None:
                    if (fmt, cutoff) in required:
                        raise FileNotFoundError(url)
                    out["missing"].append({"format": fmt, "month": month, "cutoff": cutoff})
                    print(f"missing: {url}")
                    continue
                chaos = json.loads(data)
                usages = [float(e["usage"]) for e in chaos["data"].values()]
                out["files"].append(
                    {
                        "format": fmt,
                        "month": month,
                        "cutoff": cutoff,
                        "url": url,
                        "sha256": sha256(data),
                        "bytes": len(data),
                        "battles": chaos["info"]["number of battles"],
                        "names": len(chaos["data"]),
                        "usage_sum": sum(usages),
                        "usage_min": min(usages),
                    }
                )
                problems += [f"{fmt}: {p}" for p in check_names(chaos["data"], g, sd, base_of)]
                names = out["names"].setdefault(fmt, {})
                for name in chaos["data"]:
                    names[name] = identifier(name, base_of)
                loaded[fmt, cutoff].append((month, chaos))
                print(f"{url}: {len(data):,} bytes, {len(chaos['data'])} names")
    if problems:
        raise ValueError("\n".join(sorted(set(problems))))
    for (fmt, cutoff), files in loaded.items():
        g = generation_of(fmt)
        _, base_of = showdown_typings(sd_files, g)
        typing_rows, type_rows = summarize(files, g, lambda n, b=base_of: identifier(n, b))
        keys = {"format": fmt, "generation": g, "cutoff": cutoff}
        out["typings"] += [{**keys, **r} for r in typing_rows]
        out["types"] += [{**keys, **r} for r in type_rows]
    return out


def check_all(cache: Path) -> list[str]:
    """Every Showdown species with a positive number, in Generations 1 and 9, must map."""
    files = download_showdown(cache)
    problems = []
    for g, limit in ((9, None), (1, 151)):
        sd, base_of = showdown_typings(files, g)
        dex = parse_pokedex(files["data/pokedex.ts"])
        nums = {e["name"]: e["num"] for e in dex.values() if e["name"]}
        names = [n for n in sd if limit is None or (nums.get(n) or 0) <= limit]
        if g == 1:  # Generation 1 has only the 151 base species, no formes
            names = [n for n in names if "-" not in n or n in ("Nidoran-F", "Nidoran-M")]
        problems += [f"gen{g}: {p}" for p in check_names(names, g, sd, base_of)]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="download the chaos files and write the derived table")
    b.add_argument("--formats", nargs="+", default=["gen9ou", "gen1ou"])
    b.add_argument("--months", default="2025-11..2026-10")
    b.add_argument(
        "--cutoffs",
        nargs="+",
        default=["gen9ou:0,1500,1695,1825", "gen1ou:0,1500,1630,1760"],
        help="format:cutoff,cutoff,... for each format",
    )
    b.add_argument("--required", nargs="+", default=["gen9ou:1500"], help="format:cutoff pairs")
    b.add_argument("--cache", type=Path, help="keep the raw downloads here (never committed)")
    b.add_argument("--out", type=Path, default=RESULTS / "usage-typings.json")
    c = sub.add_parser("check", help="check the name mapping against all of Showdown's species")
    c.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        with tempfile.TemporaryDirectory() as tmp:
            problems = check_all(args.cache or Path(tmp))
        print("\n".join(problems) or "every species maps, with Showdown's typing")
        return
    cutoffs = {}
    for spec in args.cutoffs:
        fmt, _, values = spec.partition(":")
        cutoffs[fmt] = [int(v) for v in values.split(",")]
    formats = {f: cutoffs[f] for f in args.formats}
    required = set()
    for spec in args.required:
        fmt, _, value = spec.partition(":")
        required.add((fmt, int(value)))
    with tempfile.TemporaryDirectory() as tmp:
        table = build(formats, months_between(args.months), args.cache or Path(tmp), required)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(table['files'])} files, {len(table['typings'])} typing rows")


if __name__ == "__main__":
    main()
