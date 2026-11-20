"""What players use in Generation 9 OU, per Pokémon, from Smogon's monthly statistics.

The team study (``teams.py``) compares its models with real use. This module reads the same
"chaos" files as ``usage.py`` (plan P1) and keeps, for the most used Pokémon of one format and
rating cutoff, three aggregates pooled over the months with equal weight:

- ``usage``: the weighted share of teams that include the Pokémon (the shares sum to about 6);
- ``abilities``: the share of its sets with each ability;
- ``moves``: the share of its sets with each move (a set has up to four, so these sum to
  about 4), kept when at least 1%.

Names map to ``pokeapi_forms.csv`` identifiers as in ``usage.py``. The raw files are not kept
(their license is unknown); only these shares and each file's SHA-256 are written.

    uv run --group pokemon python -m pokemon.teams_usage --out pokemon/results/usage-pokemon.json

smogon.com is not reachable from every machine, so the build runs on GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from pokemon import usage
from pokemon.sources import sha256

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "usage-pokemon.json"
FORMAT, CUTOFF, MONTHS = "gen9ou", 1500, "2025-11..2026-10"
KEEP = 80  # Pokémon kept, by pooled usage
MOVE_FLOOR = 0.01


def shares(entry: dict[str, float]) -> dict[str, float]:
    """A chaos-file weight table normalized to sum to 1."""
    total = sum(entry.values())
    return {k: v / total for k, v in entry.items()} if total else {}


def pooled(files: list[dict], resolve) -> dict[str, dict]:
    """{identifier: {usage, abilities, moves}} pooled with equal weight per month.

    Within a month, abilities and moves are weighted by the Pokémon's ability weights, whose
    sum is its weighted count of sets. A month in which a Pokémon is absent adds 0 to its usage.
    """
    out: dict[str, dict] = defaultdict(
        lambda: {"usage": 0.0, "abilities": defaultdict(float), "moves": defaultdict(float)}
    )
    for chaos in files:
        sets: dict[str, float] = defaultdict(float)
        for name, e in chaos["data"].items():
            ident = resolve(name)
            assert ident is not None, name
            row = out[ident]
            row["usage"] += float(e["usage"]) / len(files)
            n = sum(e["Abilities"].values())
            for a, w in e["Abilities"].items():
                row["abilities"][usage.to_id(a)] += w
            for m, w in e["Moves"].items():
                if m:
                    row["moves"][m] += w
            sets[ident] += n
        for ident, n in sets.items():
            out[ident].setdefault("sets", 0.0)
            out[ident]["sets"] += n
    result = {}
    for ident, row in out.items():
        n = row.get("sets", 0.0)
        result[ident] = {
            "usage": row["usage"],
            "abilities": {k: v / n for k, v in sorted(row["abilities"].items()) if n},
            "moves": {
                k: v / n for k, v in sorted(row["moves"].items()) if n and v / n >= MOVE_FLOOR
            },
        }
    return result


def build(cache: Path, months: list[str]) -> dict:
    """Download the files, check every name as ``usage.py`` does, and pool the aggregates."""
    with tempfile.TemporaryDirectory() as tmp:
        sd_files = usage.download_showdown(Path(tmp))
    g = usage.generation_of(FORMAT)
    sd, base_of = usage.showdown_typings(sd_files, g)
    resolve = usage.resolver(sd, base_of)
    files, chaos_files, names = [], [], {}
    for month in months:
        url = usage.chaos_url(FORMAT, month, CUTOFF)
        data = usage.download_chaos(url, cache)
        if data is None:
            raise FileNotFoundError(url)
        chaos = json.loads(data)
        problems = usage.check_names(chaos["data"], g, sd, base_of)
        if problems:
            raise ValueError("\n".join(problems))
        files.append(
            {
                "url": url,
                "sha256": sha256(data),
                "bytes": len(data),
                "battles": chaos["info"]["number of battles"],
            }
        )
        names.update({resolve(n): n for n in chaos["data"]})
        chaos_files.append(chaos)
        print(f"{url}: {len(data):,} bytes")
    rows = pooled(chaos_files, resolve)
    top = sorted(rows, key=lambda i: -rows[i]["usage"])[:KEEP]
    return {
        "source": "Smogon usage statistics, https://www.smogon.com/stats/ (chaos JSON)",
        "license": "unknown; the raw files are not redistributed",
        "format": FORMAT,
        "cutoff": CUTOFF,
        "months": months,
        "files": files,
        "pokemon": [
            {"identifier": i, "showdown": names[i], "rank": r + 1, **rows[i]}
            for r, i in enumerate(top)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--months", default=MONTHS)
    parser.add_argument("--cache", type=Path, help="keep the raw downloads here (never committed)")
    parser.add_argument("--out", type=Path, default=OUT, help="a file, or a directory for it")
    args = parser.parse_args()
    out = args.out / OUT.name if args.out.is_dir() else args.out
    with tempfile.TemporaryDirectory() as tmp:
        table = build(args.cache or Path(tmp), usage.months_between(args.months))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{out}: {len(table['pokemon'])} Pokémon from {len(table['files'])} files")


if __name__ == "__main__":
    main()
