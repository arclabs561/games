"""Download and read 17Lands public game data (CC BY 4.0), one set and format at a time.

The files are large (hundreds of megabytes compressed), so this module runs on GitHub
Actions, which can reach the 17Lands bucket. Nothing it downloads is committed.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np

from mtg_decks.decks import Decks

BUCKET = "https://17lands-public.s3.amazonaws.com/analysis_data"
LICENSE = "CC BY 4.0 (https://www.17lands.com/public_datasets)"


def game_data_url(expansion: str, event: str = "PremierDraft") -> str:
    """URL of the game-level file for one set and event type."""
    return f"{BUCKET}/game_data/game_data_public.{expansion}.{event}.csv.gz"


CARDS_URL = f"{BUCKET}/cards/cards.csv"


def download(url: str, path: Path) -> dict:
    """Download url to path; return its size and SHA-256."""
    sha = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=120) as resp, open(path, "wb") as out:
        while chunk := resp.read(1 << 20):
            sha.update(chunk)
            size += len(chunk)
            out.write(chunk)
    return {"url": url, "bytes": size, "sha256": sha.hexdigest()}


def head_rows(url: str, n: int = 3) -> tuple[list[str], list[list[str]]]:
    """The header and the first n rows of a gzipped CSV, read from the stream."""
    with urllib.request.urlopen(url, timeout=120) as resp:
        text = io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8")
        reader = csv.reader(text)
        header = next(reader)
        rows = [next(reader) for _ in range(n)]
    return header, rows


def probe(expansions: list[str], out: Path) -> None:
    """Write the columns, the first rows and the card list of each set, to plan the reader."""
    out.mkdir(parents=True, exist_ok=True)
    cards = out / "cards_full.csv"
    meta = download(CARDS_URL, cards)
    report: dict = {"cards": meta, "sets": {}}
    with open(cards, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        report["cards"]["columns"] = reader.fieldnames
        rows = list(reader)
    for x in expansions:
        url = game_data_url(x)
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = int(resp.headers.get("Content-Length", 0))
        header, first = head_rows(url)
        prefixes = sorted({c.split("_")[0] for c in header if "_" in c})
        plain = [
            c
            for c in header
            if not c.startswith(("deck_", "sideboard_", "opening_hand_", "drawn_", "tutored_"))
        ]
        set_cards = [r for r in rows if r.get("expansion") == x]
        report["sets"][x] = {
            "bytes": length,
            "n_columns": len(header),
            "prefixes": prefixes,
            "plain_columns": plain,
            "first_rows": [
                {k: v for k, v in zip(header, r, strict=True) if k in plain} for r in first
            ],
            "n_deck_columns": sum(c.startswith("deck_") for c in header),
            "deck_columns_sample": [c for c in header if c.startswith("deck_")][:40],
            "cards_in_list": len(set_cards),
            "cards_sample": set_cards[:15],
        }
    cards.unlink()
    (out / "probe.json").write_text(json.dumps(report, indent=1) + "\n")


WUBRG = "WUBRG"
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
META = ["draft_id", "build_index", "main_colors", "won", "user_game_win_rate_bucket"]


def colors(s: str) -> str:
    """A color string in WUBRG order: 'GB' -> 'BG'."""
    return "".join(sorted(s or "", key=WUBRG.index))


def read_games(path: Path, chunksize: int = 50_000) -> Decks:
    """One row per (draft, build) from a 17Lands game file: its main deck and its games.

    The main deck is the deck_ columns of the build's games (they do not change within a
    build). `group` is the main colors in WUBRG order, `skill` the player's game win
    rate bucket, and games and wins are counted over the build's games.
    """
    import pandas as pd

    header = pd.read_csv(path, nrows=0).columns
    deck_cols = [c for c in header if c.startswith("deck_")]
    cards = [c[len("deck_") :] for c in deck_cols]
    parts = []
    for chunk in pd.read_csv(
        path,
        usecols=META + deck_cols,
        chunksize=chunksize,
        dtype={c: "int16" for c in deck_cols} | {"draft_id": str, "main_colors": str},
    ):
        chunk["won"] = chunk["won"].astype(str).str.lower().eq("true").astype(np.int64)
        chunk["games"] = 1
        agg = {c: "first" for c in deck_cols} | {
            "main_colors": "first",
            "user_game_win_rate_bucket": "first",
            "won": "sum",
            "games": "sum",
        }
        parts.append(chunk.groupby(["draft_id", "build_index"], sort=False).agg(agg))
    both = pd.concat(parts)
    agg = {c: "first" for c in deck_cols} | {
        "main_colors": "first",
        "user_game_win_rate_bucket": "first",
        "won": "sum",
        "games": "sum",
    }
    both = both.groupby(level=[0, 1], sort=True).agg(agg)
    return Decks(
        cards=cards,
        counts=both[deck_cols].to_numpy(dtype=np.int16),
        group=np.array(
            [colors(str(x)) if isinstance(x, str) else "" for x in both["main_colors"]],
            dtype=object,
        ),
        games=both["games"].to_numpy(dtype=np.int64),
        wins=both["won"].to_numpy(dtype=np.int64),
        skill=pd.to_numeric(both["user_game_win_rate_bucket"], errors="coerce").to_numpy(
            dtype=np.float64
        ),
        draft=np.array([d for d, _ in both.index], dtype=object),
    )


def read_cards(path: Path, expansion: str) -> dict[str, dict]:
    """Card facts by name from 17Lands' card list, preferring the set's own printing."""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name in out and out[name]["expansion"] == expansion:
                continue
            out[name] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--sets", nargs="+", default=["BLB"])
    pr.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "probe":
        probe(a.sets, a.out)


if __name__ == "__main__":
    main()
