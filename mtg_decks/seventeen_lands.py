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
