"""Human data: association counts (EAT, and SWOW or USF if present) and human guesses.

- eat_counts(): Edinburgh Associative Thesaurus, cue -> {response: number of people}.
  About 100 people answered each of 8,400 cues, 1968-1971, British students.
- swow_counts(): Small World of Words SWOW-EN18, if the file was downloaded by hand.
- cultural_guesses(split): Cultural Codes (Shaikh et al. 2023), Codenames Duet turns:
  the clue, the unrevealed words, and the word a person picked.

Words are lowercased. Only one-word responses are kept.
"""

import ast
import csv
import json
import re
import zlib
from dataclasses import dataclass
from functools import cache

from codenames.fetch import CACHE, fetch

Counts = dict[str, dict[str, int]]


@cache
def eat_counts() -> Counts:
    raw = json.loads((fetch("eat") / "ea-thesaurus.json").read_text())
    out: Counts = {}
    for cue, responses in raw.items():
        row: dict[str, int] = {}
        for item in responses:
            for resp, n in item.items():
                r = resp.lower()
                if r.isalpha():
                    row[r] = row.get(r, 0) + int(n)
        cue = cue.lower()
        if cue.isalpha() and row:
            out[cue] = row
    return out


@cache
def swow_counts() -> Counts | None:
    """SWOW-EN18 R123 counts (cue, response, R123 count), or None if not downloaded.

    The column names (cue, response, R123, tab-separated) follow the SWOW-EN18 release
    notes. The file could not be downloaded where this was written, so this loader has
    not been run on it.
    """
    path = CACHE / "swow" / "strength.csv"
    if not path.exists():
        return None
    out: Counts = {}
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cue, resp = row["cue"].lower(), row["response"].lower()
            if cue.isalpha() and resp.isalpha():
                out.setdefault(cue, {})[resp] = int(row["R123"])
    return out


def held_out(cue: str, fraction: float = 0.2) -> bool:
    """A fixed fifth of the cues, chosen by hash, never used for training."""
    return zlib.crc32(cue.encode()) % 1000 < fraction * 1000


@dataclass(frozen=True)
class Guess:
    clue: str
    words: tuple[str, ...]
    picked: str


PATTERN = re.compile(r"remaining: (\[.*?\]), hint: (\S+)")


@cache
def cultural_guesses(split: str, scheme: str = "published") -> tuple[Guess, ...]:
    """Human guesses from Cultural Codes, in one of its splits (train, val or test) under
    a scheme of codenames.split: as published (the default), or by game, or "all"."""
    if scheme != "published" or split == "all":
        from codenames.split import turns

        return tuple(t.guess for t in turns("all" if split == "all" else f"{scheme}:{split}"))
    csv.field_size_limit(10**9)
    out = []
    with (fetch("cultural") / f"guess_{split}.csv").open() as f:
        for row in csv.DictReader(f):
            m = PATTERN.search(row["base_text"])
            if not m:
                continue
            words = tuple(w.lower() for w in ast.literal_eval(m[1]))
            clue, picked = m[2].lower(), row["output"].strip().lower()
            if picked in words and clue.isalpha():
                out.append(Guess(clue, words, picked))
    return tuple(out)
