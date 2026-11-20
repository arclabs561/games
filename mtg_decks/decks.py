"""Decks as rows of card counts, the split into fit, dev and test, and co-occurrence counts.

Nothing here knows about Magic. A deck file from any card game with the columns of
`read_long` (deck, card, count) and `read_meta` (deck, group, games, wins, skill) can be
loaded the same way; `seventeen_lands.py` is one reader that produces them.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Card = str
SPLITS = ("fit", "dev", "test")


def split_of(draft: str) -> str:
    """The split of a draft (or deck) id: half fit, a quarter dev, a quarter test, by hash.

    Every build of one draft lands in the same split, and the split does not depend on
    the order the rows were read in.
    """
    h = int(hashlib.sha256(draft.encode()).hexdigest()[:8], 16) % 4
    return "fit" if h < 2 else ("dev" if h == 2 else "test")


@dataclass
class Decks:
    """A table of decks: counts[d, c] copies of card c in deck d, plus per-deck facts.

    `group` is the deck's archetype label (for Magic, its main color pair). `games` and
    `wins` count the games played with exactly this deck. `skill` is any measure of the
    player's strength known for the deck (for 17Lands, the player's win-rate bucket),
    and `draft` ties several builds of one draft together.
    """

    cards: list[Card]
    counts: np.ndarray  # decks x cards, small nonnegative integers
    group: np.ndarray  # str per deck
    games: np.ndarray  # int per deck
    wins: np.ndarray  # int per deck
    skill: np.ndarray  # float per deck (nan when unknown)
    draft: np.ndarray  # str per deck

    def __len__(self) -> int:
        return self.counts.shape[0]

    def take(self, mask: np.ndarray) -> "Decks":
        """The decks where mask is true."""
        return Decks(
            self.cards,
            self.counts[mask],
            self.group[mask],
            self.games[mask],
            self.wins[mask],
            self.skill[mask],
            self.draft[mask],
        )

    def split(self, name: str) -> "Decks":
        """The decks of one split, by `split_of` on the draft id."""
        return self.take(np.array([split_of(str(d)) == name for d in self.draft], dtype=bool))


def cooccurrence(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set and multiset co-occurrence over decks, and the number of decks holding each card.

    set[i, j] is the number of decks holding both i and j; multiset[i, j] is the sum over
    decks of counts[i] * counts[j]. These are decksage's COUNT_SET and COUNT_MULTISET
    for i != j. Diagonals are zero here; `held[i]` is the number of decks holding i.
    """
    x = counts.astype(np.int64)
    b = (x > 0).astype(np.int64)
    s = b.T @ b
    m = x.T @ x
    held = np.diag(s).copy()
    np.fill_diagonal(s, 0)
    np.fill_diagonal(m, 0)
    return s, m, held


def write_pairs(path: Path, cards: list[Card], s: np.ndarray, m: np.ndarray) -> None:
    """Write co-occurrence in decksage's pair format: NAME_1, NAME_2, COUNT_SET, COUNT_MULTISET.

    One row per unordered pair with COUNT_SET > 0, names in sorted order, rows sorted.
    """
    order = sorted(range(len(cards)), key=lambda i: cards[i])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["NAME_1", "NAME_2", "COUNT_SET", "COUNT_MULTISET"])
        for a, i in enumerate(order):
            for j in order[a + 1 :]:
                if s[i, j] > 0:
                    w.writerow([cards[i], cards[j], int(s[i, j]), int(m[i, j])])


def read_pairs(path: Path, cards: list[Card]) -> tuple[np.ndarray, np.ndarray]:
    """Read a decksage pair file into symmetric set and multiset matrices over `cards`.

    Pairs naming a card outside `cards` are skipped.
    """
    index = {c: i for i, c in enumerate(cards)}
    n = len(cards)
    s = np.zeros((n, n), dtype=np.int64)
    m = np.zeros((n, n), dtype=np.int64)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i, j = index.get(row["NAME_1"]), index.get(row["NAME_2"])
            if i is None or j is None:
                continue
            s[i, j] = s[j, i] = int(row["COUNT_SET"])
            m[i, j] = m[j, i] = int(row["COUNT_MULTISET"])
    return s, m


def read_long(path: Path) -> dict[str, dict[Card, int]]:
    """A long deck file (columns deck, card, count) as {deck: {card: count}}."""
    decks: dict[str, dict[Card, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            decks.setdefault(row["deck"], {})[row["card"]] = int(row["count"])
    return decks


def from_long(long: dict[str, dict[Card, int]], meta: dict[str, dict] | None = None) -> Decks:
    """Build `Decks` from {deck: {card: count}} and optional per-deck facts.

    meta[deck] may hold group, games, wins, skill and draft; missing facts become
    "" (group), 0 (games, wins), nan (skill) and the deck id (draft).
    """
    meta = meta or {}
    ids = sorted(long)
    cards = sorted({c for d in long.values() for c in d})
    index = {c: i for i, c in enumerate(cards)}
    counts = np.zeros((len(ids), len(cards)), dtype=np.int16)
    for r, d in enumerate(ids):
        for c, k in long[d].items():
            counts[r, index[c]] = k

    def col(key, default, dtype):
        return np.array([meta.get(d, {}).get(key, default) for d in ids], dtype=dtype)

    return Decks(
        cards,
        counts,
        col("group", "", object),
        col("games", 0, np.int64),
        col("wins", 0, np.int64),
        col("skill", np.nan, np.float64),
        np.array([meta.get(d, {}).get("draft", d) for d in ids], dtype=object),
    )
