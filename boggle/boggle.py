"""Boggle boards and word search, used by boggle.ipynb and the tests."""

import hashlib
import multiprocessing as mp
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import product
from math import ceil, log, sqrt
from pathlib import Path

type Board = list[list[str]]  # 4 rows of 4 faces; a face is a letter, or "qu"
type Trie = dict  # letter -> Trie, with the key None marking the end of a word

# The 16 cubes of a standard Boggle set, one tuple of six faces per cube.
CUBES: list[tuple[str, ...]] = [
    ("a", "a", "e", "e", "g", "n"),
    ("a", "b", "b", "j", "o", "o"),
    ("a", "c", "h", "o", "p", "s"),
    ("a", "f", "f", "k", "p", "s"),
    ("a", "o", "o", "t", "t", "w"),
    ("c", "i", "m", "o", "t", "u"),
    ("d", "e", "i", "l", "r", "x"),
    ("d", "e", "l", "r", "v", "y"),
    ("d", "i", "s", "t", "t", "y"),
    ("e", "e", "g", "h", "n", "w"),
    ("e", "e", "i", "n", "s", "u"),
    ("e", "h", "r", "t", "v", "w"),
    ("e", "i", "o", "s", "s", "t"),
    ("e", "l", "r", "t", "t", "y"),
    ("h", "i", "m", "n", "qu", "u"),
    ("h", "l", "n", "n", "r", "z"),
]

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def rand_board(rng: random.Random | None = None, cubes: list[tuple[str, ...]] = CUBES) -> Board:
    """Shake the cubes into the 16 slots and show a random face of each."""
    rng = rng or random.Random()
    order = rng.sample(cubes, len(cubes))
    faces = [rng.choice(cube) for cube in order]
    return [faces[i : i + 4] for i in range(0, 16, 4)]


def make_trie(words: Iterable[str]) -> Trie:
    """A nested dict of letters; the key None marks the end of a word."""
    root: Trie = {}
    for word in words:
        node = root
        for letter in word:
            node = node.setdefault(letter, {})
        node[None] = None
    return root


def get_words(board: Board, trie: Trie, min_len: int = 3) -> Iterator[str]:
    """Each word of at least `min_len` letters that the board spells, once."""

    def recur(i: int, j: int, path: str, seen: frozenset, node: Trie) -> Iterator[str]:
        if (i, j) in seen or not (0 <= i < 4 and 0 <= j < 4):  # no reuse, no wrapping
            return
        face = board[i][j]
        for letter in face:  # a face can hold two letters, "qu"
            child = node.get(letter)
            if child is None:  # not a prefix of any word
                return
            node = child
        path += face
        if None in node:
            yield path
        for di, dj in NEIGHBORS:  # keep going even after a word: it may be a prefix too
            yield from recur(i + di, j + dj, path, seen | {(i, j)}, node)

    found = set()
    for i, j in product(range(4), range(4)):
        for word in recur(i, j, "", frozenset(), trie):
            if len(word) >= min_len and word not in found:
                found.add(word)
                yield word


def sample_boards(n: int, seed: int) -> list[Board]:
    """n boards drawn one after another from a single random.Random(seed) stream."""
    rng = random.Random(seed)
    return [rand_board(rng) for _ in range(n)]


def count_words(
    boards: list[Board], trie: Trie, min_len: int = 3, processes: int | None = None
) -> Counter:
    """How many of the boards contain each word (a word counts once per board).

    The workers get the trie once each, through the pool initializer. Because the work
    function lives in this module, this also runs where processes start by spawn or
    forkserver (macOS, and Linux from Python 3.14), not only by fork.
    """
    counts: Counter = Counter()
    with mp.Pool(processes, initializer=_set_trie, initargs=(trie, min_len)) as pool:
        for found in pool.imap(_board_words, boards, chunksize=500):
            counts.update(found)
    return counts


_WORKER: dict = {}


def _set_trie(trie: Trie, min_len: int) -> None:
    _WORKER.update(trie=trie, min_len=min_len)


def _board_words(board: Board) -> list[str]:
    return list(get_words(board, _WORKER["trie"], _WORKER["min_len"]))


def load_words(path: str | Path, sha256: str | None = None) -> list[str]:
    """The words of a word list, one per line, after checking the file's SHA-256 digest."""
    data = Path(path).read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA-256 digest does not match")
    return [w for line in data.decode().splitlines() if (w := line.strip().lower()).isalpha()]


def points(word: str) -> int:
    """Official score of a word: 3-4 letters 1, 5 letters 2, 6 letters 3, 7 letters 5, 8+ 11."""
    n = len(word)
    return 0 if n < 3 else 1 if n <= 4 else {5: 2, 6: 3, 7: 5}.get(n, 11)


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a binomial proportion (95% for z = 1.96)."""
    p = successes / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    margin = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return center - margin, center + margin


def hoeffding_boards(epsilon: float, delta: float, words: int = 1) -> int:
    """Boards needed so that every one of `words` fixed words is within epsilon of its
    probability, with chance at least 1 - delta (Hoeffding, with a union bound)."""
    return ceil(log(2 * words / delta) / (2 * epsilon**2))


assert make_trie(["at"]) == {"a": {"t": {None: None}}}
assert set(
    get_words(
        [list("cat."), list("...."), list("...."), list("....")], make_trie(["cat", "act", "tac"])
    )
) == {"cat", "tac"}
assert [points(w) for w in ("tea", "tees", "teens", "tenets", "setters", "teenager")] == [
    1,
    1,
    2,
    3,
    5,
    11,
]
assert hoeffding_boards(0.01, 0.1) == 14979
