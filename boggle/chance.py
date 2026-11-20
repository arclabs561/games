"""The chance that a shake of the cubes shows a given word: exact for short words, and an
upper bound for every word.

**Shakes.** A shake puts the cubes in the slots in a uniformly random order and turns up one
of the six faces of each, independently. Given which face each cube turned up, every order of
those faces on the board is equally likely.

**Exact chance, words of two or three faces.** Color each slot by which of the word's faces
it shows, or "other". Whether the word is on the board depends only on this coloring, and
colorings with the same number of slots of each color are equally likely. So

    P(word) = sum over counts k of P(the cubes show counts k) * present(k) / colorings(k),

where present(k) is the number of colorings with counts k that show the word, and colorings(k)
is the multinomial coefficient. `present_colorings` counts them with bit masks; P(counts) is
a product over cubes (`count_ways`).

**Upper bound, every word.** A word is on the board when some path spells it, so its chance is
at most the expected number of paths that spell it (Markov's inequality). Every path of L
slots shows a given sequence of L faces with the same chance, so

    expected paths = paths(L) * placements(faces) / ((16)_L * 6^L),

where paths(L) counts the directed paths of L slots on the grid (`path_counts`) and
placements(faces) counts the ways to give each face its own cube, times the faces of that cube
that show it (`placements`). For rare words the bound is close to the chance, since two paths
that spell the same rare word seldom show up together.

    uv run python -m boggle.chance          # writes results/chances.json, about 20 seconds
"""

import argparse
import json
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
from itertools import permutations, product
from math import comb, factorial, perm, prod
from pathlib import Path

import numpy as np

from boggle.board import CUBES, find_path, load_words, possible, word_faces

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
type Cubes = Sequence[tuple[str, ...]]


def grid_adjacent(rows: int, cols: int) -> list[list[int]]:
    """The slots next to each slot (sideways or diagonally), numbered row by row."""
    return [
        [(i + di) * cols + j + dj for di, dj in product((-1, 0, 1), repeat=2)
         if (di or dj) and 0 <= i + di < rows and 0 <= j + dj < cols]
        for i in range(rows) for j in range(cols)
    ]  # fmt: skip


def path_counts(adjacent: list[list[int]]) -> list[int]:
    """Entry L is the number of directed paths through L different slots, each slot next to
    the one before."""
    counts = [0] * (len(adjacent) + 1)

    def walk(k: int, seen: int, length: int) -> None:
        counts[length] += 1
        for m in adjacent[k]:
            if not seen >> m & 1:
                walk(m, seen | 1 << m, length + 1)

    for k in range(len(adjacent)):
        walk(k, 1 << k, 1)
    return counts


PATHS_4X4 = path_counts(grid_adjacent(4, 4))


def placements(faces: Sequence[str], cubes: Cubes = CUBES, exact: bool = True) -> int | float:
    """The number of ways to give each face of the sequence its own cube, counting each way
    once for every side of the cube that shows the face: a sum over one-to-one maps from
    positions to cubes of the product of face counts. A dynamic program over the cubes, whose
    state is how many positions of each different face are filled."""
    need = Counter(faces)
    keys = sorted(need)
    table = np.zeros([need[f] + 1 for f in keys], dtype=object if exact else float)
    table[(0,) * len(keys)] = 1
    for cube in cubes:
        carried = [(i, cube.count(f)) for i, f in enumerate(keys) if f in cube]
        if not carried:
            continue
        new = table.copy()
        for i, c in carried:
            m = need[keys[i]]
            src = tuple(slice(0, m) if a == i else slice(None) for a in range(len(keys)))
            dst = tuple(slice(1, m + 1) if a == i else slice(None) for a in range(len(keys)))
            shape = tuple(m if a == i else 1 for a in range(len(keys)))
            # A cube taking one of the m - s open positions of face i, with c sides showing it.
            weight = (c * (m - np.arange(m))).reshape(shape)
            new[dst] = new[dst] + table[src] * (weight.astype(object) if exact else weight)
        table = new
    return table[tuple(need[f] for f in keys)]


def c_placements(sequences: Sequence[Sequence[str]], cubes: Cubes = CUBES) -> list[float]:
    """placements for many face sequences at once, by `bound.c placements`, in floating point
    (every term is positive, so the relative error is near machine precision)."""
    from boggle import bound

    faces = ["".join("q" if f == "qu" else f for f in cube) for cube in cubes]
    text = "".join("".join("q" if f == "qu" else f for f in seq) + "\n" for seq in sequences)
    out = subprocess.run([str(bound.binary()), "placements", *faces], input=text,
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return [float(x) for x in out.split()]


def expected_paths(
    word: str, cubes: Cubes = CUBES, paths: Sequence[int] = PATHS_4X4, exact: bool = True
) -> Fraction | float:
    """The expected number of paths that spell `word` on a shaken board: an upper bound on
    the chance that the board shows it."""
    faces = word_faces(word)
    if faces is None or len(faces) > len(cubes):
        return Fraction(0) if exact else 0.0
    n, length = len(cubes), len(faces)
    ways = placements(faces, cubes, exact)
    if exact:
        return Fraction(paths[length] * int(ways), perm(n, length) * 6**length)
    return paths[length] * float(ways) / (perm(n, length) * 6.0**length)


def word_bounds(words: Sequence[str], cubes: Cubes = CUBES) -> dict[str, float]:
    """expected_paths for every word of 4 by 4 boards, in floating point, by c_placements;
    anagrams share a bound, so each multiset of faces is computed once."""
    keys = sorted({tuple(sorted(word_faces(w) or ())) for w in words})
    n = len(cubes)
    per_key = {
        k: PATHS_4X4[len(k)] * ways / (perm(n, len(k)) * 6.0 ** len(k))
        for k, ways in zip(keys, c_placements(keys, cubes), strict=True)
    }
    return {w: per_key[tuple(sorted(word_faces(w) or ()))] for w in words}


def pattern(word: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The word's different faces in order of first use, and each face's index among them,
    for a word of two or three faces. A word reads the same backward along the reversed path,
    so "tee" (0, 1, 1) is handled as "eet" (0, 0, 1)."""
    faces = word_faces(word)
    assert faces is not None and 2 <= len(faces) <= 3, word
    if len(faces) == 3 and faces[0] != faces[1] and faces[1] == faces[2]:
        faces = faces[::-1]
    letters = tuple(dict.fromkeys(faces))
    return letters, tuple(letters.index(f) for f in faces)


def _masks(adjacent: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """For every set of slots m (as a bit mask): the slots next to some slot of m, and the
    slots next to at least two slots of m."""
    m = np.arange(1 << len(adjacent), dtype=np.int64)
    near, near2 = np.zeros_like(m), np.zeros_like(m)
    for s, adj in enumerate(adjacent):
        count = np.bitwise_count(m & sum(1 << a for a in adj))
        near |= (count >= 1).astype(np.int64) << s
        near2 |= (count >= 2).astype(np.int64) << s
    return near, near2


def _submasks(mask: int) -> np.ndarray:
    """Every subset of the bit mask, as an array."""
    bits = [b for b in range(mask.bit_length()) if mask >> b & 1]
    r = np.arange(1 << len(bits), dtype=np.int64)
    out = np.zeros_like(r)
    for i, b in enumerate(bits):
        out |= ((r >> i) & 1) << b
    return out


def present_colorings(shape: tuple[int, ...], rows: int = 4, cols: int = 4) -> dict:
    """For a word pattern (from `pattern`), the number of colorings of the grid that show the
    word, by the number of slots of each color (one entry per face, the rest "other")."""
    adjacent = grid_adjacent(rows, cols)
    n = len(adjacent)
    full = (1 << n) - 1
    near, near2 = _masks(adjacent)
    every = np.arange(1 << n, dtype=np.int64)
    size = np.bitwise_count(every).astype(np.int64)
    absent: Counter = Counter()
    if shape in {(0, 0), (0, 0, 0)}:  # the word needs two (three) slots of A in a row
        hit = (every & (near if shape == (0, 0) else near2)) != 0
        return {(a,): int(c) for a, c in enumerate(np.bincount(size[hit], minlength=n + 1)) if c}
    if shape in {(0, 1), (0, 1, 0), (0, 0, 1)}:
        # The word is missing exactly when B avoids a set of slots fixed by A: next to A for
        # "ab"; next to two slots of A for "aba"; next to a slot of A that is next to A, "aab".
        blocked = {(0, 1): near, (0, 1, 0): near2, (0, 0, 1): near[every & near]}[shape]
        free = np.bitwise_count(full & ~every & ~blocked).astype(np.int64)
        for a, f in zip(size, free, strict=True):
            for j in range(f + 1):
                absent[(int(a), j)] += comb(int(f), j)
        return {
            (a, j): comb(n, a) * comb(n - a, j) - absent[(a, j)]
            for a in range(n + 1) for j in range(n - a + 1)
            if comb(n, a) * comb(n - a, j) - absent[(a, j)]
        }  # fmt: skip
    assert shape == (0, 1, 2), shape
    # "abc" is missing exactly when C avoids every slot next to a B that is next to an A.
    hist = np.zeros((n + 1, n + 1, n + 1), dtype=np.int64)  # |A|, |B|, free slots for C
    for a_mask in range(1 << n):
        b = _submasks(full & ~a_mask)
        x = b & near[a_mask]
        free = np.bitwise_count(full & ~a_mask & ~b & ~near[x]).astype(np.int64)
        flat = np.bincount(size[b] * (n + 1) + free, minlength=(n + 1) ** 2)
        hist[size[a_mask]] += flat.reshape(n + 1, n + 1)
    out = {}
    for a, b, j in product(range(n + 1), repeat=3):
        if a + b + j > n:
            continue
        missing = sum(int(hist[a, b, f]) * comb(f, j) for f in range(j, n + 1))
        total = factorial(n) // (
            factorial(a) * factorial(b) * factorial(j) * factorial(n - a - b - j)
        )
        if total - missing:
            out[(a, b, j)] = total - missing
    return out


def count_ways(letters: Sequence[str], cubes: Cubes = CUBES) -> Counter:
    """How many of the 6^n face choices turn up exactly k_i cubes showing letters[i], by k."""
    ways: Counter = Counter({(0,) * len(letters): 1})
    for cube in cubes:
        sides = [cube.count(f) for f in letters]
        rest = 6 - sum(sides)
        new: Counter = Counter()
        for k, v in ways.items():
            if rest:
                new[k] += v * rest
            for i, c in enumerate(sides):
                if c:
                    new[k[:i] + (k[i] + 1,) + k[i + 1 :]] += v * c
        ways = new
    return ways


def word_chance(word: str, cubes: Cubes = CUBES, rows: int = 4, cols: int = 4,
                present: dict | None = None) -> Fraction:  # fmt: skip
    """The exact chance that a shake shows `word`, a word of two or three faces. `present`
    can pass in present_colorings for the word's pattern, which is the slow part."""
    letters, shape = pattern(word)
    n = rows * cols
    assert len(cubes) == n
    present = present_colorings(shape, rows, cols) if present is None else present
    total = Fraction(0)
    for k, ways in count_ways(letters, cubes).items():
        hits = present.get(k, 0)
        if hits:
            colorings = factorial(n) // (factorial(n - sum(k)) * prod(factorial(x) for x in k))
            total += Fraction(ways * hits, 6**n * int(colorings))
    return total


def short_chances(words: Iterable[str]) -> dict[str, Fraction]:
    """word_chance for every word of two or three faces among `words`."""
    shapes: dict = {}
    out = {}
    for w in sorted(words):
        if len(word_faces(w) or ()) <= 3:
            _, shape = pattern(w)
            if shape not in shapes:
                shapes[shape] = present_colorings(shape)
            out[w] = word_chance(w, present=shapes[shape])
    return out


def coloring_chance(word: str, cubes: Cubes, rows: int, cols: int) -> Fraction:
    """A check on word_chance for small grids: every coloring of the slots, each tested by a
    path search, instead of present_colorings."""
    letters, shape = pattern(word)
    adjacent = grid_adjacent(rows, cols)

    def spells(colors: tuple[int, ...], k: int, slot: int, used: int) -> bool:
        if colors[slot] != shape[k]:
            return False
        if k == len(shape) - 1:
            return True
        used |= 1 << slot
        return any(spells(colors, k + 1, m, used) for m in adjacent[slot] if not used >> m & 1)

    present: Counter = Counter()
    for colors in product(range(len(letters) + 1), repeat=rows * cols):
        if any(spells(colors, 0, s, 0) for s in range(rows * cols)):
            present[tuple(colors.count(i) for i in range(len(letters)))] += 1
    return word_chance(word, cubes, rows, cols, dict(present))


def shake_chance(word: str, cubes: Cubes, rows: int, cols: int) -> Fraction:
    """A check for square grids of up to four slots: every order of the cubes and every choice
    of faces, each board searched for the word by board.find_path. Faces that the word does not use are merged into one."""
    letters = set(word_faces(word) or ())
    sides = [Counter(f if f in letters else "#" for f in cube) for cube in cubes]
    hits = 0
    for order in permutations(range(len(cubes))):
        for faces in product(*(sides[c].items() for c in order)):
            weight = prod(w for _, w in faces)
            board = [[faces[r * cols + j][0] for j in range(cols)] for r in range(rows)]
            if find_path(board, word) is not None:
                hits += weight
    return Fraction(hits, factorial(len(cubes)) * 6 ** len(cubes))


def boards_needed(bounds: np.ndarray, target: float) -> int:
    """The fewest boards T with sum over words of 1 - (1 - bound)^T at least `target`: since
    that sum bounds the expected number of different words seen in T boards, fewer boards
    cannot show `target` different words on average."""
    p = np.minimum(bounds, 1.0)

    def seen(t: float) -> float:
        return float(-np.expm1(t * np.log1p(-p)).sum())

    lo, hi = 1, 2
    while seen(hi) < target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if seen(mid) < target else (lo, mid)
    return hi


def write(out: Path) -> None:
    """Write results/chances.json (or `out`)."""
    words = sorted({w for w in load_words(HERE / "enable1.txt") if len(w) >= 3 and possible(w)})
    short = short_chances(words)
    bound = word_bounds(words)
    b = np.array([bound[w] for w in words])
    edges = [10.0**e for e in range(-20, 1)]
    longer = [w for w in words if len(word_faces(w) or ()) >= 4]
    top_long = max(longer, key=lambda w: (bound[w], w))
    top_short = max(short, key=lambda w: (short[w], w))
    result = {
        "possible words": len(words),
        "most likely word": {
            "word": top_short,
            "chance": str(short[top_short]),
            "value": float(short[top_short]),
        },
        "largest bound, four or more faces": {
            "word": top_long,
            "bound": bound[top_long],
        },
        "bound over chance, short words": [
            round(min(float(expected_paths(w)) / float(p) for w, p in short.items()), 3),
            round(max(float(expected_paths(w)) / float(p) for w, p in short.items()), 3),
        ],
        "paths by slots": PATHS_4X4,
        "paths": sum(PATHS_4X4),
        "short words": {
            w: {"chance": str(p), "value": float(p), "bound": float(expected_paths(w))}
            for w, p in short.items()
        },
        "bound histogram": {
            "edges": edges,
            "counts": np.histogram(np.clip(b, 1e-20, 1), bins=edges)[0].tolist(),
        },
        "words with bound below": {
            f"{t:g}": int((b < t).sum()) for t in (1e-4, 1e-6, 1e-7, 1e-8, 1e-10)
        },
        "expected words seen, at most": {
            str(t): round(float(-np.expm1(t * np.log1p(-np.minimum(b, 1))).sum()), 1)
            for t in sorted(
                {100, 1_000, 15_000, 10**8, 10**10, 10**12}
                | {round(10 ** (2 + k / 10)) for k in range(41)}
            )
        },  # fmt: skip
        "boards needed for a share of the words, at least": {
            str(s): boards_needed(b, s * len(words)) for s in (0.5, 0.9)
        },
    }
    out.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=RESULTS / "chances.json")
    write(parser.parse_args().out)


if __name__ == "__main__":
    main()
