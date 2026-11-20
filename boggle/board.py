"""Boggle boards and word search, used by boggle.ipynb and the tests."""

import hashlib
import multiprocessing as mp
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from fractions import Fraction
from itertools import product
from math import ceil, factorial, log, sqrt
from pathlib import Path

type Board = list[list[str]]  # n rows of n faces (4 in Boggle); a face is a letter, or "qu"
type Trie = dict  # letter -> Trie, with the key None marking the end of a word

# The 16 cubes of the "New Boggle" set, sold from 1987 (or 1992; sources differ) until the
# sealed-case edition of about 2008, one tuple of six faces per cube. Listed alike by
# bananagrammer.com (2013), luke-g.com/boggle/dice.html, robertgamble.net (2016) and Princeton's
# algs4 BoggleBoard.java ("1992 version").
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

# The 16 cubes of the classic set, sold from about 1976 to 1986, from the same four listings
# (Princeton calls it the "1983 version").
CLASSIC_CUBES: list[tuple[str, ...]] = [
    tuple("qu" if c == "q" else c for c in cube)
    for cube in (
        "aaciot abilty abjmoq acdemp acelrs adenvz ahmors biforx "
        "denosw dknotu eefhiy egintv egkluy ehinps elpstu gilruw"
    ).split()
]

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def rand_layout(
    rng: random.Random | None = None, cubes: list[tuple[str, ...]] = CUBES
) -> tuple[list[int], Board]:
    """Shake the cubes into the 16 slots and show a random face of each. Returns which
    cube went into each slot (as an index into `cubes`, row by row) and the board."""
    rng = rng or random.Random()
    order = rng.sample(range(len(cubes)), len(cubes))  # same draws as sampling the cubes
    faces = [rng.choice(cubes[c]) for c in order]
    return order, [faces[i : i + 4] for i in range(0, 16, 4)]


def rand_board(rng: random.Random | None = None, cubes: list[tuple[str, ...]] = CUBES) -> Board:
    """Shake the cubes into the 16 slots and show a random face of each."""
    return rand_layout(rng, cubes)[1]


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

    n = len(board)  # 4 for Boggle, 5 for Big Boggle

    def recur(i: int, j: int, path: str, seen: frozenset, node: Trie) -> Iterator[str]:
        if (i, j) in seen or not (0 <= i < n and 0 <= j < n):  # no reuse, no wrapping
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
    for i, j in product(range(n), range(n)):
        for word in recur(i, j, "", frozenset(), trie):
            if len(word) >= min_len and word not in found:
                found.add(word)
                yield word


def sample_layouts(n: int, seed: int) -> list[tuple[list[int], Board]]:
    """n layouts (cube order and board) drawn from a single random.Random(seed) stream."""
    rng = random.Random(seed)
    return [rand_layout(rng) for _ in range(n)]


def sample_boards(n: int, seed: int) -> list[Board]:
    """n boards drawn one after another from a single random.Random(seed) stream."""
    return [board for _, board in sample_layouts(n, seed)]


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


def _layout_losses(layout: tuple[list[int], Board]) -> list[int]:
    return cube_losses(*layout, _WORKER["trie"], _WORKER["min_len"])


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


def score(board: Board, trie: Trie) -> int:
    """Total points of the words on a board, each word counted once."""
    return sum(map(points, get_words(board, trie)))


def parse_board(letters: str) -> Board:
    """A square board from its letters read row by row (16 for Boggle, 25 for Big Boggle);
    "q" stands for the "qu" face, and spaces are ignored."""
    faces = ["qu" if c == "q" else c for c in letters.lower() if c != " "]
    n = round(len(faces) ** 0.5)
    assert n * n == len(faces), letters
    return [faces[i : i + n] for i in range(0, n * n, n)]


def show(board: Board) -> str:
    """The board as rows of letters, with Qu written as one face."""
    return "\n".join(" ".join(f"{face.capitalize():2}" for face in row).rstrip() for row in board)


def find_path(board: Board, word: str) -> list[tuple[int, int]] | None:
    """One path of neighboring squares, none used twice, that spells `word` on the board,
    as (row, column) pairs; None if there is none."""
    n = len(board)

    def extend(path: list[tuple[int, int]], rest: str) -> list[tuple[int, int]] | None:
        if not rest:
            return path
        i, j = path[-1]
        for di, dj in NEIGHBORS:
            nxt = (i + di, j + dj)
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and nxt not in path:
                face = board[nxt[0]][nxt[1]]
                if rest.startswith(face) and (found := extend([*path, nxt], rest[len(face) :])):
                    return found
        return None

    for i, j in product(range(n), range(n)):
        if word.startswith(board[i][j]) and (found := extend([(i, j)], word[len(board[i][j]) :])):
            return found
    return None


# The letter distribution that Norvig's Inverse Boggle draws from ("the distribution of
# letters in the 16-cube version of the game"), as in his pytudes notebook Boggle.ipynb.
NORVIG_LETTERS = 3 * "AABCDEEEGHIILMNOOPRSTUY" + "AADEFFIJKKLLNNQRSSTTUVVWWXZ"


def hill_climb(
    board: Board, trie: Trie, steps: int, rng: random.Random, letters: str = NORVIG_LETTERS
) -> tuple[Board, list[int]]:
    """Norvig's hill climbing for Inverse Boggle: change one random square to a random
    letter, and keep the change if the score does not fall. Returns the final board and
    the score after each step (the first entry is the starting score)."""
    board = [list(row) for row in board]
    n = len(board)
    best = score(board, trie)
    history = [best]
    for _ in range(steps):
        i, j = rng.randrange(n), rng.randrange(n)
        old = board[i][j]
        letter = rng.choice(letters).lower()
        board[i][j] = "qu" if letter == "q" else letter
        new = score(board, trie)
        if new >= best:
            best = new
        else:
            board[i][j] = old
        history.append(best)
    return board, history


def orientations(board: Board) -> list[Board]:
    """The different boards among the 4 rotations and 4 reflections of `board`."""
    seen, out = set(), []
    b = board
    for _ in range(4):
        b = [list(row) for row in zip(*b[::-1], strict=True)]  # a quarter turn clockwise
        for c in (b, [row[::-1] for row in b]):  # and its mirror image
            key = tuple(map(tuple, c))
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


def permanent(matrix: list[list[int]]) -> int:
    """The permanent of a square matrix, by Ryser's formula with a Gray code:
    perm(A) = (-1)^n sum over column sets S of (-1)^|S| prod_i sum_{j in S} A[i][j]."""
    n = len(matrix)
    sums = [0] * n  # row sums over the current column set S
    total = 0
    for k in range(1, 2**n):
        j = (k & -k).bit_length() - 1  # the column that enters or leaves S at step k
        sign = 1 if (k ^ (k >> 1)) >> j & 1 else -1  # is it in the Gray code set now?
        for i in range(n):
            sums[i] += sign * matrix[i][j]
        prod = 1
        for s in sums:
            prod *= s
            if not prod:
                break
        total += -prod if (k ^ (k >> 1)).bit_count() % 2 else prod
    return total * (-1) ** n


def face_counts(faces: list[str], cubes: Sequence[tuple[str, ...]] = CUBES) -> list[list[int]]:
    """Entry [i][j] is how many faces of cube i show the face wanted in slot j."""
    return [[cube.count(face) for face in faces] for cube in cubes]


def roll_ways(board: Board, cubes: Sequence[tuple[str, ...]] = CUBES) -> int:
    """How many of the len(cubes)! * 6**len(cubes) outcomes of a shake (which cube goes in
    each slot, and which face of it is up) give exactly this board: the permanent of the
    face counts, since an outcome is a matching of cubes to slots and a face per cube."""
    return permanent(face_counts([face for row in board for face in row], cubes))


def roll_chance(
    board: Board, cubes: Sequence[tuple[str, ...]] = CUBES, turned: bool = True
) -> Fraction:
    """The exact chance that a shake gives this board; with `turned`, the chance that it
    gives the board in any of its orientations (rotated or reflected)."""
    boards = orientations(board) if turned else [board]
    outcomes = factorial(len(cubes)) * 6 ** len(cubes)
    return Fraction(sum(roll_ways(b, cubes) for b in boards), outcomes)


# The slots adjacent to each slot, numbering the slots 0 to 15 row by row.
ADJACENT = [
    [4 * (i + di) + j + dj for di, dj in NEIGHBORS if 0 <= i + di < 4 and 0 <= j + dj < 4]
    for i, j in product(range(4), range(4))
]


def word_slots(board: Board, trie: Trie, min_len: int = 3) -> dict[str, int]:
    """Each word on the board, with the slots that every path spelling it passes through,
    as a bit mask (bit k for slot k, row by row). Blanking the cube in one of those slots
    takes the word off the board; blanking any other cube leaves a path for it."""
    faces = [face for row in board for face in row]
    masks: dict[str, int] = {}

    def recur(k: int, path: str, seen: int, node: Trie) -> None:
        for letter in faces[k]:
            child = node.get(letter)
            if child is None:
                return
            node = child
        path += faces[k]
        seen |= 1 << k
        if None in node and len(path) >= min_len:
            masks[path] = masks.get(path, seen) & seen
        for n in ADJACENT[k]:
            if not seen >> n & 1:
                recur(n, path, seen, node)

    for k in range(16):
        recur(k, "", 0, trie)
    return masks


def cube_losses(order: list[int], board: Board, trie: Trie, min_len: int = 3) -> list[int]:
    """Points the board loses when each cube is replaced by a blank, indexed by cube
    (`order` says which cube is in each slot, row by row, as rand_layout returns it)."""
    losses = [0] * len(order)
    for word, mask in word_slots(board, trie, min_len).items():
        for k in range(16):
            if mask >> k & 1:
                losses[order[k]] += points(word)
    return losses


def all_cube_losses(
    layouts: list[tuple[list[int], Board]], trie: Trie, processes: int | None = None
) -> list[list[int]]:
    """cube_losses for each layout, computed in a pool of processes like count_words."""
    with mp.Pool(processes, initializer=_set_trie, initargs=(trie, 3)) as pool:
        return list(pool.imap(_layout_losses, layouts, chunksize=500))


def one_square_changes(
    board: Board, trie: Trie, processes: int | None = None
) -> dict[tuple[int, int, str], int]:
    """The score of every board that differs from `board` in one square, keyed by (row,
    column, new face), with the 26 faces a to z and "qu" for q. Scored in a pool of
    processes like count_words."""
    n = len(board)
    changes = [
        (i, j, "qu" if c == "q" else c)
        for i in range(n)
        for j in range(n)
        for c in "abcdefghijklmnopqrstuvwxyz"
        if board[i][j] != ("qu" if c == "q" else c)
    ]
    variants = []
    for i, j, face in changes:
        b = [list(row) for row in board]
        b[i][j] = face
        variants.append(b)
    with mp.Pool(processes, initializer=_set_trie, initargs=(trie, 3)) as pool:
        scores = pool.map(_board_score, variants, chunksize=8)
    return dict(zip(changes, scores, strict=True))


def _board_score(board: Board) -> int:
    return sum(map(points, get_words(board, _WORKER["trie"], _WORKER["min_len"])))


def words_by_paths(board: Board, words: set[str], max_len: int) -> set[str]:
    """Brute force, as a check on get_words: spell every path of up to `max_len` letters,
    with no trie, and keep those in `words`."""
    n = len(board)
    found: set[str] = set()

    def walk(path: list[tuple[int, int]], spelled: str) -> None:
        if spelled in words:
            found.add(spelled)
        i, j = path[-1]
        for di, dj in NEIGHBORS:
            nxt = (i + di, j + dj)
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and nxt not in path:
                more = spelled + board[nxt[0]][nxt[1]]
                if len(more) <= max_len:
                    walk([*path, nxt], more)

    for i, j in product(range(n), range(n)):
        walk([(i, j)], board[i][j])
    return found


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


def word_faces(word: str) -> list[str] | None:
    """The faces that spell `word`, with "qu" as one face; None if some q is not followed by
    u, since no face shows a q alone."""
    faces, i = [], 0
    while i < len(word):
        if word[i] == "q":
            if word[i + 1 : i + 2] != "u":
                return None
            faces.append("qu")
            i += 2
        else:
            faces.append(word[i])
            i += 1
    return faces


def cube_matching(
    faces: Sequence[str], cubes: Sequence[tuple[str, ...]] = CUBES
) -> list[int] | None:
    """A different cube for each face, each cube showing that face on one of its sides
    (as a list of cube indices), or None if there is none. Kuhn's augmenting paths."""
    owner: dict[int, int] = {}  # cube -> position of the face it is matched to

    def augment(pos: int, seen: set[int]) -> bool:
        for c, cube in enumerate(cubes):
            if faces[pos] in cube and c not in seen:
                seen.add(c)
                if c not in owner or augment(owner[c], seen):
                    owner[c] = pos
                    return True
        return False

    if len(faces) > len(cubes) or not all(augment(pos, set()) for pos in range(len(faces))):
        return None
    match = [0] * len(faces)
    for c, pos in owner.items():
        match[pos] = c
    return match


# The slots of a path through all 16 slots, row by row and turning at each end; each slot
# is a neighbor of the one before.
SNAKE = [4 * i + (j if i % 2 == 0 else 3 - j) for i in range(4) for j in range(4)]


def possible(word: str, cubes: Sequence[tuple[str, ...]] = CUBES) -> bool:
    """Whether some shake of the cubes shows `word` along some path.

    That holds exactly when each face of the word can be given its own cube carrying that
    face: the cubes can then be laid along SNAKE, which visits neighboring slots in turn,
    with the right faces up. Where the letters fall on a real board does not matter.
    """
    faces = word_faces(word)
    return faces is not None and cube_matching(faces, cubes) is not None


def witness_board(word: str, cubes: Sequence[tuple[str, ...]] = CUBES) -> Board:
    """A board the cubes can roll that shows `word` along SNAKE; the other slots get the
    unused cubes, each showing its first face. Raises ValueError if the word is impossible."""
    faces = word_faces(word)
    match = cube_matching(faces, cubes) if faces is not None else None
    if faces is None or match is None:
        raise ValueError(f"{word} cannot be rolled")
    rest = [c for c in range(len(cubes)) if c not in match]
    slots = [""] * 16
    for k, slot in enumerate(SNAKE):
        slots[slot] = faces[k] if k < len(faces) else cubes[rest[k - len(faces)]][0]
    return [slots[i : i + 4] for i in range(0, 16, 4)]


def obstruction(word: str, cubes: Sequence[tuple[str, ...]] = CUBES) -> tuple[str, ...]:
    """Why `word` is impossible: ("q",) for a q without u, ("length",) for more than 16
    faces, or else a set of faces that the word uses more often than there are cubes
    carrying any of them (Hall's condition fails for it). Empty if the word is possible."""
    faces = word_faces(word)
    if faces is None:
        return ("q",)
    if len(faces) > len(cubes):
        return ("length",)
    for end in range(1, len(faces) + 1):  # the first prefix with no matching
        if cube_matching(faces[:end], cubes) is None:
            prefix = faces[:end]
            break
    else:
        return ()
    # Faces reachable from the last face by alternating paths form a Hall violator.
    match = cube_matching(prefix[:-1], cubes) or []
    owner = {c: pos for pos, c in enumerate(match)}
    reached, stack = {len(prefix) - 1}, [len(prefix) - 1]
    while stack:
        pos = stack.pop()
        for c, cube in enumerate(cubes):
            if prefix[pos] in cube and c in owner and owner[c] not in reached:
                reached.add(owner[c])
                stack.append(owner[c])
    return tuple(sorted({prefix[pos] for pos in reached}))


assert word_faces("quit") == ["qu", "i", "t"] and word_faces("qat") is None
assert [SNAKE.index(k) for k in (0, 3, 7, 4, 8)] == [0, 3, 4, 7, 8]
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
