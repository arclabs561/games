"""How many balls are in a pile, and how many pairs of them touch.

A **pile** is a stack of equal balls in which every ball above the bottom layer rests in the
hollow between balls of the layer below. Four kinds are counted:

- a **triangle** of side n, one flat layer with rows of 1, 2, ..., n balls;
- a **square** of side n, one flat layer of n rows of n;
- a **tetrahedral pile** of n layers, triangles of side 1, 2, ..., n;
- a **square pile** of n layers, squares of side 1, 2, ..., n;
- an **oblong pile** on an a by b base (a <= b), layers a by b, a - 1 by b - 1, ..., down to
  a ridge of 1 by b - a + 1.

Each count has a formula and a brute-force check that builds the pile ball by ball. The balls
are placed at integer points of the face-centered cubic lattice (points whose coordinates add
up to an even number), where two balls touch exactly when their centers are at squared
distance 2. Square piles and tetrahedral piles are both pieces of that one lattice.

    uv run python -m cannonballs.piles
"""

from collections import Counter
from math import comb

type Ball = tuple[int, int, int]

# The twelve neighbors of a ball in the face-centered cubic lattice: every way to put +1 or -1
# in two of the three coordinates.
NEIGHBORS: tuple[Ball, ...] = tuple(
    v
    for v in ((a, b, 0) for a in (1, -1) for b in (1, -1))
    for v in (v, (v[0], 0, v[1]), (0, v[0], v[1]))
)


def triangular(n: int) -> int:
    """Balls in a triangle of side n: 1 + 2 + ... + n."""
    return n * (n + 1) // 2


def square(n: int) -> int:
    """Balls in a square of side n."""
    return n * n


def tetrahedral(n: int) -> int:
    """Balls in a tetrahedral pile of n layers: triangles of side 1 to n."""
    return n * (n + 1) * (n + 2) // 6


def pyramid(n: int) -> int:
    """Balls in a square pile of n layers: 1 + 4 + 9 + ... + n squared."""
    return n * (n + 1) * (2 * n + 1) // 6


def oblong(a: int, b: int) -> int:
    """Balls in an oblong pile on an a by b base, with a <= b."""
    assert 1 <= a <= b
    return a * (a + 1) * (3 * b - a + 1) // 6


def triangle_layer(n: int) -> list[tuple[int, int]]:
    """Positions of the balls in a triangle of side n, row by row."""
    return [(row, k) for row in range(n) for k in range(row + 1)]


def tetrahedral_pile(n: int) -> list[Ball]:
    """Centers of the balls of a tetrahedral pile of n layers.

    The three edges from one corner run along (1, 1, 0), (1, 0, 1) and (0, 1, 1), which meet
    at 60 degrees, as the edges of a regular tetrahedron do."""
    return [(i + j, i + k, j + k) for i in range(n) for j in range(n - i) for k in range(n - i - j)]


def oblong_pile(a: int, b: int) -> list[Ball]:
    """Centers of the balls of an oblong pile on an a by b base, bottom layer first.

    Within a layer the rows run along (1, 1, 0) and (1, -1, 0), at right angles. Each layer sits
    one step of (1, 0, 1) above the last: over the middle of four balls, touching all four."""
    assert 1 <= a <= b
    return [(h + i + j, i - j, h) for h in range(a) for i in range(a - h) for j in range(b - h)]


def square_pile(n: int) -> list[Ball]:
    """Centers of the balls of a square pile of n layers."""
    return oblong_pile(n, n)


def touching_pairs(balls: list[Ball]) -> int:
    """How many pairs of balls touch, found by looking at each ball's twelve neighbors."""
    where = set(balls)
    assert len(where) == len(balls), "two balls in one place"
    seen = sum((x + dx, y + dy, z + dz) in where for x, y, z in balls for dx, dy, dz in NEIGHBORS)
    return seen // 2


def touching_pairs_slow(balls: list[Ball]) -> int:
    """The same count by checking every pair's squared distance."""
    return sum(
        sum((p - q) ** 2 for p, q in zip(u, v, strict=True)) == 2
        for i, u in enumerate(balls)
        for v in balls[i + 1 :]
    )


def no_overlaps(balls: list[Ball]) -> bool:
    """No two balls closer than touching: every squared distance is at least 2."""
    return all(
        sum((p - q) ** 2 for p, q in zip(u, v, strict=True)) >= 2
        for i, u in enumerate(balls)
        for v in balls[i + 1 :]
    )


def dot(u: Ball, v: Ball) -> int:
    return sum(p * q for p, q in zip(u, v, strict=True))


def resting(balls: list[Ball], up: Ball) -> bool:
    """Every ball above the bottom layer touches at least three balls of the layer below it.

    `up` points upward: (0, 0, 1) for square and oblong piles, (-1, -1, -1) for tetrahedral
    piles, whose layers are the planes on which x + y + z is constant."""
    where = set(balls)
    bottom = min(dot(b, up) for b in balls)
    below = [d for d in NEIGHBORS if dot(d, up) < 0]
    return all(
        dot(b, up) == bottom
        or sum(tuple(p + q for p, q in zip(b, d, strict=True)) in where for d in below) >= 3
        for b in balls
    )


def square_pile_contacts(n: int) -> int:
    """Touching pairs in a square pile of n layers (proved in the note): 2 n^2 (n - 1)."""
    return 2 * n * n * (n - 1)


def tetrahedral_contacts(n: int) -> int:
    """Touching pairs in a tetrahedral pile of n layers (proved in the note): (n - 1) n (n + 1)."""
    return (n - 1) * n * (n + 1)


def contacts_per_ball(balls: list[Ball]) -> Counter[int]:
    """How many balls touch exactly k others, for each k."""
    where = set(balls)
    return Counter(
        sum((x + dx, y + dy, z + dz) in where for dx, dy, dz in NEIGHBORS) for x, y, z in balls
    )


def three_triangles(n: int) -> list[list[int]]:
    """The entrywise sum of three copies of the number triangle for 1 + 4 + ... + n^2.

    Row r (from 1) of the triangle holds r copies of r, so its entries add up to the square
    pile of n layers. Place a number at each point (a, b, c) with a + b + c = n - 1 and give it
    n - a; the two turned copies give it n - b and n - c. The three always add up to 2n + 1."""
    rows = []
    for row in range(n):
        sums = []
        for col in range(row + 1):
            a, b, c = n - 1 - row, col, row - col
            sums.append((n - a) + (n - b) + (n - c))
        rows.append(sums)
    return rows


def layer_sizes_oblong(a: int, b: int) -> list[tuple[int, int]]:
    """Layer sizes of an oblong pile, bottom first."""
    return [(a - h, b - h) for h in range(a)]


def binomial_tetrahedral(n: int) -> int:
    """The tetrahedral number as a choice of 3 things from n + 2."""
    return comb(n + 2, 3)


def main() -> None:
    print("layers  triangle  square  tetrahedral  square pile")
    for n in range(1, 11):
        print(f"{n:6}  {triangular(n):8}  {square(n):6}  {tetrahedral(n):11}  {pyramid(n):11}")
    for n in (4, 24):
        balls = square_pile(n)
        print(
            f"square pile of {n} layers: {len(balls)} balls, {touching_pairs(balls)} touching pairs"
        )
    print(
        "balls by number of neighbors, square pile of 24:",
        dict(sorted(contacts_per_ball(square_pile(24)).items())),
    )


if __name__ == "__main__":
    main()
