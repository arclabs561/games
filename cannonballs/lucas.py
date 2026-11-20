"""Which piles hold a square number of balls, and which hold another pile's number.

**Lucas's problem** asks for the square piles whose balls can be laid out flat as a square:
1 + 4 + ... + n^2 = m^2. The answers are n = 1 and n = 24 (4,900 = 70^2), as Watson (1918)
proved. This module finds them three ways:

- brute force over n up to a bound;
- the split into six **cases** that every proof starts from, with a remainder test that rules
  out four of them;
- a walk along the solutions of the Pell equation that each remaining case needs, which finds
  every n below a bound of thousands of digits.

It also searches the other coincidences between figurate numbers, truncated square piles
(runs of consecutive squares) and oblong piles that hold a square number of balls.

    uv run python -m cannonballs.lucas
"""

import sys
from collections.abc import Callable, Iterator
from math import isqrt
from time import perf_counter

from cannonballs.piles import oblong, pyramid, tetrahedral, triangular

# n, n + 1 and 2n + 1 share no factor, so if n (n + 1) (2n + 1) = 6 m^2 then each one is a
# square times a divisor of 6. The divisors used multiply to 6, and 2n + 1 is odd, so it gets
# 1 or 3. A case lists the three divisors (for n, n + 1, 2n + 1).
type Case = tuple[int, int, int]
CASES: tuple[Case, ...] = ((1, 2, 3), (2, 1, 3), (1, 6, 1), (6, 1, 1), (2, 3, 1), (3, 2, 1))


def is_square(m: int) -> bool:
    return m >= 0 and isqrt(m) ** 2 == m


def is_triangular(m: int) -> bool:
    return m >= 0 and is_square(8 * m + 1)


def icbrt(m: int) -> int:
    """The largest x with x^3 <= m, for m >= 0."""
    if m < 8:
        return int(m >= 1)
    x = 1 << ((m.bit_length() + 2) // 3)
    while True:  # the tangent method, from above
        y = (2 * x + m // (x * x)) // 3
        if y >= x:
            break
        x = y
    while x**3 > m:
        x -= 1
    while (x + 1) ** 3 <= m:
        x += 1
    return x


def is_cube(m: int) -> bool:
    return m >= 0 and icbrt(m) ** 3 == m


def square_piles_that_are_squares(limit: int) -> list[int]:
    """Every n <= limit whose square pile holds a square number of balls, by brute force."""
    found, total = [], 0
    for n in range(1, limit + 1):
        total += n * n
        if is_square(total):
            found.append(n)
    return found


def case_of(n: int) -> Case:
    """Which case a solution n falls in: the part of 6 that each factor carries."""
    parts = []
    for f in (n, n + 1, 2 * n + 1):
        s = next(d for d in (1, 2, 3, 6) if f % d == 0 and is_square(f // d))
        parts.append(s)
    return (parts[0], parts[1], parts[2])


def powers(m: int, k: int) -> set[int]:
    """The remainders mod m of k-th powers."""
    return {pow(x, k, m) for x in range(m)}


def blocked_mod(case: Case, m: int, k: int = 2) -> bool:
    """True if no n has n, n + 1, 2n + 1 equal mod m to the case's parts times k-th powers."""
    q = powers(m, k)
    allowed = [{s * x % m for x in q} for s in case]
    return not any(
        n % m in allowed[0] and (n + 1) % m in allowed[1] and (2 * n + 1) % m in allowed[2]
        for n in range(m)
    )


def first_block(case: Case, k: int = 2, up_to: int = 200) -> int | None:
    """The smallest modulus that rules a case out, or None if none up to `up_to` does."""
    return next((m for m in range(2, up_to + 1) if blocked_mod(case, m, k)), None)


def pell(d: int, x: int, y: int, step: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """Solutions of X^2 - d Y^2 = c, from one solution (x, y), multiplying by a unit."""
    u, v = step
    assert u * u - d * v * v == 1
    while True:
        yield x, y
        x, y = x * u + d * y * v, x * v + y * u


def surviving_case_solutions(digits: int) -> dict[Case, list[int]]:
    """Every n < 10^digits in the two cases that remainders do not rule out.

    Case (6, 1, 1): n = 6x^2 and n + 1 = y^2, so y^2 - 6x^2 = 1. Every solution in positive
    integers is a power of the unit 5 + 2 sqrt 6 (a classical theorem), so walking the powers finds every
    candidate; each is kept if 2n + 1 = 12x^2 + 1 is a square.
    Case (1, 2, 3): n = x^2 and n + 1 = 2y^2, so x^2 - 2y^2 = -1. Every solution is an odd power
    of 1 + sqrt 2; each is kept if 2n + 1 = 2x^2 + 1 is three times a square."""
    bound = 10**digits
    found: dict[Case, list[int]] = {(6, 1, 1): [], (1, 2, 3): []}
    for _y, x in pell(6, 5, 2, (5, 2)):
        n = 6 * x * x
        if n >= bound:
            break
        if is_square(12 * x * x + 1):
            found[(6, 1, 1)].append(n)
    for x, _y in pell(2, 1, 1, (3, 2)):  # (1 + sqrt 2)^(2k+1) = (1 + sqrt 2)(3 + 2 sqrt 2)^k
        n = x * x
        if n >= bound:
            break
        if (2 * n + 1) % 3 == 0 and is_square((2 * n + 1) // 3):
            found[(1, 2, 3)].append(n)
    return found


def pell_candidates(digits: int) -> dict[Case, int]:
    """How many Pell solutions each surviving case walks through below 10^digits."""
    bound = 10**digits
    counts = {(6, 1, 1): 0, (1, 2, 3): 0}
    for _, x in pell(6, 5, 2, (5, 2)):
        if 6 * x * x >= bound:
            break
        counts[(6, 1, 1)] += 1
    for x, _ in pell(2, 1, 1, (3, 2)):
        if x * x >= bound:
            break
        counts[(1, 2, 3)] += 1
    return counts


def first_candidates(k: int = 4) -> dict[Case, list[tuple[int, int, int, bool]]]:
    """The first k Pell solutions of each surviving case, for the note's table: (x, n, the
    number that must be a square, whether it is one)."""
    out: dict[Case, list[tuple[int, int, int, bool]]] = {(6, 1, 1): [], (1, 2, 3): []}
    for (_, x), _i in zip(pell(6, 5, 2, (5, 2)), range(k), strict=False):
        n = 6 * x * x
        out[(6, 1, 1)].append((x, n, 2 * n + 1, is_square(2 * n + 1)))
    for (x, _), _i in zip(pell(2, 1, 1, (3, 2)), range(k), strict=False):
        n = x * x
        third = (2 * n + 1) // 3
        out[(1, 2, 3)].append((x, n, third, is_square(third)))
    return out


def square_piles_that_are_cubes(xmax: int) -> list[int]:
    """Every n <= xmax^3 whose square pile holds a cube number of balls.

    As for squares, n, n + 1 and 2n + 1 are each a cube times a divisor of 6, in one of the
    six cases. In each case walk x and y upward together until s1 x^3 + 1 = s2 y^3, then test
    2n + 1. The walk takes about xmax steps per case and no cube roots."""
    found = set()
    for s1, s2, s3 in CASES:
        x, y = 1, 1
        while x <= xmax:
            left, right = s1 * x**3 + 1, s2 * y**3
            if left < right:
                x += 1
            elif left > right:
                y += 1
            else:
                n = s1 * x**3
                if (2 * n + 1) % s3 == 0 and is_cube((2 * n + 1) // s3):
                    found.add(n)
                x += 1
    return sorted(found)


# ---------- other coincidences ----------

PILES: dict[str, Callable[[int], int]] = {"tetrahedral": tetrahedral, "square pile": pyramid}
TESTS: dict[str, Callable[[int], bool]] = {
    "square": is_square,
    "triangle": is_triangular,
    "cube": is_cube,
}


def coincidences(limit: int) -> dict[tuple[str, str], list[int]]:
    """For each pile kind and each test, the n <= limit whose pile passes the test."""
    return {
        (pile, test): [n for n in range(1, limit + 1) if ok(f(n))]
        for pile, f in PILES.items()
        for test, ok in TESTS.items()
    }


def tetrahedral_equals_square_pile(limit: int) -> list[tuple[int, int]]:
    """Pairs (n, k) with tetrahedral(n) = pyramid(k) and n <= limit, by merging the lists."""
    found, k = [], 1
    for n in range(1, limit + 1):
        t = tetrahedral(n)
        while pyramid(k) < t:
            k += 1
        if pyramid(k) == t:
            found.append((n, k))
    return found


# ---------- truncated piles: runs of consecutive squares ----------


def run_of_squares(top: int, layers: int) -> int:
    """Balls in a square pile with its top cut off: layers of side top, top + 1, ..., in all
    `layers` layers."""
    return pyramid(top + layers - 1) - pyramid(top - 1)


def smallest_top(layers: int, search: int) -> int | None:
    """The smallest top >= 1 whose truncated pile of `layers` layers is a square, if one is
    at most `search`."""
    return next((t for t in range(1, search + 1) if is_square(run_of_squares(t, layers))), None)


def run_blocked_mod(layers: int, m: int) -> bool:
    """True if no truncated pile with this many layers is a square mod m, whatever its top."""
    q = powers(m, 2)
    return not any(run_of_squares(t, layers) % m in q for t in range(1, m + 1))


def run_first_block(layers: int, up_to: int = 200) -> int | None:
    return next((m for m in range(2, up_to + 1) if run_blocked_mod(layers, m)), None)


def runs_with_square_layers(k: int) -> list[int]:
    """Every top t >= 0 for which k^2 consecutive squares from t^2 add up to a square.

    With L = k^2 layers the sum is L (t + (L-1)/2)^2 + L (L^2 - 1)/12, so it is a square s^2
    exactly when (s/k)^2 - u^2 = (L^2 - 1)/12 with u = t + (L-1)/2: a difference of two squares,
    which factors, so there are finitely many. Needs L odd and not a multiple of 3."""
    layers = k * k
    assert layers % 2 == 1 and layers % 3 != 0
    c = (layers * layers - 1) // 12  # an integer when L is prime to 6
    tops = []
    for d in range(1, isqrt(c) + 1):  # (w - u)(w + u) = c with d = w - u
        if c % d == 0 and (d + c // d) % 2 == 0:
            u = (c // d - d) // 2
            t = u - (layers - 1) // 2
            if t >= 0:
                tops.append(t)
    return sorted(tops)


def truncated_table(max_layers: int, search: int) -> list[tuple[int, str, int | None]]:
    """For each number of layers, how it is settled: ("found", smallest top), ("blocked",
    modulus) or ("square", None) when the layer count is a square handled by factoring."""
    rows = []
    for layers in range(1, max_layers + 1):
        top = smallest_top(layers, search)
        if top is not None:
            rows.append((layers, "found", top))
            continue
        k = isqrt(layers)
        if k * k == layers and layers % 6 in (1, 5):
            tops = [t for t in runs_with_square_layers(k) if t >= 1]
            assert not tops, (layers, tops)  # a top >= 1 would have been found by the search
            rows.append((layers, "square", None))
            continue
        rows.append((layers, "blocked", run_first_block(layers)))
    return rows


# ---------- oblong piles ----------


def oblong_square(a: int) -> tuple[int, int] | None:
    """For an oblong pile of short side a, the smallest long side b >= a giving a square number
    of balls, with that square's root; None if no b does.

    Lengthening the pile by one ball adds a triangle of side a, so the counts for b = a, a + 1,
    ... step by T = a(a+1)/2. A square x^2 is among them exactly when x^2 >= count(a) and x^2
    leaves the same remainder as count(a) on division by T. If any x does, so does every
    x + jT, so either no remainder works or the smallest x at least sqrt(count(a)) works and
    lies within T of it."""
    start, step = oblong(a, a), triangular(a)
    if not any((x * x - start) % step == 0 for x in range(step)):
        return None
    x = isqrt(start - 1) + 1 if start > 0 else 0
    while (x * x - start) % step:
        x += 1
    b = a + (x * x - start) // step
    assert oblong(a, b) == x * x
    return b, x


def threes(m: int) -> int:
    """How many times 3 divides m."""
    k = 0
    while m % 3 == 0:
        m //= 3
        k += 1
    return k


def oblong_has_square(a: int) -> bool:
    """The rule proved in the note: some oblong pile of short side a holds a square number of
    balls exactly when a leaves remainder 1 on division by 3, or when the multiple of 3 among
    a and a + 1, divided by 3, is 9^k (3j + 2) for some k and j."""
    if a % 3 == 1:
        return True
    m = (a if a % 3 == 0 else a + 1) // 3
    while m % 9 == 0:
        m //= 9
    return m % 3 == 2


def oblong_contacts(a: int, b: int) -> int:
    """Touching pairs in an oblong pile on an a by b base (proved in the note)."""
    return 2 * a * a * (a - 1) + (b - a) * a * (3 * a - 2)


def oblong_square_brute(a: int, longest: int) -> int | None:
    """The smallest b with a <= b <= longest whose oblong pile is a square, by trying each b."""
    return next((b for b in range(a, longest + 1) if is_square(oblong(a, b))), None)


# ---------- distances to the nearest square, for the figure ----------


def gap_to_square(m: int) -> int:
    """How far m is from the nearest square."""
    r = isqrt(m)
    return min(m - r * r, (r + 1) ** 2 - m)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10**6
    digits = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    t = perf_counter()
    print(f"square piles that are squares, n <= {limit:,}:", square_piles_that_are_squares(limit))
    print(f"  brute force took {perf_counter() - t:.1f} s")
    for case in CASES:
        print("case", case, "ruled out mod", first_block(case))
    t = perf_counter()
    print(f"by Pell equations, n < 10^{digits}:", surviving_case_solutions(digits))
    print(f"  walked {pell_candidates(digits)} solutions in {perf_counter() - t:.1f} s")
    for case in CASES:
        print("cubes: case", case, "ruled out mod", first_block(case, k=3))
    for key, ns in coincidences(10**5).items():
        print(key, ns)
    print("tetrahedral = square pile:", tetrahedral_equals_square_pile(10**5))
    for row in truncated_table(30, 10**5):
        print(row)
    print([(a, oblong_square(a)) for a in range(1, 31)])


if __name__ == "__main__":
    main()
