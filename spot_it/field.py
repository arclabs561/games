"""Finite fields GF(q) for q a prime or a prime power, as addition and multiplication tables.

An element of GF(p^k) is a polynomial of degree below k with coefficients mod p, stored as the
integer whose base-p digits are its coefficients (constant term first). Products are reduced
modulo the least monic irreducible polynomial of degree k, found by trial division. For k = 1
this is arithmetic mod p.

    uv run python -m spot_it.field     # checks the field axioms for every q up to 16
"""

from dataclasses import dataclass
from functools import cache
from itertools import product

type Poly = tuple[int, ...]  # coefficients mod p, constant term first, no trailing zeros


def prime_power(q: int) -> tuple[int, int] | None:
    """(p, k) with q = p**k and p prime, or None if q is not a prime power."""
    if q < 2:
        return None
    p = next(d for d in range(2, q + 1) if q % d == 0)
    k = 0
    while q % p == 0:
        q //= p
        k += 1
    return (p, k) if q == 1 else None


def trim(a: Poly) -> Poly:
    """Drop zero coefficients at the top."""
    end = len(a)
    while end and a[end - 1] == 0:
        end -= 1
    return tuple(a[:end])


def poly_mod(a: Poly, m: Poly, p: int) -> Poly:
    """Remainder of a on division by the monic polynomial m, coefficients mod p."""
    r = [c % p for c in a]
    while len(trim(tuple(r))) >= len(m):
        r = list(trim(tuple(r)))
        shift, lead = len(r) - len(m), r[-1]
        for i, c in enumerate(m):
            r[shift + i] = (r[shift + i] - lead * c) % p
    return trim(tuple(r))


def monic_polys(degree: int, p: int):
    """Every monic polynomial of the given degree over GF(p)."""
    for low in product(range(p), repeat=degree):
        yield (*low, 1)


def is_irreducible(m: Poly, p: int) -> bool:
    """True if no monic polynomial of degree 1 to deg(m) / 2 divides m (trial division)."""
    deg = len(m) - 1
    return all(poly_mod(m, d, p) != () for e in range(1, deg // 2 + 1) for d in monic_polys(e, p))


def modulus(p: int, k: int) -> Poly:
    """The first monic irreducible polynomial of degree k over GF(p), in `monic_polys` order."""
    return next(m for m in monic_polys(k, p) if is_irreducible(m, p))


def to_poly(x: int, p: int) -> Poly:
    """The polynomial whose base-p digits are those of x."""
    digits = []
    while x:
        x, d = divmod(x, p)
        digits.append(d)
    return tuple(digits)


def from_poly(a: Poly, p: int) -> int:
    return sum(c * p**i for i, c in enumerate(a))


def poly_mul(a: Poly, b: Poly, p: int) -> Poly:
    out = [0] * (len(a) + len(b))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return trim(tuple(out))


@dataclass(frozen=True)
class Field:
    """GF(q) with elements 0..q-1; 0 and 1 are the field's zero and one."""

    q: int
    p: int
    modulus: Poly
    add: tuple[tuple[int, ...], ...]
    mul: tuple[tuple[int, ...], ...]

    @property
    def elements(self) -> range:
        return range(self.q)

    def neg(self, x: int) -> int:
        return self.add[x].index(0)

    def inv(self, x: int) -> int:
        return self.mul[x].index(1)


@cache
def gf(q: int) -> Field:
    """The field with q elements. Raises ValueError if q is not a prime power."""
    pk = prime_power(q)
    if pk is None:
        raise ValueError(f"no field has {q} elements")
    p, k = pk
    m = modulus(p, k)
    polys = [to_poly(x, p) for x in range(q)]
    add = tuple(
        tuple(
            from_poly(tuple((a + b) % p for a, b in zip_pad(polys[x], polys[y])), p)
            for y in range(q)
        )
        for x in range(q)
    )
    mul = tuple(
        tuple(from_poly(poly_mod(poly_mul(polys[x], polys[y], p), m, p), p) for y in range(q))
        for x in range(q)
    )
    return Field(q, p, m, add, mul)


def zip_pad(a: Poly, b: Poly):
    n = max(len(a), len(b))
    return zip(a + (0,) * (n - len(a)), b + (0,) * (n - len(b)), strict=True)


def axiom_failures(add, mul) -> list[str]:
    """Names of the field axioms that the tables break, checked on every pair and triple."""
    els = range(len(add))
    failures = []
    checks = {
        "addition commutes": all(add[x][y] == add[y][x] for x in els for y in els),
        "multiplication commutes": all(mul[x][y] == mul[y][x] for x in els for y in els),
        "addition associates": all(
            add[add[x][y]][z] == add[x][add[y][z]] for x in els for y in els for z in els
        ),
        "multiplication associates": all(
            mul[mul[x][y]][z] == mul[x][mul[y][z]] for x in els for y in els for z in els
        ),
        "distributive": all(
            mul[x][add[y][z]] == add[mul[x][y]][mul[x][z]] for x in els for y in els for z in els
        ),
        "0 and 1 are identities": all(add[0][x] == x and mul[1][x] == x for x in els),
        "every element has a negative": all(0 in add[x] for x in els),
        "every nonzero element has an inverse": all(1 in mul[x] for x in els if x != 0),
    }
    failures = [name for name, ok in checks.items() if not ok]
    return failures


def integers_mod(n: int):
    """Addition and multiplication tables of the integers mod n (a field only for prime n)."""
    els = range(n)
    return (
        tuple(tuple((x + y) % n for y in els) for x in els),
        tuple(tuple((x * y) % n for y in els) for x in els),
    )


def poly_str(a: Poly) -> str:
    """A polynomial written out, highest power first: x^3 + x + 1."""
    terms = []
    for i in reversed(range(len(a))):
        c = a[i]
        if c == 0:
            continue
        power = "" if i == 0 else "x" if i == 1 else f"x^{i}"
        coef = "" if c == 1 and i > 0 else str(c)
        terms.append(coef + power)
    return " + ".join(terms)


def main() -> None:
    for q in range(2, 17):
        pk = prime_power(q)
        ring = axiom_failures(*integers_mod(q))
        if pk is None:
            print(f"{q:2}: no field; the integers mod {q} fail: {', '.join(ring)}")
            continue
        f = gf(q)
        bad = axiom_failures(f.add, f.mul)
        assert not bad, (q, bad)
        p, k = pk
        if k == 1:
            print(f"{q:2}: the integers mod {q} pass every field axiom")
        else:
            print(
                f"{q:2}: polynomials over GF({p}) modulo {poly_str(f.modulus)} pass every field"
                f" axiom; the integers mod {q} fail: {', '.join(ring)}"
            )


if __name__ == "__main__":
    main()
