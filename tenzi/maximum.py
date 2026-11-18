"""Why H_n / ln(6/5) + 1/2 gives the named-face mean to ten decimal places.

When the face is named first, each die needs a Geometric(1/6) number of rolls and the game
lasts T = the longest of the n waits. A geometric wait is an exponential time rounded up:
if X has P(X > x) = e^(-lam x) with lam = ln(6/5), then P(ceil(X) > k) = (5/6)^k. So
T = ceil(M), where M is the largest of n independent exponential times. Two facts about M
give everything:

- E[M] = H_n / lam, since the gaps between successive order statistics are independent
  exponentials with rates n lam, (n - 1) lam, ..., lam (Renyi's representation);
- for the same reason the characteristic function of M is phi(s) = prod_j j lam / (j lam - i s).

Rounding up adds 1 - frac(M) on average, and the Fourier series of the sawtooth turns
E[frac(M)] into a sum over phi(2 pi k). The result is the exact identity

    E[T] = H_n / lam + 1/2 + sum over k != 0 of phi(2 pi k) / (2 pi i k),

the form Eisenberg (2008) studies, and the same as Rice's method gives for the alternating sum.
This module checks the identity against the exact mean in 60-digit decimal arithmetic, and
measures the size of each term. The term for k = +-1 has size
(2 / (lam y)) prod_j (1 + y^2 / j^2)^(-1/2) with y = 2 pi / lam = 34.46; as n grows it tends to
(2 / lam) |Gamma(i y)|, the amplitude of the periodic wobble in log n that Szpankowski and Rego
(1990) found, about 1.4e-23.

    uv run python -m tenzi.maximum
"""

from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb

getcontext().prec = 60
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628")

type Complex = tuple[Decimal, Decimal]  # (real, imaginary), in Decimal arithmetic


def exact_mean(dice: int, faces: int = 6) -> Fraction:
    """E[T] for the longest of `dice` Geometric(1/faces) waits, by the alternating sum."""
    q = Fraction(faces - 1, faces)
    terms = ((-1) ** (j + 1) * comb(dice, j) / (1 - q**j) for j in range(1, dice + 1))
    return sum(terms, Fraction(0))


def tail_sum_mean(dice: int, faces: int = 6, upto: int = 400) -> Fraction:
    """The same mean summed the slow way, sum over k < upto of P(T > k) = 1 - (1 - q^k)^dice."""
    q = Fraction(faces - 1, faces)
    return sum((1 - (1 - q**k) ** dice for k in range(upto)), Fraction(0))


assert exact_mean(1) == 6  # one die waits six rolls on average
assert exact_mean(2) == Fraction(96, 11)  # 2 * 6 - 1 / (1 - (5/6)^2) = 12 - 36/11
assert abs(tail_sum_mean(3) - exact_mean(3)) < Fraction(1, 10**25)


def decimal(x: Fraction) -> Decimal:
    """A Fraction as a Decimal."""
    return Decimal(x.numerator) / Decimal(x.denominator)


def rate(faces: int = 6) -> Decimal:
    """lam = ln(faces / (faces - 1)), the rate of the exponential time behind one die."""
    return (Decimal(faces) / Decimal(faces - 1)).ln()


def harmonic_approximation(dice: int, faces: int = 6) -> Decimal:
    """H_n / lam + 1/2."""
    harmonic = sum((Fraction(1, j) for j in range(1, dice + 1)), Fraction(0))
    return decimal(harmonic) / rate(faces) + Decimal(1) / 2


def times(a: Complex, b: Complex) -> Complex:
    """The product of two complex numbers."""
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def characteristic(dice: int, faces: int, s: Decimal) -> Complex:
    """phi(s) = E[exp(i s M)] = prod_j j lam / (j lam - i s), for the longest exponential time M."""
    lam, value = rate(faces), (Decimal(1), Decimal(0))
    for j in range(1, dice + 1):
        a, b = j * lam, -s  # j lam - i s
        norm = a * a + b * b
        value = times(value, (j * lam * a / norm, -j * lam * b / norm))
    return value


def fourier_term(dice: int, faces: int, k: int) -> Decimal:
    """The terms for k and -k together: 2 Re(phi(2 pi k) / (2 pi i k)) = Im(phi(2 pi k)) / (pi k)."""
    return characteristic(dice, faces, 2 * PI * k)[1] / (PI * k)


def fourier_sum(dice: int, faces: int = 6, terms: int = 200) -> Decimal:
    """The sum of the Fourier terms for 1 <= k <= terms."""
    return sum((fourier_term(dice, faces, k) for k in range(1, terms + 1)), Decimal(0))


def term_size(dice: int, faces: int = 6, k: int = 1) -> Decimal:
    """|terms k and -k| = (2 / (lam y)) prod_j (1 + y^2 / j^2)^(-1/2), with y = 2 pi k / lam."""
    lam = rate(faces)
    y = 2 * PI * k / lam
    size = 2 / (lam * y)
    for j in range(1, dice + 1):
        size /= (1 + y * y / (j * j)).sqrt()
    return size


def wobble_amplitude(faces: int = 6, k: int = 1) -> Decimal:
    """The limit of term_size as dice -> infinity: (2 / lam) |Gamma(i y)|.

    |Gamma(i y)|^2 = pi / (y sinh(pi y)), from Gamma(z) Gamma(1 - z) = pi / sin(pi z).
    """
    lam = rate(faces)
    y = 2 * PI * k / lam
    sinh = ((PI * y).exp() - (-PI * y).exp()) / 2
    return 2 / lam * (PI / (y * sinh)).sqrt()


def dice_for_wobble(faces: int = 6, within: Decimal = Decimal(2)) -> int:
    """The fewest dice for which the k = 1 term is within a factor `within` of its limit."""
    lam = rate(faces)
    y = 2 * PI / lam
    size, limit, n = 2 / (lam * y), wobble_amplitude(faces), 0
    while size > within * limit:
        n += 1
        size /= (1 + y * y / (n * n)).sqrt()
    return n


def main() -> None:
    print("Named face, n dice: exact mean against H_n / ln(6/5) + 1/2 (60-digit arithmetic)")
    print("   n  exact mean          exact - approximation  Fourier sum, 200 terms  k = 1 alone")
    for n in (1, 2, 3, 5, 10, 20, 50, 100):
        exact = decimal(exact_mean(n))
        gap = exact - harmonic_approximation(n)
        print(
            f"  {n:3}  {float(exact):.12f}  {float(gap):21.6e}"
            f"  {float(fourier_sum(n)):22.6e}  {float(fourier_term(n, 6, 1)):11.4e}"
        )
    n = 10
    gap = decimal(exact_mean(n)) - harmonic_approximation(n)
    print(f"\nTen dice: exact - approximation = {gap:.6e};")
    print(f"  minus the Fourier sum: {gap - fourier_sum(n):.1e} (the sum stops at k = 200)")
    print(f"  size of the k = 1 term {term_size(n):.4e}, of the k = 2 term {term_size(n, k=2):.2e}")
    print(
        f"\nWobble amplitude as n -> infinity, (2 / lam) |Gamma(2 pi i / lam)|: {wobble_amplitude():.3e}"
    )
    print(f"  the k = 1 term is within a factor 2 of it from n = {dice_for_wobble()} dice on")
    print("  size of the k = 1 term:", end="")
    for n in (10, 100, 1000, 10_000):
        print(f"  n = {n}: {float(term_size(n)):.2e}", end="")
    print()


if __name__ == "__main__":
    main()
