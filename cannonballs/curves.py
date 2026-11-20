"""Oblong piles of a fixed shape that hold a square number of balls, exactly, as the
whole-number points of elliptic curves. Needs Sage, which the repository does not install:

    uv run --with passagemath-schemes --with passagemath-pari --with passagemath-flint \\
        --with passagemath-eclib --with passagemath-symbolics python -m cannonballs.curves

It writes `curves.json`, which the tests check against a plain search (`lucas.py`).

**The curves.** An oblong pile on an a by a + c base holds a(a + 1)(2a + k)/6 balls, with
k = 3c + 1 (`piles.oblong`). With X = 12a and Y = 72m, the count is m^2 exactly when

    Y^2 = X (X + 12) (X + 6k),

an elliptic curve in integral Weierstrass form. Every pile that is a square gives a point of it
with whole-number coordinates, so the list of all such points (Sage's `integral_points`, which
uses a Mordell-Weil basis that mwrank proves, and elliptic logarithms) gives every a. Lucas's
problem is c = 0. The pile on an a by 2a - 1 base, whose ridge is as long as the base is wide,
holds a(a + 1)(5a - 2)/6 balls; with X = 30a and Y = 180m it is Y^2 = X (X + 30) (X - 12), the
curve of Puzzling Stack Exchange question 99084.
"""

import importlib
import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
CURVES = FOLDER / "curves.json"
LONGEST = 20  # c = 0, 1, ..., LONGEST


def shape_curve(c: int) -> tuple[list[int], int, int]:
    """The coefficients [a1, a2, a3, a4, a6] of the curve for the base a by a + c, and the
    scales s, t with X = s a and Y = t m."""
    k = 3 * c + 1
    # X (X + 12) (X + 6k) = X^3 + (12 + 6k) X^2 + 72k X
    return [0, 12 + 6 * k, 0, 72 * k, 0], 12, 72


def ridge_curve() -> tuple[list[int], int, int]:
    """The curve for the base a by 2a - 1: X (X + 30) (X - 12) = X^3 + 18 X^2 - 360 X."""
    return [0, 18, 0, -360, 0], 30, 180


def solve(coefficients: list[int], s: int, t: int) -> dict:
    sage = importlib.import_module("sage.all__sagemath_schemes")
    curve = sage.EllipticCurve(coefficients)
    basis = curve.gens(proof=True)  # mwrank fails rather than return a basis it cannot prove
    points = sorted((int(p[0]), abs(int(p[1]))) for p in curve.integral_points(mw_base=basis))
    layers = sorted(x // s for x, y in points if x >= s and x % s == 0 and y % t == 0)
    return {
        "coefficients": coefficients,
        "scale": [s, t],
        "rank": len(basis),
        "generators": [[str(g[0]), str(g[1])] for g in basis],
        "integral_points": [list(p) for p in points],
        "layers": layers,
    }


def results() -> dict:
    out: dict = {"by_longer_side": {}}
    for c in range(LONGEST + 1):
        out["by_longer_side"][str(c)] = solve(*shape_curve(c))
    out["ridge"] = solve(*ridge_curve())
    return out


def version() -> str:
    return str(importlib.import_module("sage.version").version)


def main() -> None:
    data = results()
    lines = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in data["by_longer_side"].items()]
    text = '{\n "by_longer_side": {\n' + ",\n".join(" " + line for line in lines) + "\n },\n"
    text += f' "ridge": {json.dumps(data["ridge"])},\n "sage": {json.dumps(version())}\n}}\n'
    CURVES.write_text(text)
    print("wrote", CURVES.name)
    for c, row in data["by_longer_side"].items():
        print(f"c = {c}: rank {row['rank']}, layers {row['layers']}")
    print("ridge:", data["ridge"]["layers"])


if __name__ == "__main__":
    main()
