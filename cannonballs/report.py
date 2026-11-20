"""Write every number the README, its fragments in docs/ and the note quote to results.json,
and the figure data.

    uv run python -m cannonballs.report     # about half a minute; writes results.json, gaps.csv

`quick=True` reruns the same computations with smaller bounds, for the tests: every list of
answers comes out the same, and only the recorded bounds differ.
"""

import json
from pathlib import Path

from cannonballs import lucas, piles

FOLDER = Path(__file__).resolve().parent
RESULTS = FOLDER / "results.json"
GAPS = FOLDER / "gaps.csv"

# The bounds of the searches, full and quick. Each key names a search in results().
FULL = {
    "brute": 10**7,  # n for the brute-force search of Lucas's problem
    "digits": 10_000,  # 10^digits bounds n in the Pell walk
    "coincide": 10**6,  # n for the other coincidences
    "cubes": 10**6,  # x for the search of square piles that are cubes; n <= x^3
    "tops": 10**5,  # largest top tried for truncated piles
    "oblong_rule": 10**4,  # short sides a checked against the rule by count
    "oblong_residues": 300,  # short sides a for which the rule is checked by residues
    "shapes": 10**5,  # widths a tried for oblong piles of a fixed shape
    "euler": 10**4,  # |u| for the search of u^3 + 1 = 2 v^3
}
QUICK = {
    "brute": 10**5,
    "digits": 300,
    "coincide": 10**4,
    "cubes": 10**4,
    "tops": 10**5,
    "oblong_residues": 120,
    "oblong_rule": 10**4,
    "shapes": 10**4,
    "euler": 10**3,
}
GAP_LAYERS = 60
SHAPES = (
    20  # oblong piles on an a by a + c base are searched for c = 1, ..., SHAPES (curves.LONGEST)
)
# Results that change with the bounds rather than with the answers.
DEPENDS_ON_BOUNDS = {"bounds", "pell_walked"}


def results(quick: bool = False) -> dict:
    b = QUICK if quick else FULL
    big = piles.square_pile(24)
    neighbors = piles.contacts_per_ball(big)
    pell = lucas.surviving_case_solutions(b["digits"])
    tops = lucas.truncated_table(30, b["tops"])
    oblong_rows = {a: lucas.oblong_square(a) for a in range(1, 31)}
    residue_agree = all(
        lucas.oblong_has_square(a) == (lucas.oblong_square(a) is not None)
        for a in range(1, b["oblong_residues"] + 1)
    )
    have = sum(lucas.oblong_has_square(a) for a in range(1, b["oblong_rule"] + 1))
    shapes = {str(c): lucas.oblong_shape_squares(c, b["shapes"]) for c in range(1, SHAPES + 1)}
    return {
        "bounds": b,
        "table": [
            [n, piles.triangular(n), piles.square(n), piles.tetrahedral(n), piles.pyramid(n)]
            for n in range(1, 11)
        ],
        "pile24": {
            "balls": len(big),
            "root": 70,
            "touching_pairs": piles.touching_pairs(big),
            "balls_by_neighbors": dict(sorted(neighbors.items())),
        },
        "tetrahedral48": piles.tetrahedral(48),
        "lucas_brute": lucas.square_piles_that_are_squares(b["brute"]),
        "cases": {"".join(map(str, c)): lucas.first_block(c) for c in lucas.CASES},
        "case_of_solutions": {str(n): "".join(map(str, lucas.case_of(n))) for n in (1, 24)},
        "lucas_pell": {"".join(map(str, c)): ns for c, ns in pell.items()},
        "pell_first": {"".join(map(str, c)): rows for c, rows in lucas.first_candidates().items()},
        "pell_walked": {
            "".join(map(str, c)): k for c, k in lucas.pell_candidates(b["digits"]).items()
        },
        "coincidences": {
            f"{pile} / {test}": ns for (pile, test), ns in lucas.coincidences(b["coincide"]).items()
        },
        "tetrahedral_is_square_pile": lucas.tetrahedral_equals_square_pile(b["coincide"]),
        "square_pile_cubes": lucas.square_piles_that_are_cubes(b["cubes"]),
        "cube_cases_blocked_mod_7": ["".join(map(str, c)) for c in lucas.cube_case_blocked_mod_7()],
        "euler_cube_solutions": [list(p) for p in lucas.euler_cube_solutions(b["euler"])],
        "cube_cases": {"".join(map(str, c)): lucas.first_block(c, k=3) for c in lucas.CASES},
        "truncated": [list(row) for row in tops],
        "truncated_found": [n for n, status, _ in tops if status == "found"],
        "runs_of_25": lucas.runs_with_square_layers(5),
        "oblong_smallest": {str(a): (list(r) if r else None) for a, r in oblong_rows.items()},
        "oblong_rule_agrees_with_residues": residue_agree,
        "oblong_with_square": have,
        "oblong_100": [
            [a, b2] for a in range(1, 11) for b2 in range(a, 200) if piles.oblong(a, b2) == 100
        ],
        "oblong_shapes": shapes,
        "oblong_shapes_with_square": [c for c in range(1, SHAPES + 1) if shapes[str(c)]],
        "oblong_ridge": lucas.oblong_ridge_squares(b["shapes"]),
        "contacts": {
            str(n): [piles.square_pile_contacts(n), piles.tetrahedral_contacts(n)]
            for n in (2, 3, 4, 5, 10, 24)
        },
        "tetra10_balls_by_neighbors": dict(
            sorted(piles.contacts_per_ball(piles.tetrahedral_pile(10)).items())
        ),
    }


def gaps_csv() -> str:
    """For each square pile of 1 to GAP_LAYERS layers: its balls, the side of the nearest
    square, and how many balls it is from that square (0 for a square)."""
    lines = ["layers,balls,root,distance"]
    for n in range(1, GAP_LAYERS + 1):
        m = piles.pyramid(n)
        r = lucas.isqrt(m)
        root = r if m - r * r <= (r + 1) ** 2 - m else r + 1
        assert abs(m - root * root) == lucas.gap_to_square(m)
        lines.append(f"{n},{m},{root},{lucas.gap_to_square(m)}")
    return "\n".join(lines) + "\n"


def dump(data: dict) -> str:
    """One line per result, so that a change shows up as a one-line diff."""
    lines = [f" {json.dumps(k)}: {json.dumps(v)}" for k, v in data.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main() -> None:
    RESULTS.write_text(dump(results()))
    GAPS.write_text(gaps_csv())
    print(f"wrote {RESULTS.name} and {GAPS.name}")


if __name__ == "__main__":
    main()
