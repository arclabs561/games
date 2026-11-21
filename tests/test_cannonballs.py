"""Cannonballs: counts against brute force, searches against each other, and quoted numbers."""

import json
import re
from math import comb
from pathlib import Path

import pytest

from cannonballs import curves, figures, lucas, piles, report

FOLDER = Path(piles.__file__).parent
COMMITTED = json.loads(report.RESULTS.read_text())


# ---------- counting balls ----------


@pytest.mark.parametrize("n", range(1, 13))
def test_formulas_count_the_balls_that_are_built(n):
    assert len(piles.triangle_layer(n)) == piles.triangular(n)
    assert len(piles.tetrahedral_pile(n)) == piles.tetrahedral(n) == comb(n + 2, 3)
    assert len(piles.square_pile(n)) == piles.pyramid(n) == sum(k * k for k in range(1, n + 1))
    for b in range(n, n + 5):
        assert len(piles.oblong_pile(n, b)) == piles.oblong(n, b)
        assert piles.oblong(n, b) == sum(p * q for p, q in piles.layer_sizes_oblong(n, b))


def test_hand_examples():
    # 1 + 4 + 9 + 16 = 30; the triangle of side 4 is a bowling rack of 10 pins
    assert piles.pyramid(4) == 30 and piles.triangular(4) == 10 == piles.tetrahedral(3)
    assert piles.pyramid(24) == 4900 == 70**2
    assert piles.oblong(4, 11) == piles.oblong(5, 8) == 100


@pytest.mark.parametrize("n", range(1, 40))
def test_identities_in_the_note(n):
    t, te, p = piles.triangular, piles.tetrahedral, piles.pyramid
    assert 2 * t(n) == n * (n + 1)  # two triangles make an n by n + 1 rectangle
    assert t(n) + t(n - 1) == n * n  # a square is two triangles
    assert p(n) == te(n) + te(n - 1)  # a square pile is two tetrahedral piles
    assert te(2 * n) == 4 * p(n)  # so 48 layers hold four times 4,900
    assert 3 * p(n) == (2 * n + 1) * t(n)  # three turned triangles of numbers
    assert piles.oblong(n, n) == p(n)
    assert piles.oblong(n, n + 1) - piles.oblong(n, n) == t(n)  # one ball longer adds a triangle
    rows = piles.three_triangles(n)
    assert all(v == 2 * n + 1 for row in rows for v in row) and sum(map(len, rows)) == t(n)


def test_piles_are_stable_and_do_not_overlap():
    for n in range(1, 7):
        sq, te = piles.square_pile(n), piles.tetrahedral_pile(n)
        assert piles.no_overlaps(sq) and piles.no_overlaps(te)
        assert piles.resting(sq, (0, 0, 1)) and piles.resting(te, (-1, -1, -1))
        # every center is a point of the face-centered cubic lattice
        assert all(sum(b) % 2 == 0 for b in sq + te)


def test_twelve_neighbors():
    assert len(set(piles.NEIGHBORS)) == 12
    assert all(sum(c * c for c in v) == 2 for v in piles.NEIGHBORS)


@pytest.mark.parametrize("n", range(1, 9))
def test_touching_pairs(n):
    sq, te = piles.square_pile(n), piles.tetrahedral_pile(n)
    assert (
        piles.touching_pairs(sq) == piles.touching_pairs_slow(sq) == piles.square_pile_contacts(n)
    )
    assert (
        piles.touching_pairs(te) == piles.touching_pairs_slow(te) == piles.tetrahedral_contacts(n)
    )
    for b in range(n, n + 4):
        o = piles.oblong_pile(n, b)
        assert piles.touching_pairs(o) == lucas.oblong_contacts(n, b)


@pytest.mark.parametrize("n", range(4, 15))
def test_balls_that_touch_twelve_form_a_smaller_pile(n):
    assert piles.contacts_per_ball(piles.square_pile(n))[12] == piles.pyramid(n - 3)
    if n >= 5:
        assert piles.contacts_per_ball(piles.tetrahedral_pile(n))[12] == piles.tetrahedral(n - 4)


def test_touching_pairs_square_only_when_proved():
    # 2 n^2 (n - 1) is a square exactly when 2(n - 1) is; (n - 1) n (n + 1) never is for n >= 2
    for n in range(2, 3000):
        assert lucas.is_square(piles.square_pile_contacts(n)) == lucas.is_square(2 * (n - 1))
        assert not lucas.is_square(piles.tetrahedral_contacts(n))


# ---------- Lucas's problem ----------


def test_brute_force_and_pell_walk_agree():
    brute = lucas.square_piles_that_are_squares(10**5)
    assert brute == [1, 24]
    found = lucas.surviving_case_solutions(12)
    assert sorted(n for ns in found.values() for n in ns) == [1, 24]
    assert lucas.case_of(1) == (1, 2, 3) and lucas.case_of(24) == (6, 1, 1)


def test_pell_walk_lists_every_candidate():
    # Every n up to 10^6 with n = 6x^2 and n + 1 square is in the walk, and every n = x^2 with
    # n + 1 twice a square: brute force over x against the walk.
    walk_611 = {
        6 * x * x
        for _, x in zip(range(12), (x for _, x in lucas.pell(6, 5, 2, (5, 2))), strict=False)
    }
    walk_123 = {
        x * x for _, x in zip(range(12), (x for x, _ in lucas.pell(2, 1, 1, (3, 2))), strict=False)
    }
    for x in range(1, 2000):
        n = 6 * x * x
        assert lucas.is_square(n + 1) == (n in walk_611)
        n = x * x
        assert ((n + 1) % 2 == 0 and lucas.is_square((n + 1) // 2)) == (n in walk_123)


def test_cases_ruled_out_by_remainders():
    blocked = {c: lucas.first_block(c) for c in lucas.CASES}
    assert blocked == {
        (1, 2, 3): None,
        (2, 1, 3): 3,
        (1, 6, 1): 3,
        (6, 1, 1): None,
        (2, 3, 1): 3,
        (3, 2, 1): 3,
    }
    # brute force agrees: no n up to 10^5 falls in a blocked case with all three parts
    for n in range(1, 10**5):
        for s1, s2, s3 in lucas.CASES:
            if blocked[(s1, s2, s3)] and n % s1 == 0 and lucas.is_square(n // s1):
                assert not (
                    (n + 1) % s2 == 0
                    and lucas.is_square((n + 1) // s2)
                    and (2 * n + 1) % s3 == 0
                    and lucas.is_square((2 * n + 1) // s3)
                )


def test_other_coincidences_match_published_lists():
    c = lucas.coincidences(2000)
    assert c[("tetrahedral", "square")] == [1, 2, 48]  # Meyl (1878)
    assert c[("tetrahedral", "triangle")] == [1, 3, 8, 20, 34]  # Avanesov (1967)
    assert c[("square pile", "square")] == [1, 24]  # Watson (1918)
    assert c[("square pile", "triangle")] == [1, 5, 6, 85]  # Finkelstein and London (1972)
    assert c[("tetrahedral", "cube")] == c[("square pile", "cube")] == [1]
    assert lucas.tetrahedral_equals_square_pile(2000) == [(1, 1)]  # Beukers and Top (1988)


def test_cube_search_against_brute_force():
    assert lucas.square_piles_that_are_cubes(40) == [
        n for n in range(1, 40**3 + 1) if lucas.is_cube(piles.pyramid(n))
    ]
    for m in range(0, 5000):
        assert lucas.icbrt(m) ** 3 <= m < (lucas.icbrt(m) + 1) ** 3


def test_cube_proof_steps():
    # the two cases the proof rules out on division by 7, and no other case is ruled out there
    assert lucas.cube_case_blocked_mod_7() == [(2, 3, 1), (3, 2, 1)]
    cubes = lucas.powers(7, 3)
    assert cubes == {0, 1, 6}
    for s1, s2, _ in [(2, 3, 1), (3, 2, 1)]:
        left = {s1 * c % 7 for c in cubes}  # n = s1 x^3
        right = {(s2 * c - 1) % 7 for c in cubes}  # n = s2 y^3 - 1
        assert all((2 * n + 1) % 7 not in cubes for n in left & right)
    # the case of Euler's theorem the proof uses, against brute force over small v as well
    assert lucas.euler_cube_solutions(2000) == [(-1, 0), (1, 1)]
    assert [(u, v) for u in range(-60, 61) for v in range(-60, 61) if u**3 + 1 == 2 * v**3] == [
        (-1, 0),
        (1, 1),
    ]
    # the four cases that reduce to it give n in {-2, -1, 0, 1}; only n = 1 is a pile
    assert lucas.square_piles_that_are_cubes(200) == [1]


def test_oblong_shapes_against_brute_force_and_puzzling_se():
    for c in range(1, 8):
        assert lucas.oblong_shape_squares(c, 600) == [
            a for a in range(2, 601) if lucas.is_square(piles.oblong(a, a + c))
        ]
    # Puzzling Stack Exchange question 99084: an a by 2a - 1 base gives squares at 1, 6, 49
    assert lucas.oblong_ridge_squares(3000) == [1, 6, 49]
    assert piles.oblong(6, 11) == 14**2 and piles.oblong(49, 97) == 315**2
    assert all(6 * piles.oblong(a, 2 * a - 1) == a * (a + 1) * (5 * a - 2) for a in range(1, 50))


def test_curves_file_agrees_with_search():
    """curves.json comes from Sage, which CI does not install: check what can be checked
    without it. Every point lies on its curve, the widths follow from the points, and they
    agree with the plain search below its bound."""
    data = json.loads(curves.CURVES.read_text())
    assert curves.LONGEST == report.SHAPES
    shapes = {str(c): curves.shape_curve(c) for c in range(curves.LONGEST + 1)}
    for key, row in [*data["by_longer_side"].items(), ("ridge", data["ridge"])]:
        coefficients, s, t = curves.ridge_curve() if key == "ridge" else shapes[key]
        assert row["coefficients"] == coefficients and row["scale"] == [s, t]
        _, a2, _, a4, a6 = coefficients
        for x, y in row["integral_points"]:
            assert y * y == x**3 + a2 * x * x + a4 * x + a6
        assert row["layers"] == sorted(
            x // s for x, y in row["integral_points"] if x >= s and x % s == 0 and y % t == 0
        )
        assert len(row["generators"]) == row["rank"]
        widths = [a for a in row["layers"] if a >= 2]
        if key == "ridge":
            assert row["layers"] == COMMITTED["oblong_ridge"] == lucas.oblong_ridge_squares(1000)
        elif key == "0":
            assert row["layers"] == COMMITTED["lucas_brute"] == [1, 24]
        else:
            assert widths == COMMITTED["oblong_shapes"][key]
            assert widths == lucas.oblong_shape_squares(int(key), 1000)
    assert sorted(data["by_longer_side"], key=int) == [str(c) for c in range(curves.LONGEST + 1)]


def test_counts_match_oeis():
    # first terms copied from the OEIS: A035006 (rook moves on an n by n board), A007531
    # (n (n - 1) (n - 2)), A053611 and A224421 (complete lists)
    a035006 = [0, 8, 36, 96, 200, 360, 588, 896, 1296, 1800, 2420, 3168, 4056, 5096, 6300]
    assert [piles.square_pile_contacts(n) for n in range(1, 16)] == a035006
    a007531 = [0, 0, 0, 6, 24, 60, 120, 210, 336, 504, 720, 990, 1320, 1716, 2184]
    assert [piles.tetrahedral_contacts(n - 1) for n in range(1, 15)] == a007531[1:15]
    assert COMMITTED["coincidences"]["square pile / triangle"] == [1, 5, 6, 85]  # A053611
    assert COMMITTED["coincidences"]["tetrahedral / triangle"] == [1, 3, 8, 20, 34]  # A224421


def test_lucas_cli_help():
    with pytest.raises(SystemExit) as stop:
        lucas.main(["--help"])
    assert stop.value.code == 0


def test_truncated_piles_match_oeis_a001032():
    rows = lucas.truncated_table(30, 10**4)
    found = [layers for layers, how, _ in rows if how == "found"]
    assert found == [1, 2, 11, 23, 24, 26]  # OEIS A001032, terms up to 30
    assert lucas.runs_with_square_layers(5) == [0]  # 0^2 + ... + 24^2 is the only run of 25
    assert lucas.run_of_squares(18, 11) == 77**2 and lucas.run_of_squares(3, 2) == 25
    for layers, how, m in rows:
        if how == "blocked":
            assert m is not None and lucas.run_blocked_mod(layers, m)


def test_short_search_distinguishes_unresolved_from_blocked():
    assert lucas.truncated_table(3, 1) == [
        (1, "found", 1),
        (2, "unresolved", None),  # the witness is 3² + 4² = 5²
        (3, "blocked", 3),
    ]
    assert lucas.truncated_table(2, 3)[1] == (2, "found", 3)


def test_oblong_rule_against_residues_and_brute_force():
    for a in range(1, 151):
        r = lucas.oblong_square(a)
        assert lucas.oblong_has_square(a) == (r is not None), a
        if r is not None and a <= 40:
            assert lucas.oblong_square_brute(a, r[0]) == r[0]
        if r is None and a <= 40:
            assert lucas.oblong_square_brute(a, a + 3000) is None
    # the share of short sides that work tends to 7/12
    share = sum(lucas.oblong_has_square(a) for a in range(1, 12 * 10**4 + 1))
    assert abs(share - 7 * 10**4) <= 10


# ---------- committed files ----------


def test_results_file_is_current():
    quick = json.loads(report.dump(report.results(quick=True)))
    for key, value in quick.items():
        if key not in report.DEPENDS_ON_BOUNDS:
            assert COMMITTED[key] == value, key
    assert COMMITTED["bounds"] == report.FULL


def test_gaps_csv_is_current():
    assert report.GAPS.read_text() == report.gaps_csv()


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_figures_are_current(name):
    assert (FOLDER / name).read_text() == figures.FIGURES[name]()


def readme() -> str:
    return " ".join((FOLDER / "README.md").read_text().split())


def note() -> str:
    return " ".join((FOLDER / "cannonballs.typ").read_text().split())


def test_readme_quotes_the_results():
    """The README's numbers come from results.json through tools/sync_docs.py
    (tests/test_shared_docs.py checks that); these are the ones it must still quote."""
    r, c = readme(), COMMITTED
    assert f"{c['pile24']['balls']:,} balls" in r and f"{c['pile24']['touching_pairs']:,}" in r
    assert f"10<sup>{report.FULL['digits']:,}</sup>" in r
    assert (
        f"up to {report.FULL['brute']:,}" in r and f"up to {report.FULL['coincide']:,} layers" in r
    )
    assert f"({c['pile24']['balls_by_neighbors']['12']:,} of the 4,900)" in r
    assert "7/12 of all short sides" in r and "10<sup>18</sup>" in r
    assert report.FULL["cubes"] ** 3 == 10**18
    # the near misses the README names are the rows of gaps.csv one ball from a square
    rows = [line.split(",") for line in report.gaps_csv().split()[1:]]
    near = [(int(n), int(m), int(root)) for n, m, root, d in rows if d == "1"]
    assert [n for n, _, _ in near] == [2, 47, 48]
    for n, m, root in near:
        assert f"{n} layers ({m:,}" in r and f"{root}²)" in r
    assert "1, 2, 11, 23, 24 and 26 layers" in r


def test_note_quotes_the_results():
    n, c = note(), COMMITTED
    # numbers the note writes out; the rest it reads from results.json
    # each phrase is counted, so that a change to one copy of a repeated phrase fails
    assert n.count("holds 4,900 balls") == 2
    assert n.count('$"19,600" = 4 times "4,900" = 140^2$') == 1
    assert c["coincidences"]["square pile / triangle"] == [1, 5, 6, 85]
    assert n.count("[1; 5; 6; 85][1; 55; 91; 208,335]") == 1
    assert c["coincidences"]["tetrahedral / triangle"] == [1, 3, 8, 20, 34]
    assert n.count("[1; 3; 8; 20; 34][1; 10; 120; 1,540; 7,140]") == 1
    assert c["tetrahedral48"] == 19600 and c["oblong_ridge"] == [1, 6, 49]
    assert n.count("$196 = 14^2$") == 1 and n.count('$"99,225" = 315^2$') == 1
    assert f'$10^"{report.FULL["digits"]:,}"$' in n and "7 in 12" in n and "$10^18$" in n
    assert c["oblong_100"] == [[1, 100], [4, 11], [5, 8]] and "five rows of 8" in n
    assert c["contacts"]["3"][0] == 36 and "= 36$ touching pairs" in n
    assert c["runs_of_25"] == [0]
    for key in ("pell_walked", "pile24.touching_pairs", "bounds.brute", "oblong_with_square"):
        assert "R." + key.split(".")[0] in n, key


def test_root_readme_row():
    root = (FOLDER.parent / "README.md").read_text()
    assert "[`cannonballs/`](cannonballs)" in root
    assert re.search(r"\*\*Cannonballs\*\*", root)
