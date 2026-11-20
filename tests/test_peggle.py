"""Peggle: the chance of being the better player, by coin flips and by the Beta integral."""

import json
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

from peggle import peggle


def test_coin_flips_equal_the_beta_integral():
    for n in range(1, 16):
        for k in range(n + 1):
            assert peggle.better(k, n) == peggle.better_by_integral(k, n)


def test_worked_example_and_symmetry():
    assert peggle.better(6, 10) == Fraction(743, 1024)
    assert peggle.better(5, 10) + peggle.better(5, 10) == 1


def test_wins_needed_table():
    assert [peggle.wins_needed(n) for n in (10, 20, 50, 100, 200, 1000)] == [
        8,
        14,
        31,
        59,
        112,
        527,
    ]


def test_committed_csv_is_current(tmp_path, monkeypatch):
    monkeypatch.setattr(peggle, "CSV", tmp_path / "better.csv")
    peggle.main()
    committed = (Path(peggle.__file__).parent / "better.csv").read_text()
    assert (tmp_path / "better.csv").read_text() == committed


def test_posterior_slope_beats_the_fitted_one():
    posterior, fitted = (
        [peggle.largest_error(slope, lowest)[0] for lowest in (1, 30)]
        for slope in (peggle.posterior_slope, peggle.fitted_slope)
    )
    assert posterior[0] < fitted[0] / 4
    assert COMMITTED["coins"]["logistic_error"] == {"posterior": posterior, "fitted": fitted}


# ---------- results.json, which the README and the note read ----------

FOLDER = Path(peggle.__file__).parent
COMMITTED = json.loads((FOLDER / "results.json").read_text())


def test_results_file_is_current():
    from peggle import report

    quick = json.loads(report.dump(report.results(quick=True)))
    assert set(COMMITTED) == set(quick) | report.SLOW
    for key, value in quick.items():
        assert COMMITTED[key] == value, key


def test_simulation_in_the_results_file_is_current():
    from peggle import report

    assert COMMITTED["simulation"] == report.simulation()
    assert COMMITTED["simulation"]["inside"] == COMMITTED["simulation"]["of"] == 8


def floats(value, path=""):
    if isinstance(value, float):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from floats(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from floats(v, f"{path}.{i}")


# 0.57475, best of 3 at p = 0.55, is an exact tie at four decimals; it is printed with three.
TIES = {(".formats.best of 3.0", 4)}


def test_no_quoted_float_sits_on_a_rounding_tie():
    """Python rounds a float's binary value; Typst rounds its shortest decimal form, half away
    from zero (2.675 gives 2.67 in Python and 2.68 in Typst). They agree unless the shortest
    form ends in a 5 just past the digits printed."""
    for path, x in floats(COMMITTED):
        for digits in range(5):
            typst = Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
            if (path, digits) not in TIES:
                assert f"{x:.{digits}f}" == f"{typst:.{digits}f}", (path, digits)


def test_note_reads_its_numbers_from_results():
    note = (FOLDER / "peggle.typ").read_text()
    assert '#let R = json("results.json")' in note
    for name in ("rules", "results", "checks", "open"):
        assert f'shared(readme, "{name}"' in note
    # numbers once typed by hand, now read from results.json
    for stale in (
        "[0.528], [], [0.472]",
        "[34.9], [0.550]",
        "[no edge], [0.5], [0.0886]",
        "is 0.016 with",
    ):
        assert stale not in note, stale
    # every Try-it puzzle has a rating, and its answer a rating and a source
    assert note.count("#tryit(") == 5 and "#tryit[" not in note
    answers = note[note.index("= Answers to the puzzles") :]
    assert answers.count(", rated ") == 5 and answers.count("_Source:_") == 5
