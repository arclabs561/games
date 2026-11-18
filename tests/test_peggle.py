"""Peggle: the chance of being the better player, by coin flips and by the Beta integral."""

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


def test_posterior_slope_beats_the_fitted_one_and_matches_the_note():
    note = " ".join((Path(peggle.__file__).parent / "peggle.typ").read_text().split())
    posterior, fitted = (
        [peggle.largest_error(slope, lowest)[0] for lowest in (1, 30)]
        for slope in (peggle.posterior_slope, peggle.fitted_slope)
    )
    assert f"is {posterior[0]:.3f} with $a(n)$ and {fitted[0]:.3f} with the fitted slope" in note
    assert f"it is {posterior[1]:.3f} and {fitted[1]:.3f}" in note
    assert posterior[0] < fitted[0] / 4
