"""The committed SVG figures of the README are what `drive_ya_nuts/figures.py` draws, byte for byte."""

import pytest

from drive_ya_nuts import figures


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_committed_figure_is_current(name):
    committed = (figures.OUT / f"{name}.svg").read_bytes()
    assert figures.FIGURES[name]().encode("utf-8") == committed


def test_the_figure_search_finds_the_one_solution():
    # A second search, by sides and turns, with the center unturned: one solution, as solver.py finds.
    (solution,) = figures.solutions()
    assert sorted(order for order, _ in solution) == sorted(n.order for n in figures.nuts)
