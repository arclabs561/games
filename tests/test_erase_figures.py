"""The committed SVG figure of the README is what `erase/figures.py` draws, byte for byte."""

import figures
import pytest


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_committed_figure_is_current(name):
    committed = (figures.OUT / f"{name}.svg").read_bytes()
    assert figures.FIGURES[name]().encode("utf-8") == committed
