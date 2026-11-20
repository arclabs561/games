"""The committed SVG figures of the README are what `codenames/figures.py` draws, byte for byte."""

import pytest

from codenames import figures


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_committed_figure_is_current(name):
    committed = (figures.OUT / f"{name}.svg").read_bytes()
    assert figures.FIGURES[name]().encode("utf-8") == committed
