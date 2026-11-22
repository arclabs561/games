"""The figure comparison that tests/nbval_figures.py adds to nbval."""

import io

import matplotlib
import numpy as np
import pytest
from PIL import Image

from tests.nbval_figures import TOLERANCE, figure_mismatches, rms_difference

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

RNG = np.random.default_rng(0)
X, Y = RNG.random(200), RNG.random(200)


def scatter(order=range(200), size=(7, 4.5)) -> bytes:
    """A PNG like the notebooks' scatter plots: faint points, with a few bold ones on top."""
    fig, ax = plt.subplots(figsize=size)
    order = list(order)
    ax.scatter(X[order], Y[order], s=10, alpha=0.3, color="#0072B2", linewidths=0)
    ax.scatter(X[:20], Y[:20], s=30, color="#D55E00")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


@pytest.fixture(scope="module")
def base():
    return scatter()


def test_same_bytes_match(base, tmp_path):
    assert scatter() == base
    assert figure_mismatches([base], [scatter()], TOLERANCE, tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_drawing_order_is_within_tolerance(base, tmp_path):
    shuffled = scatter(RNG.permutation(200))
    assert shuffled != base
    assert rms_difference(base, shuffled) < 0.01 < TOLERANCE  # the README says about 0.004
    assert figure_mismatches([base], [shuffled], TOLERANCE, tmp_path) == []


def test_one_missing_point_fails(base, tmp_path):
    assert rms_difference(base, scatter(range(199))) > 0.2 > TOLERANCE  # the README says about 0.3
    problems = figure_mismatches([base], [scatter(range(199))], TOLERANCE, tmp_path)
    assert len(problems) == 1 and problems[0].startswith("figure 1:")
    assert (tmp_path / "figure1-failed-diff.png").exists()


def test_disappearing_figure_fails(base, tmp_path):
    with Image.open(io.BytesIO(base)) as image:
        image.putalpha(0)
        transparent = io.BytesIO()
        image.save(transparent, format="PNG")
    assert figure_mismatches([base], [transparent.getvalue()], TOLERANCE, tmp_path)


def test_other_size_fails(base, tmp_path):
    assert len(figure_mismatches([base], [scatter(size=(7, 4))], TOLERANCE, tmp_path)) == 1


def test_figure_count_must_match(base, tmp_path):
    assert figure_mismatches([base], [], TOLERANCE, tmp_path) == [
        "0 figures, but the notebook has 1"
    ]
