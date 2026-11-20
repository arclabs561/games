"""Maps: how many bits per dimension keep the guesser's pick."""

import numpy as np

from maps import quantize, study


def test_round_to_cells_by_hand():
    # One column 0, 1, 2, 3 at 1 bit: two cells of width 1.5, midpoints 0.75 and 2.25.
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert quantize.round_to_cells(x, 1).ravel().tolist() == [0.75, 0.75, 2.25, 2.25]


def test_rounding_error_is_at_most_half_a_cell():
    x = np.random.default_rng(0).normal(size=(50, 7))
    for bits in (1, 3, 6):
        step = (x.max(axis=0) - x.min(axis=0)) / 2**bits
        assert (np.abs(quantize.round_to_cells(x, bits) - x) <= step / 2 + 1e-12).all()


def test_rotation_is_orthogonal():
    r = quantize.rotation(16, 3)
    assert np.allclose(r @ r.T, np.eye(16), atol=1e-12)


def test_rabitq_at_eight_bits_is_close():
    x = quantize.unit_words()[:30]
    approx = quantize.rabitq_distances(x, 8, 0)
    assert np.abs(approx - quantize.sq_distances(x, x)).max() < 0.01


def test_exact_distances_change_nothing():
    assert (
        quantize.change_rate(quantize.sq_distances(quantize.unit_words(), quantize.unit_words()))
        < 1e-9
    )
    assert study.stored()["quantize"]["float32"] < 1e-4


def test_some_rates_are_current():
    res = study.stored()["quantize"]
    bits = res["bits"]
    for b in bits:
        rate = quantize.change_rate(quantize.distances("rounding", b))
        assert study.same(round(rate, 4), res["methods"]["rounding"][bits.index(b)]["rate"])
    for method, b in (("rotated rounding", 2), ("RaBitQ", 4)):
        rate = quantize.change_rate(quantize.distances(method, b, 0))
        assert study.same(round(rate, 4), res["methods"][method][bits.index(b)]["rates"][0])


def test_summaries_match_the_stored_seeds():
    res = study.stored()["quantize"]
    for method in ("RaBitQ", "rotated rounding"):
        for row in res["methods"][method]:
            assert study.same(study.rounded(quantize.summary(row["rates"])), row["change"])
