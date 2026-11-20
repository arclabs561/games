"""Plan C13: the regenerated picks and the stratified bootstrap (no model calls)."""

import numpy as np
import pytest

from codenames import c13


def vectors_present() -> bool:
    from codenames import study

    return len(study.models()) == 8


@pytest.mark.skipif(not vectors_present(), reason="needs the vectors of outputs/codenames-embed")
def test_picks_equal_c3s_on_its_first_three_test_boards():
    assert c13.check_c3(3)["passed"]


def fake_values(lead: float, n: int = 40, seed: int = 0) -> dict:
    """Played-out values where population leads self-play by `lead` on every board."""
    rng = np.random.default_rng(seed)
    v = {}
    for a in c13.all_arms():
        base = rng.normal(0, 1, n)
        for m in c13.GUESSERS:
            v[(a, m)] = base.copy()
    for g in c13.TASKS:
        s = c13.short(g)
        for k in c13.SEEDS:
            for m in c13.GUESSERS:
                v[(f"{s}|population|{k}", m)] = v[(f"{s}|self|{k}", m)] + lead
    return v


def test_stratified_bootstrap_pairs_population_and_self_by_seed():
    v = fake_values(0.25)
    reps = c13.stratified(v, np.arange(40), np.random.default_rng(0), resamples=200)
    # Paired by seed and board, the difference is the same on every unit: no spread.
    assert np.allclose(reps["h_self"], 0.25)
    d = c13.point(v, np.ones(40, bool))
    assert np.allclose(d["h_self"], 0.25)


def test_arm_names():
    names = c13.arm_names("potion")
    assert names[:2] == ["potion|start", "potion|count1"] and len(names) == 12
