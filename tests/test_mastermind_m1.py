"""Plan M1: screens of common words. The slices themselves run on GitHub Actions."""

import json
import random

import pytest

from mastermind import fallout3 as f3
from mastermind import m1

# Plan M1's table: ENABLE words with Zipf frequency >= 3.0 in wordfreq 3.1.1, 4 to 15 letters.
PLAN_SIZES = [1842, 2686, 3414, 3649, 3325, 2698, 1952, 1161, 665, 359, 140, 48]


def test_common_list_is_a_subset_of_enable_with_the_plans_sizes():
    common, enable = f3.common_by_length(3.0), f3.words_by_length()
    assert [len(common[n]) for n in range(4, 16)] == PLAN_SIZES
    for n, words in common.items():
        assert set(words) <= set(enable[n])


def test_rows_draw_the_same_screens_as_before():
    """row_screens with no pool is the study's old loop: one generator per row."""
    rng = random.Random(f"{f3.SEED} Very Easy 20")
    old = [f3.sample_screen(rng, rng.choice(f3.LEVELS["Very Easy"]), 20) for _ in range(5)]
    assert list(f3.row_screens("Very Easy", 20, 5, f3.SEED)) == old


def committed_slices():
    if not m1.SLICES_DIR.exists():
        pytest.skip("slices not committed yet")
    return m1.load_slices()


@pytest.mark.parametrize("level,n", [("Very Easy", 5), ("Very Hard", 13)])
def test_rows_recompute_the_committed_slice(level, n):
    slices = committed_slices()
    for k, (words, zipf, seed, screens, rows) in m1.SLICES.items():
        if (level, n) not in rows or screens > 1000 and n > 5:
            continue
        got = m1.score_row(level, n, screens, seed, m1.pool_of(words, zipf))
        want = next(r for r in slices[k]["rows"] if (r["level"], r["words"]) == (level, n))
        assert json.loads(json.dumps(got)) == want, k


def test_seed_3_slice_reproduces_the_readme_table():
    slices = committed_slices()
    summ = {m1.row_key(r["level"], r["words"]): m1.row_summary(r) for r in slices[4]["rows"]}
    assert m1.readme_check(summ)["mismatches"] == []


def test_bootstrap_of_a_histogram():
    """The multinomial draw resamples screens: equal lists give an interval around 0, and
    the mean difference is the difference of means."""
    a = {"18": 30, "19": 50, "20": 20}
    d = m1.difference(a, a, 20)
    assert d["points"] == 0 and d["low"] < 0 < d["high"]
    b = {"20": 100}
    d = m1.difference(b, a, 20)
    assert d["points"] == pytest.approx(100 * (1 - (18 * 30 + 19 * 50 + 20 * 20) / 2000))
    assert d["low"] > 0


def test_committed_results_are_what_score_writes():
    committed_slices()
    want = json.loads(m1.OUT.read_text())
    assert json.loads(json.dumps(m1.rounded(m1.score()), sort_keys=True)) == want
