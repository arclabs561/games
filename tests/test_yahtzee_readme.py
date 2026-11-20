"""The Yahtzee README and its notes keep every number the README held before the notes."""

from itertools import product

from tests.readme_moves import lost, unexplained

NOTES = [
    "README.md",
    "optimal.md",
    "variants.md",
    "openings.md",
    "card-search.md",
    "advice.md",
    "method.md",
]


def test_no_number_lost():
    assert lost("yahtzee", NOTES) == []


NEW = {
    "7,056",  # ordered rolls with a pair or more (counted below)
    "1,304",  # impossible upper pairs without the cap at 63 (counted below)
    "217.31",  # greedy play's range over tie rules, rounded from 217.3074 and 219.05
    "219",
    "35.54",  # 254.5896 - 219.05
    "139",  # digits of the numerator on Verhoeff's page
    "2017",  # the page's date for that fraction
    "205",  # the paper's first page
}


def test_no_number_new():
    assert unexplained("yahtzee", NOTES, NEW) == []


def test_rolls_with_a_pair():
    rolls = list(product(range(1, 7), repeat=5))
    assert sum(len(set(r)) < 5 for r in rolls) == 7056


def test_impossible_upper_pairs_with_and_without_the_cap():
    capped = uncapped = 0
    for mask in range(64):
        sums = {0}
        for face in range(1, 7):
            if mask >> (face - 1) & 1:
                sums = {s + k * face for s in sums for k in range(6)}
        capped += sum(t not in {min(s, 63) for s in sums} for t in range(64))
        uncapped += sum(t not in sums for t in range(64))
    assert (capped, uncapped) == (1302, 1304)
