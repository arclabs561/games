"""The Yahtzee README and its notes keep every number the README held before the notes."""

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


def test_no_number_new():
    assert unexplained("yahtzee", NOTES, set()) == []
