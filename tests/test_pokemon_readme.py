"""The Pokémon README and its notes keep every number the README held before the notes."""

from tests.readme_moves import lost, unexplained

NOTES = [
    "README.md",
    "chart.md",
    "games.md",
    "usage-test.md",
    "ranking.md",
    "team-study.md",
    "data.md",
]


def test_no_number_lost():
    assert lost("pokemon", NOTES) == []


def test_no_number_new():
    assert unexplained("pokemon", NOTES, set()) == []
