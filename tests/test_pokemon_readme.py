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
    # years and an arXiv number of two added citations, and the revision date of Schmitz's post
    assert unexplained("pokemon", NOTES, {"2017", "2022", "2206.07520", "29"}) == []
