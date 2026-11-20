"""The Hanabi README and its notes keep every number the README held before the notes."""

from tests.readme_moves import lost, unexplained

NOTES = [
    "README.md",
    "information.md",
    "open-game.md",
    "h1.md",
    "humans.md",
    "h2.md",
    "h4.md",
    "checks.md",
    "open-questions.md",
    "usage.md",
]


def test_no_number_lost():
    assert lost("hanabi", NOTES) == []


def test_no_number_new():
    assert unexplained("hanabi", NOTES, set()) == []
