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


# Numbers of the old README corrected after review: the share of SAD's and SPARTA's gaps
# (1.7% and 4.1%, from the plan's bound, are 1.8% and 4.2% with the settled one), the
# three-player bound 24.998 (written 24.9990 everywhere), and the Forkel et al. v5 date
# (2026-05-05, which is 2026-06-04 on arXiv).
CORRECTED = {"1.7", "4.1", "24.998", "05"}

NEW = {
    "0.01625",  # points lost per two-player deal, the sum behind the bound (h1.md)
    "24.986",  # the bound's upper 95% limit, 24.9838 + 0.0021
    "3.1",  # 0.01625 / 0.52, the IPPO row of h1.md's table, recomputed
    "4.2",  # 0.01625 / 0.39, SPARTA's share
    "2016",  # the year of WTFWThat (hanabi.rs)
    "75",  # Cox et al.'s "more than 75 percent"
    "04",  # 2026-06-04, the date of Forkel et al.'s v5
    "06",
}


def test_no_number_lost():
    assert lost("hanabi", NOTES) == sorted(CORRECTED)


def test_no_number_new():
    assert unexplained("hanabi", NOTES, NEW) == []
