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


# Numbers added with plan P2's section of team-study.md and the README's row for it. Each
# is checked against results/p2.json below, except the plan's own settings.
P2_SETTINGS = {"−2", "200", "300", "2,000", "2,520", "78,120", "158,760", "0.10"}
P2_NUMBERS = {
    "0.036", "0.101", "0.111", "0.18", "0.199", "0.20", "0.23", "0.232", "0.349", "0.35",
    "0.37", "0.38", "0.43", "0.431", "0.44", "0.45", "0.47", "0.62", "0.76", "0.95", "1,171",
    "1,349", "12.8", "15.5", "15.9", "18.5", "217,589", "229,818", "23.5", "25.6", "29.2",
    "30.0", "35.1", "51.6", "53.5", "55.5", "8,387",
}  # fmt: skip


def test_no_number_new():
    assert unexplained("pokemon", NOTES, P2_NUMBERS | P2_SETTINGS) == []


def test_p2_numbers_come_from_p2_json():
    import json
    from pathlib import Path

    def leaves(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from leaves(v)
        elif isinstance(x, list):
            for v in x:
                yield from leaves(v)
        elif isinstance(x, int | float) and not isinstance(x, bool):
            yield float(x)

    d = json.loads((Path(__file__).parent.parent / "pokemon/results/p2.json").read_text())
    values = list(leaves(d))
    for text in P2_NUMBERS:
        x = float(text.replace(",", ""))
        places = len(text.partition(".")[2])
        ok = any(
            round(v * k, places) == x or round(v * k + 0.0, places) == x
            for v in values
            for k in (1, 100)
        ) or any(abs(v * k - x) <= 0.5 * 10**-places for v in values for k in (1, 100))
        assert ok, text
