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


# Numbers added with plan H5's README section. Each is checked against results/h5.json
# below, except the plan's settings and the cost range, which h5-runs.json holds.
H5_SETTINGS = {"16,384", "3.45", "1.85", "1.90"}
H5_NUMBERS = {
    "1,941", "10.7", "11.47", "12.6", "13.47", "14.3", "16.23", "2.75", "2.83", "22.17", "9.67",
}  # fmt: skip


def test_no_number_new():
    assert unexplained("hanabi", NOTES, H5_NUMBERS | H5_SETTINGS) == []


def test_h5_numbers_come_from_h5_json():
    import json
    from pathlib import Path

    d = json.loads((Path(__file__).parent.parent / "hanabi/results/h5.json").read_text())
    s = d["secondary"]
    values = [
        d["primary"]["gap_to_information_strategy"],
        *d["primary"]["interval"],
        d["primary"]["detectable_effect"],
        s["gap_to_ceiling"]["mean"],
        *s["gap_to_ceiling"]["interval"],
        s["score"],
        s["information_strategy"],
        s["ceiling"] - s["information_strategy"],
        sum(x["calls"] for x in d["deals"]),
    ]
    for text in H5_NUMBERS:
        x = float(text.replace(",", ""))
        places = len(text.partition(".")[2])
        assert any(round(v, places) == x for v in values), text
