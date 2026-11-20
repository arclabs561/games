"""The committed results and figures of mtg_decks are what the scripts write from the data."""

import json
import re

import numpy as np
import pytest

from mtg_decks import cards as cardfacts
from mtg_decks import figures, study

SET = "BLB"
RESULTS = json.loads((study.RESULTS / f"{SET}.json").read_text())
DEV = json.loads((study.RESULTS / f"{SET}-dev.json").read_text())
README = (study.HERE / "README.md").read_text()


@pytest.fixture(scope="module")
def data():
    return study.load(study.DATA / SET)


def roundtrip(x):
    """A value as it reads back from the committed JSON."""
    return json.loads(json.dumps(x, sort_keys=True))


def test_aggregates_are_consistent(data):
    assert np.array_equal(data.s, data.s.T) and np.all(np.diag(data.s) == 0)
    assert data.sizes.sum() == data.held.sum()  # the null needs these margins to agree
    assert np.all(data.s <= np.minimum.outer(data.held, data.held))
    for (g, outcome), v in data.targets.items():
        assert g in cardfacts.PAIRS and outcome in ("won", "lost")
        assert v.sum() == pytest.approx(1.0) and np.all(v >= 0)
    source = json.loads((study.DATA / SET / "source.json").read_text())
    assert source["fit_decks_for_cooccurrence"] == data.n_decks
    assert source["vocabulary"] == len(data.cards)
    assert "CC BY 4.0" in source["license"]


def test_question1_ppmi_and_chance(data):
    q = RESULTS["question1"]
    base, intervals = study.q1_ppmi(data, cardfacts.BLB_TRIBES)
    assert roundtrip(base) == q["ppmi"]
    assert roundtrip(intervals) == q["ppmi_intervals"]
    assert roundtrip(study.chance_tests(data, cardfacts.BLB_TRIBES)) == q["chance"]


def test_question1_first_seeds(data):
    """Seed 1 of skip-gram and of the null, recomputed (about a minute each)."""
    q = RESULTS["question1"]
    assert roundtrip(study.q1_sgns(data, cardfacts.BLB_TRIBES, 1)) == q["sgns_runs"][0]
    assert roundtrip(study.q1_null(data, cardfacts.BLB_TRIBES, 1)) == q["null_runs"][0]


def test_questions_2_and_3(data):
    x = study.ppmi_embedding(data.s, data.held, data.n_decks)
    size = RESULTS["mean_vocabulary_copies_per_deck"]
    assert roundtrip(study.question2(data, x, size)) == RESULTS["question2"]
    assert roundtrip(study.question3(data, x)) == RESULTS["question3"]


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_figures_are_current(name, tmp_path):
    with figures.plt.style.context("default"):
        figures.save_svg(figures.FIGURES[name](), tmp_path / name)
    assert (tmp_path / name).read_bytes() == (study.HERE / "figures" / name).read_bytes()


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def test_readme_quotes_the_results():
    """Every number the opening and the results table quote comes from the result files."""
    q1 = RESULTS["question1"]
    quoted = [
        pct(q1["ppmi"]["pair"]),
        pct(q1["chance"]["pair"]),
        pct(q1["chance"]["pair_within_color"]),
        pct(q1["null"]["pair"]["mean"]),
        pct(q1["ppmi"]["tribe_partner_offcolor"]),
        pct(q1["ppmi"]["tribe_partner"]),
        f"{DEV['auc']['frequency_difference|pair_and_skill']['auc']:.3f}",
        f"{DEV['auc']['transport|pair_and_skill']['auc']:.3f}",
        f"{DEV['auc']['skill|pair']['auc']:.3f}",
        f"{DEV['auc_differences']['transport - frequency_difference|pair_and_skill']['diff']:.4f}".replace(
            "-", "−"
        ),
        pct(DEV["pairing"]["transport"]["same_kind"]["mean"]),
        pct(DEV["pairing"]["baseline"]["same_kind"]["mean"]),
        pct(DEV["pairing"]["random"]["same_kind"]["mean"]),
    ]
    lists = RESULTS["question3"]["mean"]
    at = {r["lambda"]: r for r in lists}
    quoted += [
        pct(at[0.7]["relevance_kept"]),
        f"{at[1.0]['redundancy']:.2f}",
        f"{at[0.7]['redundancy']:.2f}",
    ]
    text = re.sub(r"\s+", " ", README)
    missing = [v for v in quoted if v not in text]
    assert not missing, missing
