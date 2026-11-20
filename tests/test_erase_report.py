"""erase/results.json is what erase/report.py writes. The cheap sections are recomputed in
full; the others (over an hour in all) are recomputed in part or checked for consistency."""

import json
from math import comb
from pathlib import Path

import report
import simulate
import targets

COMMITTED = json.loads(report.RESULTS.read_text())


def test_committed_file_is_in_the_script_format():
    assert report.dump(COMMITTED) == report.RESULTS.read_text()
    assert list(COMMITTED) == report.ORDER


def test_cheap_sections_are_current():
    for name in ("no_zero", "wild", "certificates", "twenty_four"):
        assert json.loads(json.dumps(report.SECTIONS[name](COMMITTED))) == COMMITTED[name], name


def test_deals_agree_with_the_formulas():
    d = COMMITTED["deals"]
    assert d["total"] == comb(44, 7) * comb(37, 4)
    assert d["one_in"] == round(d["total"] / d["stuck"])
    assert COMMITTED["certificates"]["cases"] == sum(COMMITTED["certificates"]["by_size"].values())


def test_small_sizes_of_every_target_are_current():
    """Exceptions of up to six cards, for every target, as the full run counted them."""
    for t, x in COMMITTED["targets"]["by_target"].items():
        r = targets.study(int(t), 6)
        assert [len(r["exceptions"][n]) for n in range(1, 7)] == x["exceptions_by_size"][:6], t
        targets.make.cache_clear()
    s = report.targets_summary(COMMITTED["targets"]["by_target"])
    assert s == COMMITTED["targets"] and len(s["by_target"]) == 30


def test_simulated_section_has_the_paper_settings():
    sim = COMMITTED["simulated"]
    assert all(sim[k]["deals"] == simulate.N for k in sim if k != "seeds")
    small = simulate.run(50)  # the same code at a small size: shapes and bounds only
    assert small.keys() == {k for k in sim if k != "seeds"}
    assert all(0 <= v <= 50 for x in small.values() for v in x.values())


def test_countdown_section_is_consistent():
    c = COMMITTED["countdown"]
    assert c["problems"] == c["selections"] * c["targets"] == 13_243 * 899
    assert round(100 * c["exact"] / c["problems"], 2) == c["exact_percent"]
    assert c["reach_none"] == [[1, 1, 2, 2, 3, 3]]


def test_results_file_sits_with_the_scripts():
    assert report.RESULTS == Path(report.__file__).parent / "results.json"
