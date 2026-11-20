"""Chess: committed results, figures and the README's quoted numbers and transcript."""

import hashlib
import json
from pathlib import Path

import pytest

from chess_ratings import figures, settle

HERE = Path(__file__).parent.parent / "chess_ratings"
RESULTS = HERE / "results"

# SHA-256 of the files that the Hugging Face jobs wrote, as the jobs printed them. The files in
# results/ were copied from the jobs' outputs; these hashes tie each copy to its run.
JOB_OUTPUTS = {
    "trajectories.json": "eba24bdbfbdd2f12a055a5f8bd506efbe0326b129b293dc2842496701ca54afd",
    "whatwould.txt": "db720985eca38a957b8d0dc45b69c2256126b971fe2b25b54fce2e09369fd0f4",
    "common.json": "01071dafb40e516f32bf1f6f0e5e0ef153452541472010bf99cb82d6f383db1b",
    "estimate.json": "4d228344947857ffed2f4e6a1d9e7acd09216ae9e8cfc9e7424bfdae66703fa2",
    "middlegame.json": "c7e705ad60eccf6d58731a720cc8f175bb6aac6dd354ee9b42be16e71309062a",
    "calibration-pilot.json": "46c299fcfdd09df40d9ab3eb299a3a1555e27ce891a31f1a296095b599c78e1f",
    "calibration-fit.json": "68759397df8de6df38496be943baf5b4852c9010726bc7a675539d0b321b9dbb",
}


@pytest.mark.parametrize("name", sorted(JOB_OUTPUTS))
def test_copied_job_outputs_match_their_hashes(name):
    assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == JOB_OUTPUTS[name]


def test_comparison_with_lichess_is_what_the_script_writes():
    committed = json.loads((RESULTS / "compare.json").read_text())
    assert committed == json.loads(settle.to_text(settle.compare_with_lichess()))


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_committed_figure_is_current(name):
    committed = (figures.OUT / f"{name}.svg").read_bytes()
    assert figures.FIGURES[name]().encode("utf-8") == committed


def test_readme_transcript_is_the_job_output():
    readme = (HERE / "README.md").read_text()
    first = (RESULTS / "whatwould.txt").read_text().split("\n\n")[0]
    assert first in readme


def load(name):
    return json.loads((RESULTS / name).read_text())


def test_numbers_quoted_in_the_readme():
    readme = (HERE / "README.md").read_text()
    s, c, e = load("settle.json"), load("compare.json"), load("estimate.json")
    assert s["games_to_establish"]["60"]["lichess"] == 10
    assert [s["games_to_establish"][k]["lichess"] for k in ("80", "110", "200", "500")] == [
        11,
        11,
        14,
        34,
    ]
    sim = s["simulated"]
    games90 = [sim[k]["games_until_90_percent_within_100"] for k in ("1100", "1900", "800", "2300")]
    assert games90 == [36, 35, 43, 48] and "36; 35; 43; 48" in readme
    assert round(s["exact"]["1500"]["20"]["within_100"], 3) == 0.797
    assert (
        round(s["noise_floor_linear"], 1) == 31.4 and round(s["equal_players_rating_gap_95"]) == 87
    )
    assert [s["games_to_decide"][g]["median"] for g in ("50", "100", "200")] == [127, 29, 10]
    at20 = next(r for r in c["settling"] if r["game"] == 20)
    assert (round(at20["real"], 2), round(at20["model"], 2)) == (0.70, 0.84)
    assert round(e["posterior_mean_mae"]) == 199 and round(e["baseline_mean_rating_mae"]) == 451
    assert [round(x) for x in e["posterior_mean_mae_interval"]] == [183, 215]
    assert round(100 * e["coverage_posterior_interval"]) == 86
    assert (e["move_matching"]["hits"], e["move_matching"]["positions"]) == (8541, 15186)
    assert "199 (183 to 215)" in readme and "8,541 of 15,186" in readme


def test_blunder_totals_quoted_in_the_readme():
    totals = figures.common_totals()
    assert sum(v[1] for v in totals.values()) == 209
    assert round(sum(v[2] for v in totals.values())) == 307
    assert sum(v[0] for v in totals.values()) == 39_975
    low = [totals[b] for b in ("600-1000", "1000-1400")]
    moves, played, expected = (sum(x[i] for x in low) for i in range(3))
    assert (
        moves == 9805
        and f"{100 * played / moves:.2f}" == "1.03"
        and f"{100 * expected / moves:.2f}" == "1.74"
    )
    m = load("middlegame.json")["by_band"]
    assert (
        sum(b["observed"] for b in m.values()) == 40
        and round(sum(b["maia_expected"] for b in m.values())) == 52
    )


def test_calibration_numbers_quoted_in_the_readme():
    readme = (HERE / "README.md").read_text()
    t = load("calibration-test.json")
    p, arms = t["primary"], t["secondary"]["arms"]
    assert (p["chosen"], p["bias_arm"]) == ("F2", "F7")
    assert (p["player_games"], p["games"], p["player_games_lower_bands"]) == (1000, 500, 400)
    assert [round(x) for x in (p["bias_lower_bands_old"], *p["bias_lower_bands_old_interval"])] == [
        166,
        142,
        190,
    ]
    assert [round(x) for x in (p["bias_lower_bands_new"], *p["bias_lower_bands_new_interval"])] == [
        88,
        62,
        114,
    ]
    assert [round(x) for x in (p["bias_reduction"], *p["bias_reduction_interval"])] == [78, 74, 82]
    assert round(p["holm"]["bias"], 4) == 0.0001
    assert (p["covered_by_new_only"], p["covered_by_old_only"]) == (78, 0)
    assert f"{100 * p['coverage_old']:.1f}" == "88.9" and f"{100 * p['coverage_new']:.1f}" == "96.7"
    assert [f"{100 * x:.1f}" for x in p["coverage_new_wilson"]] == ["95.4", "97.6"]
    assert [round(arms[a]["all"]["mae"]) for a in ("F0", "F7", "F2")] == [189, 195, 205]
    for a, row in (("F0", "+174, +158, +43, -1 and -31"), ("F7", "+81, +95, -4, -24 and -2")):
        bands = ("600-1000", "1000-1400", "1400-1800", "1800-2200", "2200-2700")
        assert ", ".join(f"{arms[a][b]['mean_error']:+.0f}" for b in bands[:4]) + (
            f" and {arms[a][bands[4]]['mean_error']:+.0f}"
        ) == row.replace("-0", "+0")
        assert row in readme
    pop = t["secondary"]["population"]
    assert round(pop["F2+P"]["all"]["mae"]) == 163 and round(pop["F0"]["all"]["mae"]) == 189
    assert "+166 (142 to 190)" in readme and "96.7% (95.4 to 97.6)" in readme
    pilot = load("calibration-pilot.json")
    assert pilot["chosen"] == "F2" and pilot["fitted_constants"]["F2"]["tau"] == 0.56
    assert round(pilot["diagnosis"]["h7_sandwich"]["tau_centered"], 2) == 0.58


def test_population_baseline_and_clustered_blunder_gap_quoted_in_the_readme():
    readme = (HERE / "README.md").read_text()
    pop = load("calibration-test.json")["secondary"]
    assert round(pop["population_baseline_mean_rating_mae"]) == 282
    assert round(pop["population"]["F0"]["all"]["mae"]) == 189
    assert round(pop["population"]["F2+P"]["all"]["mae"]) == 163
    assert "| 189; 163; 282 |" in readme
    low = figures.position_interval(("600-1000", "1000-1400"))
    assert [f"{x:.2f}" for x in low] == ["-0.71", "-1.44", "0.01"]
    below_1800 = figures.position_interval(("600-1000", "1000-1400", "1400-1800"))
    assert [f"{x:.2f}" for x in below_1800] == ["-0.38", "-0.76", "-0.01"]
    assert "−0.71 points (−1.44 to 0.01)" in readme and "−0.38 (−0.76 to −0.01)" in readme
