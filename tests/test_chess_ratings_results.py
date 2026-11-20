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
