"""Plan P2: the switching policy ported from poke-env, and the exploitability analysis."""

import json
from fractions import Fraction

import numpy as np
import pytest

from pokemon import p2, teams, teams_model
from pokemon import teams_showdown as ts


def sets():
    ss = teams_model.sets(teams.POOL)
    return {s.name: s for s in ss}


def test_matchup_follows_poke_env():
    s = sets()
    tusk, gholdengo = s["great-tusk"], s["gholdengo"]
    # Great Tusk (ground/fighting) against Gholdengo (steel/ghost): ground hits it 2x;
    # Gholdengo's best type against Great Tusk is neutral (1x). Base Speed 87 vs 84.
    assert ts.matchup(tusk, gholdengo, 1.0, 1.0) == pytest.approx(2 - 1 + 0.1)
    assert ts.matchup(gholdengo, tusk, 1.0, 0.5) == pytest.approx(1 - 2 - 0.1 + 0.4 - 0.2)


def test_switch_rule_on_hand_made_states():
    me = sets()["great-tusk"]
    assert ts.should_switch(me, {}, -2.5, [-1.0, 0.3])
    assert not ts.should_switch(me, {}, -2.5, [-1.0, 0.0])
    assert not ts.should_switch(me, {}, -1.5, [0.3])
    assert ts.should_switch(me, {"def": -3}, 0.5, [0.3])
    assert ts.should_switch(me, {"spd": -4}, 0.5, [0.3])
    # Great Tusk's Attack is above its Special Attack: only an Attack drop counts.
    assert ts.should_switch(me, {"atk": -3}, 0.5, [0.3])
    assert not ts.should_switch(me, {"spa": -3}, 0.5, [0.3])


def test_hp_fraction():
    assert ts.hp_fraction("123/456 par") == 123 / 456 and ts.hp_fraction("0 fnt") == 0


def test_policies_follow_the_team_across_sides():
    assert ts.side_policies("p2-x-0-1-0", ("switch", "damage")) == ("switch", "damage")
    assert ts.side_policies("p2-x-0-1-3", ("switch", "damage")) == ("damage", "switch")
    jobs = ts.mirror_schedule([[0, 1], [2, 3]], 2, "p2-strength")
    assert jobs[0] == ("p2-strength-0-0-0", [0, 1], [0, 1]) and len(jobs) == 4
    assert ts.schedule([[0], [1]], 2, "p2-switch")[1] == ("p2-switch-0-1-1", [1], [0])


def test_summary_reads_labels_and_switches(tmp_path):
    d = {
        "pool": 2, "teams": [[0], [1]], "games": 2, "seed": "p2-switch", "format": "f",
        "showdown_commit": "c", "policies": ["switch", "switch"], "turn_limit": 300,
        "battles": [
            {"key": "p2-switch-0-1-0", "winner": "p1", "hp": [1, 2], "turns": 1, "checks": {},
             "stat_problems": [], "errors": 0, "switches": {"p1": 2, "p2": 0}, "limited": False},
            {"key": "p2-switch-0-1-1", "winner": "limit", "hp": None, "turns": 300, "checks": {},
             "stat_problems": [], "errors": 0, "switches": {"p1": 5, "p2": 1}, "limited": True},
        ],
    }  # fmt: skip
    path = tmp_path / "battles-1.json"
    path.write_text(json.dumps(d))
    row = teams.summarize([path])["pairs"]["0-1"]
    assert (row["wins"], row["ties"], row["turn_limit"]) == (1, 1, 1)
    # Game 1 puts team 0 on p2: its switches are p2's.
    assert (row["switches_i"], row["switches_j"]) == (2 + 1, 0 + 5)


def test_exploitability_and_its_vector_form():
    m = [
        [Fraction(0), Fraction(1), Fraction(-1)],
        [Fraction(-1), Fraction(0), Fraction(1)],
        [Fraction(1), Fraction(-1), Fraction(0)],
    ]
    sigma = [Fraction(1, 3)] * 3
    assert p2.exploitability(m, sigma) == (0, 0)
    sigma = [Fraction(1, 2), Fraction(1, 2), Fraction(0)]
    assert p2.exploitability(m, sigma) == (Fraction(1, 2), 0)  # team 0: (0 + 1) / 2
    pairs = [(0, 1), (0, 2), (1, 2)]
    x = np.array([1.0, -1.0, 1.0])  # payoffs to the first team of each pair, as in m
    want = [sum(float(sigma[i]) * float(m[j][i]) for i in range(3)) for j in range(3)]
    assert list(x @ p2.against(pairs, np.array([0.5, 0.5, 0.0]), k=3).T) == want


def test_wilson():
    lo, hi = p2.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)


def test_committed_results_are_what_the_module_writes():
    if not all(p2.summary_path(x).exists() for x in p2.LABELS[:2]):
        pytest.skip("the P2 matrices are not committed yet")
    assert json.loads(json.dumps(p2.run(), sort_keys=True)) == json.loads(p2.OUT.read_text())
