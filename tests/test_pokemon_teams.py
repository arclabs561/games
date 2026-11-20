"""The team study: the sets, the four model levels, the team game and its solver."""

import itertools
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from pokemon import teams, teams_data, teams_model, teams_showdown
from pokemon.equilibrium import duel_score, exact_equilibrium, lp_strategy

F = Fraction
SETS = teams_model.sets(teams.POOL)
BY_NAME = {s.name: s for s in SETS}


# The data and the rule for moves.


def entry(**kw):
    base = {"nonstandard": None, "flags": set(), "keys": set(), "handlers": set(), "id": "x"}
    return base | kw


def test_rule_for_moves():
    assert teams_data.reason_excluded(entry()) == ""
    assert teams_data.reason_excluded(entry(flags={"recharge", "contact"})) == "recharge"
    assert teams_data.reason_excluded(entry(keys={"selfSwitch"})) == "selfSwitch"
    assert (
        teams_data.reason_excluded(entry(handlers={"onBasePower", "onAfterHit"})) == "onBasePower"
    )
    assert teams_data.reason_excluded(entry(handlers={"onHit"})) == ""
    hurricane = entry(id="hurricane", handlers={"onModifyMove"})
    assert teams_data.reason_excluded(hurricane) == ""
    assert teams_data.reason_excluded(entry(id="photongeyser", handlers={"onModifyMove"})) != ""


def test_hits_and_drain():
    assert teams_data.showdown_hits({"multihit": "[2, 5]"}) == F(31, 10)
    # 2 or 3 hits with chance 7/20 each, 4 or 5 with 3/20 each.
    assert F(7, 20) * 2 + F(7, 20) * 3 + F(3, 20) * 4 + F(3, 20) * 5 == F(31, 10)
    assert teams_data.showdown_hits({"multihit": "2"}) == 2
    assert teams_data.showdown_drain({"drain": "[1, 2]", "recoil": None}) == F(1, 2)
    assert teams_data.showdown_drain({"drain": None, "recoil": "[33, 100]"}) == F(-33, 100)


def test_committed_tables_follow_the_rule():
    moves = teams_model.all_moves()
    assert "close-combat" in moves and dict(moves["close-combat"].boosts) == {"def": -1, "spd": -1}
    for excluded in ("knock-off", "u-turn", "sucker-punch", "body-press", "hyper-beam"):
        assert excluded not in moves
    assert moves["tachyon-cutter"].hits == 2 and moves["tachyon-cutter"].accuracy == 1
    assert moves["wave-crash"].drain == F(-33, 100)
    assert len(SETS) == teams.POOL
    assert all(1 <= len(s.moves) <= 4 for s in SETS)
    assert all(m.name in s.pokemon.moves for s in SETS for m in s.moves)


# Level 4's formulas.


def test_stats_match_showdown():
    # Showdown's request for Great Tusk (base 115/131/131/53/53/87) gave HP 392, Atk 319, Spe 231.
    st = BY_NAME["great-tusk"].pokemon.stats
    assert (st["hp"], st["atk"], st["spa"], st["spe"]) == (392, 319, 163, 231)


def test_modify_rounds_half_down():
    assert teams_model.modify(100, 3, 2) == 150
    assert teams_model.modify(101, 3, 2) == 151  # 151.5 rounds down
    assert teams_model.modify(103, 3, 2) == 154  # 154.5 rounds down
    assert teams_model.modify(10, 1, 2) == 5


def test_stages():
    assert [teams_model.stage(100, s) for s in (-2, -1, 0, 1, 2, 6)] == [50, 66, 100, 150, 200, 400]


def test_damage_by_hand():
    # Great Tusk's Headlong Rush (120, Ground, physical, STAB) against Gholdengo (Steel/Ghost,
    # 2x): base = floor(floor(42 * 120 * 319 / Def) / 50) + 2 with Gholdengo's Def 2*95+57.
    tusk, gold = BY_NAME["great-tusk"], BY_NAME["gholdengo"]
    move = next(m for m in tusk.moves if m.name == "headlong-rush")
    fa, fd = teams_model.fighters(tusk, gold)
    dfn = 2 * 95 + 57
    base = 42 * 120 * 319 // dfn // 50 + 2
    want = [teams_model.modify(base * r // 100, 3, 2) * 2 for r in range(85, 101)]
    assert teams_model.damage_rolls(move, fa, fd, {}, {}, True) == want


def test_abilities_change_matchups():
    weezing = BY_NAME["weezing-galar"]  # Poison/Fairy with Levitate
    tusk = BY_NAME["great-tusk"]
    assert teams_model.effectiveness("ground", tusk, weezing, 2) == 2
    assert teams_model.effectiveness("ground", tusk, weezing, 3) == 0


# The matchups.


def test_level_1_is_schmitz_duel():
    a, b = BY_NAME["primarina"], BY_NAME["heatran"]  # Water/Fairy against Fire/Steel
    # Water hits Fire/Steel for 2 * 1 = 2. Heatran's better type is Steel: 1/2 against Water
    # times 2 against Fairy = 1 (Fire would give 1/2). The winner keeps 1 - 1/2.
    assert teams_model.matchup(a, b, 1) == duel_score(F(2), F(1)) == F(1, 2)


@pytest.mark.parametrize("level", teams_model.LEVELS)
def test_matchups_are_antisymmetric(level):
    ss = SETS[:10]
    g = teams_model.matrix(ss, level)
    for i in range(10):
        for j in range(10):
            assert g[i][j] == -teams_model.matchup(ss[j], ss[i], level)
            assert -1 <= g[i][j] <= 1


def test_speed_tie_is_averaged():
    # Two copies of a Pokémon with different moves: same speed, so both orders count half.
    p = BY_NAME["garchomp"].pokemon
    moves = teams_model.all_moves()
    a = teams_model.Set(p, "sand-veil", (moves["earthquake"],))
    b = teams_model.Set(
        teams_model.Pokemon("x", "X", p.typing, p.base, p.abilities, p.moves),
        "sand-veil",
        (moves["outrage"],),
    )
    m, turns = teams_model.level4(a, b)
    fa, fb = teams_model.fighters(a, b)
    first = teams_model.race(
        teams_model.Side(fa, F(fa.stats["hp"]), {}), teams_model.Side(fb, F(fb.stats["hp"]), {})
    )
    assert m == first[0] and m == -teams_model.level4(b, a)[0]
    assert turns >= 1


# The team game.


def small(n=9, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.integers(-6, 7, (n, n))
    g = [[F(int(a[i][j] - a[j][i]), 12) for j in range(n)] for i in range(n)]
    return g


def test_payoff_is_antisymmetric_and_hand_checked():
    m = small()
    for t, u in itertools.combinations(list(itertools.combinations(range(9), 6))[:20], 2):
        assert teams.payoff(m, t, u) == -teams.payoff(m, u, t)
    # Two Pokémon a beats b by 1/2: {a} answers b for 1/2, and b's best answer to a is -1/2.
    m2 = [[F(0), F(1, 2)], [F(-1, 2), F(0)]]
    assert teams.payoff(m2, (0,), (1,)) * teams.TEAM == 1


def test_field_and_score_match_payoffs():
    m = small()
    ts = list(itertools.combinations(range(9), 6))
    sigma = [F(1, len(ts))] * len(ts)
    w, c = teams.field(m, ts, sigma)
    for t in ts[:10]:
        assert teams.score(m, w, c, t) == sum(teams.payoff(m, t, u) for u in ts) / len(ts)


def test_uniform_best_answer_by_brute_force():
    m = small()
    ts = list(itertools.combinations(range(9), 6))
    for t in range(9):
        brute = sum((max(m[u][t] for u in team) for team in ts), F(0)) / len(ts)
        assert teams.uniform_best_answer(m, t, 9) == brute


def test_milp_finds_the_best_team():
    rng = np.random.default_rng(0)
    m = np.array(small(10, 1), dtype=float)
    ts = teams.all_teams(10)
    assert len(ts) == math.comb(10, 6)
    for _ in range(5):
        w, c = rng.random(10), rng.random(10) * 0.3
        t = teams.best_team_milp(m, w, c)
        brute = teams.brute_scores(m, w, c, ts)
        assert abs(teams.score(m.tolist(), list(w), list(c), t) - brute.max()) < 1e-9


def test_double_oracle_against_the_full_game():
    # With 10 Pokémon there are 210 teams: solve the whole game and compare.
    m = small(10, 2)
    ts = [tuple(int(x) for x in t) for t in teams.all_teams(10)]
    sol = teams.solve(m, [ts[0]])
    assert sol["best_score"] <= 0
    g = teams.game(m, ts)
    sigma = [F(0)] * len(ts)
    for t, p in sol["support"]:
        sigma[ts.index(t)] = p
    for i in range(len(ts)):  # no team beats the mix, exactly
        assert sum((g[i][j] * sigma[j] for j in range(len(ts))), F(0)) <= 0
    _, v = lp_strategy(g)
    assert abs(v) < 1e-9


def test_exact_equilibrium_of_rock_paper_scissors_teams():
    rps = [[F(0), F(1), F(-1)], [F(-1), F(0), F(1)], [F(1), F(-1), F(0)]]
    x, _, v = exact_equilibrium(rps)
    assert x == [F(1, 3)] * 3 and v == 0


def test_ladder_field_is_seeded():
    use = [0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
    assert teams.ladder_field(use, 50, 0) == teams.ladder_field(use, 50, 0)
    assert all(len(set(t)) == 6 for t in teams.ladder_field(use, 50, 0))


# The simulator driver.


def test_seed_and_schedule():
    assert teams_showdown.seed("a").startswith("sodium,") and len(teams_showdown.seed("a")) == 71
    jobs = teams_showdown.schedule([[0], [1], [2]], 4, 0)
    assert len(jobs) == 3 * 4
    assert jobs[0] == ("0-0-1-0", [0], [1]) and jobs[1] == ("0-0-1-1", [1], [0])


def test_log_reading():
    b = teams_showdown.Battle("k", ([0], [1]), teams.POOL)
    chunk = "\n".join(
        [
            "update",
            "|split|p1",
            "|switch|p1a: P0|Great Tusk|392/392",
            "|switch|p1a: P0|Great Tusk|100/100",
            "|split|p2",
            "|switch|p2a: P1|Gholdengo|336/336",
            "|switch|p2a: P1|Gholdengo|100/100",
            "|move|p1a: P0|Headlong Rush|p2a: P1",
            "|-supereffective|p2a: P1",
            "|split|p2",
            "|-damage|p2a: P1|0 fnt",
            "|-damage|p2a: P1|0 fnt",
            "|faint|p2a: P1",
            "|win|p1",
        ]
    )
    b.read(chunk)
    assert b.active == {"p1": "P0", "p2": "P1"} and b.winner == "p1"
    assert b.hp[("p1", "P0")] == (392, 392) and b.fainted["p2"] == {"P1"}
    assert b.winner_hp() == [392, 392]


def test_summary_counts_sides(tmp_path):
    d = {
        "pool": 2, "teams": [[0], [1]], "games": 2, "seed": 0, "format": "f",
        "showdown_commit": "c",
        "battles": [
            {"key": "0-0-1-0", "winner": "p1", "hp": [1, 2], "turns": 1, "checks": {},
             "stat_problems": [], "errors": 0},
            {"key": "0-0-1-1", "winner": "p1", "hp": [1, 4], "turns": 1, "checks": {},
             "stat_problems": [], "errors": 0},
        ],
    }  # fmt: skip
    path = tmp_path / "battles-1.json"
    path.write_text(json.dumps(d))
    s = teams.summarize([path])
    # Game 0: team 0 on p1 wins with 1/2; game 1: team 1 on p1 wins with 1/4.
    assert s["pairs"]["0-1"] == {
        "games": 2,
        "wins": 1,
        "losses": 1,
        "ties": 0,
        "hp": "1/4",
        "turns": 2,
        "each": [0.5, -0.25],
    }
    assert teams.pair_matrix(s, 2, "hp")[0][1] == F(1, 8)


# The committed results.


def results():
    return json.loads((teams.RESULTS / "teams.json").read_text())


def test_committed_unit_results_recompute():
    r = results()
    names = [s.name for s in SETS]
    assert r["pool"] == names
    for level in teams_model.LEVELS:
        m = teams_model.matrix(SETS, level)
        u = teams.unit_summary(m, names)
        lv = r["levels"][str(level)]
        assert lv["unit_gradient_share"] == str(u["gradient_share"])
        assert lv["unit_ranking"] == u["ranking"]


def test_committed_team_equilibria_hold():
    # Each level's committed mix: no team in the committed restricted support beats it, and
    # the brute-force check over every team found a best score of exactly 0.
    r = results()
    for lv in r["levels"].values():
        assert lv["team_best_score"] == "0"
        assert lv["teams_checked"] == math.comb(teams.POOL, 6)
        assert sum(F(p) for _, p in lv["team_support"]) == 1


def test_simulator_checks_passed():
    r = results()
    for part in (r["duels"], r["common"]["battles"]):
        assert part["stat_problems"] == [] and part["choice_errors"] == 0
        assert part["damage_checks"].get("differ", 0) == 0 and part["damage_checks"]["agree"] > 1000


def test_note_is_what_the_script_writes():
    assert (teams.RESULTS / "teams.md").read_text() == teams.note(results())


def test_committed_figures_are_what_the_script_writes(tmp_path):
    from pokemon import teams_figures

    teams_figures.main(tmp_path)
    for name in ("teams-cards.svg", "teams-graph.svg", "teams-ranks.svg"):
        assert (tmp_path / name).read_bytes() == (
            teams_figures.HERE / "figures" / name
        ).read_bytes()


def test_readme_quotes_the_results():
    r = results()
    readme = "\n".join(  # the README and the note that holds its section on teams
        (teams.HERE / f).read_text() for f in ("README.md", "results/team-study.md")
    )
    lv = r["levels"]
    b = r["common"]["battles"]
    eq = sorted(b["equilibrium"].values(), key=lambda p: -F(p))
    quoted = [
        f"{math.comb(teams.POOL, 6):,}",
        *(f"{float(F(lv[k]['unit_gradient_share'])):.1%}" for k in "12345"),
        f"τ = {r['kendall_tau']['1-4']:.2f}",
        f"τ = {r['kendall_tau']['4-5']:.2f}",
        f"{float(F(eq[0])):.1%}",
        f"{float(F(b['equilibrium']['0'])):.1%}",
        f"{float(F(b['gradient_share'])):.1%}",
        f"{r['duels']['damage_checks']['agree'] + b['damage_checks']['agree']:,}",
        f"{r['sets_vs_usage']['same_ability']} of the 40",
        f"{sum(r['sets_vs_usage']['move_overlap']) / 40:.2f}",
        *(f"−{-float(F(lv[k]['usage_team_score'])):.3f}" for k in "12345"),
    ]
    for q in quoted:
        assert q in readme, q
    assert [len(lv[k]["team_support"]) for k in "12345"] == [3, 11, 11, 15, 11]
    assert len(b["equilibrium"]) == 15 and b["unique"]
