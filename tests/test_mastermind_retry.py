"""Fallout 3 exit and retry: both readings against brute force on small screens, and the study."""

import itertools
import json
from fractions import Fraction
from pathlib import Path

import pytest

from mastermind import fallout3 as fo
from mastermind import retry as rt

SMALL = ["hard", "herd", "hero", "zero", "jump", "love", "dive", "cave", "test"]
# Six-word screens of SMALL, where one entry is always enough, and seven-word ENABLE screens
# where it is not: the first six that `fallout3.sample_screen` drew with the seed "retry
# tests" on which no strategy pins every password down within one entry.
SCREENS = [list(s) for s in itertools.combinations(SMALL, 6)][::21] + [
    "boot mall sits gray colt emic tang".split(),
    "carl zonk leno spur moly avow dyed".split(),
    "noir twas mair cadi yill levo oped".split(),
    "nebs rolf smog rump vide ogam cane".split(),
    "roil duma hued pixy grum fops eddo".split(),
    "hoke kind resh sulu lint wair pams".split(),
]
RESULTS = Path(__file__).parent.parent / "mastermind" / "results" / "retry.json"


def parts(screen, guess, cands):
    out = {}
    for s in cands:
        out.setdefault(fo.likeness(guess, s), []).append(s)
    return out


def brute_a(screen, cands, used):
    """Reading A by plain recursion over every screen word, with no memo: the least
    (guesses, extra entries) summed over `cands`."""
    if len(cands) == 1:
        return (1, 0)
    if used == rt.SAFE:
        g, e = brute_a(screen, cands, 0)
        return (g, e + len(cands))
    best = None
    for g in screen:
        split = parts(screen, g, cands)
        if g not in cands and len(split) == 1:
            continue
        total = (len(cands), 0)
        for part in split.values():
            if part != [g]:
                a, b = brute_a(screen, part, used + 1)
                total = (total[0] + a, total[1] + b)
        best = total if best is None or total < best else best
    return best


def brute_b_value(screen, cands, used, restart):
    """Reading B: the least guesses + restart × exits summed over `cands`, in floats, by plain
    recursion over every action (leave, or guess any screen word)."""
    best = restart * len(cands)
    if used == rt.SAFE and len(cands) > 1:
        return best
    for g in screen:
        if used == rt.SAFE and g not in cands:
            continue
        split = parts(screen, g, cands)
        if g not in cands and len(split) == 1:
            continue
        total = len(cands)
        for part in split.values():
            if part != [g]:
                total += brute_b_value(screen, part, used + 1, restart)
        best = min(best, total)
    return best


def brute_b(screen):
    """Reading B's best expected guesses as the fixed point restart = value / n, found by
    bisection: a second method beside Dinkelbach's iteration."""
    n = len(screen)
    lo, hi = 1.0, float(4 * n)
    for _ in range(60):
        mid = (lo + hi) / 2
        if brute_b_value(screen, list(screen), 0, mid) / n < mid:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@pytest.mark.parametrize("screen", SCREENS, ids=" ".join)
def test_reading_a_matches_brute_force(screen):
    table = fo.likeness_table(screen)
    g, e = rt.fewest_guesses(table)
    bg, be = brute_a(screen, list(screen), 0)
    assert (g, e) == (bg, be + len(screen))
    assert g == rt.fewest_guesses_unlimited(table)  # the limit never binds under reading A


@pytest.mark.parametrize("screen", SCREENS, ids=" ".join)
def test_reading_b_matches_bisection(screen):
    table = fo.likeness_table(screen)
    b = rt.new_password(table)
    assert abs(float(b["guesses"]) - brute_b(screen)) < 1e-9
    assert b["entries"] == 1 / b["first entry"]


@pytest.mark.parametrize("screen", SCREENS, ids=" ".join)
def test_one_entry_lies_between_three_and_four_attempts(screen):
    table = fo.likeness_table(screen)
    n = len(screen)
    one = rt.first_entry_chance(table)
    assert Fraction(max(fo.most_cracked(table, 3)), n) <= one
    assert one <= Fraction(max(fo.most_cracked(table, 4)), n)
    assert rt.new_password(table)["first entry"] <= one


def test_the_best_policy_under_reading_b_can_leave_early():
    # On this screen one entry can pin down 5 of 7 passwords, but the fewest expected
    # guesses come from leaving sooner: 3 guesses in all, against 19/7 under reading A.
    s = rt.score_screen([w.upper() for w in SCREENS[-6]])
    assert (s["one entry"], s["B one entry"]) == (Fraction(5, 7), Fraction(3, 7))
    assert (s["A guesses"], s["B guesses"], s["B entries"]) == (
        19 / Fraction(7),
        3,
        7 / Fraction(3),
    )


def test_readings_agree_when_one_entry_is_enough():
    # On the README's eighteen-word screen every password is pinned down within one entry,
    # so leaving is never needed and both readings give the same expected guesses.
    s = rt.score_screen([w.upper() for w in rt.EXAMPLE])
    assert s["one entry"] == 1
    assert s["A guesses"] == s["B guesses"] == Fraction(55, 18)
    assert s["A entries"] == s["B entries"] == 1


def test_a_screen_where_the_readings_differ():
    # Every pair of these words has likeness 0, so each guess rules out only itself. Under
    # reading A the player guesses the words in turn; under reading B a fresh password makes
    # each entry a new three-guess game, and the fourth guess of an entry is never certain.
    screen = ["ABC", "BCA", "CAB", "DDD", "EEE", "FFF"]
    table = fo.likeness_table(screen)
    assert {table[g][s] for g in range(6) for s in range(6) if g != s} == {0}
    g, e = rt.fewest_guesses(table)
    assert (g, e) == (1 + 2 + 3 + 4 + 5 + 6, 3 * 1 + 3 * 2)  # words in turn; two entries
    b = rt.new_password(table)
    # An entry guesses three words in turn: it cracks with chance 1/2, after 1, 2 or 3
    # guesses, and otherwise leaves after 3, so c = (1 + 2 + 3) / 6 + 3 / 2 = 5 / 2 and
    # c / q = 5. Leaving earlier never helps: each next guess is likelier than a fresh one.
    assert b["first entry"] == Fraction(1, 2)
    assert b["guesses"] == 5 and b["entries"] == 2
    assert b["guesses"] > Fraction(g, 6)


def test_committed_study_matches_a_recomputation():
    data = json.loads(RESULTS.read_text())
    for row in data["rows"]:
        k = rt.ROWS.index((row["level"], row["words"]))
        screens = rt.screens_of_row(row["level"], row["words"], 2, row["seed"])
        for i, screen in enumerate(screens):
            for name, value in rt.score_screen(screen).items():
                assert row["values"][name][i] == str(value), (k, i, name)
    assert data["summary"] == rt.summarize(data["rows"])


def test_readme_transcript(capsys):
    readme = (Path(__file__).parent.parent / "mastermind" / "README.md").read_text()
    command = "$ uv run python -m mastermind.retry boot mall sits gray colt emic tang\n"
    block = readme.split(command, 1)[1].split("```", 1)[0]
    rt.main("boot mall sits gray colt emic tang".split())
    assert capsys.readouterr().out == block


def test_readme_table_is_the_committed_study():
    readme = (Path(__file__).parent.parent / "mastermind" / "README.md").read_text()
    for line in rt.readme_rows(json.loads(RESULTS.read_text())):
        assert line in readme, line


def test_half_up():
    assert rt._half_up(Fraction(8275, 100), 1) == "82.8"
    assert rt._half_up(Fraction(1, 8), 2) == "0.13"
    assert rt._half_up(Fraction(1, 3), 3) == "0.333"
