"""FreeCell: the deal generator, the rules, both solvers, brute force, fences and certificates."""

import pytest

from freecell import native
from freecell.brute import brute_winnable, reachable
from freecell.certificates import blocked_at_start, static_stuck
from freecell.deals import (
    deal_columns,
    ms_deal,
    ms_order,
    ms_rand,
    name,
    parse_card,
    parse_layout,
    show,
)
from freecell.fence import check, fence_a, fence_b
from freecell.report import FAQ_ZERO_CELLS, published_least
from freecell.rules import Rules, apply, autoplay, can_stack, replay, safe, start
from freecell.small import orders
from freecell.solver import run_moves, solve
from freecell.survey import survey_a, survey_b

needs_cc = pytest.mark.skipif(native.compiler() is None, reason="Solver B needs a C compiler")

# Rosetta Code, "Deal cards for FreeCell": the cards of games 1 and 617 in the order dealt.
GAME_1 = (
    "JD 2D 9H JC 5D 7H 7C 5H KD KC 9S 5S AD QC KH 3H 2S KS 9D QD JS AS AH 3C 4C 5C TS QH "
    "4H AC 4D 7S 3S TD 4S TH 8H 2C JH 7D 6D 8S 8D QS 6C 3D 8C TC 6S 9C 2H 6H"
)
GAME_617 = (
    "7D AD 5C 3S 5S 8C 2D AH TD 7S QD AC 6D 8H AS KH TH QC 3H 9D 6S 8D 3D TC KD 5H 9S 3C "
    "8S 7H 4D JS 4C QS 9C 9H 7C 6H 2C 2S 4S TS 2H 5D JC 6C JH QH JD KS KC 4H"
)
# The fc-solve FAQ ("I found a solution for MS Deal No. 11,982"): the layout, in rows.
DEAL_11982 = """
AH AS 4H AC 2D 6S TS JS
3D 3H QS QC 8S 7H AD KS
KD 6H 5S 4D 9H JH 9S 3C
JC 5D 5C 8C 9D TD KH 7C
6C 2C TH QH 6D TC 4S 7S
JD 7D 8H 9C 2H QD 4C 5H
KC 8D 2S 3S
"""
FROZEN_11982 = {"AC", "AS", "3H", "4H", "6H", "6S", "QC", "QS"}


def test_microsoft_rand_matches_the_c_runtime():
    rand = ms_rand(1)
    assert [next(rand) for _ in range(4)] == [41, 18467, 6334, 26500]


def test_deals_match_published_layouts():
    assert " ".join(map(name, ms_order(1))) == GAME_1
    assert " ".join(map(name, ms_order(617))) == GAME_617
    assert ms_deal(11982) == parse_layout(DEAL_11982)
    assert parse_layout(show(ms_deal(5))) == ms_deal(5)
    assert [len(col) for col in ms_deal(1)] == [7, 7, 7, 7, 6, 6, 6, 6]


def test_rules():
    assert can_stack(parse_card("5H"), parse_card("6S"))
    assert not can_stack(parse_card("5H"), parse_card("6D"))
    assert not can_stack(parse_card("5H"), parse_card("7S"))
    rules = Rules()
    found = (1, 1, 1, 1)  # every ace home
    assert safe(rules, found, parse_card("2H"))
    assert not safe(rules, (2, 1, 1, 2), parse_card("3S"))
    assert safe(rules, (2, 2, 2, 2), parse_card("3S"))
    pos = start(rules, ms_deal(1))
    with pytest.raises(ValueError):
        apply(rules, pos, (parse_card("AS"), "home"))  # the ace of spades is buried
    assert not replay(rules, ms_deal(1), [(parse_card("6H"), "home")])


def test_autoplay_plays_aces_and_twos():
    rules = Rules(cells=0, columns=2, ranks=2, suits=(0, 1))
    layout = ((parse_card("2C"), parse_card("AD")), (parse_card("2D"), parse_card("AC")))
    pos, made = autoplay(rules, start(rules, layout))
    assert len(made) == 4 and sum(pos[2]) == 4


def test_supermove_expands_to_legal_single_moves():
    # 9S 8H 7C 6D on one column, a 10D on another, one free cell and two empty columns.
    rules = Rules(cells=1, columns=4, ranks=10, suits=(0, 1, 2, 3))
    run = [parse_card(c) for c in ("9S", "8H", "7C", "6D")]
    rest = [c for c in rules.deck() if c not in run and c != parse_card("TD")]
    layout = (tuple(rest[:-1]) + tuple(run), (rest[-1], parse_card("TD")), (), ())
    pos = start(rules, layout)
    line = run_moves(run, parse_card("TD"), free=1, temps=2)
    for move in line:
        pos = apply(rules, pos, move)
    assert pos[0][1][-5:] == bytes([parse_card("TD"), *run])


def test_deal_11982_is_lost_with_four_cells_and_won_with_five():
    res = solve(Rules(), ms_deal(11982), use_supermoves=False)
    assert (res.verdict, res.states, res.most_home) == ("lost", 61643, 2)  # Sells (2026)
    assert {name(c) for c in res.frozen} == FROZEN_11982
    win = solve(Rules(cells=5), ms_deal(11982))
    assert win.verdict == "won" and replay(Rules(cells=5), ms_deal(11982), win.line)


def test_brute_force_on_11982_reaches_the_published_number_of_positions():
    # Columns in fixed places, free cells as a set, no autoplay: 84,767 in Sells (2026).
    assert reachable(ms_deal(11982), 4) == (84767, False, 2)


@needs_cc
def test_solver_b_on_11982_and_deal_1():
    lost, won = native.run([native.ms_problem(11982, 4), native.ms_problem(1, 4)])
    assert (lost.verdict, lost.states, lost.most_home) == ("lost", 83239, 2)
    assert {name(c) for c in lost.frozen} == FROZEN_11982
    assert won.verdict == "won" and replay(Rules(), ms_deal(1), won.line)
    # With its own autoplay, Solver B reaches exactly Solver A's 61,643 positions.
    auto, won = native.run([native.ms_problem(11982, 4), native.ms_problem(1, 4)], autoplay=True)
    assert (auto.verdict, auto.states, auto.frozen) == ("lost", 61643, lost.frozen)
    assert won.verdict == "won" and replay(Rules(), ms_deal(1), won.line)


# Small games: (ranks, suits, columns, cells). Every 7th deal of each.
SMALL = [
    (3, (0, 1), 2, 0),
    (3, (0, 1), 2, 1),
    (4, (0, 1), 2, 1),
    (4, (0, 1), 2, 2),
    (2, (0, 1, 2, 3), 2, 1),
    (4, (0, 1), 3, 0),
]


@needs_cc
@pytest.mark.parametrize("game", SMALL)
def test_solvers_agree_with_brute_force_on_small_decks(game):
    ranks, suits, columns, cells = game
    rules = Rules(cells=cells, columns=columns, ranks=ranks, suits=suits)
    layouts = [deal_columns(list(o), columns) for o in orders(rules, None, step=7)][:600]
    answers = native.run([native.deck_problem(rules, lay) for lay in layouts])
    for lay, b in zip(layouts, answers, strict=True):
        a = solve(rules, lay)
        brute = brute_winnable(lay, cells)
        assert brute == (a.verdict == "won") == (b.verdict == "won"), lay
        if brute:
            assert replay(rules, lay, a.line) and replay(rules, lay, b.line)
        else:
            assert a.frozen == b.frozen
            if cells == 0:
                assert static_stuck(lay) <= a.frozen
        if static_stuck(lay) and cells == 0 or blocked_at_start(lay, cells):
            assert not brute


def test_blocked_at_the_start_matches_the_faq():
    # FreeCell FAQ: with zero free cells, 1,325 of the first 32,000 deals are blocked at the
    # start, and the first few are 1, 23, 25, 28, 46 and 51.
    blocked = [d for d in range(1, 32001) if blocked_at_start(ms_deal(d), 0)]
    assert len(blocked) == 1325 and blocked[:6] == [1, 23, 25, 28, 46, 51]
    assert not blocked_at_start(ms_deal(1), 1)


def test_no_zero_cell_win_has_stuck_cards():
    assert not any(static_stuck(ms_deal(d)) for d in FAQ_ZERO_CELLS)


def test_fences_of_both_solvers_check_and_a_broken_one_fails():
    for deal, cells in ((1, 1), (6, 2)):
        fa = fence_a(deal, cells)
        assert check(ms_deal(deal), cells, fa, autoplay=True)[0]
        broken = set(fa)
        broken.discard(max(broken))
        assert not check(ms_deal(deal), cells, broken, autoplay=True)[0]
        if native.compiler():
            assert check(ms_deal(deal), cells, fence_b(deal, cells), autoplay=False)[0]


def test_published_list_and_the_survey_agree_on_the_first_deals():
    pub = published_least()
    assert len(pub) == 32000 and pub[11982] == 5
    assert sorted(d for d, c in pub.items() if c == 0) == FAQ_ZERO_CELLS
    for deal in (1, 2, 3, 6):
        recs = survey_a(deal)
        assert recs[-1]["verdict"] == "won" and recs[-1]["replay"]
        assert recs[-1]["cells"] == pub[deal]


@needs_cc
def test_survey_b_on_the_first_deals():
    pub = published_least()
    recs = survey_b(list(range(1, 13)))
    won = {r["deal"]: r["cells"] for r in recs if r["verdict"] == "won"}
    assert won == {d: pub[d] for d in range(1, 13)}
    assert all(r["replay"] for r in recs if r["verdict"] == "won")
