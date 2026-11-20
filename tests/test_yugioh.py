"""Yu-Gi-Oh opening hands: exact chances checked by closed forms, brute force and simulation."""

import re
from fractions import Fraction
from math import comb
from pathlib import Path

from yugioh import calc, draws, figures, odds, tables
from yugioh.draws import Engine, p_engine, simulate_engine
from yugioh.odds import FIRST, NEEDS, SECOND, Deck, Name, by_role

FOLDER = Path(__file__).parent.parent / "yugioh"


def test_hand_checked_values():
    # One given card in 40, five drawn: 5/40.
    assert odds.p_at_least_one(40, 1, FIRST) == Fraction(1, 8)
    # Three copies: 1 − C(37, 5)/C(40, 5) = 1 − 435,897/658,008 = 667/1,976.
    assert odds.p_at_least_one(40, 3, FIRST) == Fraction(667, 1976)
    # 14 starters in 40 is the first count to reach 90% going first, by about 21 hands in 658,008.
    assert odds.p_at_least_one(40, 14, FIRST) == Fraction(592228, 658008)
    assert odds.fewest(Fraction(9, 10), 40, FIRST) == 14


def test_sum_over_hands_matches_closed_forms():
    for size in (40, 41, 45, 50, 60):
        for s in range(0, 16, 3):
            for e in range(0, 13, 4):
                for b in (0, 3):
                    deck = Deck(size, s, e, b)
                    for n in (FIRST, SECOND):
                        assert odds.p_hand(deck, n, odds.starter) == odds.p_at_least_one(size, s, n)
                        assert odds.p_hand(deck, n, odds.no_brick) == Fraction(
                            comb(size - b, n), comb(size, n)
                        )
                        assert odds.p_hand(
                            deck, n, odds.starter_and_extender
                        ) == odds.p_starter_and_extender(deck, n)


def test_hand_probabilities_sum_to_one():
    deck = Deck(40, 12, 9, 3)
    assert sum(w for _, w in odds.hands(deck.counts(), FIRST)) == comb(40, FIRST)


def small_decks() -> list[list[Name]]:
    """Small named decks with copies from 1 to 3."""
    return [
        [Name("s1", "starter", 3), Name("s2", "starter", 1), Name("e1", "extender", 2),
         Name("b1", "brick", 2), Name("o1", "other", 3), Name("o2", "other", 1)],
        [Name("s1", "starter", 2), Name("s2", "starter", 2), Name("s3", "starter", 1),
         Name("e1", "extender", 3), Name("b1", "brick", 1), Name("o1", "other", 3),
         Name("o2", "other", 3)],
        odds.named(Deck(14, 5, 3, 2), copies=2),
    ]  # fmt: skip


def test_brute_force_agrees_with_dynamic_program_and_sum():
    for names in small_decks():
        size = sum(c.copies for c in names)
        counts = [sum(c.copies for c in names if c.role == r) for r in odds.ROLES]
        deck = Deck(size, *counts[:3])
        for n in (FIRST, SECOND):
            for need in NEEDS.values():
                brute = odds.p_brute(names, n, by_role(need))
                assert brute == odds.p_named(names, n, by_role(need)) == odds.p_hand(deck, n, need)
            named_need = odds.two_different_starters
            assert odds.p_brute(names, n, named_need) == odds.p_named(names, n, named_need)


def test_copies_do_not_change_role_needs():
    deck = Deck(40, 12, 9, 3)
    for copies in (1, 2, 3):
        names = odds.named(deck, copies)
        assert all(1 <= c.copies <= copies for c in names)
        for need in NEEDS.values():
            assert odds.p_named(names, FIRST, by_role(need)) == odds.p_hand(deck, FIRST, need)
    # But they change "two different starters": fewer names, more duplicates.
    two = [
        odds.p_named(odds.named(Deck(40, 12), c), FIRST, odds.two_different_starters)
        for c in (3, 2, 1)
    ]
    assert two[0] < two[1] < two[2]


def test_small_simulation_agrees():
    deck = Deck(40, 12, 9, 3)
    trials = 20_000
    for need in (odds.starter, odds.starter_and_extender):
        hits = odds.simulate(odds.named(deck), FIRST, by_role(need), trials, seed=7)
        lo, hi = odds.wilson(hits, trials)
        assert lo <= odds.p_hand(deck, FIRST, need) <= hi


def test_engine_by_hand():
    # Deck {S, draw-2, O}, one card drawn: S, or the draw-2 then both others: 2/3.
    assert p_engine(Deck(3, starters=1), Engine(draw2=1), 1, odds.starter) == Fraction(2, 3)
    # Deck {S, looker, O, O}, one card: S (1/4), or the looker (1/4) seeing S in 2 of 3: 5/12.
    p = p_engine(Deck(4, starters=1), Engine(lookers=1, look=2), 1, odds.starter)
    assert p == Fraction(5, 12)
    # A second draw-2 is dead: {draw-2, draw-2, S, O, O}, one card.
    p = p_engine(Deck(5, starters=1), Engine(draw2=2), 1, odds.starter)
    # S first: 1/5. A draw-2 first (2/5): two draws from {draw-2, S, O, O}; the other
    # draw-2 is dead, so success is S among the two cards: 1/2.
    assert p == Fraction(1, 5) + Fraction(2, 5) * Fraction(1, 2)


def test_no_engine_is_the_plain_chance():
    deck = Deck(40, 12, 9, 3)
    for n in (FIRST, SECOND):
        for need in NEEDS.values():
            assert p_engine(deck, Engine(), n, need) == odds.p_hand(deck, n, need)


def test_cantrips_cancel_exactly():
    # Proved in the README: c cantrips in N + c cards give the hands of N cards without them.
    for s, e, b in ((12, 9, 3), (8, 4, 6)):
        for c in (1, 3):
            for n in (FIRST, SECOND):
                for need in NEEDS.values():
                    with_cantrips = p_engine(Deck(40 + c, s, e, b), Engine(cantrips=c), n, need)
                    assert with_cantrips == odds.p_hand(Deck(40, s, e, b), n, need)


def test_engine_simulation_agrees():
    deck = Deck(40, 12, 9, 3)
    trials = 20_000
    for eng in (Engine(draw2=3), Engine(lookers=3), Engine(cantrips=2, draw2=2, lookers=1)):
        for need in (odds.starter, odds.starter_and_extender):
            hits = simulate_engine(deck, eng, FIRST, need, trials, seed=7)
            lo, hi = odds.wilson(hits, trials)
            assert lo <= p_engine(deck, eng, FIRST, need) <= hi


def test_starter_equivalent_is_zero_without_engine():
    assert draws.starter_equivalent(Deck(40, 12), Engine(), FIRST, odds.starter) == 0


def test_published_values():
    assert tables.agrees("33.76", odds.p_at_least_one(40, 3, FIRST))
    assert not tables.agrees("77.8", odds.p_at_least_one(40, 9, FIRST))
    verdicts = [
        tables.agrees(v, odds.p_at_least_one(s, k, n)) for src, q, v, s, k, n in tables.PUBLISHED
        if src == tables.LUMDERA
    ]  # fmt: skip
    assert verdicts.count(False) == 10 and len(verdicts) == 13


def test_committed_tables_are_current():
    assert (FOLDER / "results" / "tables.md").read_text() == tables.exact_tables()


def test_committed_simulation_matches_exact_values():
    text = (FOLDER / "results" / "simulation.md").read_text()
    assert f"Seed {tables.SEED}" in text and f"{tables.TRIALS:,} hands" in text
    rows = [line.split(" | ") for line in text.splitlines() if line.startswith("| ")]
    body = [r for r in rows if r[0] not in ("| Need",) and not r[0].startswith("|---")]
    assert len(body) == 5 + 2 * (len(tables.ENGINES) - 1)
    assert all(r[-1].strip(" |") == "yes" for r in body)
    for label, need in NEEDS.items():
        assert f"| {label} |" in text
        assert tables.pct(odds.p_hand(tables.EXAMPLE, FIRST, need), 4) in text


def test_committed_figures_are_current():
    for name, draw in figures.FIGS.items():
        assert (FOLDER / "figures" / name).read_text() == draw()


def test_readme_transcripts():
    readme = (FOLDER / "README.md").read_text()
    blocks = re.findall(
        r"```console\n\$ uv run python -m yugioh\.calc(.*?)\n(.*?)```", readme, re.S
    )
    assert len(blocks) == 2
    for args, output in blocks:
        assert output == "\n".join(calc.report(calc.parser().parse_args(args.split()))) + "\n"


def test_readme_percentages_are_written_by_the_scripts():
    readme = re.sub(r"```.*?```", "", (FOLDER / "README.md").read_text(), flags=re.S)
    written = (FOLDER / "results" / "tables.md").read_text()
    written += (FOLDER / "results" / "simulation.md").read_text()
    for p in set(re.findall(r"\d+\.\d+%", readme)):
        assert p in written or f"| {p[:-1]} |" in written, p  # printed by a calculator
