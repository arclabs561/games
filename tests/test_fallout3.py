"""Fallout 3 terminal hacking: the likeness, the strategies, the optimum and the study."""

import itertools
import random
from fractions import Fraction

import pytest

from mastermind import fallout3 as fo

# Nine four-letter words with many letters in common, so screens drawn from them have
# partitions of every shape. Every 5-word screen of them (126 screens) is checked below.
SMALL = ["hard", "herd", "hero", "zero", "jump", "love", "dive", "cave", "test"]


def brute_best(screen, candidates, left):
    """Most candidates any strategy can crack within `left` guesses: try every screen word at
    every step, with no memo and no shortcuts."""
    if left == 0 or not candidates:
        return 0
    top = 0
    for g in screen:
        total = 0
        for k in {fo.likeness(g, s) for s in candidates}:
            part = [s for s in candidates if fo.likeness(g, s) == k]
            total += 1 if part == [g] else brute_best(screen, part, left - 1)
        top = max(top, total)
    return top


def brute_random(candidates, password, left):
    """Chance that random candidate guessing cracks `password` within `left` guesses, by
    following every sequence of choices."""
    if left == 0:
        return Fraction(0)
    chance = Fraction(0)
    for g in candidates:
        if g == password:
            chance += Fraction(1, len(candidates))
        else:
            k = fo.likeness(g, password)
            rest = [s for s in candidates if fo.likeness(g, s) == k]
            chance += Fraction(1, len(candidates)) * brute_random(rest, password, left - 1)
    return chance


def replay(screen, strategy, password):
    """Guesses needed for one password, playing the game turn by turn."""
    table = fo.likeness_table(screen)
    candidates, guesses = list(range(len(screen))), 0
    while True:
        g = strategy(table, candidates)
        guesses += 1
        if screen[g] == password:
            return guesses
        k = fo.likeness(screen[g], password)
        candidates = [s for s in candidates if table[g][s] == k]


def test_likeness_counts_letters_in_place_only():
    assert fo.likeness("ABLE", "BALE") == 2  # the README of brianburton's clone
    assert fo.likeness("RELEASED", "DETECTOR") == 2
    assert fo.likeness("COPY", "PICK") == 0
    for a, b in itertools.combinations(SMALL, 2):
        assert fo.likeness(a, b) == fo.likeness(b, a)


def test_terminal_rejects_mixed_lengths():
    with pytest.raises(ValueError):
        fo.Terminal(["four", "five", "sixsix"])


def test_expected_eliminations_of_math_goes_pop_part_1():
    # Their expected eliminations are n - (sum of squared part sizes)/n, plus 1/n when the
    # guess is on the list (finding the password eliminates it too): 4.2, 3.8, 3.8, 4.2, 1.8.
    screen = ["hard", "herd", "hero", "zero", "jump"]
    table, n = fo.likeness_table(screen), len(screen)
    sizes = [[len(p) for p in fo.partition(table, g, range(n)).values()] for g in range(n)]
    eliminations = [n - Fraction(sum(k * k for k in s), n) + Fraction(1, n) for s in sizes]
    assert eliminations == [Fraction(x, 10) for x in (42, 38, 38, 42, 18)]
    assert screen[fo.expected_size(table, list(range(n)))] == "hard"
    assert screen[fo.most_in_common(table, list(range(n)))] == "herd"  # the "closest" word


def test_minimax_on_the_example_of_eriksson_2016():
    # Largest parts 2, 3, 2, 2 for love, test, dive, cave: love, dive and cave tie.
    screen = ["love", "test", "dive", "cave"]
    table = fo.likeness_table(screen)
    largest = [max(len(p) for p in fo.partition(table, g, range(4)).values()) for g in range(4)]
    assert largest == [2, 3, 2, 2]
    assert screen[fo.minimax(table, [0, 1, 2, 3])] == "love"


def test_fallout_4_screens_match_math_goes_pop_part_2():
    # Their table: expected eliminations wins 100% on all three screens, averaging 2.75,
    # 2.58 and 2.58 guesses; the closest word wins 91.67%, 100% and 100%, averaging 2.92,
    # 2.58 and 2.58; random guessing (simulated, 1.2 million games per screen) wins 92.79%,
    # 96.21% and 97.93%.
    published = [(2.75, 11, 2.92, 0.9279), (2.58, 12, 2.58, 0.9621), (2.58, 12, 2.58, 0.9793)]
    for screen, (exp_avg, common_won, common_avg, random_won) in zip(
        fo.FALLOUT4_SCREENS, published, strict=True
    ):
        table = fo.likeness_table(screen)
        expected = fo.guess_counts(table, fo.expected_size)
        common = fo.guess_counts(table, fo.most_in_common)
        assert max(expected) <= 4 and round(sum(expected) / 12, 2) == exp_avg
        assert sum(c <= 4 for c in common) == common_won
        assert round(sum(common) / 12, 2) == common_avg
        assert abs(float(fo.random_candidate(table)) / 12 - random_won) < 0.001


def test_every_small_screen_exhaustively():
    """On all 126 five-word screens of SMALL: the optimum matches brute force for every
    attempt limit, no strategy beats it, each compiled guess tree matches turn-by-turn play,
    and the exact random-candidate chance matches enumeration of every choice."""
    for screen in itertools.combinations(SMALL, 5):
        table = fo.likeness_table(screen)
        n = len(screen)
        for left in (1, 2, 3, 4):
            values = fo.most_cracked(table, left)
            assert max(values) == brute_best(screen, list(screen), left), (screen, left)
        assert max(fo.most_cracked(table, n)) == n
        best = max(fo.most_cracked(table))
        for strategy in fo.STRATEGIES.values():
            counts = fo.guess_counts(table, strategy)
            assert counts == [replay(screen, strategy, p) for p in screen]
            assert sum(c <= fo.ATTEMPTS for c in counts) <= best
        for left in (1, 2, 4):
            chance = sum(brute_random(list(screen), p, left) for p in screen)
            assert fo.random_candidate(table, left) == chance


def test_optimum_matches_brute_force_on_sampled_screens():
    rng = random.Random(7)
    for _ in range(40):
        screen = fo.sample_screen(rng, rng.choice([4, 5, 6]), rng.randint(3, 8))
        table = fo.likeness_table(screen)
        assert max(fo.most_cracked(table)) == brute_best(screen, screen, 4)


def test_random_candidate_guesses_on_two_and_three_words():
    # Two words: guess right first half the time, else second: 1.5 each, 3 in total.
    assert fo.random_candidate_guesses(fo.likeness_table(["ab", "cd"])) == 3
    # Three words with likeness 0 to each other: 1, 2 or 3 guesses, 2 on average, 6 in total.
    assert fo.random_candidate_guesses(fo.likeness_table(["ab", "cd", "ef"])) == 6
    assert fo.random_candidate(fo.likeness_table(["ab", "cd", "ef"]), 2) == 2


def test_best_next_guess_and_command_line(capsys):
    screen = ["DOCTRINES", "BATTERIES", "SCORPIONS", "GENERATED", "OCCASIONS"]
    guess, sure, left = fo.best_next_guess(screen, {})
    assert (sure, left) == (5, 5)
    fo.main(["--best", *screen])
    assert capsys.readouterr().out.strip() == guess
    guess, sure, left = fo.best_next_guess(screen, {"DOCTRINES": 3})
    assert left == 2 and sure == 2 and guess in {"BATTERIES", "OCCASIONS"}
    fo.main(["--best", "doctrines=3", "batteries", "scorpions", "generated", "occasions"])
    assert capsys.readouterr().out.strip() == guess


def test_best_next_guess_may_be_a_ruled_out_word():
    # After IRON gets likeness 0, the candidates are LUNY, SPUD, BABU and FEDS, with three
    # attempts left. Any candidate guess leaves the other three with likeness 0 (or LUNY and
    # SPUD apart), so it cracks at most 3. BEAN is ruled out but splits them into LUNY, SPUD
    # (likeness 0) and BABU, FEDS (likeness 1, the B or the E): two attempts crack each pair.
    screen = ["luny", "iron", "spud", "bean", "babu", "feds"]
    candidates = [0, 2, 4, 5]
    values = fo.most_cracked(fo.likeness_table(screen), 3, candidates)
    assert [values[i] for i in candidates] == [3, 3, 3, 3]
    assert values[3] == 4
    assert fo.best_next_guess(screen, {"iron": 0}) == ("bean", 4, 4)


def test_word_count_formula():
    assert fo.word_count(50, 2) == 20  # at the lowest Science that can try the lock
    assert fo.word_count(75, 2) == 12.5
    assert fo.word_count(100, 0) == 5
    assert fo.word_count(100, 4) == 12.5


def test_sample_screen_is_seeded_and_valid():
    a = fo.sample_screen(random.Random(1), 9, 20)
    assert a == fo.sample_screen(random.Random(1), 9, 20)
    assert len(set(a)) == 20 and {len(w) for w in a} == {9}


def test_first_guess_rules_are_shares_of_the_optimum():
    table = fo.likeness_table(fo.FALLOUT4_SCREENS[0])
    shares = fo.first_guess_rules(table)
    assert all(0 < s <= 1 for s in shares.values())
    assert shares["expected size"] == 1 and shares["most in common"] == 1
