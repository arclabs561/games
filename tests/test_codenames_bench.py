"""Tests for codenames/bench.py and bench_data.py: the framework's deal, its cosine
distance, its legality rule, and its codemaster and guesser, each checked against a
line-by-line transliteration of the framework's code (stepmat/Codenames_GPT, players/
codemaster_glove.py and guesser_glove.py) on random vectors; the scoring rules; and the
word2vec and GloVe readers."""

import gzip
import io
import itertools
import math
import random
import struct

import numpy as np
import pytest

from codenames import bench, bench_data
from codenames.game import Board, play
from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM

# Transliterations of the framework's code, kept as close to the original as possible.


def framework_deal_reference(pool, seed):
    random.seed(int(seed))
    temp = list(pool)
    random.shuffle(temp)
    words = temp[:25]
    key = ["Red"] * 9 + ["Blue"] * 8 + ["Civilian"] * 7 + ["Assassin"] * 1
    random.shuffle(key)
    return words, key


def cos_dist(u, v):
    """scipy.spatial.distance.cosine for 1-d float64 vectors."""
    uv, uu, vv = float(np.dot(u, v)), float(np.dot(u, u)), float(np.dot(v, v))
    return min(max(1.0 - uv / math.sqrt(uu * vv), 0.0), 2.0)


def arr_not_in_word(word, arr, forms):
    if word in arr:
        return False
    lemm, lancas = forms[word]
    for i in arr:
        if i == lemm or i == lancas:
            return False
        if i.find(word) != -1:
            return False
        if word.find(i) != -1:
            return False
    return True


def get_clue_reference(words, maps, vecs, cm_wordlist, forms, threshold=0.7):
    """codemaster_glove.AICodemaster.get_clue on a fresh codemaster (no cache)."""
    red_words, bad_words = [], []
    for i in range(25):
        if words[i][0] == "*":
            continue
        elif maps[i] != "Red":
            bad_words.append(words[i].lower())
        else:
            red_words.append(words[i].lower())
    bad_word_dists = {w: {v: cos_dist(vecs[v], vecs[w]) for v in cm_wordlist} for w in bad_words}
    red_word_dists = {w: {v: cos_dist(vecs[v], vecs[w]) for v in cm_wordlist} for w in red_words}
    bests = {}
    for clue_num in range(1, 3 + 1):
        best_per_dist = np.inf
        best_per = ""
        best_red_word = ""
        for red_word in list(itertools.combinations(red_words, clue_num)):
            best_word = ""
            best_dist = np.inf
            for word in cm_wordlist:
                if not arr_not_in_word(word, red_words + bad_words, forms):
                    continue
                bad_dist = np.inf
                for bad_word in bad_word_dists:
                    if bad_word_dists[bad_word][word] < bad_dist:
                        bad_dist = bad_word_dists[bad_word][word]
                worst_red = 0
                for red in red_word:
                    dist = red_word_dists[red][word]
                    if dist > worst_red:
                        worst_red = dist
                if worst_red < best_dist and worst_red < bad_dist:
                    best_dist = worst_red
                    best_word = word
                    if best_dist < best_per_dist:
                        best_per_dist = best_dist
                        best_per = best_word
                        best_red_word = red_word
        bests[clue_num] = (best_red_word, best_per, best_per_dist)
    chosen_clue = bests[1]
    chosen_num = 1
    for clue_num, clue in bests.items():
        best_red_word, combined_clue, combined_score = clue
        worst = -np.inf
        for word in best_red_word:
            dist = cos_dist(vecs[word], vecs[combined_clue])
            if dist > worst:
                worst = dist
        if worst < threshold and worst != -np.inf:
            chosen_clue = clue
            chosen_num = clue_num
    return chosen_clue[1], chosen_num


def guess_reference(clue, board, vecs):
    """guesser_glove.AIGuesser._compute_distance, then sorted_words[0][1]."""
    w2v = []
    for word in board:
        if word[0] == "*":
            continue
        if word.lower() not in vecs or clue not in vecs:
            continue
        w2v.append((cos_dist(vecs[clue], vecs[word.lower()]), word))
    return list(sorted(w2v))


# A synthetic world: board words, clue words and random vectors.


def world(seed, n_clues=60, dims=8):
    rng = np.random.default_rng(seed)
    board_pool = [f"b{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]
    clues = [f"c{i:03d}" for i in range(n_clues)] + [
        "xbaa",
        "ba",
    ]  # illegal when a board holds baa, or any ba-word
    words = board_pool + clues
    vecs = {w: rng.standard_normal(dims) for w in words}
    forms = {c: (c, c[:3]) for c in clues}
    forms["c000"] = ("baa", "c00")  # its "lemma" is a word that is never on a board
    return board_pool, clues, vecs, forms


def as_vectors(vecs):
    names = sorted(vecs)
    return bench.Vectors(names, np.stack([vecs[w] for w in names]))


def framework_view(board: Board):
    """The board as the framework's codemaster sees it: capitals, revealed words starred."""
    names = {TEAM: "Red", OPPONENT: "Blue", NEUTRAL: "Civilian", ASSASSIN: "Assassin"}
    words = [
        f"*{names[k]}*" if w in board.revealed else w.upper()
        for w, k in zip(board.words, board.kinds, strict=True)
    ]
    return words, [names[k] for k in board.kinds]


def test_deal_matches_the_framework():
    pool = [f"W{i}" for i in range(394)]
    for seed in (0, 1, 7, 99, 3442):
        words, key = framework_deal_reference(pool, seed)
        b = bench.framework_deal(pool, seed)
        assert b.words == tuple(w.lower() for w in words)
        assert b.kinds == tuple(bench.KINDS[k] for k in key)
    assert sorted(bench.framework_deal(pool, 5).kinds).count(TEAM) == 9


def test_cosine_matches_the_scalar_formula():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((5, 7)), rng.standard_normal((4, 7))
    got = bench.cosine(a, b)
    want = np.array([[cos_dist(x, y) for y in b] for x in a])
    assert np.allclose(got, want, atol=1e-14)
    assert np.allclose(bench.cosine(a, a).diagonal(), 0, atol=1e-12)
    f = bench.cosine(a.astype(np.float32), b.astype(np.float32))
    assert f.dtype == np.float32 and np.allclose(f, want, atol=1e-6)


def test_legality_rule():
    forms = ("dog", "dog")
    assert not bench.framework_legal("dogs", forms, ["dog"])  # lemma is a board word
    assert not bench.framework_legal("fire", ("fire", "fir"), ["firefly"])  # part of one
    assert not bench.framework_legal("firefly", ("firefly", "firefly"), ["fire"])  # contains one
    assert not bench.framework_legal("cat", ("cat", "cat"), ["cat"])
    assert bench.framework_legal("tiger", ("tiger", "tig"), ["lion", "bear"])


@pytest.mark.parametrize("seed", range(6))
def test_codemaster_matches_the_framework_loops(seed):
    pool, clues, vecs, forms = world(seed)
    cm = bench.FrameworkCodemaster("glove", as_vectors(vecs), clues, forms)
    board = bench.framework_deal(pool, seed)
    rng = random.Random(seed)
    for _ in range(4):  # the full board, then with some words revealed
        words, key = framework_view(board)
        clue, count, _ = cm.plan(board)
        assert (clue, count) == get_clue_reference(words, key, vecs, clues, forms)
        left = [w for w in board.unrevealed() if board.kind(w) in (TEAM, NEUTRAL)]
        board.revealed |= set(rng.sample(left, 2))


@pytest.mark.parametrize("seed", range(4))
def test_codemaster_with_a_threshold_that_allows_every_count(seed):
    pool, clues, vecs, forms = world(seed + 10)
    cm = bench.FrameworkCodemaster("glove", as_vectors(vecs), clues, forms, threshold=2.0)
    board = bench.framework_deal(pool, seed)
    words, key = framework_view(board)
    clue, count, bests = cm.plan(board)
    assert count == max(k for k, b in enumerate(bests, start=1) if b[0])  # the largest valid k
    assert (clue, count) == get_clue_reference(words, key, vecs, clues, forms, threshold=2.0)


def test_guesser_matches_the_framework():
    pool, clues, vecs, _ = world(3)
    g = bench.FrameworkGuesser("glove", as_vectors(vecs))
    board = bench.framework_deal(pool, 3)
    for clue in clues[:20]:
        ref = [w.lower() for _, w in guess_reference(clue, [w.upper() for w in board.words], vecs)]
        assert g(clue, 2, board.words) == ref
    assert g("not-a-word", 2, board.words) == []


def test_guesser_breaks_ties_alphabetically():
    v = bench.Vectors(["c", "zeta", "alpha", "mid"], np.array([[1.0, 0], [1, 1], [1, 1], [0, 1]]))
    assert bench.FrameworkGuesser("t", v)("c", 3, ("zeta", "mid", "alpha")) == [
        "alpha",
        "zeta",
        "mid",
    ]


def test_concat_keeps_words_in_every_table():
    a = bench.Vectors(["x", "y", "z"], np.eye(3, dtype=np.float32))
    b = bench.Vectors(["z", "x"], np.ones((2, 2)))
    c = bench.Vectors.concat([a, b])
    assert c.words == ["x", "z"] and c.matrix.shape == (2, 5) and c.matrix.dtype == np.float64


def tiny_board():
    words = tuple(f"w{i:02d}" for i in range(25))
    kinds = tuple([TEAM] * 9 + [OPPONENT] * 8 + [NEUTRAL] * 7 + [ASSASSIN])
    return Board(words, kinds)


def fixed(order):
    """A guesser that always picks in this order."""
    return lambda clue, n, words: [w for w in order if w in words]


def test_scores():
    b = tiny_board()
    win = play(b, lambda _: bench.Clue("c", 1), fixed(b.words))
    assert win.won and win.turns == 9 and bench.score(win, b) == 9
    b = tiny_board()
    assassin = play(b, lambda _: bench.Clue("c", 1), fixed([b.words[24]]))
    assert bench.loss_kind(assassin, b) == ASSASSIN
    assert bench.score(assassin, b) == bench.score(assassin, b, "table") == 25
    b = tiny_board()
    blue = play(b, lambda _: bench.Clue("c", 1), fixed(b.words[9:17]))
    assert bench.loss_kind(blue, b) == OPPONENT and blue.turns == 8
    assert bench.score(blue, b) == 25 and bench.score(blue, b, "table") == 8
    assert bench.transcript(blue)[0] == ("c", 1, ["W09"])


def test_word2vec_reader():
    rows = {"cat": [1.0, 2.0], "dog": [3.0, -1.5], "über": [0.5, 0.25]}
    raw = io.BytesIO()
    raw.write(b"3 2\n")
    for w, v in rows.items():
        raw.write(w.encode() + b" " + struct.pack("<2f", *v) + b"\n")
    got = dict(bench_data.read_word2vec(io.BytesIO(raw.getvalue())))
    assert set(got) == set(rows)
    assert all(np.array_equal(got[w], np.float32(v)) for w, v in rows.items())
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb") as fh:
        fh.write(raw.getvalue())
    got = dict(
        bench_data.read_word2vec(
            io.BufferedReader(gzip.GzipFile(fileobj=io.BytesIO(gz.getvalue()))), {"dog"}
        )
    )
    assert list(got) == ["dog"]


def test_glove_parser_keeps_float64():
    v = bench_data.parse_glove(["the 0.418 0.24968 -0.41242", "of 0.70853 0.57088 -0.4716"])
    assert v.words == ["the", "of"] and v.matrix.dtype == np.float64
    assert v.matrix[0, 1] == 0.24968


def test_published_transcript():
    moves = {
        "clue_0": {"clue": "imperial", "clue_num": "2"},
        "guess_1": {"guess_answer": "RULER"},
        "guess_2": {"guess_answer": "CHINA"},
        "clue_3": {"clue": "creek", "clue_num": "1"},
        "guess_4": {"guess_answer": "BEACH"},
    }
    assert bench_data.published_transcript(moves) == [
        ("imperial", 2, ["RULER", "CHINA"]),
        ("creek", 1, ["BEACH"]),
    ]


def synthetic_space(seed=0, n=120, dims=16):
    from codenames.players import Space

    rng = np.random.default_rng(seed)
    words = [f"w{i:03d}" for i in range(n)]
    v = rng.standard_normal((n, dims)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return Space("synthetic", v, v, words), words


def test_policy_actions_match_rl_on_a_fresh_board():
    from codenames import bench_play, rl

    sp, words = synthetic_space()
    board = bench.framework_deal(words[:60], 4)
    pool = tuple(words[60:])
    a_clues, a_phi = rl.actions(sp, 0.05, board, pool, 10)
    b_clues, b_phi = bench_play.policy_actions(sp, 0.05, board, pool, 10)
    assert a_clues == b_clues and np.array_equal(a_phi, b_phi)
    # With two team words left, only counts 1 and 2 are offered.
    team = [w for w in board.words if board.kind(w) == TEAM]
    board.revealed |= set(team[:7])
    clues, phi = bench_play.policy_actions(sp, 0.05, board, pool, 10)
    assert {n for _, n in clues} == {1, 2} and phi.shape == (len(clues), len(rl.FEATURES))


def test_bootstrap_interval_and_summary():
    from codenames import bench_play

    s = bench_play.summarize([5.0] * 50)
    assert s == {"mean": 5.0, "ci": [5.0, 5.0], "n": 50}
    x = np.r_[np.full(50, 4.0), np.full(50, 6.0)]
    lo, hi = bench_play.boot(x)
    assert lo < 5.0 < hi and hi - lo == pytest.approx(2 * 1.96 * 1.0 / 10, abs=0.06)


def test_shares_form():
    assert bench.shares_form("residency", "residence")
    assert bench.shares_form("governing", "governor")
    assert bench.shares_form("teaching", "teacher")
    assert bench.shares_form("fire", "firefly")  # containment, at any length
    assert not bench.shares_form("tiger", "tight")  # four letters in common
    assert not bench.shares_form("ocean", "water")
    assert bench.flagged("teaching", ["dog", "teacher"]) and not bench.flagged("teaching", ["dog"])


def test_strict_codemaster_skips_flagged_clues():
    pool, clues, vecs, forms = world(2)
    board = bench.framework_deal(pool, 2)
    loose = bench.FrameworkCodemaster("glove", as_vectors(vecs), clues, forms)
    strict = bench.FrameworkCodemaster("glove", as_vectors(vecs), clues, forms, strict=True)
    _, illegal_loose = loose.board_tables(board)
    _, illegal_strict = strict.board_tables(board)
    assert (illegal_strict >= illegal_loose).all()
    clue = strict(board).word
    assert not bench.flagged(clue, board.unrevealed())
