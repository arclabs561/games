"""Boggle word search (boggle/board.py, which boggle.ipynb imports)."""

import random
from collections import Counter

from boggle.board import (
    CUBES,
    count_words,
    get_words,
    hoeffding_boards,
    load_words,
    make_trie,
    points,
    rand_board,
    sample_boards,
    wilson,
)


def test_qu_face_is_matched_as_one_unit():
    board = [
        ["qu", "i", "t", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
    ]
    trie = make_trie(["quit", "quite", "qit"])
    assert set(get_words(board, trie, min_len=3)) == {"quit"}


def test_single_letter_faces_still_match():
    board = [
        ["c", "a", "t", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
    ]
    trie = make_trie(["cat", "act", "tac", "dog"])
    assert set(get_words(board, trie)) == {"cat", "tac"}


def test_a_cube_is_used_once_but_a_letter_may_repeat():
    board = [
        ["e", "y", "e", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
    ]
    assert set(get_words(board, make_trie(["eye", "eyey"]))) == {"eye"}


def test_random_board_uses_each_cube_once():
    cubes = [(letter,) * 6 for letter in "abcdefghijklmnop"]  # one letter per cube
    board = rand_board(random.Random(0), cubes)
    assert sorted(face for row in board for face in row) == list("abcdefghijklmnop")
    assert rand_board(random.Random(0)) == rand_board(random.Random(0))
    assert all(len(row) == 4 for row in rand_board(random.Random(1), CUBES))


def test_count_words_matches_a_serial_count_in_any_number_of_processes():
    trie = make_trie(["tee", "toe", "ten", "net", "tea", "set", "eat", "the"])
    boards = sample_boards(40, seed=1)
    one = count_words(boards, trie, min_len=3, processes=1)
    two = count_words(boards, trie, min_len=3, processes=2)
    serial = Counter()
    for board in boards:
        serial.update(get_words(board, trie, 3))
    assert one == two == serial and sum(one.values()) > 0


def test_sample_boards_is_one_seeded_stream():
    rng = random.Random(20161126)
    assert sample_boards(3, 20161126) == [rand_board(rng) for _ in range(3)]


def test_points_wilson_and_hoeffding():
    assert [points(w) for w in ("at", "tea", "teen", "teens", "tenets", "setters", "teenager")] == [
        0,
        1,
        1,
        2,
        3,
        5,
        11,
    ]
    low, high = wilson(50, 100)
    assert abs((low + high) / 2 - 0.5) < 1e-12 and 0.40 < low < high < 0.60
    assert hoeffding_boards(0.01, 0.1) == 14979
    assert hoeffding_boards(0.01, 0.1, words=20) > 14979


def test_load_words_checks_the_digest(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("Tea\nteen\nx-ray\n")
    assert load_words(path) == ["tea", "teen"]
    try:
        load_words(path, sha256="0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong digest must be rejected")
