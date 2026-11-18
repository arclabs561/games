"""Boggle word search (boggle/boggle.py, which boggle.ipynb imports)."""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "boggle"))

from boggle import CUBES, count_words, get_words, make_trie, rand_board  # noqa: E402


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


def test_count_words_is_reproducible_across_processes():
    trie = make_trie(["tee", "toe", "ten", "net", "tea", "set", "eat", "the"])
    one = count_words(40, trie, min_len=3, processes=1)
    two = count_words(40, trie, min_len=3, processes=2)
    assert one == two and sum(one.values()) > 0
    serial = Counter()
    for i in range(40):
        serial.update(get_words(rand_board(random.Random(i)), trie, 3))
    assert one == serial
