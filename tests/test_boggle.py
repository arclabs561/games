"""Boggle word search (boggle/board.py, which boggle.ipynb imports)."""

import random
from collections import Counter
from fractions import Fraction
from itertools import permutations, product
from math import factorial, prod
from pathlib import Path

import pytest

from boggle import anneal, figures
from boggle.board import (
    CUBES,
    all_cube_losses,
    count_words,
    cube_losses,
    face_counts,
    find_path,
    get_words,
    hill_climb,
    hoeffding_boards,
    load_words,
    make_trie,
    one_square_changes,
    orientations,
    parse_board,
    permanent,
    points,
    rand_board,
    rand_layout,
    roll_chance,
    roll_ways,
    sample_boards,
    sample_layouts,
    score,
    show,
    wilson,
    word_slots,
    words_by_paths,
)

HERE = Path(__file__).parent.parent / "boggle"


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


def brute_permanent(matrix):
    """The permanent from its definition: a sum over all n! permutations."""
    n = len(matrix)
    return sum(prod(matrix[i][p[i]] for i in range(n)) for p in permutations(range(n)))


def test_permanent_matches_its_definition():
    assert permanent([[1, 2], [3, 4]]) == 1 * 4 + 2 * 3 == 10
    assert permanent([[0]]) == 0 and permanent([[5]]) == 5
    rng = random.Random(0)
    for n in range(1, 8):
        for _ in range(10):
            m = [[rng.randint(0, 6) for _ in range(n)] for _ in range(n)]
            assert permanent(m) == brute_permanent(m)
    assert permanent([[1] * 8 for _ in range(8)]) == factorial(8)


def test_roll_ways_counts_every_shake_of_a_small_set():
    """Three cubes in three slots: enumerate all 3! * 6**3 shakes and count each outcome."""
    cubes = [
        ("a", "a", "b", "c", "d", "e"),
        ("a", "b", "b", "b", "c", "c"),
        ("c", "c", "c", "a", "a", "qu"),
    ]
    outcomes = Counter()
    for order in permutations(cubes):
        for faces in product(*order):
            outcomes[faces] += 1
    assert sum(outcomes.values()) == factorial(3) * 6**3
    for faces in product("abcdq", repeat=3):
        faces = tuple("qu" if f == "q" else f for f in faces)
        assert permanent(face_counts(list(faces), cubes)) == outcomes[faces]


def test_orientations_and_roll_chance():
    best = parse_board("perslatgsineters")
    assert len(orientations(best)) == 8
    assert len(orientations(parse_board("a" * 16))) == 1
    assert len(orientations(parse_board("abba" + "bccb" * 2 + "abba"))) == 1
    # "abcd" * 4 is its own top-to-bottom mirror image, so it has four orientations, not eight.
    assert len(orientations(parse_board("abcd" * 4))) == 4
    assert parse_board("q" + "a" * 15)[0][0] == "qu"
    # The count depends only on which faces are wanted, so every orientation gives the same.
    assert {roll_ways(b) for b in orientations(best)} == {1_054_080}
    assert roll_chance(best) == 8 * roll_chance(best, turned=False)
    assert roll_chance(best) == Fraction(8 * 1_054_080, factorial(16) * 6**16)
    assert roll_ways(parse_board("z" * 16)) == 0  # one cube has a z


def test_roll_chances_add_up_on_a_small_set():
    """Every outcome of two two-faced cubes (padded to six faces) is some board."""
    cubes = [("a", "b") * 3, ("c", "d") * 3]
    boards = {tuple(faces) for order in permutations(cubes) for faces in product(*order)}
    total = sum(Fraction(permanent(face_counts(list(b), cubes)), 2 * 36) for b in boards)
    assert total == 1


def test_cube_losses_match_blanking_each_cube():
    trie = make_trie(load_words("boggle/enable1.txt"))
    rng = random.Random(3)
    for _ in range(5):
        order, board = rand_layout(rng)
        assert set(word_slots(board, trie)) == set(get_words(board, trie))
        losses = cube_losses(order, board, trie)
        for k in range(16):
            blanked = [row[:] for row in board]
            blanked[k // 4][k % 4] = "."
            assert score(board, trie) - score(blanked, trie) == losses[order[k]]
    layouts = sample_layouts(4, seed=2)
    assert [b for _, b in layouts] == sample_boards(4, seed=2)
    assert all_cube_losses(layouts, trie, processes=2) == [
        cube_losses(o, b, trie) for o, b in layouts
    ]


def test_best_board_scores():
    trie = make_trie(w for w in load_words("boggle/enable1.txt") if len(w) >= 3)
    best = parse_board("perslatgsineters")
    found = set(get_words(best, trie))
    assert len(found) == 1044 and score(best, trie) == 3623
    assert "tenge" not in found and "tenge" in set(get_words(best, make_trie(["tenge"])))


def brute_possible(word, cubes):
    """Whether some choice of distinct cubes shows the word's faces, by trying them all."""
    from boggle.board import word_faces

    faces = word_faces(word)
    if faces is None or len(faces) > len(cubes):
        return False
    return any(
        all(f in cubes[c] for f, c in zip(faces, pick, strict=True))
        for pick in permutations(range(len(cubes)), len(faces))
    )


def test_possible_matches_trying_every_choice_of_cubes():
    from boggle.board import possible

    rng = random.Random(4)
    for _ in range(300):
        cubes = [tuple(rng.choice("abcq") for _ in range(6)) for _ in range(rng.randint(1, 5))]
        cubes = [tuple("qu" if f == "q" else f for f in cube) for cube in cubes]
        word = "".join(rng.choice(["a", "b", "c", "qu", "q"]) for _ in range(rng.randint(1, 6)))
        assert possible(word, cubes) == brute_possible(word, cubes), (word, cubes)


def test_possible_words_have_witness_boards():
    from boggle.board import CLASSIC_CUBES, possible, roll_ways, witness_board

    words = sorted({w for w in load_words("boggle/enable1.txt") if len(w) >= 3})
    for cubes in (CUBES, CLASSIC_CUBES):
        for word in words[::997]:
            if possible(word, cubes):
                board = witness_board(word, cubes)
                assert roll_ways(board, cubes) > 0
                assert list(get_words(board, make_trie([word]))) == [word]
            elif len(word) <= 4:
                assert not brute_possible(word, cubes)


def test_obstructions_are_hall_violations():
    from boggle.board import CLASSIC_CUBES, obstruction, possible, word_faces

    words = sorted({w for w in load_words("boggle/enable1.txt") if len(w) >= 3})
    for cubes in (CUBES, CLASSIC_CUBES):
        for word in words[::101]:
            why = obstruction(word, cubes)
            assert (why == ()) == possible(word, cubes), word
            if why and why not in (("q",), ("length",)):
                faces = word_faces(word) or []
                need = sum(f in why for f in faces)
                have = sum(any(f in cube for f in why) for cube in cubes)
                assert need > have, (word, why)
    assert obstruction("jazz") == ("z",) and obstruction("fake") == ("f", "k")
    assert obstruction("qat") == ("q",) and obstruction("a" * 17) == ("length",)


def test_possible_word_counts_and_the_qu_rule():
    from boggle.board import CLASSIC_CUBES, possible, word_faces

    one = {w for w in load_words("boggle/enable1.txt") if len(w) >= 3}
    two = {w for w in load_words("boggle/enable2k.txt") if len(w) >= 3}
    assert len(one) == 172_723 and len(two) == 173_432
    # Vanderkam's hybrid-boggle wordlists/README.md: 173,402 ENABLE2K words are "Boggle OK".
    assert sum(word_faces(w) is not None for w in two) == 173_402
    assert sum(possible(w) for w in one) == 160_719
    assert sum(possible(w, CLASSIC_CUBES) for w in one) == 169_378
    assert sum(possible(w) for w in two) == 161_361
    assert sum(possible(w, CLASSIC_CUBES) for w in two) == 170_084


def test_classic_cubes_and_the_best_board():
    from boggle.board import CLASSIC_CUBES

    assert len(CLASSIC_CUBES) == 16 and all(len(c) == 6 for c in CLASSIC_CUBES)
    assert sum(c.count("qu") for c in CLASSIC_CUBES) == 1
    best = parse_board("perslatgsineters")
    assert roll_ways(best, CLASSIC_CUBES) == 221_616
    assert roll_chance(best, CLASSIC_CUBES) == Fraction(8 * 221_616, factorial(16) * 6**16)


def test_best_board_scores_with_enable2k():
    trie = make_trie(w for w in load_words("boggle/enable2k.txt") if len(w) >= 3)
    best = parse_board("perslatgsineters")
    assert len(set(get_words(best, trie))) == 1045 and score(best, trie) == 3625


def test_boards_of_any_size_and_show():
    board = parse_board("abcde fghij klmno pqrst uvwxy")
    assert len(board) == 5 and board[3][1] == "qu"
    assert set(get_words(board, make_trie(["agm", "ysx", "eij"]))) == {"agm", "ysx", "eij"}
    assert set(get_words(board, make_trie(["ay", "abc"]), min_len=2)) == {"abc"}  # no wrapping
    assert show(parse_board("qest ansi xweo yrsn")).splitlines()[0] == "Qu E  S  T"


def test_find_path_spells_the_word_along_neighbors():
    board = parse_board("qest ansi xweo yrsn")
    path = find_path(board, "quest")
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert find_path(board, "tsetse") is None  # would need a cube twice
    for word in ("swan", "wrens", "session"):
        path = find_path(board, word)
        assert path is not None and len(set(path)) == len(path)
        assert "".join(board[i][j] for i, j in path) == word
        assert all(
            max(abs(a - c), abs(b - d)) == 1 for (a, b), (c, d) in zip(path, path[1:], strict=False)
        )


def test_words_by_paths_agrees_with_get_words():
    words = load_words("boggle/enable1.txt")
    trie = make_trie(w for w in words if len(w) >= 3)
    for seed in range(3):
        board = rand_board(random.Random(seed))
        fast = {w for w in get_words(board, trie) if len(w) <= 6}
        assert words_by_paths(board, {w for w in words if len(w) >= 3}, 6) == fast


def test_norvig_boards_score_as_he_reports():
    # Norvig, pytudes ipynb/Boggle.ipynb (commit 3b4c5dd): points and words per board.
    trie = make_trie(w for w in load_words("boggle/enable1.txt") if len(w) >= 3)
    for letters, expected in {
        "PUZZL WORDE BOGGL SEARC FINDH": (449, 252),
        "QEST ANSI XWEO YRSN": (245, 125),
        "RSTCS DEIAE GNLRP EATES MSSID": (10_112, 2_290),
    }.items():
        found = set(get_words(parse_board(letters), trie))
        assert (sum(map(points, found)), len(found)) == expected


def test_hill_climb_never_loses_points_and_is_seeded():
    trie = make_trie(["tea", "tee", "ten", "net", "toe", "eaten", "teen"])
    start = rand_board(random.Random(3))
    board, history = hill_climb(start, trie, 60, random.Random(5))
    assert history == sorted(history) and len(history) == 61
    assert history[-1] == score(board, trie)
    assert (board, history) == hill_climb(start, trie, 60, random.Random(5))
    assert start == rand_board(random.Random(3))  # the start is not changed


def test_one_square_changes_rescore_each_variant():
    trie = make_trie(["cat", "cot", "act", "tact"])
    board = parse_board("catx xxxx xxxx xxxx")
    changes = one_square_changes(board, trie, processes=2)
    assert len(changes) == 16 * 25
    assert changes[(0, 0, "t")] == score(parse_board("tatx xxxx xxxx xxxx"), trie)
    assert changes[(0, 3, "y")] == score(board, trie)
    assert changes[(1, 0, "o")] == score(parse_board("catx oxxx xxxx xxxx"), trie)


def test_committed_svgs_are_what_the_script_writes(tmp_path):
    figures.main(tmp_path)
    for name in figures.FIGURES:
        assert (tmp_path / name).read_bytes() == (figures.HERE / "figures" / name).read_bytes(), (
            name
        )


@pytest.mark.skipif(anneal.compiler_missing(), reason="anneal.c needs a C compiler")
def test_c_scorer_agrees_with_get_words():
    words = [w for w in load_words(HERE / "enable1.txt") if len(w) >= 3]
    trie = make_trie(words)
    rng = random.Random(5)
    boards = ["rstcsdeiaegnlrpeatesmssid", "ligdrmanesietildsracsepes", "quqqquqqqquqqqqquuuuuuuuu"]
    boards += ["".join(rng.choice("aeeinorstlqdxz") for _ in range(25)) for _ in range(30)]
    boards += [
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(16)) for _ in range(30)
    ]
    for letters in boards:
        assert anneal.c_score(letters) == anneal.py_score(letters, trie), letters


@pytest.mark.skipif(anneal.compiler_missing(), reason="anneal.c needs a C compiler")
def test_boyans_board_resists_every_change_of_one_square_and_every_swap():
    # 625 one-square changes and 300 swaps, each scored; none beats 10,112.
    assert anneal.c_climb("rstcsdeiaegnlrpeatesmssid") == (10112, 2290, "rstcsdeiaegnlrpeatesmssid")
    # Vanderkam's best 5 by 5 board for ENABLE2K, scored with enable1.txt, beats it.
    assert anneal.c_score("ligdrmanesietildsracsepes") == (10397, 2338)
    assert anneal.c_score("ligdrmanesietildsracsepes", HERE / "enable2k.txt") == (10406, 2344)


@pytest.mark.skipif(anneal.compiler_missing(), reason="anneal.c needs a C compiler")
def test_annealing_is_seeded():
    assert anneal.anneal(7, 1, steps=2000) == anneal.anneal(7, 1, steps=2000)
