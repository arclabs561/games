from codenames.c4 import score_reply
from codenames.game import Board


def test_score_only_explicit_guesses():
    board = Board(("whale", "ship", "pen"), ("T", "T", "N"))
    assert score_reply(board, "whale", 2) == (1, 1)
    assert score_reply(board, "", 2) == (0, 0)
