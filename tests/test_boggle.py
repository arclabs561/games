"""Boggle word search, loaded from the notebook's code cells."""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "boggle" / "boggle.ipynb"


def _load_boggle():
    ns = {}
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for cell in cells:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and ("def make_trie" in src or "def get_words" in src):
            exec(compile(src, str(NOTEBOOK), "exec"), ns)
    return ns["make_trie"], ns["get_words"]


def test_qu_face_is_matched_as_one_unit():
    make_trie, get_words = _load_boggle()
    board = [
        ["qu", "i", "t", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
    ]
    trie = make_trie(["quit", "quite", "qit"])
    assert set(get_words(board, trie, min_len=3)) == {"quit"}


def test_single_letter_faces_still_match():
    make_trie, get_words = _load_boggle()
    board = [
        ["c", "a", "t", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
        ["x", "x", "x", "x"],
    ]
    trie = make_trie(["cat", "act", "tac", "dog"])
    assert set(get_words(board, trie)) == {"cat", "tac"}
