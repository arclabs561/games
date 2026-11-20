"""Plan C14: strictly legal candidates, the fallback and the reading of cut replies."""

from codenames import c4, c14
from codenames.actions import read_action_lines
from codenames.legality import strict_legal
from codenames.rerank import read_choice


def test_every_candidate_is_strictly_legal_and_f_is_listed():
    lists = read_action_lines(c14.ACTIONS.read_text())
    f = c4.read_clue_lines(c14.F_LINES.read_text())
    for seed in (c14.PILOT, c14.VAL):
        for b, clues, (fc, fk, _) in zip(
            c4.boards(seed, c14.N[seed]), lists[seed], f[("F", seed)], strict=True
        ):
            assert len(clues) == 30
            assert all(strict_legal(c, b.words) for c in clues)
            assert fc in clues and 1 <= fk <= 4


def test_a_cut_reply_is_not_a_pick():
    clues = ["galaxy", "orbit"]
    assert read_choice("galaxy 2", clues, "length")["why"] == "truncated"
    assert read_choice("galaxy 2", clues, "stop")["clue"] == "galaxy"


def test_caps_and_rules():
    assert c14.CAPS == {"qwen3.8-27b": (256, 1024), "gpt-oss-120b": (1024, 2048)}
    for m in c14.CHOOSERS:
        assert c14.chooser_model(m).provider == "deepinfra"
