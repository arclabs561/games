"""Plan H5: the information strategy is deterministic, and the replay reproduces games."""

from hanabi import h2, h5
from hanabi.information import play_info_game
from hanabi.rules import Rules, seeded_deck


def test_information_strategy_is_deterministic_on_the_test_deals():
    rules = Rules(players=3)
    a = [play_info_game(rules, seeded_deck(rules, s), hansim=False).score for s in range(30)]
    b = [play_info_game(rules, seeded_deck(rules, s), hansim=False).score for s in range(30)]
    assert a == b and max(a) <= 25


def test_replay_reproduces_a_stub_models_game():
    rules = Rules(players=3)
    for seed in (0, 1):
        deck = seeded_deck(rules, seed)
        # A stub that never answers: every move is the fallback.
        state, rec = h2.play_game(rules, deck, lambda _prompt: "no move", deduction=True)
        b = h5.behavior(rules, deck, rec["moves"], rec["fallback_turns"])
        assert b["score"] == rec["score"] == state.score
        assert b["fireworks"] == sum(rec["board"])
        assert b["lives_lost_by_model"] == 0
        assert sum(sum(c.values()) - c.get("misplays", 0) for c in b["seats"]) == rec["turns"]


def test_last_move_and_first_prompts(tmp_path):
    assert h5.last_move("I think PLAY 2... no, hint 1 red") == "HINT 1 RED"
    assert h5.last_move("no move here") is None
    import json

    for seed, prompts in ((1001, ["c", "d"]), (1000, ["a", "b"])):
        (tmp_path / f"seed-{seed}.jsonl").write_text(
            "".join(json.dumps({"prompt": p}) + "\n" for p in prompts)
        )
    assert h5.first_prompts(tmp_path, 3) == ["a", "b", "c"]
