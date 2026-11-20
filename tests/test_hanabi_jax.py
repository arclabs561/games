"""Hanabi, plan H4: JaxMARL's two-color game against rules.py, and the IPPO loop end to end.
Skipped where JAX is not installed."""

import json

import pytest

pytest.importorskip("jax")
pytest.importorskip("jaxmarl")

from hanabi import h4  # noqa: E402
from hanabi.jax_bridge import H4_RULES  # noqa: E402
from hanabi.rules import seeded_deck  # noqa: E402


def test_jaxmarl_deals_as_rules_py_deals():
    from hanabi.jax_bridge import from_jaxmarl_state, make_env, to_jaxmarl

    env = make_env(H4_RULES)
    for seed in range(400_000, 400_010):
        deck = seeded_deck(H4_RULES, seed)
        _, state = env.reset_from_deck_of_pairs(to_jaxmarl(H4_RULES, deck))
        assert from_jaxmarl_state(H4_RULES, state) == deck


def test_engines_agree_on_random_play():
    """The agreement test of plan H4: 1,000 games of random legal play, and 1,000 of
    careful random play that reach the end of the deck, on seeded decks and on JaxMARL's
    own deals. Legal moves, game length and score must agree at every turn."""
    from hanabi.jax_bridge import agreement_test

    report = agreement_test(1000)
    for name in ("seeded", "jaxmarl", "seeded-careful", "jaxmarl-careful"):
        assert report[name]["agree"], report[name]["problems"]
    assert report["seeded"]["ends"]["lives_out"] > 500
    assert report["seeded-careful"]["ends"]["deck_out"] == 1000
    assert report["seeded-careful"]["ends"]["top_score"] > 500


def test_training_and_evaluation_run_end_to_end(tmp_path):
    from hanabi import ippo

    cfg = ippo.Config(num_envs=4, num_steps=8, total_steps=4 * 8 * 2, fc_dim=8, hidden_dim=8)
    rec = ippo.train(cfg, [1, 2], tmp_path / "run", chunk=1)
    assert rec["updates_done"] == 2 and (tmp_path / "run" / "params-seed2.msgpack").exists()
    ippo.evaluate([str(tmp_path / "run")], [400_000, 400_031], tmp_path / "eval")
    h4.write_summary(tmp_path / "eval", [str(tmp_path / "run")])
    summary = json.loads((tmp_path / "eval" / "h4.json").read_text())
    assert summary["greedy"]["replayed_in_rules_py"] == 2 * 2 * 2
    assert len(summary["greedy"]["matrix"]) == 2
    assert summary["ceiling_mean"] == pytest.approx((10 + 9) / 2)
    assert set(summary["curves"]) == {"0", "1"}
