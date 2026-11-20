"""Tests for codenames/split.py and the clustered bootstrap of codenames/heldout.py.

The rows are made up; nothing here downloads the data.
"""

import json
import re
from collections import Counter

import numpy as np
import pytest

from codenames.split import (
    SPLITS_FILE,
    Row,
    assign,
    build_turns,
    first_refs,
    from_refs,
    game_keys,
    game_split,
    games,
    half,
    player_id,
    player_split,
    shuffled_split,
    to_refs,
)
from codenames.stats import (
    cluster_draws,
    clustered,
    clustered_kendall,
    exact_distribution,
    kendall_tau_b,
    replicates,
)

SURVEY = "marriage: single, education: bachelor, race: white, age: 22 - 30"
TRAITS = "care: 4.0, fairness: 3.5, political: liberal, openness: 3.0"


def profile(age: str, survey: str = SURVEY) -> str:
    return f"{survey}, age: {age}, gender: female, country: united states, native: true, {TRAITS}"


BOARD_A = tuple(f"a{i}" for i in range(25))
BOARD_B = tuple(f"b{i}" for i in range(25))
ANN, BEN = profile("25"), profile("40", "marriage: married, education: graduate, race: asian")


def row(i, words, giver, guesser, picked=None, clue="hint"):
    return Row("train", i, tuple(words), clue, picked or words[0], giver, guesser)


def test_player_id_ignores_the_typed_fields_but_not_the_survey():
    assert player_id(profile("25")) == player_id(profile("26"))
    assert player_id(profile("25")) != player_id(profile("25", "marriage: divorced"))
    typed_only = "None, age:  56, gender: female, country: united states, native: true, None"
    assert player_id(typed_only).startswith("p")
    assert player_id(profile("25")).startswith("s")


def test_both_sides_of_a_game_get_one_id_and_other_games_another():
    rows = [
        row(0, BOARD_A, ANN, BEN),  # Ann gives, side one
        row(1, BOARD_A[1:], ANN, BEN),
        row(2, BOARD_A, BEN, ANN),  # Ben gives, side two, same 25 words
        row(3, BOARD_A[3:], BEN, ANN),
        row(4, BOARD_B, ANN, BEN),  # a second game of the same pair
        row(5, BOARD_B[:20], BEN, ANN),
        row(6, BOARD_A, ANN, profile("30", "marriage: widowed")),  # board A, another pair
    ]
    g = games(rows)
    assert g[0] == g[1] == g[2] == g[3]
    assert g[4] == g[5] != g[0]
    assert g[6] not in (g[0], g[4])


def test_a_turn_on_no_board_or_on_two_boards_is_an_error():
    with pytest.raises(ValueError):
        games([row(0, BOARD_A, ANN, BEN), row(1, BOARD_B[:5], ANN, BEN)])
    other = BOARD_A[:12] + BOARD_B[12:]  # a second board of the pair, sharing 12 words
    with pytest.raises(ValueError):
        games([row(0, BOARD_A, ANN, BEN), row(1, other, ANN, BEN), row(2, BOARD_A[:12], ANN, BEN)])


def test_turns_keep_one_word_picks_only():
    rows = [row(0, BOARD_A, ANN, BEN), row(1, BOARD_A[1:], ANN, BEN, picked="a1, a2")]
    t = build_turns(rows)
    assert len(t) == 1 and t[0].guess.picked == "a0" and t[0].guess.words == BOARD_A


def many_turns(n_games=40, players=12, seed=0):
    rng = np.random.default_rng(seed)
    people = [profile(str(20 + i), f"marriage: m{i}") for i in range(players)]
    rows = []
    for k in range(n_games):
        board = tuple(f"g{k}w{i}" for i in range(25))
        a, b = rng.choice(players, 2, replace=False)
        for j in range(int(rng.integers(2, 8))):
            giver, guesser = (people[a], people[b]) if j % 2 else (people[b], people[a])
            rows.append(row(len(rows), board[j:] if j else board, giver, guesser))
        rows.append(row(len(rows), board, people[b], people[a]))
    return build_turns(rows)


def test_shuffled_split_is_a_seeded_partition_of_the_games():
    t = many_turns()
    s = shuffled_split(t, seed=3)
    assert shuffled_split(t, seed=3) == s != shuffled_split(t, seed=4)
    ids = s["train"] + s["val"] + s["test"]
    assert sorted(ids) == sorted({x.game for x in t}) and len(set(ids)) == len(ids)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (32, 4, 4)


def test_plan_split_follows_the_crc_of_the_game_key():
    rows = [row(0, BOARD_A, ANN, BEN), row(1, BOARD_A[2:], BEN, ANN)]
    key = game_keys(rows)[0]
    assert key == "|".join(sorted(BOARD_A)) + "#" + "#".join(sorted([ANN, BEN]))
    t = many_turns(400)
    labels = Counter(game_split(x.key) for x in {x.game: x for x in t}.values())
    assert set(labels) == {"train", "val", "test"} and 0.7 < labels["train"] / 400 < 0.9
    assert all(game_split(x.key) == game_split(x.key) for x in t)  # no seed, no state
    assert {half(x) for x in t} == {"held", "sibling"}


def test_player_split_keeps_held_out_people_out_of_training():
    t = many_turns()
    s = player_split(t, seed=1, frac=0.15)
    labels = assign(t, "player", s)
    held = set(s["test"]) | set(s["val"])
    train = [x for x, lab in zip(t, labels, strict=True) if lab == "train"]
    test = [x for x, lab in zip(t, labels, strict=True) if lab == "test"]
    assert train and test
    assert not {p for x in train for p in (x.giver, x.guesser)} & held
    assert all(x.guesser in s["test"] for x in test)
    assert not {x.game for x in train} & {x.game for x in test}
    assert sum(lab == "test" for lab in labels) >= 0.15 * len(t)


def test_positions_stand_for_the_same_games_and_players_as_the_ids():
    t = many_turns()
    for scheme, ids in (
        ("shuffled", shuffled_split(t, seed=2)),
        ("player", player_split(t, 1, 0.15)),
    ):
        refs = to_refs(t, scheme, ids)
        assert from_refs(t, scheme, refs) == ids
        assert assign(t, scheme, from_refs(t, scheme, refs)) == assign(t, scheme, ids)
    game, guesser = first_refs(t)
    assert len(set(game.values())) == len(game) and len(set(guesser.values())) == len(guesser)
    first = t[0]
    assert game[first.game] == guesser[first.guesser] == f"{first.source}:{first.row}"


def test_committed_splits_hold_positions_and_no_hashes():
    """splits.json names games and players by "file:row" only, never by a hash of the
    data: a hash of a typed-fields-only profile can be inverted by enumeration."""
    text = SPLITS_FILE.read_text()
    s = json.loads(text)
    lists = [v for scheme in ("game", "shuffled", "player") for v in s[scheme].values()]
    assert all(re.fullmatch(r"(train|val|test):\d+", r) for refs in lists for r in refs)
    assert not re.search(r"[0-9a-f]{8}", text)


def test_clustered_interval_matches_explicit_resampling():
    games_of = ["g1", "g1", "g2", "g3", "g3", "g3"]
    values = [1, 0, 1, 0, 0, 1]
    index, counts = cluster_draws(games_of, draws=200, seed=5)
    got = clustered(values, index, counts)
    by_game = {
        g: [v for x, v in zip(games_of, values, strict=True) if x == g] for g in "g1 g2 g3".split()
    }
    reps = []
    for c in counts:
        drawn = [
            v for g, k in zip(sorted(by_game), c, strict=True) for _ in range(k) for v in by_game[g]
        ]
        reps.append(np.mean(drawn))
    assert got["ci"] == pytest.approx(list(np.percentile(reps, [2.5, 97.5])))
    assert got["mean"] == pytest.approx(0.5)


def test_clustering_widens_the_interval_when_games_agree_within():
    rng = np.random.default_rng(0)
    per_game = rng.integers(0, 2, 60)
    values = np.repeat(per_game, 10)  # every turn of a game has the same outcome
    games_of = [f"g{i}" for i in range(60) for _ in range(10)]
    singles = [f"t{i}" for i in range(600)]
    wide = clustered(values, *cluster_draws(games_of, 2000))
    narrow = clustered(values, *cluster_draws(singles, 2000))
    assert wide["ci"][1] - wide["ci"][0] > 2.5 * (narrow["ci"][1] - narrow["ci"][0])


def test_committed_splits_are_disjoint_and_add_up():
    s = json.loads(SPLITS_FILE.read_text())
    for scheme in ("game", "shuffled"):
        g = s[scheme]
        assert not set(g["train"]) & set(g["test"]) and not set(g["val"]) & set(g["test"])
        assert len(g["train"]) + len(g["val"]) + len(g["test"]) == s["games"]
    assert not set(s["player"]["test"]) & set(s["player"]["val"])
    for scheme, c in s["counts"].items():
        assert sum(c.values()) == s["turns"], scheme
    assert Counter(s["counts"]["published"]) == Counter(train=5407, val=376, test=305)
    assert s["counts"]["game"] == {"test": 551, "train": 4820, "val": 717}


def test_draws_match_the_exact_bootstrap_on_three_and_four_games():
    for games_of, values in (
        (["a", "a", "b", "c", "c", "c"], [1, 0, 1, 0, 0, 1]),
        (["a", "b", "b", "c", "d", "d", "d"], [1, 1, 0, 0, 1, 1, 0]),
    ):
        exact = exact_distribution(values, games_of)
        assert sum(exact.values()) == pytest.approx(1.0)
        reps = replicates(values, *cluster_draws(games_of, draws=200_000, seed=1))
        means = np.array(sorted(exact))
        probs = np.array([exact[m] for m in sorted(exact)])
        assert reps.mean() == pytest.approx((means * probs).sum(), abs=2e-3)
        assert reps.std() == pytest.approx(
            np.sqrt((probs * (means - (means * probs).sum()) ** 2).sum()), abs=2e-3
        )
        for q in (0.025, 0.975):  # percentiles of the exact distribution
            exact_q = means[np.searchsorted(np.cumsum(probs), q)]
            assert np.quantile(reps, q, method="inverted_cdf") == pytest.approx(exact_q)


def test_paired_differences_resample_both_guessers_together():
    games_of = ["a", "a", "b", "c"]
    x, y = np.array([1, 1, 0, 1]), np.array([0, 1, 0, 1])
    draws = cluster_draws(games_of, 1000)
    d = clustered(x - y, *draws)
    assert np.allclose(replicates(x - y, *draws), replicates(x, *draws) - replicates(y, *draws))
    assert d["mean"] == pytest.approx(0.25) and d["ci"][0] >= 0


def test_kendall_tau_b():
    assert kendall_tau_b([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau_b([1, 2, 3], [30, 20, 10]) == -1.0
    assert kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    assert kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / np.sqrt(2 * 3))
    hits = np.array([[1, 0], [1, 0], [1, 1], [0, 1]])
    k = clustered_kendall(hits, [0.9, 0.1], *cluster_draws(["a", "b", "c", "d"], 500))
    assert k["tau"] == 1.0 and -1.0 <= k["ci"][0] <= k["ci"][1] <= 1.0
