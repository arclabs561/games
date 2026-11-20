"""Hanabi: the human games of AH2AC2, the ceiling of a game in progress, and the CLI."""

import io
import json
import random
import struct

import pytest

from hanabi.ceiling import live_states, parse_deck, parse_live, report, time_witness
from hanabi.humans import (
    Ceiling,
    analyze,
    ceiling,
    free_order_lost,
    is_critical,
    reachable,
    states_of,
    summarize,
    trajectory,
)
from hanabi.humans_data import (
    HumanGame,
    decode,
    encode,
    games_from_tensors,
    read_safetensors,
    replay,
)
from hanabi.rules import (
    Rules,
    card_name,
    discard,
    full_deck,
    hint,
    legal,
    legal_actions,
    make_card,
    new_game,
    play,
    playable,
    seeded_deck,
    step,
)
from hanabi.solver import brute_winnable, winnable
from hanabi.upper_bound import multiset_orders

# --- the dataset format ----------------------------------------------------------------------


def test_action_ids_follow_jaxmarl():
    """JaxMARL, two players, hand 5: 0-4 discard, 5-9 play, 10-14 color hints, 15-19 rank
    hints, 20 no-op. With three players the hints to the next seat come first."""
    two = Rules(players=2)
    assert decode(3, 0, two) == discard(3)
    assert decode(7, 1, two) == play(2)
    assert decode(10, 0, two) == hint(1, "color", 0)
    assert decode(19, 1, two) == hint(0, "rank", 5)
    with pytest.raises(ValueError):
        decode(20, 0, two)
    three = Rules(players=3)  # 10-14 colors to the next seat, 15-19 to the one after
    assert decode(15, 2, three) == hint(1, "color", 0)
    assert decode(12, 2, three) == hint(0, "color", 2)
    assert decode(20, 1, three) == hint(2, "rank", 1)
    assert decode(29, 1, three) == hint(0, "rank", 5)
    for rules in (two, three):
        n = 2 * rules.hand + (rules.players - 1) * (rules.colors + rules.ranks)
        for player in range(rules.players):
            assert [encode(decode(a, player, rules), player, rules) for a in range(n)] == list(
                range(n)
            )


def test_action_ids_match_the_jaxmarl_bridge():
    """The same numbering as `hanabi.jax_bridge`, which plan H4 checked against JaxMARL."""
    pytest.importorskip("numpy")
    from hanabi.jax_bridge import to_jaxmarl_action, to_rules_action

    for players in (2, 3):
        rules = Rules(players=players)
        n = 2 * rules.hand + (players - 1) * (rules.colors + rules.ranks)
        for player in range(players):
            for a in range(n):
                assert decode(a, player, rules) == to_rules_action(a, player, rules)
                assert encode(decode(a, player, rules), player, rules) == to_jaxmarl_action(
                    decode(a, player, rules), player, rules
                )


def careless(state, rng):
    """A test player who sees its own hand, plays what fits, and otherwise discards or
    hints at random."""
    hand = state.hands[state.current]
    for i, c in enumerate(hand):
        if playable(c, state.board, state.rules):
            return play(i)
    moves = [a for a in legal_actions(state) if a.kind != "play"]
    return rng.choice(moves)


def recorded(players: int, seed: int, policy=careless) -> HumanGame:
    """A game played by `policy`, stored as the dataset stores it."""
    rules = Rules(players=players)
    deck = seeded_deck(rules, seed)
    s = new_game(rules, deck)
    rng = random.Random(seed)
    ids = []
    while not s.over:
        a = policy(s, rng)
        ids.append(encode(a, s.current, rules))
        step(s, a)
    return HumanGame(seed, players, tuple(deck), tuple(ids), s.score, "test")


def safetensors_bytes(tensors: dict) -> bytes:
    header, body = {}, b""
    for name, (dtype, shape, values) in tensors.items():
        code = {"I64": "q", "I32": "i", "I8": "b"}[dtype]
        data = struct.pack(f"<{len(values)}{code}", *values)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(body), len(body) + len(data)],
        }
        body += data
    head = json.dumps(header).encode()
    return struct.pack("<Q", len(head)) + head + body


def as_tensors(games: list[HumanGame], pad: int = 89) -> dict:
    p = games[0].players
    rules = Rules(players=p)
    noop = 2 * rules.hand + (p - 1) * (rules.colors + rules.ranks)
    decks, acts = [], []
    for g in games:
        for c in g.deck:
            decks += [c // rules.ranks, c % rules.ranks]
        for t in range(pad):
            for q in range(p):
                acts.append(g.actions[t] if t < len(g.actions) and t % p == q else noop)
    return {
        "game_ids": ("I64", [len(games)], [g.game_id for g in games]),
        "scores": ("I64", [len(games)], [g.score for g in games]),
        "decks": ("I8", [len(games), 50, 2], decks),
        "actions": ("I32", [len(games), pad, p], acts),
        "num_actions": ("I64", [len(games)], [len(g.actions) for g in games]),
        "num_players": ("I64", [], [p]),
    }


@pytest.mark.parametrize("players", [2, 3])
def test_files_round_trip(tmp_path, players):
    games = [recorded(players, s) for s in range(3)]
    path = tmp_path / "games.safetensors"
    path.write_bytes(safetensors_bytes(as_tensors(games)))
    back = games_from_tensors(read_safetensors(path), "test")
    assert back == games
    for g in back:
        final = replay(g)
        assert final.over and final.score == g.score


def test_replay_rejects_an_illegal_action():
    g = recorded(2, 0)
    bad = HumanGame(g.game_id, 2, g.deck, (15,) * 9, 0, "test")  # nine rank-1 hints
    with pytest.raises(ValueError):
        replay(bad)


# --- the ceiling of a game in progress -----------------------------------------------------

SMALL = [
    Rules(players=2, colors=2, counts=(2, 1), hand_size=1, max_hints=2, lives=1),
    Rules(players=2, colors=2, counts=(2, 1), hand_size=2, max_hints=1, lives=2),
    Rules(players=3, colors=2, counts=(2, 1), hand_size=1, max_hints=1, lives=2),
]


@pytest.mark.parametrize("rules", SMALL)
def test_search_from_a_game_in_progress_agrees_with_brute_force(rules):
    """From every prefix of a random game on every deal of a small deck, the fast solver
    and brute force agree on every target."""
    rng = random.Random(1)
    for deck in multiset_orders(full_deck(rules)):
        s = new_game(rules, deck)
        while not s.over:
            for target in range(s.score + 1, rules.max_score + 1):
                fast, line = winnable(rules, deck, target=target, start=s)
                assert fast == brute_winnable(rules, deck, target, start=s), (deck, s.turn)
            step(s, rng.choice(legal_actions(s)))


@pytest.mark.parametrize("rules", SMALL)
def test_the_free_order_game_with_a_target_never_rules_out_a_reachable_score(rules):
    """From every prefix of a random game on every deal of a small deck, and for every
    target, a free-order game that cannot reach the target means brute force cannot."""
    rng = random.Random(3)
    lost = 0
    for deck in multiset_orders(full_deck(rules)):
        s = new_game(rules, deck)
        while not s.over:
            for target in range(s.score + 1, rules.max_score + 1):
                if free_order_lost(rules, deck, target, s, 0) is False:
                    lost += 1
                    assert not brute_winnable(rules, deck, target, start=s), (deck, s.turn)
            step(s, rng.choice(legal_actions(s)))
    assert lost > 0


def test_the_ceiling_never_rises_in_a_game():
    rules = Rules(players=2, colors=2, counts=(2, 1), hand_size=2, max_hints=1, lives=2)
    rng = random.Random(2)
    for deck in list(multiset_orders(full_deck(rules)))[::7]:
        s, best = new_game(rules, deck), rules.max_score
        while not s.over:
            c = ceiling(rules, deck, s)
            assert c.exact and c.low <= best
            best = c.low
            step(s, rng.choice(legal_actions(s)))
        assert s.score <= best


def test_known_deals():
    """Plan H1: two-player seed 161 is lost to the pooled game and allows 24."""
    rules = Rules(players=2)
    assert ceiling(rules, seeded_deck(rules, 161)) == Ceiling(24, 24, ("pooled",))
    assert ceiling(rules, seeded_deck(rules, 0)) == Ceiling(25, 25)


def test_critical_cards():
    rules = Rules(players=2)
    s = new_game(rules, seeded_deck(rules, 0))
    r5, r2, r1 = make_card(0, 5, rules), make_card(0, 2, rules), make_card(0, 1, rules)
    assert is_critical(r5, s)  # one copy
    assert not is_critical(r2, s)  # another copy is left
    s.discards = [r2]
    assert is_critical(r2, s)
    s.discards = [r1, r1, r1]  # red is dead: nothing red above 0 counts any more
    assert not is_critical(r5, s) and not is_critical(r2, s)
    s.board[0] = 2
    assert not is_critical(r2, s)


def five_thrower(state, rng):
    """Throws away the first 5 it holds when it can, and otherwise plays carelessly."""
    hand = state.hands[state.current]
    rules = state.rules
    for i, c in enumerate(hand):
        if c % rules.ranks == rules.ranks - 1 and legal(state, discard(i)):
            return discard(i)
    return careless(state, rng)


def test_a_discarded_five_is_the_first_drop():
    g = recorded(2, 0, five_thrower)
    states = states_of(g)
    first = next(
        t + 1
        for t in range(len(g.actions))
        if states[t + 1].discards
        and states[t + 1].discards[-1] % 5 == 4
        and len(states[t + 1].discards) > len(states[t].discards)
    )
    row = analyze(g)
    assert row["ceiling_low"] == row["ceiling_high"] == 25
    assert row["drops"][0]["turn"] == first
    assert row["drops"][0]["kind"] == "critical discard"
    assert len(row["drops"]) == 25 - g.score
    traj = trajectory(g, 25)
    assert traj is not None and sorted(traj) == [d["turn"] for d in row["drops"]]
    assert [e["kind"] for e in row["events"]].count("critical discard") >= 1
    other = analyze(recorded(2, 1))
    s = summarize([row, other])
    lost = [r["ceiling_low"] - r["score"] for r in (row, other)]
    assert s["games"] == 2 and s["below_ceiling"] == sum(lost) / 2
    assert s["drops"] == sum(lost)
    assert s["mean_gap"] == s["ruled_out_by_deal"] + s["below_ceiling"]
    assert sum(s["drops_by_phase"].values()) == s["drops"]
    assert 0 <= row["info_score"] <= 25


def test_drops_add_up_on_random_games():
    for seed in range(3):
        g = recorded(2, seed)
        row = analyze(g)
        assert row["replayed"] == g.score
        if row["ceiling_low"] == row["ceiling_high"]:
            assert len(row["drops"]) == row["ceiling_low"] - g.score
            for d in row["drops"]:  # just before a drop the ceiling is still higher
                before = states_of(g)[d["turn"] - 1]
                assert reachable(Rules(players=2), g.deck, g.score + 1, before) is not False


# --- the command line ------------------------------------------------------------------------


def live_export(game: HumanGame, style: str = "spec") -> dict:
    """The game as a hanab.live export: "spec" as the format's description writes clues,
    "site" as the site's /export pages do (receiverIndex, color or rank, variantId 0), or
    "numbered" with the older numbered actions."""
    numbered = style == "numbered"
    rules = Rules(players=game.players)
    p, h = rules.players, rules.hand
    deck = [{"suitIndex": c // 5, "rank": c % 5 + 1} for c in game.deck]
    s = new_game(rules, list(game.deck))
    held = [list(range(q * h, (q + 1) * h)) for q in range(p)]
    nxt, actions = p * h, []
    for a_id in game.actions:
        a = decode(a_id, s.current, rules)
        if a.kind in ("play", "discard"):
            o = held[s.current].pop(a.index)
            if nxt < 50:
                held[s.current].append(nxt)
                nxt += 1
            if numbered:
                actions.append({"type": 0 if a.kind == "play" else 1, "target": o})
            else:
                actions.append({"type": a.kind, "order": o})
        elif numbered:
            kind = 2 if a.attr == "color" else 3
            actions.append({"type": kind, "target": a.target, "value": a.value})
        elif style == "site":
            clue = {"type": a.attr, a.attr: a.value}
            actions.append({"type": "clue", "receiverIndex": a.target, "clue": clue})
        else:
            clue = {"type": a.attr, "value": a.value}
            actions.append({"type": "clue", "target": a.target, "clue": clue})
        step(s, a)
    names = ["Alice", "Bob", "Cathy"][:p]
    out = {"playerNames": names, "deck": deck, "actions": actions, "gameId": game.game_id}
    if style == "site":
        out["options"] = {"variantId": 0, "deckPlays": False, "detrimentalCharacters": False}
    return out


@pytest.mark.parametrize(
    ("players", "style"), [(2, "spec"), (3, "spec"), (2, "site"), (3, "site"), (3, "numbered")]
)
def test_hanab_live_exports_replay(players, style):
    g = recorded(players, 4)
    live = parse_live(json.loads(json.dumps(live_export(g, style))))
    assert live.deck == list(g.deck)
    states, acts = live_states(live)
    assert states[-1].score == g.score and states[-1].over
    assert [s.hands for s in states] == [s.hands for s in states_of(g)]


@pytest.mark.parametrize(
    "options", [{"startingPlayer": 1}, {"variantId": 1}, {"detrimentalCharacters": True}]
)
def test_exports_with_other_rules_are_refused(options):
    data = live_export(recorded(2, 4))
    data["options"] = options
    with pytest.raises(ValueError):
        parse_live(data)


def test_unsupported_variant():
    with pytest.raises(ValueError):
        parse_live(
            {"playerNames": ["a", "b"], "deck": [], "options": {"variantName": "Rainbow (6 Suits)"}}
        )


def test_parse_deck():
    rules = Rules(players=2)
    deck = seeded_deck(rules, 7)
    text = " ".join(card_name(c, rules) for c in deck)
    assert parse_deck(text, rules) == deck
    assert parse_deck(text.replace("W", "P"), rules) == deck
    with pytest.raises(ValueError):
        parse_deck(text.rsplit(" ", 1)[0], rules)


def test_report_prints_a_certificate():
    rules = Rules(players=5)
    out = io.StringIO()
    c = report(rules, seeded_deck(rules, 18105), "RYGBW", ["a"] * 5, file=out)
    assert c == Ceiling(24, 24, ("pooled", "time"))
    text = out.getvalue()
    assert "Best possible score: 24 (exact)." in text
    assert "before the 16th draw (R1)" in text
    bound, i, g = time_witness(rules, seeded_deck(rules, 18105))
    assert (bound, i) == (24, 16)


def test_report_on_a_recorded_game():
    g = recorded(2, 0, five_thrower)
    live = parse_live(live_export(g))
    states, acts = live_states(live)
    out = io.StringIO()
    report(live.rules, live.deck, "RYGBP", live.names, states, acts, file=out)
    text = out.getvalue()
    assert f"Recorded score: {g.score}" in text
    assert text.count("the ceiling fell below") == 25 - g.score
    assert "the last copy" in text


def test_report_and_figure_from_blocks(tmp_path):
    from hanabi.humans import GAMES_FILE, figure, read_games, report, summary_of

    runs = tmp_path / "runs" / "seed-0"
    runs.mkdir(parents=True)
    for p in (2, 3):
        rows = [analyze(recorded(p, s)) for s in range(3)]
        (runs / f"p{p}-0-of-1.json").write_text(json.dumps(rows))
    out = tmp_path / "humans.json"
    summary = report(tmp_path / "runs", out)
    games = read_games(tmp_path / GAMES_FILE)
    assert len(games) == 6 and json.loads(out.read_text()) == json.loads(json.dumps(summary))
    assert summary_of(games) == summary
    svg = figure(games)
    assert svg.startswith("<svg") and "2 players, 3 games" in svg


def test_committed_results_are_what_the_scripts_write():
    """The summary and the picture are recomputed from the committed per-game records."""
    from pathlib import Path

    from hanabi.humans import GAMES_FILE, figure, read_games, summary_of

    folder = Path(__file__).parent.parent / "hanabi" / "results"
    if not (folder / GAMES_FILE).exists():
        pytest.skip("no committed records yet")
    games = read_games(folder / GAMES_FILE)
    committed = json.loads((folder / "humans.json").read_text())
    assert json.loads(json.dumps(summary_of(games))) == committed
    assert (folder / "humans-ceiling.svg").read_text() == figure(games)
    two, three = committed["players"]["2"], committed["players"]["3"]
    assert (two["games"], three["games"]) == (1858, 1221)  # the dataset card's counts
    for s in (two, three):
        assert s["replay_matches"] == s["games"] and s["score_above_ceiling"] == 0
        assert s["perfect_games_with_ceiling_below_25"] == 0
