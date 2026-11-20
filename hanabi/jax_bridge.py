"""JaxMARL's Hanabi and `rules.py` side by side: the same deal in both, and an action list
from one replayed in the other (plan H4).

    uv run --group hanabi python -m hanabi.jax_bridge --games 1000     # the agreement test

JaxMARL (`jaxmarl` 0.2.0) numbers the actions of a player with a hand of h cards as:
discard slot 0 to h - 1, then play slot 0 to h - 1, then a color hint to each other player
in turn (the next seat first), then a rank hint to each, and a last action, "no move",
which only a player whose turn it is not may take. Both engines keep a hand oldest card
first and put a drawn card at the end, so a slot means the same card in both. A card is a
one-hot (color, rank) array in JaxMARL and an int in `rules.py`; `to_jaxmarl` turns a deck
into JaxMARL's (color, rank - 1) pairs, dealt in blocks from the front as `rules.new_game`
deals.

`random_games` plays games with uniformly random legal actions in JaxMARL, records each
action and the legal set at each turn, and `agree` replays them in `rules.py`. The test
asks for the same legal actions at every turn, the same game length and the same score.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from hanabi.rules import (
    Action,
    Rules,
    State,
    color_of,
    discard,
    hint,
    legal_actions,
    make_card,
    new_game,
    play,
    rank_of,
    seeded_deck,
    step,
)

H4_RULES = Rules(players=2, colors=2)  # the two-color game of plan H4: 20 cards, top score 10


def max_turns(rules: Rules) -> int:
    """An upper bound on the number of turns of a game. Each play or discard uses a card,
    and each hint a token; tokens come from the start, from discards and from played top
    ranks."""
    cards = rules.deck_size
    return cards + rules.max_hints + cards + rules.colors


def check_supported(rules: Rules) -> None:
    """JaxMARL models the official rules only, with the same tokens and lives."""
    assert not rules.discard_when_full, "JaxMARL forbids a discard with all tokens on the table"
    assert rules.zero_on_last_life, "JaxMARL scores 0 when the last life is lost"
    assert 2 <= rules.players <= 5


def make_env(rules: Rules = H4_RULES):
    """JaxMARL's Hanabi with the deck, hand, tokens and lives of `rules`."""
    from jaxmarl.environments.hanabi import Hanabi

    check_supported(rules)
    return Hanabi(
        num_agents=rules.players,
        num_colors=rules.colors,
        num_ranks=rules.ranks,
        hand_size=rules.hand,
        max_info_tokens=rules.max_hints,
        max_life_tokens=rules.lives,
        num_cards_of_rank=np.array(rules.counts),
    )


def to_jaxmarl(rules: Rules, deck: list[int]) -> np.ndarray:
    """The deck as JaxMARL's (color, rank - 1) pairs, for `reset_from_deck_of_pairs`."""
    return np.array([(color_of(c, rules), rank_of(c, rules) - 1) for c in deck], dtype=np.int32)


def from_jaxmarl_state(rules: Rules, env_state) -> list[int]:
    """The deck of a fresh JaxMARL state, in `rules.py` order: the hands in seat order,
    then the undealt cards. (JaxMARL blanks the dealt cards in its own deck array.)"""
    hands = np.asarray(env_state.player_hands)  # players x hand x colors x ranks
    rest = np.asarray(env_state.deck)[rules.players * rules.hand :]
    cards = [c for hand in hands for c in hand] + list(rest)
    out = []
    for card in cards:
        color, rank = np.argwhere(card)[0]
        out.append(make_card(int(color), int(rank) + 1, rules))
    return out


def noop(rules: Rules) -> int:
    """JaxMARL's "no move" action."""
    return 2 * rules.hand + (rules.players - 1) * (rules.colors + rules.ranks)


def to_rules_action(a: int, current: int, rules: Rules) -> Action:
    """JaxMARL action number `a` of the player `current`, as a `rules.Action`."""
    h, p, nc, nr = rules.hand, rules.players, rules.colors, rules.ranks
    if a < h:
        return discard(a)
    if a < 2 * h:
        return play(a - h)
    a -= 2 * h
    if a < (p - 1) * nc:
        offset, value = divmod(a, nc)
        return hint((current + 1 + offset) % p, "color", value)
    a -= (p - 1) * nc
    if a < (p - 1) * nr:
        offset, value = divmod(a, nr)
        return hint((current + 1 + offset) % p, "rank", value + 1)
    raise ValueError(f"action {a + 2 * h + (p - 1) * nc} is not a move")


def to_jaxmarl_action(action: Action, current: int, rules: Rules) -> int:
    """The inverse of `to_rules_action`."""
    h, p, nc, nr = rules.hand, rules.players, rules.colors, rules.ranks
    if action.kind == "discard":
        return action.index
    if action.kind == "play":
        return h + action.index
    offset = (action.target - current - 1) % p
    if action.attr == "color":
        return 2 * h + offset * nc + action.value
    return 2 * h + (p - 1) * nc + offset * nr + action.value - 1


def replay(actions, rules: Rules, deck: list[int]) -> tuple[State, list]:
    """Play the JaxMARL actions of the players whose turn it is, in order, in `rules.py`.
    Returns the final state and the events. Stops at the end of the game; an illegal
    action raises ValueError."""
    state, events = new_game(rules, deck), []
    for a in actions:
        if state.over:
            break
        events.append(step(state, to_rules_action(int(a), state.current, rules)))
    return state, events


def rules_legal_mask(state: State) -> np.ndarray:
    """The legal moves of the current player in `rules.py`, as JaxMARL's mask (no-move
    action excluded)."""
    rules = state.rules
    mask = np.zeros(noop(rules) + 1, dtype=bool)
    for action in legal_actions(state):
        mask[to_jaxmarl_action(action, state.current, rules)] = True
    return mask


def random_games(rules: Rules, decks: list[list[int]] | None, n: int, seed: int, careful=False):
    """n games of random legal play in JaxMARL, on the given decks or, if None, on
    JaxMARL's own random deals. A careful player looks at its own cards: it plays a card
    that fits if it holds one, never plays one that does not, and otherwise acts at
    random, so that games last to the end of the deck and some reach the top score. Returns the decks (in `rules.py` form), the action of the
    current player at each turn, the current player's legal mask at each turn, the
    number of turns and the final score, as numpy arrays."""
    import jax
    import jax.numpy as jnp

    env = make_env(rules)
    turns = max_turns(rules)
    key = jax.random.PRNGKey(seed)
    if decks is None:
        keys = jax.random.split(key, n + 1)
        key = keys[0]
        _, states = jax.vmap(env.reset)(keys[1:])
        decks = [
            from_jaxmarl_state(rules, jax.tree.map(lambda x, i=i: x[i], states)) for i in range(n)
        ]
    pairs = jnp.asarray(np.stack([to_jaxmarl(rules, d) for d in decks]))
    _, states = jax.vmap(env.reset_from_deck_of_pairs)(pairs)

    def one_turn(carry, k):
        state = carry
        legal = env.get_legal_moves(state)
        cur = jnp.argmax(state.cur_player_idx)
        mask = jnp.stack([legal[a] for a in env.agents])[cur]
        allowed = mask
        if careful:
            hand = state.player_hands[cur]  # hand x colors x ranks
            height = state.fireworks.sum(axis=1).astype(jnp.int32)  # cards played per color
            fits = (hand * jax.nn.one_hot(height, rules.ranks)[None]).sum(axis=(1, 2)) > 0
            plays = jnp.zeros_like(mask).at[rules.hand : 2 * rules.hand].set(fits)
            safe = mask.at[rules.hand : 2 * rules.hand].multiply(fits)
            allowed = jnp.where(plays.sum() > 0, plays * mask, safe)
        logits = jnp.where(allowed > 0, 0.0, -jnp.inf)
        a = jax.random.categorical(k, logits)
        actions = {
            agent: jnp.where(i == cur, a, env.num_moves - 1) for i, agent in enumerate(env.agents)
        }
        _, new, _, dones, _ = env.step_env(k, state, actions)
        done = state.done
        new = jax.tree.map(lambda x, y: jnp.where(done, x, y), state, new)
        return new, (jnp.where(done, -1, a), mask > 0, done)

    def one_game(state, k):
        final, (acts, masks, dones) = jax.lax.scan(one_turn, state, jax.random.split(k, turns))
        return acts, masks, (~dones).sum(), final.score, final.done

    acts, masks, lengths, scores, finished = jax.jit(jax.vmap(one_game))(
        states, jax.random.split(key, n)
    )
    assert bool(np.all(np.asarray(finished))), "a game outlasted the turn bound"
    return decks, np.asarray(acts), np.asarray(masks), np.asarray(lengths), np.asarray(scores)


def agree(rules: Rules, decks, acts, masks, lengths, scores) -> tuple[list[str], dict]:
    """Replay each JaxMARL game in `rules.py`. Returns the differences found, as text
    (none: []), and how the games ended: how many drew the last card, lost the last life,
    or reached the top score, and how many turns found all tokens on the table."""
    problems = []
    ends = {"deck_out": 0, "lives_out": 0, "top_score": 0, "turns_tokens_full": 0}
    for g, deck in enumerate(decks):
        state = new_game(rules, deck)
        for t in range(int(lengths[g])):
            if not np.array_equal(rules_legal_mask(state), masks[g, t]):
                problems.append(f"game {g} turn {t}: legal moves differ")
                break
            ends["turns_tokens_full"] += state.hints == rules.max_hints
            step(state, to_rules_action(int(acts[g, t]), state.current, rules))
        else:
            ends["deck_out"] += state.pos == len(state.deck)
            ends["lives_out"] += state.lives == 0
            ends["top_score"] += state.score == rules.max_score
            if not state.over:
                problems.append(f"game {g}: rules.py has not ended after {lengths[g]} turns")
            elif state.score != int(scores[g]):
                problems.append(
                    f"game {g}: score {state.score} in rules.py, {scores[g]} in JaxMARL"
                )
    return problems, ends


def agreement_test(n: int = 1000, seed: int = 0, rules: Rules = H4_RULES) -> dict:
    """The agreement test of plan H4: n random games on seeded decks and n on JaxMARL's own
    deals, and, beyond the plan, n games of careful random play on each, which reach the
    end of the deck."""
    report = {"games": n, "seed": seed}
    seeded = [seeded_deck(rules, 900_000 + i) for i in range(n)]
    for name, decks, careful in (
        ("seeded", seeded, False),
        ("jaxmarl", None, False),
        ("seeded-careful", seeded, True),
        ("jaxmarl-careful", None, True),
    ):
        t0 = time.time()
        out = random_games(rules, decks, n, seed, careful)
        problems, ends = agree(rules, *out)
        scores, lengths = out[4], out[3]
        report[name] = {
            "problems": problems[:20],
            "agree": not problems,
            "mean_score": float(np.mean(scores)),
            "scores": {
                int(s): int(c) for s, c in zip(*np.unique(scores, return_counts=True), strict=True)
            },
            "turns": [int(lengths.min()), int(lengths.max())],
            "ends": ends,
            "seconds": round(time.time() - t0, 1),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = agreement_test(args.games, args.seed)
    for name in ("seeded", "jaxmarl", "seeded-careful", "jaxmarl-careful"):
        r = report[name]
        print(
            f"{name} decks: {args.games} random games, agree: {r['agree']}, mean score "
            f"{r['mean_score']:.3f}, turns {r['turns']}, {r['ends']}, {r['seconds']} s"
        )
        for p in r["problems"]:
            print("  ", p)


if __name__ == "__main__":
    main()
