"""Solver B of plan H1: the open game as a time-indexed constraint model, solved by CP-SAT.

It shares no code with `hanabi.solver`; it uses `hanabi.rules` for the deck and to replay
winning lines. A deal is **won** if the model has a solution, **lost** if CP-SAT proves it
has none, and **undecided** if the time limit runs out first.

The model, for P players, hands of H cards and N = 50 - P * H cards left to draw:

- *Cards* are deck positions k = 0, ..., 49. Card k < P * H starts in the hand of player
  k // H; card P * H + j is the j-th card drawn.
- *Turns* t = 0, ..., T - 1, where turn t belongs to player t mod P. Each turn is a play,
  a discard, a misplay, a pass, or idle (the game is over). Idle turns come last.
- `use[k, t]` says that card k leaves its holder's hand on turn t, by a play, discard or
  misplay; each card leaves at most once. A card can leave only on a turn of its holder,
  after it was drawn.
- `drawn[j, t]` says that the j-th draw happens on turn t. Every play, discard or misplay
  draws while the deck lasts, so the j-th draw is on the turn of the (j + 1)-th such action.
  The holder of a drawn card is the player whose turn drew it.
- Piles: `pile[c, t]` is the height of color c before turn t. A play of card k needs its
  pile at rank(k) - 1 and raises it by one; a misplay needs it elsewhere.
- Tokens start at 8. A pass needs and spends one; a discard needs fewer than 8 and returns
  one; a played 5 returns one unless all 8 are on the table.
- Lives start at 3; a misplay costs one, and at least one must remain.
- The turn that makes the last draw is followed by exactly P more turns.
- The goal: every pile at 5 at the end.

T is an upper bound on the length of a winning game: at most N + P turns draw or end the
game, and a pass spends a token, of which there are at most 8 at the start, one per discard
and one per played 5. A winning line has 25 plays, so at most N + P - 25 discards, and
T = 2(N + P) - 12.
"""

from __future__ import annotations

from hanabi.rules import Action, Rules, hint, new_game, step


def max_turns(rules: Rules) -> int:
    n = rules.deck_size - rules.players * rules.hand
    fives = rules.colors
    discards = n + rules.players - rules.max_score
    return n + rules.players + rules.max_hints + discards + fives


def build(rules: Rules, deck: list[int]):
    """The model and its variables, as a dict."""
    from ortools.sat.python import cp_model  # ty: ignore[import-not-found]

    m = cp_model.CpModel()
    players, nh, nr, nc = rules.players, rules.hand, rules.ranks, rules.colors
    nk = rules.deck_size
    nd = nk - players * nh
    nt = max_turns(rules)
    color = [deck[k] // nr for k in range(nk)]
    rank = [deck[k] % nr + 1 for k in range(nk)]

    kinds = ("play", "discard", "misplay", "pass", "idle")
    kind = {(x, t): m.new_bool_var(f"{x}{t}") for x in kinds for t in range(nt)}
    for t in range(nt):
        m.add_exactly_one(kind[x, t] for x in kinds)
        if t + 1 < nt:
            m.add_implication(kind["idle", t], kind["idle", t + 1])
    acts = [sum(kind[x, t] for x in ("play", "discard", "misplay")) for t in range(nt)]

    # draws: the j-th draw is on the turn of the (j + 1)-th action, for j < nd
    drawn = {}
    for j in range(nd):
        for t in range(j, nt):
            drawn[j, t] = m.new_bool_var(f"d{j}_{t}")
    done = [m.new_int_var(0, nt, f"a{t}") for t in range(nt + 1)]  # actions before turn t
    m.add(done[0] == 0)
    for t in range(nt):
        m.add(done[t + 1] == done[t] + acts[t])
    for t in range(nt):
        before = done[t]
        for j in range(nd):
            if (j, t) not in drawn:
                continue
            # drawn[j, t] <=> turn t acts and exactly j actions came before it
            eq = m.new_bool_var("")
            m.add(before == j).only_enforce_if(eq)
            m.add(before != j).only_enforce_if(~eq)
            act = m.new_bool_var("")
            m.add(acts[t] == 1).only_enforce_if(act)
            m.add(acts[t] == 0).only_enforce_if(~act)
            m.add_bool_and([eq, act]).only_enforce_if(drawn[j, t])
            m.add_bool_or([~eq, ~act]).only_enforce_if(~drawn[j, t])
    for j in range(nd):
        m.add_at_most_one(drawn[j, t] for t in range(j, nt))
    # after the last draw, exactly players more turns
    for t in range(nd - 1, nt):
        for u in range(t + players + 1, nt):
            m.add_implication(drawn[nd - 1, t], kind["idle", u])

    # which card leaves on which turn, and how
    how = ("play", "discard", "misplay")
    use = {}
    for k in range(nk):
        for t in range(nt):
            if k < players * nh:
                if t % players != k // nh:
                    continue
            elif t <= k - players * nh:
                continue
            for x in how:
                use[k, t, x] = m.new_bool_var(f"u{k}_{t}{x}")
    for t in range(nt):
        for x in how:
            m.add(sum(use[k, t, x] for k in range(nk) if (k, t, x) in use) == kind[x, t])
    for k in range(nk):
        m.add_at_most_one(v for (k2, _, _), v in use.items() if k2 == k)
        if k >= players * nh:
            j = k - players * nh
            for t in range(nt):
                if (k, t, "play") not in use:
                    continue
                earlier = [drawn[j, u] for u in range(j, t) if u % players == t % players]
                for x in how:
                    if earlier:
                        m.add(sum(earlier) >= 1).only_enforce_if(use[k, t, x])
                    else:
                        m.add(use[k, t, x] == 0)

    # piles
    pile = {(c, t): m.new_int_var(0, nr, f"p{c}_{t}") for c in range(nc) for t in range(nt + 1)}
    for c in range(nc):
        m.add(pile[c, 0] == 0)
        m.add(pile[c, nt] == nr)
    for t in range(nt):
        for c in range(nc):
            ups = [use[k, t, "play"] for k in range(nk) if color[k] == c and (k, t, "play") in use]
            m.add(pile[c, t + 1] == pile[c, t] + sum(ups))
        for (k, t2, x), v in use.items():
            if t2 != t:
                continue
            if x == "play":
                m.add(pile[color[k], t] == rank[k] - 1).only_enforce_if(v)
            elif x == "misplay":
                m.add(pile[color[k], t] != rank[k] - 1).only_enforce_if(v)

    # tokens and lives
    top = rules.max_hints
    tok = [m.new_int_var(0, top, f"h{t}") for t in range(nt + 1)]
    m.add(tok[0] == top)
    for t in range(nt):
        five = sum(use[k, t, "play"] for k in range(nk) if rank[k] == nr and (k, t, "play") in use)
        full = m.new_bool_var("")
        m.add(tok[t] == top).only_enforce_if(full)
        m.add(tok[t] < top).only_enforce_if(~full)
        gain = m.new_bool_var("")
        m.add(gain <= five)
        m.add_implication(full, ~gain)
        m.add(gain >= five - full)
        m.add(tok[t + 1] == tok[t] - kind["pass", t] + kind["discard", t] + gain)
        if not rules.discard_when_full:
            m.add_implication(kind["discard", t], ~full)
    lives_lost = sum(kind["misplay", t] for t in range(nt))
    m.add(lives_lost <= rules.lives - 1)
    return m, {"use": use, "kind": kind, "T": nt}


def winnable_cp(rules: Rules, deck: list[int], time_limit: float = 60.0):
    """('won', line) | ('lost', None) | ('undecided', None). A line is a list of
    (player, kind, deck position) with kind 'play', 'discard', 'misplay' or 'pass'.
    One worker and a fixed random seed make the search repeatable."""
    from ortools.sat.python import cp_model  # ty: ignore[import-not-found]

    m, v = build(rules, deck)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(m)
    if status == cp_model.INFEASIBLE:
        return "lost", None
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "undecided", None
    line = []
    for t in range(v["T"]):
        x = next(
            x
            for x in ("play", "discard", "misplay", "pass", "idle")
            if solver.value(v["kind"][x, t])
        )
        if x == "idle":
            break
        k = -1
        if x != "pass":
            k = next(
                k for (k, t2, x2), b in v["use"].items() if t2 == t and x2 == x and solver.value(b)
            )
        line.append((t % rules.players, x, k))
    return "won", line


def replay_cp(rules: Rules, deck: list[int], line) -> int:
    """Play a line of deck positions with the rules engine and return the score."""
    s = new_game(rules, deck)
    where = {k: k // rules.hand for k in range(rules.players * rules.hand)}  # holder
    slots = [list(range(p * rules.hand, (p + 1) * rules.hand)) for p in range(rules.players)]
    for player, kind, k in line:
        if s.over:  # the model may go on after the goal is reached
            break
        assert s.current == player
        if kind == "pass":
            q = next(q for q in range(rules.players) if q != player and s.hands[q])
            step(s, hint(q, "rank", s.hands[q][0] % rules.ranks + 1))
            continue
        assert where[k] == player
        i = slots[player].index(k)
        slots[player].pop(i)
        if s.pos < len(s.deck):
            where[s.pos] = player
            slots[player].append(s.pos)
        step(s, Action("discard" if kind == "discard" else "play", i))
    return s.score
