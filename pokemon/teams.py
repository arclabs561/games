"""The best team of six, at each level of ``teams_model.py``: the team game, solved exactly.

A **team** is six different Pokémon from the pool, each with its fixed set. The **team game**
is symmetric and zero-sum. Given the matchups M of a level, team T scores against team U

    P(T, U) = (1/6) Σ_{u ∈ U} max_{t ∈ T} M(t, u)  -  (1/6) Σ_{t ∈ T} max_{u ∈ U} M(u, t):

each opposing Pokémon is met by T's best answer to it, and each of T's Pokémon by U's best
answer to it. This is perfect switching: every Pokémon faces the opposing one that does best
against it. P(U, T) = -P(T, U), so the game's value is 0.

**Solving.** The game has C(n, 6) teams, 3,838,380 for a pool of 40, so it is solved by
**double oracle** (McMahan, Gordon and Blum 2003): solve the game restricted to a few teams,
find the best team against that mix, add it, and repeat until no team beats the mix. A
team's score against a mix σ is

    (1/6) Σ_u w_u max_{t ∈ T} M(t, u)  -  (1/6) Σ_{t ∈ T} c_t,

with w_u = the chance that u is on σ's team and c_t = σ's expected best answer to t. The first
sum is a facility-location problem, solved as an integer program (scipy's HiGHS). At the
end, the restricted game is solved again in exact rationals (``equilibrium.exact_equilibrium``),
and every one of the C(n, 6) teams is scored against the exact mix by brute force, the best
ones again in rationals, so that no team beats it is checked exactly, not taken from the
integer program.

    uv run --group pokemon python -m pokemon.teams              # all levels, pool of 40
    uv run --group pokemon python -m pokemon.teams --pool 20    # a smaller pool, in seconds
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import kendalltau, spearmanr

from pokemon import hodge, teams_model
from pokemon.equilibrium import exact_equilibrium, has_unique_certificate, lp_strategy

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TEAM = 6
POOL = 40
FIELD_TEAMS, FIELD_SEED = 20_000, 0
RANDOM_TEAMS, RANDOM_SEED = 8, 1

Team = tuple[int, ...]  # sorted indices into the pool
Matrix = list[list[Fraction]]


# The team game.


def payoff(m: Matrix, t: Team, u: Team) -> Fraction:
    """P(T, U): T's best answers to U's Pokémon, minus U's best answers to T's."""
    mine = sum((max(m[a][b] for a in t) for b in u), Fraction(0))
    theirs = sum((max(m[b][a] for b in u) for a in t), Fraction(0))
    return (mine - theirs) / TEAM


def game(m: Matrix, teams: Sequence[Team]) -> Matrix:
    """P over a list of teams, computed once per unordered pair."""
    k = len(teams)
    g = [[Fraction(0)] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            g[i][j] = payoff(m, teams[i], teams[j])
            g[j][i] = -g[i][j]
    return g


def field(m: Matrix, teams: Sequence[Team], sigma: Sequence) -> tuple[list, list]:
    """(w, c) of a mix: w_u = chance u is on the team, c_t = expected best answer to t."""
    n = len(m)
    zero = type(sigma[0])(0)
    w = [zero] * n
    c = [zero] * n
    for team, s in zip(teams, sigma, strict=True):
        if not s:
            continue
        for u in team:
            w[u] += s
        for t in range(n):
            c[t] += s * max(m[u][t] for u in team)
    return w, c


def score(m: Matrix, w: Sequence, c: Sequence, t: Team):
    """A team's payoff against the mix with these (w, c)."""
    return (
        sum(w[u] * max(m[a][u] for a in t) for u in range(len(m)) if w[u]) - sum(c[a] for a in t)
    ) / TEAM


def best_team_milp(mf: np.ndarray, w: np.ndarray, c: np.ndarray) -> Team:
    """The team with the highest score against (w, c), by integer programming.

    Variables: x_t (t is on the team) and y_tu (t answers u), for the u with w_u > 0.
    Maximize Σ w_u M(t, u) y_tu - Σ c_t x_t, with Σ_t y_tu = 1, y_tu <= x_t and Σ x_t = 6.
    """
    n = len(mf)
    us = [u for u in range(n) if w[u] > 1e-12]
    k = len(us)
    nv = n + n * k
    obj = np.zeros(nv)
    obj[:n] = c
    for j, u in enumerate(us):
        obj[n + np.arange(n) * k + j] = -w[u] * mf[:, u]
    rows, lo, hi = [], [], []
    size = np.zeros(nv)
    size[:n] = 1
    rows.append(size)
    lo.append(TEAM)
    hi.append(TEAM)
    for j in range(k):
        r = np.zeros(nv)
        r[n + np.arange(n) * k + j] = 1
        rows.append(r)
        lo.append(1)
        hi.append(1)
    for t in range(n):
        for j in range(k):
            r = np.zeros(nv)
            r[n + t * k + j] = 1
            r[t] = -1
            rows.append(r)
            lo.append(-np.inf)
            hi.append(0)
    integrality = np.zeros(nv)
    integrality[:n] = 1
    res = milp(
        obj,
        constraints=LinearConstraint(np.array(rows), lo, hi),
        integrality=integrality,
        bounds=Bounds(0, 1),
    )
    assert res.status == 0, res.message
    return tuple(int(t) for t in np.flatnonzero(res.x[:n] > 0.5))


def all_teams(n: int) -> np.ndarray:
    """Every team of six from n Pokémon, one per row, in lexicographic order."""
    return np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n), TEAM)),
        dtype=np.int8,
        count=math.comb(n, TEAM) * TEAM,
    ).reshape(-1, TEAM)


def brute_scores(mf: np.ndarray, w: np.ndarray, c: np.ndarray, teams: np.ndarray) -> np.ndarray:
    """Every team's score against (w, c), in floating point, in chunks."""
    us = np.flatnonzero(w > 0)
    sub = mf[:, us]
    out = np.empty(len(teams))
    for s in range(0, len(teams), 50_000):
        chunk = teams[s : s + 50_000]
        best = sub[chunk].max(axis=1)  # (chunk, |us|)
        out[s : s + 50_000] = (best @ w[us] - c[chunk].sum(axis=1)) / TEAM
    return out


def solve(m: Matrix, start: Sequence[Team], teams: np.ndarray | None = None) -> dict:
    """An exact equilibrium of the full team game, by double oracle, checked by brute force.

    Returns the restricted teams, the exact mix, the best score any team gets against it
    (exactly 0 when the mix is an equilibrium), the number of teams within 1e-9 of that
    best, and whether the restricted game's equilibrium is unique.
    """
    mf = np.array(m, dtype=float)
    pool = list(dict.fromkeys(tuple(sorted(t)) for t in start))
    rounds = 0
    while True:
        while True:
            rounds += 1
            g = np.array([[float(payoff(m, a, b)) for b in pool] for a in pool])
            x, _ = lp_strategy(g.tolist())
            x = np.clip(x, 0, None)
            w, c = field(mf.tolist(), pool, list(x / x.sum()))
            bt = best_team_milp(mf, np.array(w), np.array(c))
            if score(mf.tolist(), w, c, bt) <= 1e-9 or bt in pool:
                break
            pool.append(bt)
        gx = game(m, pool)
        xs, ys, v = exact_equilibrium(gx)
        assert v == 0, v
        w, c = field(m, pool, xs)
        if teams is None:
            teams = all_teams(len(m))
        scores = brute_scores(mf, np.array(w, dtype=float), np.array(c, dtype=float), teams)
        near = np.flatnonzero(scores > scores.max() - 1e-9)
        exact = {tuple(int(i) for i in teams[k]): score(m, w, c, tuple(teams[k])) for k in near}
        best = max(exact.values())
        if best <= 0:
            break
        pool += [t for t, s in exact.items() if s > 0 and t not in pool]
    support = [(pool[i], p) for i, p in enumerate(xs) if p]
    support.sort(key=lambda tp: -tp[1])
    ties = sorted(t for t, s in exact.items() if s == 0)
    return {
        "rounds": rounds,
        "restricted": len(pool),
        "support": support,
        "best_score": best,
        "best_responses": ties,
        "restricted_unique": has_unique_certificate(gx, xs, ys, v),
        "w": w,
        "c": c,
        "teams_checked": len(teams),
    }


# Fields to play against.


def ladder_field(usage: Sequence[float], k: int = FIELD_TEAMS, seed: int = FIELD_SEED):
    """k teams drawn from the pool, each Pokémon in turn with chance proportional to its
    usage share among those not yet drawn, with a fixed seed."""
    rng = np.random.default_rng(seed)
    p = np.array(usage, dtype=float)
    out = []
    for _ in range(k):
        out.append(
            tuple(sorted(int(i) for i in rng.choice(len(p), TEAM, replace=False, p=p / p.sum())))
        )
    return out


def best_against(
    m: Matrix, teams: Sequence[Team], sigma: Sequence[Fraction]
) -> tuple[Team, Fraction]:
    """The best team against a mix, by integer program, with its exact score."""
    mf = np.array(m, dtype=float)
    w, c = field(m, teams, sigma)
    t = best_team_milp(mf, np.array(w, dtype=float), np.array(c, dtype=float))
    return t, score(m, w, c, t)


def counts(teams: Sequence[Team]) -> tuple[list[Team], list[Fraction]]:
    """Distinct teams and their exact frequencies."""
    tally: dict[Team, int] = {}
    for t in teams:
        tally[t] = tally.get(t, 0) + 1
    keys = sorted(tally)
    return keys, [Fraction(tally[t], len(teams)) for t in keys]


def uniform_best_answer(m: Matrix, t: int, n: int) -> Fraction:
    """E[max_{u ∈ U} M(u, t)] over uniformly random teams U of six from the pool, exactly.

    Sort the n values M(u, t); the k-th smallest (from 0) is U's maximum exactly when U holds
    it and five of the k below it: C(k, 5) of the C(n, 6) teams.
    """
    vals = sorted(m[u][t] for u in range(n))
    return sum((v * math.comb(k, TEAM - 1) for k, v in enumerate(vals)), Fraction(0)) / math.comb(
        n, TEAM
    )


# Level 5: matchups and battles from the simulator.

DUELS_JSON = RESULTS / "teams-duels.json"
BATTLES_JSON = RESULTS / "teams-battles.json"
POOL_JSON = RESULTS / "teams-pool.json"
BOOTSTRAP, BOOTSTRAP_SEED = 1000, 0


def summarize(paths: Sequence[Path]) -> dict:
    """Per pair of teams (i < j): battles, wins of each, ties, and i's summed signed HP share.

    Reads the outputs of ``teams_showdown.py play``. Even game numbers put team i on side p1.
    """
    pairs: dict[str, dict] = {}
    meta: dict = {}
    checks: Counter = Counter()
    problems: set[str] = set()
    errors = 0
    for path in sorted(paths):
        d = json.loads(path.read_text())
        meta = {k: d[k] for k in ("pool", "teams", "games", "seed", "format", "showdown_commit")}
        for b in d["battles"]:
            _, i, j, g = b["key"].split("-")
            row = pairs.setdefault(f"{i}-{j}", {"games": 0, "wins": 0, "losses": 0, "ties": 0,
                                                "hp": "0", "turns": 0})  # fmt: skip
            i_side = "p1" if int(g) % 2 == 0 else "p2"
            row["games"] += 1
            row["turns"] += b["turns"]
            hp = Fraction(0)
            if b["winner"] == i_side:
                row["wins"] += 1
                hp = Fraction(*b["hp"])
            elif b["winner"] in ("p1", "p2"):
                row["losses"] += 1
                hp = -Fraction(*b["hp"])
            else:
                row["ties"] += 1
            row["hp"] = str(Fraction(row["hp"]) + hp)
            if len(meta["teams"][0]) == 1:  # one-on-one: keep each battle, for the bootstrap
                row.setdefault("each", []).append(round(float(hp), 4))
            checks.update({k: v for k, v in b["checks"].items() if k in ("agree", "differ")})
            checks.update({k: v for k, v in b["checks"].items() if k.startswith("differ:")})
            problems |= set(b["stat_problems"])
            errors += b["errors"]
    return {**meta, "pairs": dict(sorted(pairs.items())), "damage_checks": dict(sorted(checks.items())),
            "stat_problems": sorted(problems), "choice_errors": errors}  # fmt: skip


def pair_matrix(summary: dict, k: int, kind: str) -> Matrix:
    """An antisymmetric matrix from a summary: mean signed HP share ('hp') or wins minus
    losses over games ('wins')."""
    g = [[Fraction(0)] * k for _ in range(k)]
    for key, r in summary["pairs"].items():
        i, j = map(int, key.split("-"))
        v = Fraction(r["hp"]) if kind == "hp" else Fraction(r["wins"] - r["losses"])
        g[i][j] = v / r["games"]
        g[j][i] = -g[i][j]
    return g


def bootstrap_pairs(summary: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap copy of the wins-minus-losses matrix: each pair's games resampled."""
    g = np.zeros((k, k))
    for key, r in summary["pairs"].items():
        i, j = map(int, key.split("-"))
        n = r["games"]
        p = np.array([r["wins"], r["losses"], r["ties"]]) / n
        w, lo, _ = rng.multinomial(n, p)
        g[i, j] = (w - lo) / n
        g[j, i] = -g[i, j]
    return g


def teams_field_w(teams: Sequence[Team], sigma: Sequence[Fraction], n: int) -> list[Fraction]:
    """Chance that each Pokémon is on a team drawn from the mix."""
    w = [Fraction(0)] * n
    for t, p in zip(teams, sigma, strict=True):
        for u in t:
            w[u] += p
    return w


def gradient_share_float(g: np.ndarray) -> float:
    """The ranking part's share of the squared norm, in floating point."""
    s = g.mean(axis=1)
    grad = s[:, None] - s[None, :]
    return float((grad**2).sum() / (g**2).sum())


def interval(xs: Sequence[float]) -> tuple[float, float]:
    """The 95% percentile interval."""
    return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))


# The whole study.


def unit_summary(m: Matrix, names: Sequence[str]) -> dict:
    """The Pokémon-level game: Hodge ranking, the ranking part's share, an exact equilibrium."""
    s, _, _ = hodge.decompose(m)
    x, y, v = exact_equilibrium(m)
    return {
        "gradient_share": hodge.gradient_share(m),
        "score": s,
        "equilibrium": x,
        "unique": has_unique_certificate(m, x, y, v),
        "ranking": [names[i] for i in sorted(range(len(names)), key=lambda i: (-s[i], i))],
    }


def team_level(m: Matrix, ladder: tuple[list[Team], list[Fraction]], teams: np.ndarray) -> dict:
    """The team game at one level: best teams against the ladder and random teams, and an
    exact equilibrium of the full game."""
    n = len(m)
    w_u = [Fraction(TEAM, n)] * n
    c_u = [uniform_best_answer(m, t, n) for t in range(n)]
    bu = best_team_milp(*(np.array(x, dtype=float) for x in (m, w_u, c_u)))
    bl, sl = best_against(m, *ladder)
    sol = solve(m, [tuple(range(TEAM)), bl, bu], teams)
    return {
        **sol,
        "best_vs_ladder": bl,
        "best_vs_ladder_score": sl,
        "best_vs_random": bu,
        "best_vs_random_score": score(m, w_u, c_u, bu),
    }


def common_pool(levels: dict, n: int) -> list[Team]:
    """Teams for the cross-level tournament: the six most used Pokémon, then each level's
    equilibrium support and best teams against the ladder and against random teams, then
    RANDOM_TEAMS random teams (seed RANDOM_SEED)."""
    out: list[Team] = [tuple(range(TEAM))]
    for lv in levels.values():
        out += [t for t, _ in lv["team"]["support"]]
        out += [lv["team"]["best_vs_ladder"], lv["team"]["best_vs_random"]]
    rng = np.random.default_rng(RANDOM_SEED)
    out += [
        tuple(sorted(int(i) for i in rng.choice(n, TEAM, replace=False)))
        for _ in range(RANDOM_TEAMS)
    ]
    return list(dict.fromkeys(out))


def usage_shares(ss: Sequence[teams_model.Set]) -> list[float]:
    usage = json.loads((RESULTS / "usage-pokemon.json").read_text())
    shares = {p["identifier"]: p["usage"] for p in usage["pokemon"]}
    return [shares[s.name] for s in ss]


def spearman_interval(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Spearman's ρ with a 95% bootstrap interval over Pokémon (BOOTSTRAP resamples)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = float(spearmanr(xa, ya).statistic)
    reps = []
    for _ in range(BOOTSTRAP):
        k = rng.integers(0, len(xa), len(xa))
        if np.ptp(xa[k]) and np.ptp(ya[k]):
            reps.append(float(spearmanr(xa[k], ya[k]).statistic))
    return (rho, *interval(reps))


def run(n: int = POOL) -> dict:
    """Levels 1 to 5 of the Pokémon game and the team game, and the common tournament."""
    ss = teams_model.sets(n)
    assert len(ss) == n, f"the usage table has only {len(ss)} Pokémon the study keeps"
    names = [s.name for s in ss]
    use = usage_shares(ss)
    ladder = counts(ladder_field(use))
    teams = all_teams(n)
    mats = {level: teams_model.matrix(ss, level) for level in teams_model.LEVELS}
    duels = json.loads(DUELS_JSON.read_text()) if DUELS_JSON.exists() else None
    level5_bootstrap = None
    if duels:
        assert duels["pool"] == n and duels["teams"] == [[i] for i in range(n)]
        mats[5] = pair_matrix(duels, n, "hp")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        m4 = np.array(mats[4], dtype=float)
        shares, taus = [], []
        for _ in range(BOOTSTRAP):
            r = np.zeros((n, n))
            for key, row in duels["pairs"].items():
                i, j = map(int, key.split("-"))
                each = np.array(row["each"])
                r[i, j] = each[rng.integers(0, len(each), len(each))].mean()
                r[j, i] = -r[i, j]
            shares.append(gradient_share_float(r))
            taus.append(float(kendalltau(r.mean(axis=1), m4.mean(axis=1)).statistic))
        level5_bootstrap = {"gradient_share": interval(shares), "tau_4_5": interval(taus)}
    levels = {}
    for level, m in mats.items():
        t0 = time.time()
        levels[level] = {"unit": unit_summary(m, names), "team": team_level(m, ladder, teams)}
        print(f"level {level}: {time.time() - t0:.0f} s")
    out: dict = {"pool": names, "usage": use, "levels": levels, "matrices": mats, "duels": duels,
                 "level5_bootstrap": level5_bootstrap}  # fmt: skip
    pool_teams = common_pool(levels, n)
    out["common"] = {"teams": pool_teams, "levels": {}}
    for level, m in mats.items():
        g = game(m, pool_teams)
        x, y, v = exact_equilibrium(g)
        out["common"]["levels"][level] = {
            "equilibrium": x,
            "gradient_share": hodge.gradient_share(g),
            "score": hodge.score(g),
        }
    battles = json.loads(BATTLES_JSON.read_text()) if BATTLES_JSON.exists() else None
    if battles and battles["teams"] == [list(t) for t in pool_teams]:
        k = len(pool_teams)
        g = pair_matrix(battles, k, "wins")
        x, y, v = exact_equilibrium(g)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        reps = [bootstrap_pairs(battles, k, rng) for _ in range(BOOTSTRAP)]
        top = max(range(k), key=lambda i: x[i])
        weights = [lp_strategy(r.tolist())[0][top] for r in reps]
        boot_shares = [gradient_share_float(r) for r in reps]
        out["common"]["battles"] = {
            "equilibrium": x,
            "unique": has_unique_certificate(g, x, y, v),
            "gradient_share": hodge.gradient_share(g),
            "gradient_share_interval": interval(boot_shares),
            "gradient_share_boot_mean": float(np.mean(boot_shares)),
            "score": hodge.score(g),
            "matrix": g,
            "top": top,
            "top_weight_interval": interval(weights),
            "summary": battles,
        }
    return out


LEVEL_NAMES = {"1": "types", "2": "moves", "3": "abilities", "4": "stats", "5": "simulator"}


def pct(x) -> str:
    return f"{float(Fraction(x)):.1%}"


def short(team: Sequence[str]) -> str:
    return ", ".join(team)


def note(res: dict) -> str:
    """``results/teams.md``: every table of the study, written from ``teams.json``."""
    usage = {
        p["identifier"]: p
        for p in json.loads((RESULTS / "usage-pokemon.json").read_text())["pokemon"]
    }
    out = ["# Teams of six at five levels of detail", "",
           "Written by `uv run --group pokemon python -m pokemon.teams` from the models in "
           "`teams_model.py` and the simulator runs summarized in `teams-duels.json` and "
           "`teams-battles.json`. Exact unless marked. The README's section on teams explains "
           "the models.", ""]  # fmt: skip
    out += ["## The pool and its sets", "",
            "The 40 most used Pokémon of `gen9ou` (rating 1500, 2025-11 to 2026-10) that can be "
            "played without an item. Use is the share of teams with the Pokémon (Smogon).", "",
            "| # | Pokémon | use | ability here | most used ability | moves here |",
            "|---:|---|---:|---|---|---|"]  # fmt: skip
    for k, s in enumerate(res["sets"]):
        u = usage[s["pokemon"]]
        top = max(u["abilities"], key=u["abilities"].get)
        out.append(f"| {k + 1} | {s['pokemon']} | {u['usage']:.1%} | {s['ability']} | {top} | "
                   f"{', '.join(s['moves'])} |")  # fmt: skip
    sv = res["sets_vs_usage"]
    out += ["", f"The ability here is the most used one for {sv['same_ability']} of "
            f"{sv['pokemon']} Pokémon. Of the four moves here, on average "
            f"{sum(sv['move_overlap']) / len(sv['move_overlap']):.2f} are among the Pokémon's "
            "four most used moves.", ""]  # fmt: skip
    out += ["## The Pokémon game at each level", "",
            "M(p, q) is the share of HP the winner of a one-on-one fight keeps (level 5: the "
            "mean over the simulated battles). The ranking share is the part of M's squared "
            "norm that one score per Pokémon explains (Hodge decomposition).", "",
            "| level | ranking share | top five by ranking score | Pokémon in the equilibrium |",
            "|---|---:|---|---:|"]  # fmt: skip
    for lv, r in res["levels"].items():
        out.append(f"| {lv}: {LEVEL_NAMES[lv]} | {pct(r['unit_gradient_share'])} | "
                   f"{short(r['unit_ranking'][:5])} | {len(r['unit_equilibrium'])} |")  # fmt: skip
    keys = sorted(res["levels"])
    out += ["", "Kendall's τ between the rankings of two levels:", "",
            "| | " + " | ".join(keys[1:]) + " |", "|---|" + "---:|" * (len(keys) - 1)]  # fmt: skip
    for a in keys[:-1]:
        cells = [f"{res['kendall_tau'][f'{a}-{b}']:.2f}" if a < b else "" for b in keys[1:]]
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    out += ["", "## The team game at each level", "",
            "P(T, U) is the mean over U's six of T's best answer, minus the mean over T's six "
            "of U's best answer. The equilibrium is of the whole game over all "
            f"{res['levels']['1']['teams_checked']:,} teams, found by double oracle and checked "
            "exactly against every team. The ladder is "
            f"{res['field']['teams']:,} teams drawn by use (seed {res['field']['seed']}).", "",
            "| level | teams in the equilibrium | heaviest team (weight) | best against the ladder (score) | best against random teams (score) | six most used, against the equilibrium |",
            "|---|---:|---|---|---|---:|"]  # fmt: skip
    for lv, r in res["levels"].items():
        t, p = r["team_support"][0]
        out.append(f"| {lv} | {len(r['team_support'])} | {short(t)} ({pct(p)}) | "
                   f"{short(r['best_vs_ladder'])} ({float(Fraction(r['best_vs_ladder_score'])):.3f}) | "
                   f"{short(r['best_vs_random'])} ({float(Fraction(r['best_vs_random_score'])):.3f}) | "
                   f"{float(Fraction(r['usage_team_score'])):.3f} |")  # fmt: skip
    out += ["", "Chance that each Pokémon is on the equilibrium team (the eight largest):", "",
            "| level | Pokémon |", "|---|---|"]  # fmt: skip
    for lv, r in res["levels"].items():
        top = sorted(r["team_marginals"].items(), key=lambda kv: (-Fraction(kv[1]), kv[0]))[:8]
        out.append(f"| {lv} | " + ", ".join(f"{k} {pct(v)}" for k, v in top) + " |")
    out += ["", "Solver details: rounds of double oracle, teams in the restricted game, whether "
            "its equilibrium is unique (exact certificate), and teams tied with the best score "
            "(0) against the mix.", "",
            "| level | rounds | restricted teams | restricted equilibrium unique | teams scoring 0 |",
            "|---|---:|---:|---|---:|"]  # fmt: skip
    for lv, r in res["levels"].items():
        out.append(f"| {lv} | {r['team_rounds']} | {r['team_restricted']} | "
                   f"{'yes' if r['team_restricted_unique'] else 'no'} | {r['team_best_responses']} |")  # fmt: skip
    c = res["common"]
    out += ["", "## One tournament across the levels", "",
            f"{len(c['teams'])} teams: the six most used Pokémon (team 0), each level's "
            "equilibrium teams and best teams, and random teams (seed "
            f"{RANDOM_SEED}). Each level plays the same round robin.", "",
            "| level | ranking share | teams in the equilibrium | rank of the six most used | τ with full battles |",
            "|---|---:|---:|---:|---:|"]  # fmt: skip
    for lv, r in c["levels"].items():
        tau = res.get("kendall_tau_common", {}).get(lv)
        out.append(f"| {lv} | {pct(r['gradient_share'])} | {len(r['support'])} | "
                   f"{res['levels'][lv]['usage_team_rank']} | {'' if tau is None else f'{tau:.2f}'} |")  # fmt: skip
    if "battles" in c:
        b = c["battles"]
        lo, hi = b["gradient_share_interval"]
        out.append(f"| full battles | {pct(b['gradient_share'])} ({lo:.1%} to {hi:.1%}) | "
                   f"{len(b['equilibrium'])} | {b['usage_team_rank']} | 1 |")  # fmt: skip
        out += ["", f"Full battles: {b['games']} per pair of teams, seed {b['seed']}, sides "
                "alternating; payoff wins minus losses over battles. The interval is a 95% "
                f"bootstrap over battles within each pair ({res['bootstrap']['resamples']} "
                f"resamples, seed {res['bootstrap']['seed']}). Equilibrium "
                f"{'unique' if b['unique'] else 'not shown unique'}:", "",
                "| team | weight | Pokémon |", "|---:|---:|---|"]  # fmt: skip
        for i, p in sorted(b["equilibrium"].items(), key=lambda kv: (-Fraction(kv[1]), int(kv[0]))):
            out.append(f"| {i} | {pct(p)} | {short(c['teams'][int(i)])} |")
        top8 = sorted(b["marginals"].items(), key=lambda kv: (-Fraction(kv[1]), kv[0]))[:8]
        out += ["", "Chance that each Pokémon is on the equilibrium team (the eight largest): "
                + ", ".join(f"{k} {pct(v)}" for k, v in top8) + "."]  # fmt: skip
        lo, hi = b["top_weight_interval"]
        est = float(Fraction(b["gradient_share"]))
        out += ["", f"The heaviest team's weight over the bootstrap: {lo:.1%} to {hi:.1%}.", "",
                f"The bootstrap copies of the ranking share average {b['gradient_share_boot_mean']:.1%}, "
                f"below the estimate of {est:.1%}: noise from sampling adds to the cyclic part. The "
                "estimate is biased the same way, so the share without noise is likely higher; the "
                f"usual bias correction gives {2 * est - b['gradient_share_boot_mean']:.1%}."]  # fmt: skip
    out += ["", "## Agreement with use (exploratory)", "",
            "Spearman's ρ over the 40 Pokémon between use and each level's ranking score, and "
            "between use and the chance of being on the equilibrium team, with 95% bootstrap "
            "intervals over Pokémon. The pool is the 40 most used, so this asks about order "
            "within the top of the ladder.", "",
            "| level | ρ, ranking score | ρ, equilibrium team |", "|---|---|---|"]  # fmt: skip
    for lv, r in res["usage_spearman"].items():
        a, b2 = r["score"], r["team_weight"]
        out.append(
            f"| {lv} | {a[0]:.2f} ({a[1]:.2f} to {a[2]:.2f}) | {b2[0]:.2f} ({b2[1]:.2f} to {b2[2]:.2f}) |"
        )
    if "duels" in res:
        d = res["duels"]
        out += ["", "## Simulator checks", "",
                f"- One-on-one battles: {d['pairs']} pairs, {d['games']} battles each, seed "
                f"{d['seed']}. Damage checks: {d['damage_checks'].get('agree', 0):,} hits "
                f"agree with level 4's rolls, {d['damage_checks'].get('differ', 0)} differ. Stats "
                f"that differ from Showdown's: {len(d['stat_problems'])}. Choices Showdown "
                f"refused: {d['choice_errors']}.",
                f"- Level 5's ranking share, resampling battles within each pair: "
                f"{d['bootstrap']['gradient_share'][0]:.1%} to {d['bootstrap']['gradient_share'][1]:.1%}; "
                f"Kendall's τ between levels 4 and 5: {d['bootstrap']['tau_4_5'][0]:.2f} to "
                f"{d['bootstrap']['tau_4_5'][1]:.2f} (95%, {res['bootstrap']['resamples']} resamples, "
                f"seed {res['bootstrap']['seed']})."]  # fmt: skip
        if "battles" in c:
            b = c["battles"]
            out.append(f"- Full battles: damage checks {b['damage_checks'].get('agree', 0):,} agree, "
                       f"{b['damage_checks'].get('differ', 0)} differ; stats that differ: "
                       f"{len(b['stat_problems'])}; refused choices: {b['choice_errors']}.")  # fmt: skip
    return "\n".join(out) + "\n"


# Writing the results.


def kendall(a: Sequence[Fraction], b: Sequence[Fraction]) -> float:
    """Kendall's τ-b between two score vectors."""
    return float(kendalltau([float(x) for x in a], [float(x) for x in b]).statistic)


def usage_comparison(out: dict) -> dict:
    """Agreement of each level with real use, over the pool (exploratory)."""
    use = out["usage"]
    rows = {}
    for level, lv in out["levels"].items():
        rows[level] = {
            "score": spearman_interval(lv["unit"]["score"], use),
            "team_weight": spearman_interval(lv["team"]["w"], use),
        }
    return rows


def set_comparison(ss: Sequence[teams_model.Set]) -> dict:
    """How the fixed sets compare with what players use: ability and moves."""
    usage = {
        p["identifier"]: p
        for p in json.loads((RESULTS / "usage-pokemon.json").read_text())["pokemon"]
    }
    same_ability, overlaps = 0, []
    for s in ss:
        u = usage[s.name]
        if u["abilities"] and max(u["abilities"], key=u["abilities"].get) == s.ability.replace(
            "-", ""
        ):
            same_ability += 1
        top = sorted(u["moves"], key=lambda m: -u["moves"][m])[:4]
        overlaps.append(
            len({m.replace("-", "") for m in (x.name for x in s.moves)} & {m.lower() for m in top})
        )
    return {"same_ability": same_ability, "pokemon": len(ss), "move_overlap": overlaps}


def fr(x) -> str:
    return str(x)


def jsonable(out: dict, ss: Sequence[teams_model.Set]) -> dict:
    """The numbers the README and the note quote, with every fraction exact as a string."""

    def team(t: Team) -> list[str]:
        return [ss[i].name for i in t]

    levels = {}
    for level, lv in out["levels"].items():
        u, t = lv["unit"], lv["team"]
        levels[str(level)] = {
            "unit_gradient_share": fr(u["gradient_share"]),
            "unit_ranking": u["ranking"],
            "unit_score": [fr(x) for x in u["score"]],
            "unit_equilibrium": {ss[i].name: fr(p) for i, p in enumerate(u["equilibrium"]) if p},
            "unit_unique": u["unique"],
            "team_support": [[team(x), fr(p)] for x, p in t["support"]],
            "team_marginals": {ss[i].name: fr(w) for i, w in enumerate(t["w"]) if w},
            "team_rounds": t["rounds"],
            "team_restricted": t["restricted"],
            "team_restricted_unique": t["restricted_unique"],
            "team_best_score": fr(t["best_score"]),
            "team_best_responses": len(t["best_responses"]),
            "teams_checked": t["teams_checked"],
            "best_vs_ladder": team(t["best_vs_ladder"]),
            "best_vs_ladder_score": fr(t["best_vs_ladder_score"]),
            "best_vs_random": team(t["best_vs_random"]),
            "best_vs_random_score": fr(t["best_vs_random_score"]),
            "usage_team_score": fr(score(out["matrices"][level], t["w"], t["c"], tuple(range(TEAM)))),
            "usage_team_rank": 1 + sorted(range(len(out["common"]["teams"])),
                                          key=lambda i: -out["common"]["levels"][level]["score"][i]).index(0),
        }  # fmt: skip
    keys = list(out["levels"])
    tau = {f"{a}-{b}": kendall(out["levels"][a]["unit"]["score"], out["levels"][b]["unit"]["score"])
           for a in keys for b in keys if a < b}  # fmt: skip
    common = out["common"]
    c_levels = {str(k): {"gradient_share": fr(v["gradient_share"]),
                         "support": {str(i): fr(p) for i, p in enumerate(v["equilibrium"]) if p},
                         "score": [fr(x) for x in v["score"]]}
                for k, v in common["levels"].items()}  # fmt: skip
    res: dict = {
        "pool": out["pool"],
        "usage": out["usage"],
        "sets": [
            {"pokemon": s.name, "ability": s.ability, "moves": [m.name for m in s.moves]}
            for s in ss
        ],
        "levels": levels,
        "kendall_tau": tau,
        "usage_spearman": {str(k): v for k, v in usage_comparison(out).items()},
        "sets_vs_usage": set_comparison(ss),
        "common": dict[str, object](teams=[team(t) for t in common["teams"]], levels=c_levels),
        "field": {"teams": FIELD_TEAMS, "seed": FIELD_SEED},
        "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED},
    }
    if out.get("duels"):
        d = out["duels"]
        res["duels"] = {"games": d["games"], "seed": d["seed"], "pairs": len(d["pairs"]),
                        "damage_checks": d["damage_checks"], "stat_problems": d["stat_problems"],
                        "choice_errors": d["choice_errors"],
                        "bootstrap": out["level5_bootstrap"]}  # fmt: skip
    if "battles" in common:
        b = common["battles"]
        s = b["summary"]
        res["common"]["battles"] = {
            "games": s["games"], "seed": s["seed"], "pairs": len(s["pairs"]),
            "equilibrium": {str(i): fr(p) for i, p in enumerate(b["equilibrium"]) if p},
            "unique": b["unique"],
            "gradient_share": fr(b["gradient_share"]),
            "gradient_share_interval": b["gradient_share_interval"],
            "gradient_share_boot_mean": b["gradient_share_boot_mean"],
            "top": b["top"], "top_weight_interval": b["top_weight_interval"],
            "score": [fr(x) for x in b["score"]],
            "usage_team_rank": 1 + sorted(range(len(b["score"])), key=lambda i: -b["score"][i]).index(0),
            "marginals": {
                ss[u].name: fr(w)
                for u, w in enumerate(teams_field_w(common["teams"], b["equilibrium"], len(ss)))
                if w
            },
            "damage_checks": s["damage_checks"], "stat_problems": s["stat_problems"],
            "choice_errors": s["choice_errors"],
        }  # fmt: skip
        res["common"]["battles_matrix"] = [[fr(x) for x in row] for row in b["matrix"]]
        res["kendall_tau_common"] = {
            str(k): kendall(v["score"], b["score"]) for k, v in common["levels"].items()
        }
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    s = sub.add_parser("summarize", help="merge teams_showdown.py outputs into a summary")
    s.add_argument("paths", nargs="+", type=Path)
    s.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool", type=int, default=POOL)
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.command == "summarize":
        paths = [
            p
            for x in args.paths
            for p in (sorted(x.rglob("battles-*.json")) if x.is_dir() else [x])
        ]
        args.out.write_text(
            json.dumps(summarize(paths), sort_keys=True, separators=(",", ":")) + "\n"
        )
        print(f"{args.out}: {len(paths)} files")
        return
    out = run(args.pool)
    ss = teams_model.sets(args.pool)
    res = jsonable(out, ss)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "teams.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    pool_out = args.out / POOL_JSON.name
    pool_out.write_text(
        json.dumps({"pool": args.pool, "teams": [list(t) for t in out["common"]["teams"]]}) + "\n"
    )
    (args.out / "teams.md").write_text(note(res))
    print(json.dumps({k: res["levels"][k]["unit_ranking"][:5] for k in res["levels"]}, indent=1))


if __name__ == "__main__":
    main()
