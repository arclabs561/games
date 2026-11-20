"""Why the rating from one game is biased and overconfident, and repairs fitted on pilot games.

The plan is `results/calibration-plan.md`. Every function here is arithmetic on stored
per-move log-probabilities: for each player-game, an array (moves, ratings) of
log P(move | position, both players at rating R) on the wide grid 400 to 3000.

Vocabulary:
- a **record** is one player-game: its true rating, band, game key, fold and features, and its
  per-move log-probabilities `logp`;
- a **curve** is a log-likelihood over the grid (one number per grid rating);
- an **arm** turns a record into a curve on the grid of true ratings, with a prior, and then
  into a posterior mean and a central 95% interval;
- the **score** of a move at rating t is the slope of its log-probability there, per 100
  points. It is zero on average if the model at t describes players rated t.
"""

from dataclasses import dataclass, field

import numpy as np

GRID = np.arange(400, 3001, 100, dtype=float)
OLD = (GRID >= 600) & (GRID <= 2600)  # the old grid, 600 to 2600, is a subset
STEP = 100.0
BANDS = ((600, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 2700))
LOWER = ("600-1000", "1000-1400")
TIME_CONTROLS = ("180+0", "180+2", "300+0", "300+3", "other")
FLOOR = 1e-3  # log scores mix the posterior with this share of a uniform over 400 to 3000


@dataclass
class Record:
    """One player-game: true rating, labels, and per-move arrays."""

    rating: float
    band: str
    game: str  # the same for both players of a game, the bootstrap unit
    fold: int  # 0 for even pilot files, 1 for odd
    time_control: str
    logp: np.ndarray  # (moves, len(GRID))
    ply: np.ndarray  # (moves,)
    clock: np.ndarray  # (moves,) share of the starting time left before the move
    month_change: float = float("nan")  # last rating of the month minus this game's rating
    month_games: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def curve(self) -> np.ndarray:
        return self.logp.sum(axis=0)


def time_control_group(tc: str) -> str:
    return tc if tc in TIME_CONTROLS else "other"


# ---------------------------------------------------------------- posteriors on a grid


def normalize(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - logw.max())
    return w / w.sum()


def summarize(w: np.ndarray, grid: np.ndarray) -> tuple[float, float, float]:
    """Mean and central 95% interval of grid weights, each point's mass spread evenly over the
    cell of one step around it (as `estimate.posterior` does)."""
    step = grid[1] - grid[0]
    edges = np.concatenate([[grid[0] - step / 2], grid + step / 2])
    cdf = np.concatenate([[0.0], np.cumsum(w)])

    def quantile(q: float) -> float:
        i = int(np.searchsorted(cdf, q)) - 1
        i = min(max(i, 0), len(w) - 1)
        return float(edges[i] + (q - cdf[i]) / w[i] * step) if w[i] > 0 else float(edges[i])

    return float(w @ grid), quantile(0.025), quantile(0.975)


def log_score(w: np.ndarray, grid: np.ndarray, truth: float) -> float:
    """Log of the posterior density at the true rating, per 100 points, mixed with a small
    uniform over 400 to 3000 so that a rating off the grid scores a finite value."""
    step = grid[1] - grid[0]
    i = int(np.floor((truth - (grid[0] - step / 2)) / step))
    dens = w[i] / step if 0 <= i < len(w) else 0.0
    uniform = 1.0 / (GRID[-1] + STEP / 2 - (GRID[0] - STEP / 2))
    return float(np.log(STEP * ((1 - FLOOR) * dens + FLOOR * uniform)))


def at(curve: np.ndarray, x: float | np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """A curve (or per-move curves, last axis the grid) at ratings x, by linear interpolation
    and constant beyond the grid."""
    if curve.ndim == 1:
        return np.interp(x, grid, curve)
    return np.stack([np.interp(x, grid, c) for c in curve])


def scores(logp: np.ndarray, truth: float, h: float = 50.0) -> np.ndarray:
    """Each move's slope of log-probability at the true rating, per 100 points."""
    lo, hi = at(logp, np.array([truth - h, truth + h])).T
    return (hi - lo) * (STEP / (2 * h))


def information(logp: np.ndarray, truth: float) -> float:
    """Observed information at the true rating: minus the second difference of the total
    log-likelihood over one step, per 100 points squared."""
    c = logp.sum(axis=0)
    a, b, d = at(c, np.array([truth - STEP, truth, truth + STEP]))
    return float(-(a - 2 * b + d))


# ---------------------------------------------------------------- the arms


def kernel(sigma: float, grid: np.ndarray = GRID) -> np.ndarray:
    """Rows: for each rating t, normal weights over grid ratings r around t, summing to one."""
    if sigma <= 0:
        return np.eye(len(grid))
    k = np.exp(-0.5 * ((grid[None, :] - grid[:, None]) / sigma) ** 2)
    return k / k.sum(axis=1, keepdims=True)


def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis=axis))


def game_mixture(curve: np.ndarray, sigma: float) -> np.ndarray:
    """The likelihood of the game when the player's strength in it is spread around t."""
    if sigma <= 0:
        return curve
    return logsumexp(curve[None, :] + np.log(np.clip(kernel(sigma), 1e-300, None)), axis=1)


def move_mixture(logp: np.ndarray, sigma: float) -> np.ndarray:
    """The likelihood when each move's probability is averaged over ratings spread around t."""
    if sigma <= 0:
        return logp.sum(axis=0)
    p = np.exp(logp)  # (moves, grid)
    return np.log(np.clip(p @ kernel(sigma).T, 1e-300, None)).sum(axis=0)


def mapped(curve: np.ndarray, h: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The likelihood at true rating t is Maia-3's at h(t)."""
    return at(curve, apply_map(h, GRID))


def apply_map(h: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    """A piecewise linear map through knots (xs, ys), with slope 1 beyond the ends."""
    xs, ys = h
    out = np.interp(x, xs, ys)
    out = np.where(x < xs[0], ys[0] + (x - xs[0]), out)
    return np.where(x > xs[-1], ys[-1] + (x - xs[-1]), out)


def invert_map(h: tuple[np.ndarray, np.ndarray], y: np.ndarray) -> np.ndarray:
    xs, ys = h
    return apply_map((ys, xs), y)


@dataclass(frozen=True)
class Arm:
    """A repair: which grid, tempering, mixtures, rating map and correction after the fact."""

    name: str
    old_grid: bool = False
    tau: float = 1.0
    game_sigma: float = 0.0
    move_sigma: float = 0.0
    rating_map: tuple | None = None  # knots (xs, ys) of h
    correction: tuple | None = None  # knots (xs, ys) of g, the mean estimate at true rating
    prior: dict | None = None  # time-control group -> log prior on GRID
    skip_moves: int = 0  # drop each player's first moves

    def curve(self, r: Record) -> np.ndarray:
        logp = r.logp[self.skip_moves :] if self.skip_moves else r.logp
        if logp.shape[0] == 0:
            return np.zeros(len(GRID))
        c = move_mixture(logp, self.move_sigma) if self.move_sigma else logp.sum(axis=0)
        if self.rating_map is not None:
            c = mapped(c, self.rating_map)
        c = game_mixture(c, self.game_sigma)
        return self.tau * c

    def posterior(self, r: Record) -> tuple[np.ndarray, np.ndarray]:
        c = self.curve(r)
        if self.prior is not None:
            c = c + self.prior[time_control_group(r.time_control)]
        if self.old_grid:
            return normalize(c[OLD]), GRID[OLD]
        return normalize(c), GRID

    def evaluate(self, r: Record) -> dict:
        w, grid = self.posterior(r)
        mean, lo, hi = summarize(w, grid)
        ls = log_score(w, grid, r.rating)
        if self.correction is not None:
            mean, lo, hi = (float(v) for v in invert_map(self.correction, np.array([mean, lo, hi])))
            ls = corrected_log_score(w, grid, r.rating, self.correction)
        return {"estimate": mean, "lo": lo, "hi": hi, "log_score": ls}


def corrected_log_score(w, grid, truth, g) -> float:
    """Log score after the posterior is pushed through the inverse of g."""
    step = grid[1] - grid[0]
    edges = invert_map(g, np.concatenate([[grid[0] - step / 2], grid + step / 2]))
    i = int(np.searchsorted(edges, truth)) - 1
    dens = w[i] / (edges[i + 1] - edges[i]) if 0 <= i < len(w) else 0.0
    uniform = 1.0 / (GRID[-1] + STEP / 2 - (GRID[0] - STEP / 2))
    return float(np.log(STEP * ((1 - FLOOR) * dens + FLOOR * uniform)))


# ---------------------------------------------------------------- fitting on pilot games


def isotonic(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted least-squares nondecreasing fit (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [value, weight, count]
    for v, wt in zip(y.astype(float), weight.astype(float), strict=True):
        blocks.append([v, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[v] * int(n) for v, _, n in blocks])


def strictly_increasing(ys: np.ndarray, gap: float = 1.0) -> np.ndarray:
    out = ys.astype(float).copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    return out


def rating_bins(records: list[Record], width: float = 100.0, least: int = 5) -> dict:
    """Records grouped by true rating in bins of `width`, keeping bins with `least` or more."""
    groups: dict[float, list[Record]] = {}
    for r in records:
        groups.setdefault(width * np.floor(r.rating / width) + width / 2, []).append(r)
    return {c: rs for c, rs in sorted(groups.items()) if len(rs) >= least}


def best_rating(curve: np.ndarray) -> float:
    """The grid maximum of a pooled curve, refined by a parabola through its neighbors."""
    i = int(np.argmax(curve))
    if 0 < i < len(curve) - 1:
        y0, y1, y2 = curve[i - 1 : i + 2]
        d = y0 - 2 * y1 + y2
        if d < 0:
            return float(GRID[i] + STEP * (y0 - y2) / (2 * d))
    return float(GRID[i])


def fit_rating_map(records: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    """h: for each 100-point bin of true ratings, the rating at which Maia-3 gives the pooled
    moves their highest likelihood; made monotone by isotonic regression."""
    bins = rating_bins(records)
    xs = np.array([np.mean([r.rating for r in rs]) for rs in bins.values()])
    ys = np.array([best_rating(sum(r.curve for r in rs)) for rs in bins.values()])
    n = np.array([len(rs) for rs in bins.values()], dtype=float)
    return xs, strictly_increasing(isotonic(ys, n))


def fit_correction(records: list[Record], arm: Arm) -> tuple[np.ndarray, np.ndarray]:
    """g: the arm's mean estimate as a function of the true rating, by bins, isotonic."""
    bins = rating_bins(records)
    xs = np.array([np.mean([r.rating for r in rs]) for rs in bins.values()])
    ys = np.array([np.mean([arm.evaluate(r)["estimate"] for r in rs]) for rs in bins.values()])
    n = np.array([len(rs) for rs in bins.values()], dtype=float)
    return xs, strictly_increasing(isotonic(ys, n))


TAUS = tuple(np.round(np.arange(0.10, 1.001, 0.02), 2))
SIGMAS = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0)


def coverage(arm: Arm, records: list[Record]) -> float:
    return float(np.mean([(e := arm.evaluate(r))["lo"] <= r.rating <= e["hi"] for r in records]))


def mean_log_score(arm: Arm, records: list[Record]) -> float:
    return float(np.mean([arm.evaluate(r)["log_score"] for r in records]))


def fit_tau(arm: Arm, records: list[Record], target: float = 0.95) -> float:
    """The power of the likelihood whose coverage on these records is closest to the target."""
    cov = {t: coverage(replace(arm, tau=t), records) for t in TAUS}
    return float(min(TAUS, key=lambda t: (abs(cov[t] - target), -t)))


def fit_sigma(arm: Arm, records: list[Record], which: str) -> float:
    """The mixture spread with the best mean log score on these records."""
    return float(max(SIGMAS, key=lambda s: mean_log_score(replace(arm, **{which: s}), records)))


def replace(arm: Arm, **changes) -> Arm:
    from dataclasses import replace as _replace

    return _replace(arm, **changes)


def fit_arm(name: str, records: list[Record]) -> Arm:
    """Arm `name` (F0 to F8 of the plan) with its constants fitted on `records`."""
    f1 = Arm("F1")
    if name == "F0":
        return Arm("F0", old_grid=True)
    if name == "F1":
        return f1
    if name == "F2":
        return replace(f1, name="F2", tau=fit_tau(f1, records))
    if name == "F3":
        return replace(f1, name="F3", game_sigma=fit_sigma(f1, records, "game_sigma"))
    if name == "F4":
        return replace(f1, name="F4", move_sigma=fit_sigma(f1, records, "move_sigma"))
    if name in ("F5", "F7"):
        f5 = replace(f1, name="F5", rating_map=fit_rating_map(records))
        return f5 if name == "F5" else replace(f5, name="F7", tau=fit_tau(f5, records))
    if name == "F6":
        return replace(f1, name="F6", correction=fit_correction(records, f1))
    if name == "F8":
        f2 = replace(f1, tau=fit_tau(f1, records))
        return replace(f2, name="F8", correction=fit_correction(records, f2))
    raise ValueError(name)


ARMS = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
CONSTANTS = {"F0": 0, "F1": 0, "F2": 1, "F3": 1, "F4": 1, "F5": 1, "F6": 1, "F7": 2, "F8": 2}


def cross_validate(records: list[Record], names=ARMS) -> dict:
    """Twofold cross-validation by file parity: fit on one fold, evaluate on the other."""
    out = {}
    for name in names:
        rows = []
        for fold in (0, 1):
            fit = [r for r in records if r.fold != fold]
            held = [r for r in records if r.fold == fold]
            arm = fit_arm(name, fit)
            rows += [(r, arm.evaluate(r)) for r in held]
        ls = np.array([e["log_score"] for _, e in rows])
        out[name] = {
            "log_score": float(ls.mean()),
            "log_score_se": float(ls.std(ddof=1) / np.sqrt(len(ls))),
            "coverage": float(np.mean([e["lo"] <= r.rating <= e["hi"] for r, e in rows])),
            "by_band": by_band([(r, e) for r, e in rows]),
        }
    return out


def choose(cv: dict) -> str:
    """The plan's rule: best log score with coverage in [0.93, 0.97]; an arm within one standard
    error of it with fewer fitted constants is preferred."""
    ok = [a for a in cv if 0.93 <= cv[a]["coverage"] <= 0.97]
    if not ok:
        ok = list(cv)
    best = max(ok, key=lambda a: cv[a]["log_score"])
    near = [a for a in ok if cv[a]["log_score"] >= cv[best]["log_score"] - cv[best]["log_score_se"]]
    return min(near, key=lambda a: (CONSTANTS[a], -cv[a]["log_score"]))


# ---------------------------------------------------------------- summaries


def by_band(rows: list[tuple[Record, dict]]) -> dict:
    out = {}
    for b in sorted({r.band for r, _ in rows}) + ["all"]:
        sel = [(r, e) for r, e in rows if b in ("all", r.band)]
        err = np.array([e["estimate"] - r.rating for r, e in sel])
        out[b] = {
            "n": len(sel),
            "mae": float(np.abs(err).mean()),
            "mean_error": float(err.mean()),
            "median_error": float(np.median(err)),
            "coverage": float(np.mean([e["lo"] <= r.rating <= e["hi"] for r, e in sel])),
            "median_width": float(np.median([e["hi"] - e["lo"] for _, e in sel])),
            "log_score": float(np.mean([e["log_score"] for _, e in sel])),
        }
    return out


def game_bootstrap(records: list[Record], stat, seed: int, resamples: int = 10_000):
    """Percentile 95% interval of stat(indices) when games are resampled with replacement.
    Returns the interval and the resampled values."""
    games = sorted({r.game for r in records})
    members: dict[str, list[int]] = {g: [] for g in games}
    for i, r in enumerate(records):
        members[r.game].append(i)
    rng = np.random.default_rng(seed)
    vals = np.empty(resamples)
    for k in range(resamples):
        pick = rng.integers(0, len(games), len(games))
        idx = np.concatenate([members[games[j]] for j in pick])
        vals[k] = stat(idx)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))), vals


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def mcnemar_one_sided(new_only: int, old_only: int) -> float:
    """Exact one-sided p-value that the new method covers more often, from discordant pairs."""
    from math import comb

    n = new_only + old_only
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(new_only, n + 1)) / 2**n)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=lambda k: pvalues[k])
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[k]))
        out[k] = running
    return out


# ---------------------------------------------------------------- diagnosis on pilot games


def ply_group(ply: int) -> str:
    for lo, hi in ((0, 10), (10, 20), (20, 40)):
        if lo <= ply < hi:
            return f"{lo}-{hi - 1}"
    return "40+"


def mean_scores(records: list[Record], key, seed: int) -> dict:
    """Mean move score at the true rating, by key(record, move index), with a game bootstrap
    interval. Positive means the moves look like a higher-rated player's to Maia-3."""
    cells: dict[str, list[tuple[int, float]]] = {}
    for i, r in enumerate(records):
        s = scores(r.logp, r.rating)
        for j, v in enumerate(s):
            cells.setdefault(key(r, j), []).append((i, float(v)))
    out = {}
    for name in sorted(cells):
        rows = cells[name]
        per_record: dict[int, list[float]] = {}
        for i, v in rows:
            per_record.setdefault(i, []).append(v)
        idx_records = [records[i] for i in sorted(per_record)]
        sums = np.array([sum(per_record[i]) for i in sorted(per_record)])
        counts = np.array([len(per_record[i]) for i in sorted(per_record)])
        (lo, hi), _ = game_bootstrap(
            idx_records,
            lambda idx, s=sums, c=counts: s[idx].sum() / c[idx].sum(),
            seed,
            resamples=2000,
        )
        out[name] = {
            "moves": int(counts.sum()),
            "mean_score": float(sums.sum() / counts.sum()),
            "interval": [lo, hi],
        }
    return out


def slope_by_cell(records: list[Record]) -> dict:
    """Mean absolute change of a move's log-probability between neighboring grid ratings."""
    d = np.concatenate([np.abs(np.diff(r.logp, axis=1)) for r in records])
    m = d.mean(axis=0)
    ref = m[int(np.searchsorted(GRID, 1500))]  # the cell from 1500 to 1600
    return {
        f"{int(GRID[i])}-{int(GRID[i + 1])}": [float(m[i]), float(m[i] / ref)]
        for i in range(len(m))
    }


def best_rating_by_bin(records: list[Record]) -> list[dict]:
    """For bins of true rating: the rating that best explains the pooled moves, and the mean
    score at the true rating per move."""
    out = []
    for c, rs in rating_bins(records).items():
        moves = sum(r.logp.shape[0] for r in rs)
        s = sum(float(scores(r.logp, r.rating).sum()) for r in rs)
        out.append(
            {
                "bin": c,
                "n": len(rs),
                "mean_rating": float(np.mean([r.rating for r in rs])),
                "best_rating": best_rating(sum(r.curve for r in rs)),
                "mean_score_per_move": s / moves,
            }
        )
    return out


def sandwich(records: list[Record]) -> dict:
    """Godambe's ratio for a composite likelihood: mean observed information at the true rating
    over the variance of the total score there. Below one, the likelihood is too sharp, and
    raising it to that power corrects the curvature."""
    s = np.array([float(scores(r.logp, r.rating).sum()) for r in records])
    info = np.array([information(r.logp, r.rating) for r in records])
    return {
        "mean_information": float(info.mean()),
        "score_variance": float(s.var(ddof=1)),
        "score_mean_square": float((s**2).mean()),
        "tau_centered": float(info.mean() / s.var(ddof=1)),
        "tau_uncentered": float(info.mean() / (s**2).mean()),
    }


def diagnose(records: list[Record], seed: int = 30) -> dict:
    """Every check of hypotheses 1 to 8 in the plan, on the stratified pilot sample."""
    f0, f1 = Arm("F0", old_grid=True), Arm("F1")
    out: dict = {"player_games": len(records), "games": len({r.game for r in records})}

    def mle(arm_grid_old: bool):
        rows = []
        for r in records:
            c = r.curve
            grid = GRID[OLD] if arm_grid_old else GRID
            c = c[OLD] if arm_grid_old else c
            i = int(np.argmax(c))
            est = float(grid[i])
            if 0 < i < len(c) - 1:
                y0, y1, y2 = c[i - 1 : i + 2]
                d = y0 - 2 * y1 + y2
                if d < 0:
                    est = float(grid[i] + STEP * (y0 - y2) / (2 * d))
            rows.append((r, {"estimate": est, "lo": -1e9, "hi": 1e9, "log_score": 0.0}))
        return {b: v["mean_error"] for b, v in by_band(rows).items()}

    out["h1_prior_and_grid"] = {
        "posterior_mean_old_grid": by_band([(r, f0.evaluate(r)) for r in records]),
        "posterior_mean_wide_grid": by_band([(r, f1.evaluate(r)) for r in records]),
        "maximum_old_grid_mean_error": mle(True),
        "maximum_wide_grid_mean_error": mle(False),
    }
    out["h2_slope_by_cell"] = slope_by_cell(records)
    out["h3_best_rating_by_bin"] = best_rating_by_bin(records)
    out["h3_mean_score_by_band"] = mean_scores(records, lambda r, j: r.band, seed)
    low = [r for r in records if r.band in LOWER]
    out["h4_time_control_lower_bands"] = by_band_key(
        low, f1, lambda r: time_control_group(r.time_control)
    )
    out["h4_score_by_clock_lower_bands"] = mean_scores(
        low, lambda r, j: "more than half" if r.clock[j] > 0.5 else "half or less", seed + 1
    )
    out["h5_score_by_ply"] = mean_scores(
        records, lambda r, j: f"{r.band} {ply_group(int(r.ply[j]))}", seed + 2
    )
    skip = Arm("F1 without first 10 moves", skip_moves=10)
    out["h5_without_first_10_moves"] = by_band([(r, skip.evaluate(r)) for r in records])
    with_month = [r for r in records if np.isfinite(r.month_change)]
    out["h6_month"] = by_band_key(
        with_month, f1, lambda r: "rose 100 or more" if r.month_change >= 100 else "rest"
    )
    out["h6_month_lower_bands"] = by_band_key(
        [r for r in with_month if r.band in LOWER],
        f1,
        lambda r: "rose 100 or more" if r.month_change >= 100 else "rest",
    )
    out["h6_median_month_games"] = (
        float(np.median([r.month_games for r in with_month])) if with_month else None
    )
    out["h7_sandwich"] = sandwich(records)
    out["h7_sandwich_by_band"] = {
        b: sandwich([r for r in records if r.band == b]) for b in sorted({r.band for r in records})
    }
    shift = {
        b: v["mean_error"] for b, v in out["h1_prior_and_grid"]["posterior_mean_wide_grid"].items()
    }
    shifted = []
    for r in records:
        e = f1.evaluate(r)
        d = shift[r.band]
        shifted.append(
            (r, {**e, "estimate": e["estimate"] - d, "lo": e["lo"] - d, "hi": e["hi"] - d})
        )
    out["h8_coverage_after_removing_band_bias"] = {
        b: v["coverage"] for b, v in by_band(shifted).items()
    }
    return out


def by_band_key(records: list[Record], arm: Arm, key) -> dict:
    groups: dict[str, list[Record]] = {}
    for r in records:
        groups.setdefault(key(r), []).append(r)
    return {
        k: by_band([(r, arm.evaluate(r)) for r in rs])["all"] | {"n": len(rs)}
        for k, rs in sorted(groups.items())
    }


def case_file(r: Record, arm: Arm, sans: list[str] | None = None) -> str:
    """A text case file: the posterior curve and every move's probability and score."""
    w, grid = arm.posterior(r)
    e = arm.evaluate(r)
    lines = [
        f"true rating {r.rating:.0f}, band {r.band}, time control {r.time_control}, "
        f"moves {r.logp.shape[0]}",
        f"estimate {e['estimate']:.0f}, 95% interval {e['lo']:.0f} to {e['hi']:.0f}",
        "posterior: "
        + " ".join(f"{int(g)}:{v:.3f}" for g, v in zip(grid, w, strict=True) if v >= 0.001),
        "ply  move     p(true)  p(est)  score  clock",
    ]
    pt = np.exp(at(r.logp, r.rating))
    pe = np.exp(at(r.logp, e["estimate"]))
    s = scores(r.logp, r.rating)
    for j in range(r.logp.shape[0]):
        name = sans[j] if sans else "?"
        lines.append(
            f"{int(r.ply[j]):3d}  {name:7s} {pt[j]:7.3f} {pe[j]:7.3f} {s[j]:+6.3f} {r.clock[j]:5.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- the population prior


def prior_from_counts(
    counts: dict[str, dict[str, int]], width: float = 25.0, smooth: float = 50.0
) -> dict:
    """Log prior on GRID for each time-control group, from counts of players' ratings in bins
    of `width` (keys are the bins' lower edges), smoothed by a normal kernel."""
    out = {}
    for group, hist in counts.items():
        lows = np.array([float(k) for k in hist])
        n = np.array([float(v) for v in hist.values()])
        centres = lows + width / 2
        fine = np.arange(GRID[0] - STEP / 2, GRID[-1] + STEP / 2, 5.0) + 2.5
        dens = (n[None, :] * np.exp(-0.5 * ((fine[:, None] - centres[None, :]) / smooth) ** 2)).sum(
            1
        )
        cells = np.floor((fine - (GRID[0] - STEP / 2)) / STEP).astype(int)
        mass = np.bincount(cells, weights=dens, minlength=len(GRID))[: len(GRID)]
        mass = mass / mass.sum()
        out[group] = np.log(np.clip(mass, 1e-12, None))
    return out
