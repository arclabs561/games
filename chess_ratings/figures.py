"""The README's figures, drawn as SVG from the committed results, with no plotting package.

    uv run python -m chess_ratings.figures      # rewrites chess_ratings/figures/*.svg

A test compares every figure with the committed file.
"""

import json
import random
from pathlib import Path

import chess

from chess_ratings.board_svg import SQUARE, draw_arrow, draw_board
from peggle.figures import BLUE, GRID, INK, MUTED, ORANGE, Svg, num

HERE = Path(__file__).parent
OUT = HERE / "figures"
RESULTS = HERE / "results"
# strength 800 in dark gray, far in lightness from the blue of 1100, so the two stay apart
# for readers who do not see red and green
COLORS = {"800": "#3d3d3d", "1100": BLUE, "1900": "#1a9a6c", "2300": ORANGE}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def transcript(block: int = 0) -> tuple[str, dict[str, dict[str, float]]]:
    """The FEN and the table {rating: {SAN: probability}} of one block of whatwould.txt."""
    lines = (RESULTS / "whatwould.txt").read_text().split("\n\n")[block].splitlines()
    fen, header = lines[1], lines[3].split()[1:]
    table: dict[str, dict[str, float]] = {r: {} for r in header}
    for line in lines[4:]:
        words = line.split()
        if words[0] == "other":
            continue
        for r, cell in zip(header, words[1:], strict=True):
            table[r][words[0]] = float(cell.rstrip("%")) / 100
    return fen, table


def board() -> str:
    """The position after 1. e4 e5 2. Qh5 Nc6 3. Bc4, with Maia-3's three likeliest replies
    for players rated 1100 (left) and 1900 (right), from the transcript in whatwould.txt."""
    fen, table = transcript(0)
    b = chess.Board(fen)
    width = 2 * (8 * SQUARE + 60) + 20
    svg = Svg(
        width,
        8 * SQUARE + 120,
        "Black to move after 1. e4 e5 2. Qh5 Nc6 3. Bc4, with Maia-3's likeliest replies at 1100 and 1900",
        "Two copies of the same board, letters for pieces. White's queen on h5 and bishop on c4 "
        "both aim at f7. Arrows show Maia-3's three likeliest moves for Black, thicker for more "
        "likely, with their probabilities: for players rated 1100 on the left and 1900 on the right.",
    )
    for i, rating in enumerate(("1100", "1900")):
        left = 40 + i * (8 * SQUARE + 60)
        top = 40
        svg.text(left, 26, f"Both players rated {rating}", 14, font_weight="bold")
        draw_board(svg, b, left, top)
        probs = table[rating]
        best = sorted(probs, key=lambda m: (-probs[m], m))[:3]
        for san in best:
            draw_arrow(svg, b.parse_san(san), probs[san], left, top)
        y = top + 8 * SQUARE + 38
        for j, san in enumerate(best):
            svg.text(left + j * 120, y, f"{san}  {100 * probs[san]:.0f}%", 13, fill=INK)
        nf6 = probs["Nf6"]
        svg.text(left, y + 22, f"Nf6, which allows Qxf7 mate: {100 * nf6:.1f}%", 12, fill=MUTED)
    return svg.render()


def settle() -> str:
    """Deviation by game (model and real new players) and the chance of being within 100."""
    s = load("settle.json")
    real = load("trajectories.json") if (RESULTS / "trajectories.json").exists() else None
    pw, ph, top = 330, 220, 50
    svg = Svg(
        2 * pw + 150,
        ph + top + 70,
        "How a new Lichess blitz rating settles",
        "Left: the rating deviation after each of the first 40 games, falling from 500 below the "
        "provisional line at 110 after game 10 in the model; dots are the median deviation backed "
        "out of real new players' rating changes. Right: in the model, the chance that a new "
        "player's rating is within 100 points of their strength, by games played, for strengths "
        "800, 1100, 1900 and 2300.",
    )
    # left panel
    left = 60

    def lx(k: float) -> float:
        return left + k / 40 * pw

    def ly(d: float) -> float:
        return top + ph - d / 500 * ph

    svg.text(left, 30, "Rating deviation after each game", 14, font_weight="bold")
    for d in (0, 100, 200, 300, 400, 500):
        svg.add("line", x1=left, y1=ly(d), x2=left + pw, y2=ly(d), stroke=GRID)
        svg.text(left - 8, ly(d) + 4, str(d), 11, fill=MUTED, text_anchor="end")
    for k in (0, 10, 20, 30, 40):
        svg.text(lx(k), top + ph + 16, str(k), 11, fill=MUTED, text_anchor="middle")
    svg.text(left + pw / 2, top + ph + 34, "games played", 12, fill=MUTED, text_anchor="middle")
    svg.add(
        "line", x1=left, y1=ly(110), x2=left + pw, y2=ly(110), stroke=ORANGE, stroke_dasharray="5 4"
    )
    svg.text(
        left + pw - 4, ly(110) - 6, "110: provisional above", 11, fill=ORANGE, text_anchor="end"
    )
    path = [500.0] + s["deviation_path"]
    pts = " ".join(f"{num(lx(k))},{num(ly(d))}" for k, d in enumerate(path))
    svg.add("polyline", points=pts, fill="none", stroke=BLUE, stroke_width=2)
    if real:
        for row in real["implied_deviation"][:40]:
            if row["median"] is not None:
                svg.add("circle", cx=lx(row["game"]), cy=ly(row["median"]), r=2.5, fill=INK)
        svg.text(lx(16), ly(330), "dots: real new players (median)", 11, fill=INK)
    svg.text(lx(16), ly(300), "line: model, opponents' deviation 60", 11, fill=BLUE)
    # right panel
    left2 = left + pw + 80

    def rx(k: float) -> float:
        return left2 + k / 100 * pw

    def ry(p: float) -> float:
        return top + ph - p * ph

    svg.text(left2, 30, "Chance the rating is within 100 of strength", 14, font_weight="bold")
    for p in (0, 0.25, 0.5, 0.75, 0.9):
        svg.add("line", x1=left2, y1=ry(p), x2=left2 + pw, y2=ry(p), stroke=GRID)
        svg.text(left2 - 8, ry(p) + 4, f"{p:g}", 11, fill=MUTED, text_anchor="end")
    for k in (0, 25, 50, 75, 100):
        svg.text(rx(k), top + ph + 16, str(k), 11, fill=MUTED, text_anchor="middle")
    svg.text(left2 + pw / 2, top + ph + 34, "games played", 12, fill=MUTED, text_anchor="middle")
    for j, (true, color) in enumerate(COLORS.items()):
        curve = s["simulated"][true]["within_100_curve"]
        pts = " ".join(f"{num(rx(k))},{num(ry(p))}" for k, p in enumerate(curve))
        svg.add("polyline", points=pts, fill="none", stroke=color, stroke_width=2)
        svg.text(rx(62), ry(0.52 - 0.08 * j), f"strength {true}", 11, fill=color)
    return svg.render()


BANDS = ("600-1000", "1000-1400", "1400-1800", "1800-2200", "2200-2700")


def common_totals() -> dict[str, tuple[int, int, float]]:
    """Per band of the mover's rating: moves, blunders played, and blunders Maia-3 expects,
    summed over the most common opening positions in common.json."""
    out = {b: [0, 0, 0.0] for b in BANDS}
    for p in load("common.json")["positions"]:
        for b in BANDS:
            n, o, e = p["observed"][b]
            out[b][0] += n
            out[b][1] += o
            out[b][2] += e
    return {b: (int(v[0]), int(v[1]), float(v[2])) for b, v in out.items()}


def position_interval(
    bands: tuple[str, ...], played_only: bool = False, draws: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """For the players in `bands`, blunders played minus blunders Maia-3 expects, in percent of
    moves (or with `played_only`, blunders played alone), with a 95% percentile interval from
    resampling the 24 positions. Moves in one position share it, so the positions, not the
    moves, are the units resampled."""
    positions = load("common.json")["positions"]

    def stat(sample: list[dict]) -> float:
        n = sum(p["observed"][b][0] for p in sample for b in bands)
        o = sum(p["observed"][b][1] for p in sample for b in bands)
        e = 0 if played_only else sum(p["observed"][b][2] for p in sample for b in bands)
        return 100 * (o - e) / n

    rng = random.Random(seed)
    values = sorted(stat([rng.choice(positions) for _ in positions]) for _ in range(draws))
    return stat(positions), values[int(0.025 * draws)], values[int(0.975 * draws) - 1]


def blunders() -> str:
    """Blunder rate by rating band: what players did and what Maia-3 expected."""
    totals = common_totals()
    left, top, pw, ph = 70, 50, 560, 220
    ymax = 4.0
    svg = Svg(
        left + pw + 40,
        top + ph + 80,
        "How often players blunder in the 24 most common opening positions, by rating",
        "Pairs of bars for five rating bands, with the count of blunders under each. Black "
        "bars: the share of moves that were blunders in Lichess blitz games, falling from about "
        "1.5% for players rated 600 to 1000 to about 0.25% above 1800, with 95% intervals from "
        "resampling the positions. Blue bars: the share Maia-3 expected, higher than observed "
        "below 1800 and close to it above. Above 1800 the counts are small.",
    )
    svg.text(
        left,
        26,
        "Blunders in the 24 most common opening positions (% of moves)",
        14,
        font_weight="bold",
    )

    def y(v: float) -> float:
        return top + ph - v / ymax * ph

    for v in (0, 1, 2, 3, 4):
        svg.add("line", x1=left, y1=y(v), x2=left + pw, y2=y(v), stroke=GRID)
        svg.text(left - 8, y(v) + 4, f"{v}%", 11, fill=MUTED, text_anchor="end")
    slot = pw / len(BANDS)
    for i, b in enumerate(BANDS):
        n, observed, expected = totals[b]
        x = left + i * slot + slot * 0.18
        _, lo, hi = position_interval((b,), played_only=True, draws=2000)
        for j, (value, color) in enumerate(((observed, INK), (expected, BLUE))):
            share = 100 * value / n
            svg.add(
                "rect",
                x=x + j * slot * 0.32,
                y=y(share),
                width=slot * 0.3,
                height=y(0) - y(share),
                fill=color,
            )
            svg.text(
                x + j * slot * 0.32 + slot * 0.15,
                y(share) - 5,
                f"{share:.2f}",
                11,
                fill=color,
                text_anchor="middle",
            )
        cx = x + slot * 0.04
        svg.add("line", x1=cx, y1=y(lo), x2=cx, y2=y(hi), stroke=ORANGE, stroke_width=2)
        svg.text(left + (i + 0.5) * slot, top + ph + 18, b, 12, fill=MUTED, text_anchor="middle")
        svg.text(
            left + (i + 0.5) * slot,
            top + ph + 34,
            f"{observed} of {n:,} moves",
            11,
            fill=MUTED,
            text_anchor="middle",
        )
    svg.text(left + pw - 210, top + 20, "played (Lichess, Sept. 2025)", 12, fill=INK)
    svg.text(
        left + pw - 210, top + 56, "orange: 95% interval, positions resampled", 12, fill=ORANGE
    )
    svg.text(left + pw - 210, top + 38, "expected by Maia-3 (5M)", 12, fill=BLUE)
    svg.text(left, top + ph + 60, "Rating band of the player to move", 12, fill=MUTED)
    return svg.render()


def estimate() -> str:
    """Error of the one-game rating estimate by the moves used, and its bias by band."""
    e = load("estimate.json")
    pw, ph, top = 300, 200, 50
    svg = Svg(
        2 * pw + 170,
        ph + top + 80,
        "How well Maia-3 guesses a rating from one game",
        "Left: the mean absolute error of the estimate, in rating points, using a player's first "
        "5, 10, 20, 30 or 40 moves of a game: 331, 263, 215, 193 and 174. Right: by rating band, "
        "the mean error (estimate minus rating), from about +200 below 1400 to about -30 above "
        "2200; Maia-3 rates weaker players' moves as stronger than their ratings.",
    )
    left = 70
    ks = ["5", "10", "20", "30", "40"]

    def lx(i: int) -> float:
        return left + i / (len(ks) - 1) * pw

    def ly(v: float) -> float:
        return top + ph - v / 500 * ph

    svg.text(left, 28, "Mean absolute error by moves used", 14, font_weight="bold")
    for v in (0, 100, 200, 300, 400, 500):
        svg.add("line", x1=left, y1=ly(v), x2=left + pw, y2=ly(v), stroke=GRID)
        svg.text(left - 8, ly(v) + 4, str(v), 11, fill=MUTED, text_anchor="end")
    pts = []
    for i, k in enumerate(ks):
        row = e["by_moves"][k]
        x, y = lx(i), ly(row["posterior_mae"])
        pts.append(f"{num(x)},{num(y)}")
        svg.add("circle", cx=x, cy=y, r=3.5, fill=BLUE)
        svg.text(x, y - 9, f"{row['posterior_mae']:.0f}", 11, fill=BLUE, text_anchor="middle")
        svg.text(x, top + ph + 16, k, 11, fill=MUTED, text_anchor="middle")
        svg.text(x, top + ph + 30, f"n={row['n']}", 10, fill=MUTED, text_anchor="middle")
    svg.add("polyline", points=" ".join(pts), fill="none", stroke=BLUE, stroke_width=2)
    base = e["baseline_mean_rating_mae"]
    svg.add(
        "line",
        x1=left,
        y1=ly(base),
        x2=left + pw,
        y2=ly(base),
        stroke=ORANGE,
        stroke_dasharray="5 4",
    )
    svg.text(
        left + pw,
        ly(base) + 14,
        f"guessing the mean: {base:.0f}",
        11,
        fill=ORANGE,
        text_anchor="end",
    )
    svg.text(
        left + pw / 2,
        top + ph + 48,
        "moves used (the player's own)",
        12,
        fill=MUTED,
        text_anchor="middle",
    )

    left2 = left + pw + 90
    bands = list(BANDS)
    slot = pw / len(bands)

    def by(v: float) -> float:
        return top + ph / 2 - v / 300 * (ph / 2)

    svg.text(left2, 28, "Mean error by rating band", 14, font_weight="bold")
    for v in (-300, -150, 0, 150, 300):
        svg.add("line", x1=left2, y1=by(v), x2=left2 + pw, y2=by(v), stroke=GRID)
        svg.text(left2 - 8, by(v) + 4, f"{v:+d}" if v else "0", 11, fill=MUTED, text_anchor="end")
    for i, b in enumerate(bands):
        v = e["by_band"][b]["posterior_mean_error"]
        x = left2 + i * slot + slot * 0.2
        y0, y1 = sorted((by(0), by(v)))
        svg.add("rect", x=x, y=y0, width=slot * 0.6, height=y1 - y0, fill=BLUE)
        svg.text(
            x + slot * 0.3,
            by(v) + (-6 if v > 0 else 15),
            f"{v:+.0f}",
            11,
            fill=BLUE,
            text_anchor="middle",
        )
        svg.text(left2 + (i + 0.5) * slot, top + ph + 16, b, 10, fill=MUTED, text_anchor="middle")
    svg.text(
        left2 + pw / 2,
        top + ph + 48,
        "estimate minus rating, 100 players a band",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    return svg.render()


CALIBRATION_ARMS = (
    ("F0", "old method", ORANGE),
    ("F2", "likelihood tempered", BLUE),
    ("F7", "rating map, tempered", "#1a9a6c"),
)


def calibration() -> str:
    """Test games: mean error and 95% interval coverage by band, for the old method and two
    repairs fitted on pilot games."""
    t = load("calibration-test.json")["secondary"]["arms"]
    pw, ph, top = 300, 200, 70
    bands = list(BANDS)
    slot = pw / len(bands)
    svg = Svg(
        2 * pw + 170,
        ph + top + 70,
        "The rating from one game on held-out August 2025 games: bias and coverage by band",
        "Left: by rating band, the mean error (estimate minus rating) of the old method, of the "
        "tempered likelihood and of the rating map with tempering. Right: the share of 95% "
        "intervals that hold the rating, with a dashed line at 95%. "
        + "; ".join(
            f"{label}: errors "
            + ", ".join(f"{t[a][b]['mean_error']:+.0f}" for b in bands)
            + ", coverage "
            + ", ".join(f"{100 * t[a][b]['coverage']:.0f}%" for b in bands)
            for a, label, _ in CALIBRATION_ARMS
        ),
    )
    left = 70
    x0 = left + 10
    for i, (_, label, color) in enumerate(CALIBRATION_ARMS):
        x = x0 + i * 210
        svg.add("rect", x=x, y=42, width=12, height=12, fill=color)
        svg.text(x + 18, 52, label, 12)

    def by(v: float) -> float:
        return top + ph / 2 - v / 300 * (ph / 2)

    svg.text(left, 26, "Mean error by rating band", 14, font_weight="bold")
    for v in (-300, -150, 0, 150, 300):
        svg.add("line", x1=left, y1=by(v), x2=left + pw, y2=by(v), stroke=GRID)
        svg.text(left - 8, by(v) + 4, f"{v:+d}" if v else "0", 11, fill=MUTED, text_anchor="end")
    width = slot * 0.8 / len(CALIBRATION_ARMS)
    for i, b in enumerate(bands):
        for k, (a, _, color) in enumerate(CALIBRATION_ARMS):
            v = max(-300.0, min(300.0, t[a][b]["mean_error"]))
            x = left + i * slot + slot * 0.1 + k * width
            y0, y1 = sorted((by(0), by(v)))
            svg.add("rect", x=x, y=y0, width=width - 1, height=max(y1 - y0, 0.5), fill=color)
        svg.text(left + (i + 0.5) * slot, top + ph + 16, b, 10, fill=MUTED, text_anchor="middle")
    svg.text(
        left + pw / 2,
        top + ph + 40,
        "estimate minus rating, 200 player-games a band",
        12,
        fill=MUTED,
        text_anchor="middle",
    )

    left2 = left + pw + 90

    def cy(v: float) -> float:
        return top + ph - (v - 0.6) / 0.4 * ph

    svg.text(left2, 26, "95% intervals that hold the rating", 14, font_weight="bold")
    for v in (0.6, 0.7, 0.8, 0.9, 1.0):
        svg.add("line", x1=left2, y1=cy(v), x2=left2 + pw, y2=cy(v), stroke=GRID)
        svg.text(left2 - 8, cy(v) + 4, f"{100 * v:.0f}%", 11, fill=MUTED, text_anchor="end")
    svg.add(
        "line",
        x1=left2,
        y1=cy(0.95),
        x2=left2 + pw,
        y2=cy(0.95),
        stroke=INK,
        stroke_dasharray="5 4",
    )
    for i, b in enumerate(bands):
        for k, (a, _, color) in enumerate(CALIBRATION_ARMS):
            v = max(0.6, t[a][b]["coverage"])
            x = left2 + i * slot + slot * 0.1 + k * width
            svg.add("rect", x=x, y=cy(v), width=width - 1, height=cy(0.6) - cy(v), fill=color)
        svg.text(left2 + (i + 0.5) * slot, top + ph + 16, b, 10, fill=MUTED, text_anchor="middle")
    svg.text(
        left2 + pw / 2,
        top + ph + 40,
        "rating band of the game",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    return svg.render()


FIGURES = {
    "board": board,
    "settle": settle,
    "blunders": blunders,
    "estimate": estimate,
    "calibration": calibration,
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw())


if __name__ == "__main__":
    main()
