"""The README's figures, drawn as SVG from the committed results, with no plotting package.

    uv run python -m chess_ratings.figures      # rewrites chess_ratings/figures/*.svg

A test compares every figure with the committed file.
"""

import json
from pathlib import Path

import chess

from chess_ratings.board_svg import SQUARE, draw_arrow, draw_board
from peggle.figures import BLUE, GRID, INK, MUTED, ORANGE, Svg, num

HERE = Path(__file__).parent
OUT = HERE / "figures"
RESULTS = HERE / "results"
SCHOLAR = "e2e4 e7e5 d1h5 b8c6 f1c4"
COLORS = {"800": "#7a4bc2", "1100": BLUE, "1900": "#1a9a6c", "2300": ORANGE}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def board() -> str:
    """The position after 1. e4 e5 2. Qh5 Nc6 3. Bc4, with Maia-3's three likeliest replies
    for players rated 1100 (left) and 1900 (right)."""
    dists = load("whatwould.json")["distributions"][SCHOLAR]
    b = chess.Board()
    for m in SCHOLAR.split():
        b.push_uci(m)
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
        probs = dists[rating]
        best = sorted(probs, key=lambda m: (-probs[m], m))[:3]
        for m in best:
            draw_arrow(svg, chess.Move.from_uci(m), probs[m], left, top)
        y = top + 8 * SQUARE + 38
        for j, m in enumerate(best):
            san = b.san(chess.Move.from_uci(m))
            svg.text(left + j * 120, y, f"{san}  {100 * probs[m]:.0f}%", 13, fill=INK)
        nf6 = probs.get("g8f6", 0.0)
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


FIGURES = {"board": board, "settle": settle}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw())


if __name__ == "__main__":
    main()
