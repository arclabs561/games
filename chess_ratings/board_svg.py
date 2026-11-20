"""A chess board as SVG, drawn here with letters for pieces (no piece artwork).

White pieces are white disks with a black letter, Black pieces black disks with a white letter
(K, Q, R, B, N, P). Arrows show candidate moves, their width and label set by a probability.
"""

from math import atan2, cos, sin

import chess

from peggle.figures import INK, MUTED, Svg, num

LIGHT, DARK = "#eeeae2", "#c9c1b1"
SQUARE = 48
ARROW = "#2a78d6"


def centre(square: chess.Square, left: float, top: float) -> tuple[float, float]:
    f, r = chess.square_file(square), chess.square_rank(square)
    return left + (f + 0.5) * SQUARE, top + (7.5 - r) * SQUARE


def draw_board(svg: Svg, board: chess.Board, left: float, top: float) -> None:
    """Squares, coordinates and pieces of `board`, White at the bottom."""
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        svg.add(
            "rect",
            x=left + f * SQUARE,
            y=top + (7 - r) * SQUARE,
            width=SQUARE,
            height=SQUARE,
            fill=LIGHT if (f + r) % 2 else DARK,
        )
    svg.add("rect", x=left, y=top, width=8 * SQUARE, height=8 * SQUARE, fill="none", stroke=INK)
    for i in range(8):
        svg.text(
            left + (i + 0.5) * SQUARE,
            top + 8 * SQUARE + 15,
            "abcdefgh"[i],
            11,
            fill=MUTED,
            text_anchor="middle",
        )
        svg.text(
            left - 8, top + (7.5 - i) * SQUARE + 4, str(i + 1), 11, fill=MUTED, text_anchor="end"
        )
    for sq, piece in sorted(board.piece_map().items()):
        x, y = centre(sq, left, top)
        white = piece.color == chess.WHITE
        svg.add(
            "circle",
            cx=x,
            cy=y,
            r=17,
            fill="#ffffff" if white else "#1f1f1f",
            stroke=INK,
            stroke_width=1.5,
        )
        svg.text(
            x,
            y + 6,
            piece.symbol().upper(),
            17,
            fill=INK if white else "#ffffff",
            font_weight="bold",
            text_anchor="middle",
        )


def draw_arrow(
    svg: Svg, move: chess.Move, weight: float, left: float, top: float, color: str = ARROW
) -> None:
    """An arrow for a move; `weight` in [0, 1] sets its width."""
    x1, y1 = centre(move.from_square, left, top)
    x2, y2 = centre(move.to_square, left, top)
    angle = atan2(y2 - y1, x2 - x1)
    head = 14
    bx, by = x2 - head * cos(angle), y2 - head * sin(angle)
    width = 2 + 10 * weight
    svg.add(
        "line",
        x1=x1,
        y1=y1,
        x2=bx,
        y2=by,
        stroke=color,
        stroke_width=width,
        stroke_opacity="0.75",
        stroke_linecap="round",
    )
    spread = 0.5
    p1 = (x2 - head * 1.6 * cos(angle - spread), y2 - head * 1.6 * sin(angle - spread))
    p2 = (x2 - head * 1.6 * cos(angle + spread), y2 - head * 1.6 * sin(angle + spread))
    svg.add(
        "polygon",
        points=f"{num(x2)},{num(y2)} {num(p1[0])},{num(p1[1])} {num(p2[0])},{num(p2[1])}",
        fill=color,
        fill_opacity="0.75",
    )
