"""Draw the README's pictures as SVG, with no third-party packages.

    uv run python -m cannonballs.figures     # writes pile.svg and gaps.svg

`pile.svg` shows 4,900 balls two ways: as a square pile of 24 layers and as a flat square of
70 by 70. `gaps.svg` shows where each square pile's count falls between two squares.
"""

from math import cos, radians, sin, sqrt
from pathlib import Path

from cannonballs import lucas, piles

FOLDER = Path(__file__).resolve().parent
GREY, RED = "#5b6770", "#D55E00"


def project(x: float, y: float, z: float, turn: float = 32, tilt: float = 24):
    """Screen position (u right, v up) and nearness to the viewer of a point in space."""
    t, p = radians(turn), radians(tilt)
    u = x * cos(t) - y * sin(t)
    w = x * sin(t) + y * cos(t)  # depth: larger is farther away
    return u, w * sin(p) + z * cos(p), -w * cos(p) + z * sin(p)


def pile_centers(n: int) -> list[tuple[float, float, float]]:
    """Centers of the balls of a square pile of n layers, radius 1, bottom layer first."""
    return [
        (2 * i + h, 2 * j + h, h * sqrt(2))
        for h in range(n)
        for i in range(n - h)
        for j in range(n - h)
    ]


def svg(width: int, height: int, body: list[str], title: str) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">'
    )
    defs = (
        "<defs>"
        '<radialGradient id="ball" cx="0.35" cy="0.32" r="0.75">'
        '<stop offset="0" stop-color="#ffffff"/><stop offset="0.45" stop-color="#8fa3b4"/>'
        '<stop offset="1" stop-color="#36444f"/></radialGradient>'
        f'<circle id="b" r="4.6" fill="url(#ball)"/>'
        '<pattern id="dot" width="4" height="4" patternUnits="userSpaceOnUse">'
        f'<circle cx="2" cy="2" r="1.7" fill="{GREY}"/></pattern>'
        "</defs>"
    )
    return (
        "\n".join(
            [
                head,
                f"<title>{title}</title>",
                defs,
                f'<rect width="{width}" height="{height}" fill="white"/>',
                *body,
                "</svg>",
            ]
        )
        + "\n"
    )


def pile_svg() -> str:
    n, scale = 24, 4.6
    pts = sorted((project(x, y, z) for x, y, z in pile_centers(n)), key=lambda p: p[2])
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    sq_top, side = 60, 70 * 4
    left = 20 - min(us) * scale + scale
    top = sq_top + side / 2 + (max(vs) + min(vs)) / 2 * scale  # pile centered beside the square
    # farthest first, so that nearer balls cover them; every ball is drawn, since the
    # hollows between balls show parts of the balls beneath
    body = [
        f'<use href="#b" x="{left + u * scale:.1f}" y="{top - v * scale:.1f}"/>' for u, v, _ in pts
    ]
    pile_right = left + max(us) * scale + scale
    sq_left = pile_right + 60
    body.append(
        f'<rect x="{sq_left}" y="{sq_top}" width="{side}" height="{side}" fill="url(#dot)"/>'
    )
    caption_y = sq_top + side + 34
    body += [
        f'<text x="{(20 + pile_right) / 2:.0f}" y="{caption_y}" text-anchor="middle" font-size="15">'
        "a square pile of 24 layers</text>",
        f'<text x="{sq_left + side / 2:.0f}" y="{caption_y}" text-anchor="middle" font-size="15">'
        "a flat square, 70 by 70</text>",
        f'<text x="{(20 + sq_left + side) / 2:.0f}" y="28" text-anchor="middle" font-size="17" '
        'font-weight="bold">4,900 balls, two ways</text>',
    ]
    width = int(sq_left + side + 20)
    return svg(
        width,
        int(caption_y + 16),
        body,
        "4,900 balls as a square pile of 24 layers and as a 70 by 70 square",
    )


def gaps_svg(layers: int = 60) -> str:
    w, h, x0, y0 = 760, 300, 60, 250
    dx, dy = (w - x0 - 20) / layers, 190
    body = [
        f'<line x1="{x0}" y1="{y0}" x2="{w - 15}" y2="{y0}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - dy}" stroke="#222"/>',
        f'<text x="{x0 - 8}" y="{y0 + 5}" text-anchor="end" font-size="13">0</text>',
        f'<text x="{x0 - 8}" y="{y0 - dy + 5}" text-anchor="end" font-size="13">1</text>',
        f'<text x="{(x0 + w) / 2:.0f}" y="{y0 + 40}" text-anchor="middle" font-size="14">layers in the square pile</text>',
        f'<text x="{x0}" y="24" font-size="14">where the count falls between the square below it (0) and the next square (1)</text>',
    ]
    for n in range(1, layers + 1):
        m = piles.pyramid(n)
        r = lucas.isqrt(m)
        share = (m - r * r) / (2 * r + 1)
        cx, cy = x0 + n * dx, y0 - share * dy
        hit = share == 0
        body.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{5 if hit else 3.2}" '
            f'fill="{RED if hit else GREY}"/>'
        )
        if n % 10 == 0 or hit:
            body.append(
                f'<text x="{cx:.1f}" y="{y0 + 18}" text-anchor="middle" font-size="13">{n}</text>'
            )
    return svg(
        w,
        h,
        body,
        "For square piles of 1 to 60 layers, where the number of balls falls between two squares; only 1 and 24 land on a square",
    )


FIGURES = {"pile.svg": pile_svg, "gaps.svg": gaps_svg}


def main() -> None:
    for name, draw in FIGURES.items():
        (FOLDER / name).write_text(draw())
        print("wrote", name)


if __name__ == "__main__":
    main()
