"""Pictures for the README, written as plain SVG text so that the bytes are the same on
every run.

- `deal_svg` draws a deal as eight fanned columns, with chosen cards shaded and outlined.
- `bars_svg` draws how many of the first 32,000 deals need each number of free cells.

    uv run python -m freecell.draw      # writes results/deal-11982.svg and results/least-cells.svg

The deal picture marks the eight cards of deal 11982 that no reachable position moves (see
`test_deal_11982_is_lost_with_four_cells_and_won_with_five`). The bar chart reads
`results/least_cells.tsv`, which `freecell.report` writes.
"""

from collections import Counter
from pathlib import Path

from freecell.deals import Card, Layout, is_red, ms_deal, parse_card, rank, suit

HERE = Path(__file__).parent
RESULTS = HERE / "results"

SYMBOL = "♣♦♥♠"  # clubs, diamonds, hearts, spades
RANK_TEXT = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
INK, INK_2, RED, GRID = "#0b0b0b", "#52514e", "#b3261e", "#e1e0d9"
MARK_FILL, MARK_EDGE, BAR = "#fbe3a1", "#8a5a00", "#2a78d6"
FONT = "font-family='Helvetica, Arial, sans-serif'"
FROZEN_11982 = ("AC", "AS", "3H", "4H", "6H", "6S", "QC", "QS")


def _card(x: int, y: int, c: Card, marked: bool, w: int, h: int) -> str:
    fill, edge, width = (MARK_FILL, MARK_EDGE, 2.5) if marked else ("#ffffff", "#9a9890", 1)
    color = RED if is_red(c) else INK
    label = RANK_TEXT[rank(c) - 1] + SYMBOL[suit(c)]
    return (
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='6' fill='{fill}' "
        f"stroke='{edge}' stroke-width='{width}'/>"
        f"<text x='{x + 7}' y='{y + 20}' font-size='17' fill='{color}' {FONT}>{label}</text>"
    )


def deal_svg(layout: Layout, marked: frozenset[Card] = frozenset(), title: str = "") -> str:
    """The deal as columns, bottom card first; cards in `marked` are shaded and outlined."""
    w, h, gap, step, left, top = 70, 96, 12, 28, 16, 44 if title else 16
    height = top + step * (max(map(len, layout)) - 1) + h + 16
    width = 2 * left + len(layout) * w + (len(layout) - 1) * gap
    out = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
    ]
    if title:
        out.append(f"<text x='{left}' y='28' font-size='16' fill='{INK}' {FONT}>{title}</text>")
    for i, col in enumerate(layout):
        x = left + i * (w + gap)
        for j, c in enumerate(col):
            out.append(_card(x, top + j * step, c, c in marked, w, h))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def bars_svg(counts: dict[str, int], title: str) -> str:
    """A bar for each label, its count written above it; one series, so no legend."""
    labels = list(counts)
    bw, gap, left, top, plot_h, bottom = 80, 28, 64, 48, 220, 52
    width = left + len(labels) * (bw + gap) + 16
    height = top + plot_h + bottom
    peak = max(counts.values())
    scale = 10 ** (len(str(peak)) - 1)
    ymax = -(-peak // scale) * scale
    out = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
        f"<text x='16' y='26' font-size='16' fill='{INK}' {FONT}>{title}</text>",
    ]
    base = top + plot_h
    for k in range(0, ymax + 1, ymax // 4):
        y = base - plot_h * k / ymax
        out.append(
            f"<line x1='{left - 6}' x2='{width - 16}' y1='{y:.1f}' y2='{y:.1f}' "
            f"stroke='{GRID}' stroke-width='1'/>"
            f"<text x='{left - 10}' y='{y + 4:.1f}' font-size='12' fill='{INK_2}' "
            f"text-anchor='end' {FONT}>{k:,}</text>"
        )
    for i, lab in enumerate(labels):
        n = counts[lab]
        x = left + gap / 2 + i * (bw + gap)
        bh = plot_h * n / ymax
        out.append(
            f"<rect x='{x:.1f}' y='{base - bh:.1f}' width='{bw}' height='{bh:.1f}' "
            f"fill='{BAR}'><title>{lab}: {n:,} deals</title></rect>"
            f"<text x='{x + bw / 2:.1f}' y='{base - bh - 6:.1f}' font-size='13' fill='{INK}' "
            f"text-anchor='middle' {FONT}>{n:,}</text>"
            f"<text x='{x + bw / 2:.1f}' y='{base + 20}' font-size='13' fill='{INK_2}' "
            f"text-anchor='middle' {FONT}>{lab}</text>"
        )
    out.append(
        f"<text x='{left + (width - left) / 2:.0f}' y='{height - 10}' font-size='13' "
        f"fill='{INK_2}' text-anchor='middle' {FONT}>least number of free cells that wins</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def least_counts(tsv: Path = RESULTS / "least_cells.tsv") -> dict[str, int]:
    """Deals by least winning number of free cells, as Solver B found them."""
    rows = [line.split("\t") for line in tsv.read_text().splitlines()[1:]]
    found = Counter(row[2] for row in rows)
    labels = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5 or more"}
    return {lab: found[k] for k, lab in labels.items()}


def figures() -> dict[str, str]:
    """Every picture, by file name."""
    frozen = frozenset(parse_card(n) for n in FROZEN_11982)
    return {
        "deal-11982.svg": deal_svg(
            ms_deal(11982), frozen, "Deal 11982; shaded cards never move in any reachable position"
        ),
        "least-cells.svg": bars_svg(
            least_counts(), "Deals among the first 32,000 (the 31,986 that Solver B decides)"
        ),
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for name, text in figures().items():
        (RESULTS / name).write_text(text)
        print(RESULTS / name)


if __name__ == "__main__":
    main()
