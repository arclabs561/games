"""The strategy cards the search found, the frontier of rules against points, and the cheat sheet.

The search (`learn.py`) runs on GitHub Actions and its cards are committed in `results/` as
JSON lines, one card per accepted step, each with its exact expected score. This module reads
them and works out, all exactly:

- the **frontier**: for each number of rules, the best card found with at most that many;
- the **cheat sheet**: the best card with at most `SHEET_RULES` rules, its score distribution
  and its average in each box, next to optimal play's;
- where the sheet loses its points: the sheet's boxes with optimal keeps, and the sheet's keeps
  with optimal boxes;
- Glenn's published rule list, scored the same way (`glenn.py`).

It writes `results/cards.json`, `frontier.svg` and `scorecard.svg`.

    uv run python -m yahtzee.cards      # about 15 minutes
"""

import json
import sys
from html import escape
from pathlib import Path

import numpy as np

from yahtzee import policy
from yahtzee.glenn import GLENN, GLENN_REPORTED
from yahtzee.human import KEEP_WORDS, SHORT, Strategy, card_values, expected_score, play
from yahtzee.learn import from_json, to_json
from yahtzee.rules import CATEGORIES, OFFICIAL
from yahtzee.solver import Tables, solve

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SHEET_RULES = 8
NAMES = {
    "aces": "Aces",
    "twos": "Twos",
    "threes": "Threes",
    "fours": "Fours",
    "fives": "Fives",
    "sixes": "Sixes",
    "three_kind": "Three of a Kind",
    "four_kind": "Four of a Kind",
    "full_house": "Full House",
    "small_straight": "Small Straight",
    "large_straight": "Large Straight",
    "yahtzee": "Yahtzee",
    "chance": "Chance",
}


def found() -> list[dict]:
    """Every card the search recorded, with its file."""
    out = []
    for path in sorted(RESULTS.glob("*-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            out.append(row | {"file": path.name})
    return out


def frontier(rows: list[dict]) -> list[dict]:
    """The cards that beat every card with fewer rules, the shortest first."""
    best: dict[int, dict] = {}
    for r in rows:
        if r["size"] not in best or r["score"] > best[r["size"]]["score"]:
            best[r["size"]] = r
    out, top = [], -np.inf
    for size in sorted(best):
        if best[size]["score"] > top + 1e-9:
            out.append(best[size])
            top = best[size]["score"]
    return out


PHRASES = {
    "its upper box is open": "the most common face's upper box is open",
    "one roll left": "one roll is left",
    "two rolls left": "two rolls are left",
    "on pace for the bonus": "you are on pace for the bonus",
    "the bonus is made": "the upper bonus is made",
    "Yahtzee box holds 50": "the Yahtzee box holds 50",
    "nine or more boxes open": "nine or more boxes are open",
    "six or fewer boxes open": "six or fewer boxes are open",
    "three or fewer boxes open": "three or fewer boxes are open",
}
PHRASES |= {f"{SHORT[c]} open": f"{NAMES[c]} is open" for c in CATEGORIES}
PHRASES |= {f"{SHORT[c]} filled": f"{NAMES[c]} is filled" for c in CATEGORIES}
MOST = "The most common face is the face on the most dice, the higher one on a tie."
PACE = (
    "Pace is the upper total minus 3 times the faces of the filled upper boxes; "
    "0 or more is on pace for the 63 points of the bonus."
)


def sheet_lines(card: Strategy) -> list[str]:
    """The cheat sheet as plain lines, the way a player reads it."""
    lines = ["Before each reroll, keep the dice of the first rule that applies.", ""]
    for i, r in enumerate(card.keep_rules, 1):
        lines.append(f"{i}. If {words(r.tests)}: keep {SHEET_KEEPS[r.keep]}.")
    lines.append(f"{len(card.keep_rules) + 1}. Otherwise keep {SHEET_KEEPS[card.default]}.")
    lines += ["", "After the last roll, score in the box of the first rule that applies.", ""]
    for i, r in enumerate(card.box_rules, 1):
        where = "the upper box of that face" if r.box == "face" else NAMES[r.box]
        lines.append(f"{i}. If {words(r.tests)}: {where}.")
    n = len(card.box_rules) + 1
    lines.append(f"{n}. Otherwise the open box whose points minus par are largest.")
    par = dict(card.par)
    lines += ["", "Par: " + ", ".join(f"{NAMES[c]} {par[c]:g}" for c in CATEGORIES) + "."]
    if card.weight != 1:
        lines.append(
            f"Until the bonus is made, count an upper box's points minus par {card.weight:g} times."
        )
    tests = {t for r in (*card.keep_rules, *card.box_rules) for t in r.tests}
    return lines + ["", MOST] + ([PACE] if "on pace for the bonus" in tests else [])


SHEET_KEEPS = KEEP_WORDS | {
    "most": "every die of the most common face",
    "most open": "every die of the most common face among those whose upper box is open "
    "(the most common face if none is)",
    "run": "one die of each face in the longest run",
    "pairs": "every die whose face shows two or more times",
}
ROLL = ("a ", "two pair", "three or more", "four or more", "five ")


def phrase(test: str) -> str:
    if test in PHRASES:
        return PHRASES[test]
    return "the roll has " + test if test.startswith(ROLL) else test


def words(tests: tuple[str, ...]) -> str:
    parts = [phrase(t) for t in tests if t != "it scores"]
    if "it scores" in tests:
        parts.append("the box scores")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


class BoxesOnly:
    """The card's boxes; the solver chooses the keeps."""

    def __init__(self, card):
        self.card = card

    def boxes(self, t, states):
        return self.card.boxes(t, states)

    def keeps(self, t, states, left):
        return None


class KeepsOnly(BoxesOnly):
    """The card's keeps; the solver chooses the boxes."""

    def boxes(self, t, states):
        return None

    def keeps(self, t, states, left):
        return self.card.keeps(t, states, left)


def analyze(card: Strategy, t: Tables) -> dict:
    """The card's exact score, distribution, box averages and the split of its loss."""
    _, values = card_values(card, t)
    g = policy.graph(t, values, rule=card)
    dist = policy.distribution(g)
    split = {}
    for name, rule in (("optimal keeps", BoxesOnly(card)), ("optimal boxes", KeepsOnly(card))):
        _, v = solve(OFFICIAL, rule=rule)
        split[name] = float(v[t.index(0, 0, 0)])
    return {
        "score": float(values[t.index(0, 0, 0)]),
        "summary": policy.summary(dist),
        "boxes": dict(zip(CATEGORIES, g.stats["points"].tolist(), strict=True)),
        "upper_bonus": float(g.stats["upper_bonus"][0]),
        "yahtzee_bonus": float(g.stats["bonus"].sum()),
        "split": split,
    }


def optimal_boxes(t: Tables) -> dict:
    t, values = solve(OFFICIAL)
    g = policy.graph(t, values)
    return {
        "score": float(values[t.index(0, 0, 0)]),
        "boxes": dict(zip(CATEGORIES, g.stats["points"].tolist(), strict=True)),
        "upper_bonus": float(g.stats["upper_bonus"][0]),
        "yahtzee_bonus": float(g.stats["bonus"].sum()),
    }


def frontier_svg(rows: list[dict], front: list[dict], optimal: float, glenn: float) -> str:
    """Expected score against rules on the card: every card found, and the frontier."""
    w, h, left, right, top, bottom = 760, 420, 60, 20, 20, 50
    xmax = max(r["size"] for r in rows) + 1
    ylo, yhi = 170, 260

    def x(v):
        return left + (w - left - right) * v / xmax

    def y(v):
        return top + (h - top - bottom) * (yhi - v) / (yhi - ylo)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
    ]
    for v in range(ylo, yhi + 1, 10):
        out.append(
            f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e5e5"/>'
        )
        out.append(f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in range(0, xmax + 1, 5):
        out.append(f'<text x="{x(v):.1f}" y="{h - bottom + 16}" text-anchor="middle">{v}</text>')
    out.append(
        f'<text x="{(w + left) / 2}" y="{h - 10}" text-anchor="middle">rules on the card</text>'
    )
    out.append(
        f'<text transform="translate(16 {(h - bottom) / 2}) rotate(-90)" text-anchor="middle">'
        "expected score (exact)</text>"
    )
    for v, label, dash in (
        (optimal, f"optimal play {optimal:.2f}", ""),
        (optimal - 5, "5 points below optimal", ' stroke-dasharray="4 4"'),
    ):
        out.append(
            f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#555"{dash}/>'
        )
        out.append(f'<text x="{w - right - 4}" y="{y(v) - 5:.1f}" text-anchor="end">{label}</text>')
    for r in rows:
        if r["score"] >= ylo:
            out.append(
                f'<circle cx="{x(r["size"]):.1f}" cy="{y(r["score"]):.1f}" r="2.5" fill="#bbb"/>'
            )
    pts = " ".join(f"{x(r['size']):.1f},{y(r['score']):.1f}" for r in front if r["score"] >= ylo)
    out.append(f'<polyline points="{pts}" fill="none" stroke="#1f5fa8" stroke-width="2"/>')
    for r in front:
        if r["score"] >= ylo:
            out.append(
                f'<circle cx="{x(r["size"]):.1f}" cy="{y(r["score"]):.1f}" r="3.5" fill="#1f5fa8"/>'
            )
    gx = min(25, xmax)
    out.append(
        f'<rect x="{x(gx) - 4:.1f}" y="{y(glenn) - 4:.1f}" width="8" height="8" fill="#c0392b"/>'
    )
    out.append(
        f'<text x="{x(gx) - 8:.1f}" y="{y(glenn) + 18:.1f}" text-anchor="end" fill="#c0392b">'
        f"Glenn's 25 rules, {glenn:.2f}</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def scorecard_svg(sheet: dict, optimal: dict) -> str:
    """A score card with the sheet's average in each box beside optimal play's."""
    rows = [(NAMES[c], sheet["boxes"][c], optimal["boxes"][c]) for c in CATEGORIES[:6]]
    rows.append(("Upper bonus", 35 * sheet["upper_bonus"], 35 * optimal["upper_bonus"]))
    rows += [(NAMES[c], sheet["boxes"][c], optimal["boxes"][c]) for c in CATEGORIES[6:]]
    rows.append(("Yahtzee bonus", sheet["yahtzee_bonus"], optimal["yahtzee_bonus"]))
    rows.append(("Grand total", sheet["score"], optimal["score"]))
    w, line, top = 420, 26, 70
    h = top + line * len(rows) + 20
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'font-family="Helvetica, Arial, sans-serif" font-size="14">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<rect x="10" y="10" width="{w - 20}" height="{h - 20}" fill="none" stroke="#222" stroke-width="2"/>',
        '<text x="24" y="40" font-weight="bold">Box</text>',
        f'<text x="{w - 150}" y="40" text-anchor="end" font-weight="bold">cheat sheet</text>',
        f'<text x="{w - 30}" y="40" text-anchor="end" font-weight="bold">optimal</text>',
        f'<text x="{w - 150}" y="58" text-anchor="end" font-size="11">average points</text>',
        f'<text x="{w - 30}" y="58" text-anchor="end" font-size="11">average points</text>',
    ]
    for i, (name, a, b) in enumerate(rows):
        y0 = top + i * line
        shade = "#f2f2f2" if name in ("Upper bonus", "Yahtzee bonus", "Grand total") else "white"
        out.append(
            f'<rect x="11" y="{y0}" width="{w - 22}" height="{line}" fill="{shade}" stroke="#999" stroke-width="0.5"/>'
        )
        weight = ' font-weight="bold"' if name == "Grand total" else ""
        out.append(f'<text x="24" y="{y0 + 18}"{weight}>{escape(name)}</text>')
        out.append(f'<text x="{w - 150}" y="{y0 + 18}" text-anchor="end"{weight}>{a:.2f}</text>')
        out.append(f'<text x="{w - 30}" y="{y0 + 18}" text-anchor="end"{weight}>{b:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    t = Tables(OFFICIAL)
    rows = found()
    front = frontier(rows)
    sheet_row = max((r for r in rows if r["size"] <= SHEET_RULES), key=lambda r: r["score"])
    sheet = from_json(sheet_row["card"])
    assert abs(expected_score(sheet, t) - sheet_row["score"]) < 1e-9
    stats = analyze(sheet, t)
    games = play(sheet, t, 20000, seed=1)
    half = 1.96 * games.std() / np.sqrt(len(games))
    stats["simulated"] = {
        "games": len(games),
        "seed": 1,
        "mean": float(games.mean()),
        "half": float(half),
    }
    best = optimal_boxes(t)
    glenn = expected_score(GLENN, t)
    data = {
        "optimal": best["score"],
        "glenn": glenn,
        "glenn_reported": GLENN_REPORTED,
        "frontier": [{k: r[k] for k in ("size", "score", "file", "card")} for r in front],
        "cheat": {"card": to_json(sheet), **stats},
        "optimal_boxes": best,
        "cards_found": len(rows),
    }
    (RESULTS / "cards.json").write_text(json.dumps(data, indent=1) + "\n")
    (HERE / "frontier.svg").write_text(frontier_svg(rows, front, best["score"], glenn))
    (HERE / "scorecard.svg").write_text(scorecard_svg({**stats, "score": stats["score"]}, best))
    for r in front:
        print(f"{r['size']:3d} rules  {r['score']:.4f}  {best['score'] - r['score']:6.2f} below")
    print("\n".join(sheet_lines(sheet)))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
