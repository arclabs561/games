"""The README's figures, drawn as SVG with no third-party packages.

    uv run python -m ev_nova.figures      # rewrites ev_nova/figures/*.svg

`evnova_routes` draws, on a schematic map, the EV Nova routes that are best for some cost of a
landing. The stops, jumps and goods come from the published routes in `guides.py`, and the
one equal round trip the guides do not list is added by hand; positions are not the game's
coordinates. `naev_landing` draws, for Naev, the rate of the three routes that
`routes.landing_envelope` finds best between landing costs 1/10 and 10. That search takes
about half a minute, so its answer is kept in `NAEV_ENVELOPE`; `main` checks it again, and a
test checks that the routes tie where the README says the best route changes.
"""

import math
from fractions import Fraction
from html import escape
from pathlib import Path

from ev_nova import naev
from ev_nova.guides import GUIDES
from ev_nova.routes import landing_envelope, route_parts

HERE = Path(__file__).parent
OUT = HERE / "figures"
# (landing cost from which the route is best, route), from routes.landing_envelope on Naev's
# map without hidden jumps, from 1/10 to 10.
NAEV_ENVELOPE = [
    (Fraction(1, 10), ("Dvaer Prime", "Eist Shipyard")),
    (Fraction(29, 168), ("Burnan", "Lora-Mor")),
    (Fraction(111, 86), ("Modus", "Tankard Cloister")),
]
SANS = "Helvetica, Arial, sans-serif"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#d9d8d4", "#2a78d6", "#d95926"


def num(x: float) -> str:
    """A coordinate with at most two decimals and no trailing zeros, so the bytes are stable."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


class Svg:
    """A list of SVG elements on a white background, with a title and a description."""

    def __init__(self, width: int, height: int, title: str, desc: str):
        self.width, self.height, self.title, self.desc = width, height, title, desc
        self.items: list[str] = []

    def add(self, tag: str, content: str = "", **attrs) -> None:
        text = "".join(
            f' {k.rstrip("_").replace("_", "-")}="{num(v) if isinstance(v, float | int) else escape(v)}"'
            for k, v in attrs.items()
        )
        self.items.append(
            f"<{tag}{text}>{escape(content)}</{tag}>" if content else f"<{tag}{text}/>"
        )

    def text(self, x: float, y: float, s: str, size: float = 13, **attrs) -> None:
        attrs.setdefault("fill", INK)
        self.add("text", s, x=x, y=y, font_size=size, **attrs)

    def render(self) -> str:
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}"'
                f' viewBox="0 0 {self.width} {self.height}" role="img" font-family="{SANS}">',
                f"<title>{escape(self.title)}</title>",
                f"<desc>{escape(self.desc)}</desc>",
                f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
                *self.items,
                "</svg>",
                "",
            ]
        )


# EV Nova's best routes on a schematic map

# Systems, placed by hand.
PLACES = {
    "Sol": (150, 150),
    "Kania": (330, 90),
    "Altair": (470, 60),
    "Tuatha": (70, 320),
    "Dani Fort": (430, 290),
    "Dani": (560, 230),
    "Dani Evera": (560, 350),
    "Ver'ashan": (250, 330),
}
# Systems whose name goes below the circle, clear of the lines.
LABEL_BELOW = {"Tuatha"}
# The routes drawn: those best somewhere in the study (README), by their names in guides.py.
SHOWN = ["Kania/Altair", "Dani/Dani Fort", "Ver'ashan", "Kania/Sol", "Sol/Tuatha"]
# Forticus and Palein tie with the best round trip but are on neither page (README).
UNLISTED = (("Forticus (Dani Fort)", "Palein (Dani Evera)"), 1, "Equipment", "Medical Supplies")


def system_of(stop: str) -> str:
    return stop[stop.index("(") + 1 : -1]


def evnova_routes() -> str:
    guides = {g.name: g for g in GUIDES}
    legs = []  # (system a, system b, jumps, goods a to b, goods b to a, listed)
    stops: dict[str, set[str]] = {s: set() for s in PLACES}
    for name in SHOWN:
        g = guides[name]
        for i, stop in enumerate(g.stops):
            stops[system_of(stop)].add(stop[: stop.index(" (")])
            nxt = g.stops[(i + 1) % len(g.stops)]
            a, b = system_of(stop), system_of(nxt)
            goods = ", ".join(c for c, _, _ in g.legs[i]) if g.legs[i] else ""
            legs.append((a, b, g.jumps[i], goods, True))
    (a_stop, b_stop), jumps, there, back = UNLISTED
    for stop in (a_stop, b_stop):
        stops[system_of(stop)].add(stop[: stop.index(" (")])
    legs.append((system_of(a_stop), system_of(b_stop), jumps, there, False))
    svg = Svg(
        660,
        440,
        "EV Nova's best trade routes on a schematic map",
        "Systems are circles, with their trading stops named inside. Lines join the stops of "
        "the best routes, labeled with the jumps per leg. Kania to Altair and Dani Fort to Dani "
        "are one jump and earn 217 credits per ton per jump. Sol to Kania is two jumps, Sol to "
        "Tuatha three. The two stations of Ver'ashan are in one system, no jump apart.",
    )
    drawn = set()
    for a, b, jumps, _, listed in legs:
        key = tuple(sorted((a, b)))
        if a == b or key in drawn:
            continue
        drawn.add(key)
        (x1, y1), (x2, y2) = PLACES[a], PLACES[b]
        dash = {} if jumps == 1 else {"stroke_dasharray": "7 5"}
        color = "#2a78d6" if listed else "#d95926"
        svg.add("line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=3, **dash)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        svg.add(
            "rect", x=mx - 30, y=my - 10, width=60, height=18, rx=9, fill="#ffffff", stroke=color
        )
        svg.text(
            mx,
            my + 3,
            f"{jumps} jump{'s' if jumps > 1 else ''}",
            11,
            fill=INK,
            text_anchor="middle",
        )
    for system, (x, y) in PLACES.items():
        names = sorted(stops[system])
        r = 30 + 6 * (len(names) - 1)
        svg.add("circle", cx=x, cy=y, r=r, fill="#f4f3ef", stroke=INK, stroke_width=1.5)
        label_y = y + r + 16 if system in LABEL_BELOW else y - r - 6
        svg.text(x, label_y, system, 13, font_weight="bold", text_anchor="middle")
        if system == "Ver'ashan":
            svg.text(x, y + r + 16, "two stations, no jump", 11, fill=MUTED, text_anchor="middle")
        for i, name in enumerate(names):
            svg.text(
                x,
                y + 4 + 14 * (i - (len(names) - 1) / 2),
                name,
                11,
                fill=MUTED,
                text_anchor="middle",
            )
    svg.add("line", x1=20, y1=410, x2=50, y2=410, stroke="#2a78d6", stroke_width=3)
    svg.text(56, 414, "on the wiki's route pages", 12, fill=MUTED)
    svg.add("line", x1=230, y1=410, x2=260, y2=410, stroke="#d95926", stroke_width=3)
    svg.text(266, 414, "equal best round trip, not listed", 12, fill=MUTED)
    svg.text(
        20,
        432,
        "Schematic: positions are not the game's. A dashed line passes systems not drawn.",
        12,
        fill=MUTED,
    )
    return svg.render()


# Naev: the best route as the cost of a landing changes


def naev_parts() -> list[tuple[tuple[str, ...], int, int, int]]:
    g = naev.galaxy(naev.load(), hidden=False)
    return [(route, *route_parts(g, route)) for _, route in NAEV_ENVELOPE]


def naev_landing() -> str:
    parts = naev_parts()
    left, top, width, height = 70, 40, 540, 280
    xlo, xhi, ylo, yhi = math.log10(0.1), math.log10(10), math.log10(5), math.log10(1000)

    def sx(x: float) -> float:
        return left + (math.log10(x) - xlo) / (xhi - xlo) * width

    def sy(y: float) -> float:
        return top + height - (math.log10(y) - ylo) / (yhi - ylo) * height

    svg = Svg(
        left + width + 30,
        top + height + 70,
        "Naev: the rate of the three routes that are best for some landing cost",
        "A log-log plot of credits per ton per unit of cost against the cost of a landing in "
        "jumps, from 1/10 to 10, for three routes. Dvaer Prime and Eist Shipyard, in one "
        "system, are best below 29/168; Burnan and Lora-Mor up to 111/86; Modus and Tankard "
        "Cloister above. The best rate is drawn thick.",
    )
    for v in (10, 100, 1000):
        svg.add("line", x1=left, y1=sy(v), x2=left + width, y2=sy(v), stroke=GRID)
        svg.text(left - 8, sy(v) + 4, f"{v:,}", 11, fill=MUTED, text_anchor="end")
    for v, label in ((0.1, "1/10"), (0.3, "0.3"), (1, "1"), (3, "3"), (10, "10")):
        svg.add("line", x1=sx(v), y1=top, x2=sx(v), y2=top + height, stroke=GRID)
        svg.text(sx(v), top + height + 16, label, 11, fill=MUTED, text_anchor="middle")
    svg.text(
        left + width / 2,
        top + height + 36,
        "cost of a landing, in jumps (log scale)",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    svg.text(10, 24, "credits per ton per unit of cost (log scale)", 12, fill=MUTED)
    colors = ["#2a78d6", "#d95926", "#1baf7a"]
    xs = [10 ** (xlo + (xhi - xlo) * i / 200) for i in range(201)]
    starts = [float(x) for x, _ in NAEV_ENVELOPE] + [10.0]
    for k, (route, gain, jumps, stops) in enumerate(parts):

        def rate(x: float, gain=gain, jumps=jumps, stops=stops) -> float:
            return gain / (jumps + x * stops)

        line = " ".join(f"{num(sx(x))},{num(sy(rate(x)))}" for x in xs if rate(x) >= 5)
        svg.add(
            "polyline",
            points=line,
            fill="none",
            stroke=colors[k],
            stroke_width=1.5,
            stroke_dasharray="4 3",
        )
        best = [x for x in xs if starts[k] <= x <= starts[k + 1]] + [starts[k + 1]]
        best = [starts[k], *best]
        line = " ".join(f"{num(sx(x))},{num(sy(rate(x)))}" for x in best)
        svg.add("polyline", points=line, fill="none", stroke=colors[k], stroke_width=4)
        y = top + 16 + 18 * k
        svg.add("rect", x=left + width - 250, y=y - 9, width=12, height=12, fill=colors[k])
        svg.text(
            left + width - 232, y + 2, f"{' and '.join(route)}: {gain} / ({jumps} + {stops}x)", 12
        )
    svg.text(
        10,
        top + height + 58,
        "Thick: the best route at that cost. The rate of a route is its gain per ton per pass over jumps plus landings.",
        11,
        fill=MUTED,
    )
    return svg.render()


# Naev's map, from the positions in its XML files

MAP_SCALE = 0.25  # pixels per unit of Naev's map coordinates
MAP_LABELS = ("Iris", "Holly", "Tau Ceti")  # the three systems with the highest betweenness


def naev_links(data: dict, hidden: bool) -> list[tuple[str, str]]:
    """Each link between two systems once, sorted: usable both ways, or with `hidden` only those
    hidden one way or both."""
    systems = data["systems"]
    out = set()
    for a, s in systems.items():
        for b, flags in s["jumps"].items():
            if b not in systems:
                continue
            back = systems[b]["jumps"].get(a)
            if "exitonly" in flags or back is None or "exitonly" in back:
                continue
            is_hidden = "hidden" in flags or "hidden" in back
            if is_hidden == hidden:
                out.add((min(a, b), max(a, b)))
    return sorted(out)


def shortest_paths(data: dict, a: str, b: str) -> list[list[str]]:
    """Every shortest path from system a to system b on links that are not hidden."""
    nbrs: dict[str, list[str]] = {}
    for x, y in naev_links(data, hidden=False):
        nbrs.setdefault(x, []).append(y)
        nbrs.setdefault(y, []).append(x)
    dist = {a: 0}
    frontier = [a]
    while frontier and b not in dist:
        nxt = []
        for x in frontier:
            for y in sorted(nbrs.get(x, [])):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    nxt.append(y)
        frontier = nxt

    def back(x: str) -> list[list[str]]:
        if x == a:
            return [[a]]
        return [
            path + [x] for y in sorted(nbrs[x]) if dist.get(y) == dist[x] - 1 for path in back(y)
        ]

    return back(b)


def naev_map() -> str:
    data = naev.load()
    systems = data["systems"]
    xs = [s["pos"][0] for s in systems.values()]
    ys = [s["pos"][1] for s in systems.values()]
    pad, top = 20, 50
    x0, y1 = min(xs), max(ys)

    def px(name: str) -> tuple[float, float]:
        x, y = systems[name]["pos"]
        return pad + (x - x0) * MAP_SCALE, top + (y1 - y) * MAP_SCALE  # y up, as Naev draws it

    width = round(2 * pad + (max(xs) - x0) * MAP_SCALE)
    height = round(top + pad + 40 + (y1 - min(ys)) * MAP_SCALE)
    svg = Svg(
        width,
        height,
        "Naev's map from its data files, with the three routes that are best for some landing cost",
        f"All {len(systems)} systems of Naev at their positions in the game's data, joined by "
        "their jumps; hidden jumps are dashed. Three routes are drawn thick: Dvaer Prime and "
        "Eist Shipyard, both in Dvaer; Burnan in Scarlet and Lora-Mor in Druss, one jump "
        "apart; and Modus in Modus Manis and Tankard Cloister in An'ku, two jumps apart by "
        "either of two paths.",
    )
    for hidden, color, dash in ((True, "#c9c8c4", "3 3"), (False, "#b5b4b0", "")):
        for a, b in naev_links(data, hidden):
            (xa, ya), (xb, yb) = px(a), px(b)
            attrs = {"stroke_dasharray": dash} if dash else {}
            svg.add("line", x1=xa, y1=ya, x2=xb, y2=yb, stroke=color, stroke_width=0.8, **attrs)
    traders = {s["system"] for s in data["spobs"].values()}
    for name in sorted(systems):
        x, y = px(name)
        fill = INK if name in traders else "#9a9995"
        svg.add("circle", cx=x, cy=y, r=2.2 if name in traders else 1.6, fill=fill)
    colors = ["#2a78d6", "#d95926", "#1baf7a"]
    labels: dict[str, str] = {}
    for k, (_, route) in enumerate(NAEV_ENVELOPE):
        where = [data["spobs"][stop]["system"] for stop in route]
        for i, a in enumerate(where):
            b = where[(i + 1) % len(where)]
            if a == b:
                continue
            for path in shortest_paths(data, a, b):
                for u, v in zip(path, path[1:], strict=False):
                    (xa, ya), (xb, yb) = px(u), px(v)
                    svg.add("line", x1=xa, y1=ya, x2=xb, y2=yb, stroke=colors[k], stroke_width=4)
        for a in where:
            x, y = px(a)
            svg.add("circle", cx=x, cy=y, r=5, fill=colors[k])
            labels[a] = colors[k]
    for name in MAP_LABELS:
        labels.setdefault(name, MUTED)
        x, y = px(name)
        svg.add("circle", cx=x, cy=y, r=4, fill="none", stroke=MUTED, stroke_width=1.5)
    for name, color in sorted(labels.items()):
        x, y = px(name)
        svg.text(x + 7, y - 6, name, 12, fill=color, font_weight="bold")
    x = pad
    for k, (_, route) in enumerate(NAEV_ENVELOPE):
        svg.add("rect", x=x, y=14, width=12, height=12, fill=colors[k])
        svg.text(x + 17, 25, " and ".join(route), 12)
        x += 230
    svg.text(
        pad,
        height - 14,
        "Dark dots: systems with a trading stop. Circled: the three systems with the highest "
        "betweenness. Dashed: hidden jumps.",
        11,
        fill=MUTED,
    )
    return svg.render()


FIGURES = {"evnova_routes": evnova_routes, "naev_landing": naev_landing, "naev_map": naev_map}


def main() -> None:
    g = naev.galaxy(naev.load(), hidden=False)
    assert landing_envelope(g, Fraction(1, 10), Fraction(10)) == NAEV_ENVELOPE
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
