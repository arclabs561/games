#import "@preview/cetz:0.4.2"
#import "@preview/lilaq:0.6.0" as lq
#import "@preview/codly:1.3.0": codly, codly-init

// Every number in this note comes from results.json and gaps.csv, which report.py writes, or
// is checked against them by tests/test_cannonballs.py. The tables read results.json directly.
#let R = json("results.json")

// ---------- editions (as in the Erase paper) ----------
// print (default): US letter. phone: a narrow page, compiled with sys input edition=phone.
// html: compiled with format "html"; drawings become inline SVG through html.frame.
#let edition = sys.inputs.at("edition", default: "print")
#let phone = edition == "phone"
#let vis(x) = context if target() == "html" { box(html.frame(x)) } else { x }    // inline drawing
#let visb(x) = context if target() == "html" { html.frame(x) } else { x }         // block drawing
#let fitw(x) = layout(size => {
  let w = measure(x).width
  if w > size.width { scale(size.width / w * 100%, reflow: true, x) } else { x }
})
#let wide(x, w: 4.8in) = if not phone { x } else { layout(size => {
  let m = measure(x, width: w)
  if m.width > size.width { scale(size.width / m.width * 100%, reflow: true, block(width: m.width, x)) } else { x }
}) }
#let beside(a, b, right: 64mm) = context if target() == "html" { a; b } else if phone { stack(spacing: 1em, align(left, a), align(center, b)) } else {
  grid(columns: (1fr, right), column-gutter: 12pt, align: (left + horizon, center + horizon), a, b)
}
#let lqd(..args) = visb(fitw(lq.diagram(..args)))

#let html-css = " :root { color-scheme: light; } body { max-width: 42rem; margin: 0 auto; padding: 1.2rem 1.1rem 4rem; font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, Georgia, serif; font-size: 1.12rem; line-height: 1.55; color: #161616; background: #fff; } h1 { font-size: 2rem; line-height: 1.2; text-align: center; margin: 1.2rem 0 0.4rem; } p.subtitle { text-align: center; font-style: italic; margin: 0; } p.date { text-align: center; color: #666; font-size: 0.9rem; margin: 0.3rem 0 1.6rem; } h2 { font-size: 1.45rem; margin: 2.2rem 0 0.6rem; line-height: 1.25; } h3 { font-size: 1.15rem; margin: 1.6rem 0 0.5rem; } p { margin: 0.7rem 0; } a { color: #0072B2; text-decoration: none; } a:hover { text-decoration: underline; } svg { max-width: 100%; height: auto !important; vertical-align: middle; } svg[style*=\"display: block\"] { margin: 0.8rem auto; } figure { margin: 1.6rem 0; text-align: center; overflow-x: auto; } figure.thm, figure:has(div.thm) { text-align: left; } figcaption { font-size: 0.9rem; color: #444; margin-top: 0.5rem; text-align: center; } table { border-collapse: collapse; margin: 0.4rem auto; font-size: 0.9rem; line-height: 1.35; border-top: 1.5px solid #222; border-bottom: 1.5px solid #222; } td, th { padding: 0.25rem 0.55rem; text-align: left; vertical-align: top; } tr:first-child td, tr:first-child th { font-weight: 600; border-bottom: 1px solid #222; } aside.box { border-left: 4px solid; padding: 0.6rem 0.95rem; margin: 1.3rem 0; border-radius: 0 5px 5px 0; font-size: 1rem; line-height: 1.45; } aside.idea { background: #fdf6e7; border-color: #E69F00; } aside.tryit { background: #f0f1f3; border-color: #5b6770; } aside.box p { margin: 0.35rem 0; } .boxlabel { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; margin-right: 0.45em; } aside.idea .boxlabel { color: #8a5a00; } aside.tryit .boxlabel { color: #3d464d; } pre { text-align: left; overflow-x: auto; font-size: 0.78rem; line-height: 1.45; background: #fafafa; border: 1px solid #ddd; border-radius: 4px; padding: 0.7rem 0.8rem; } code { font-size: 0.9em; overflow-wrap: anywhere; } pre code { overflow-wrap: normal; } math { font-size: 1.06em; font-family: \"STIX Two Math\", \"Latin Modern Math\", \"Cambria Math\", math; } math[display=\"block\"] { margin: 0.9rem 0; overflow-x: auto; overflow-y: hidden; } ul, ol { padding-left: 1.4rem; } li { margin: 0.25rem 0; } .thm { margin: 1.2rem 0; } blockquote.epigraph { margin: 0.2rem 0 1.4rem auto; max-width: 30rem; text-align: right; font-style: italic; font-size: 0.95rem; } p.mnote { font-size: 0.88rem; font-style: italic; color: #555; margin-left: 1.2rem; } blockquote.epigraph footer { font-style: normal; font-size: 0.85rem; color: #555; } "
#context if target() == "html" { html.elem("style", html-css) }
#show align: it => context if target() == "html" { it.body } else { it }

// ---------- look ----------
#set document(title: "How Many Cannonballs?", date: none)  // no date: every build is identical
#set page(..if phone { (width: 4.4in, height: 7.6in, margin: (x: 0.32in, top: 0.42in, bottom: 0.55in)) }
  else { (paper: "us-letter", margin: (x: 1.55in, y: 1.1in)) }, numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)
#set par(justify: true, leading: 0.6em, spacing: 1em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set block(above: 1.6em, below: 0.8em)
#show heading.where(level: 2): set block(above: 1.2em, below: 0.6em)
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading and type(el.numbering) == function {
    box(link(el.location(), [Appendix #numbering("A", ..counter(heading).at(el.location()))]))
  } else { it }
}
#show link: set text(fill: rgb("#0072B2"))
#show raw.where(block: true): set text(size: 8.5pt)
#show figure.caption: set text(size: 10pt)
#show figure.caption: it => context if target() == "html" { it } else { layout(size => {
  if measure(it).width <= size.width { align(center, it) } else { set par(justify: not phone); align(left, it) }
}) }
#show math.equation.where(block: true): it => if phone { fitw(it) } else { it }
#show table: it => if phone {
  let fr = type(it.columns) == array and it.columns.any(c => type(c) == fraction)
  wide(it, w: if fr { 4.8in } else { 7in })
} else { it }
#show heading: set text(hyphenate: false)
#show math.equation.where(block: false): box
#show: body => context if target() == "html" { body } else { codly-init(body) }
#codly(zebra-fill: none, stroke: 0.5pt + luma(205), fill: luma(251), radius: 2pt,
  number-format: n => text(fill: luma(160), size: 7.5pt, str(n)), display-name: false, display-icon: false)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set block(breakable: false)
#set table(stroke: none, inset: (x: 6pt, y: 4pt))
#show table: set par(justify: false)
#show table.cell.where(y: 0): set text(weight: "semibold")

// One meaning per color. Blue, green and purple: the first, second and third copy of a shape
// in a counting argument. Red: what a question is about (a count that is a square, the ball
// whose neighbors we count). Orange: Idea boxes. Slate: Try-it boxes. Steel: plain balls.
#let red = rgb("#D55E00")
#let blue = rgb("#0072B2")
#let green = rgb("#009E73")
#let purple = rgb("#AA4499")
#let orange = rgb("#E69F00")
#let slate = rgb("#5b6770")
#let steel = rgb("#8fa3b4")

#let theorem-like(kind) = (title: none, body) => figure(
  kind: "thm", supplement: kind, numbering: "1", outlined: false,
  [#if title != none [ (#title)]*.* #body],
)
#show figure.where(kind: "thm"): set block(breakable: true)
#show figure.where(kind: "thm"): it => block(
  width: 100%, breakable: true, above: 1.2em, below: 1.1em,
  { set par(justify: not phone); align(left)[*#it.supplement #it.counter.display(it.numbering)*#it.body] },
)
#let proposition = theorem-like([Proposition])
#let theorem = theorem-like([Theorem])
#let lemma = theorem-like([Lemma])
#let corollary = theorem-like([Corollary])
#let proof(title: [Proof], body) = block(above: 0.7em, below: 1.2em)[_#title._ #body#h(0.6em)#h(1fr)$qed$]

// numbers with thousands separators, as the text writes them
#let fmt(n) = {
  let s = str(n)
  let k = s.len()
  s.clusters().enumerate().map(((i, c)) => if i > 0 and calc.rem(k - i, 3) == 0 { "," + c } else { c }).join()
}

// a margin note in a child's voice (print); an indented line on a phone or in HTML
#let mnote(inline: false, body) = context if target() == "html" {
  html.elem("p", attrs: (class: "mnote"), body)
} else if phone {
  block(above: 0.5em, below: 0.7em, inset: (left: 1.2em), text(size: 9.5pt, style: "italic", fill: luma(38%), body))
} else {
  let note = block(width: 1.2in, {
    set align(left)
    set par(justify: false, leading: 0.45em)
    set text(size: 8pt, style: "italic", fill: luma(38%))
    body
  })
  if inline { box(place(top + left, dx: 5.56in, dy: -5.5pt, note)) }
  else { place(right, dx: 1.36in, dy: 2.7pt, note) }
}

// ---------- boxes for the young reader ----------
#let sidebox(label, col, title, body) = context if target() == "html" {
  html.elem("aside", attrs: (class: "box " + lower(label).replace(" ", "")), {
    html.elem("span", attrs: (class: "boxlabel"), upper(label))
    if title != none { [ ]; strong(title) }
    parbreak()
    body
  })
} else { block(
  breakable: false, width: 100%, above: 1.1em, below: 1.1em,
  inset: (left: 10pt, right: 9pt, y: 7pt), radius: (right: 3pt),
  fill: col.lighten(90%), stroke: (left: 2.2pt + col),
  {
    set par(justify: false, leading: 0.55em, spacing: 0.6em)
    set text(size: 10.5pt)
    text(size: 9pt, weight: "bold", fill: col.darken(35%), tracking: 0.04em, upper(label))
    if title != none { h(0.5em); text(weight: "bold", title) }
    parbreak()
    body
  },
) }
#let idea(title, body) = sidebox("Idea", orange, title, body)
#let tryit(body) = sidebox("Try it", slate, none, body)
#let puzzle(body) = context if target() == "html" { body; parbreak() } else {
  block(breakable: false, width: 100%, above: 1.0em, below: 1.0em, { set par(justify: false); body }) }

// ---------- drawing balls ----------
// A ball is a circle shaded as if lit from the upper left. Piles are drawn in a fixed
// orthographic view, farthest balls first, so that nearer balls cover them.
#let shade(c) = gradient.radial((white, 0%), (c, 45%), (c.darken(55%), 100%), center: (35%, 32%), radius: 72%)
#let proj(p, turn, tilt) = {
  let (x, y, z) = p
  let u = x * calc.cos(turn) - y * calc.sin(turn)
  let w = x * calc.sin(turn) + y * calc.cos(turn)
  (u, w * calc.sin(tilt) + z * calc.cos(tilt), -w * calc.cos(tilt) + z * calc.sin(tilt))
}
// balls: an array of (center, color, shaded?); width: the drawing's width
#let pile3d(balls, width: 4cm, turn: 32deg, tilt: 24deg) = {
  let pts = balls.map(b => { let (u, v, d) = proj(b.at(0), turn, tilt); (u, v, d, b.at(1), b.at(2, default: true)) })
    .sorted(key: q => q.at(2))
  let us = pts.map(q => q.at(0))
  let span = calc.max(..us) - calc.min(..us) + 2
  vis(cetz.canvas(length: width / span, {
    import cetz.draw: *
    for (u, v, d, c, lit) in pts {
      circle((u, v), radius: 1, fill: if lit { shade(c) } else { c.darken(35%) }, stroke: 0.3pt + c.darken(60%))
    }
  }))
}
// the centers of a square pile (layer h from the bottom, side n - h), with a color per ball
#let square-pile(n, col: (h, i, j) => steel, lift: 0, lit: (h, i, j) => true) = {
  let out = ()
  for h in range(n) { for i in range(n - h) { for j in range(n - h) {
    out.push(((2 * i + h, 2 * j + h, h * (calc.sqrt(2) + lift)), col(h, i, j), lit(h, i, j)))
  } } }
  out
}
#let oblong-pile(a, b, col: (h, i, j) => steel) = {
  let out = ()
  for h in range(a) { for i in range(a - h) { for j in range(b - h) {
    out.push(((2 * i + h, 2 * j + h, h * calc.sqrt(2)), col(h, i, j)))
  } } }
  out
}
// a tetrahedral pile: triangles in layers, each ball over a hollow of three below
#let tetra-pile(n, col: (k, i, j) => steel, lift: 0) = {
  let r3 = calc.sqrt(3)
  let out = ()
  for k in range(n) { for i in range(n - k) { for j in range(n - k - i) {
    out.push(((2 * i + j + k, r3 * j + k / r3, k * (2 * calc.sqrt(2 / 3) + lift)), col(k, i, j)))
  } } }
  out
}
// flat balls in the plane: an array of ((x, y), color, label)
#let flat(balls, unit: 0.32cm) = vis(cetz.canvas(length: unit, {
  import cetz.draw: *
  for b in balls {
    let (p, c) = (b.at(0), b.at(1))
    circle(p, radius: 1, fill: shade(c), stroke: 0.3pt + c.darken(60%))
    let lab = b.at(2, default: none)
    if lab != none { content(p, text(size: 7.5pt, weight: "bold", lab)) }
  }
}))
// a triangle of side n, rows from the top, with a color per ball
#let tri(n, col: (r, k) => steel, dx: 0, label: none) = {
  let r3 = calc.sqrt(3)
  let out = ()
  for r in range(n) { for k in range(r + 1) {
    out.push(((dx + 2 * k - r, -r3 * r), col(r, k), if label != none { label(r, k) }))
  } }
  out
}

// ---------- title ----------
#context if target() == "html" {
  html.elem("h1")[How Many Cannonballs?]
  html.elem("p", attrs: (class: "subtitle"))[Counting stacked balls, and the one pile that is also a square]
  html.elem("p", attrs: (class: "date"))[November 2026]
} else [
  #align(center)[
    #set par(justify: false)
    #text(size: 21pt, weight: "bold")[How Many Cannonballs?]
    #v(2pt)
    #text(size: 11.5pt, style: "italic")[Counting stacked balls, and the one pile that is also a square]
    #v(4pt)
    #text(size: 9.5pt, fill: luma(90))[November 2026]
  ]
]

#v(4pt)

Cannonballs used to be stored in piles. A *square pile* has one ball on top, four under it, then nine, and so on: each layer is a square, one ball wider than the layer above. A pile of 24 layers holds 4,900 balls, and 4,900 balls can also be laid out flat as a square, 70 by 70. No other square pile can do this, except the pile of one ball. This was guessed in 1875 and proved in 1918.

This note counts the balls in piles of every common shape and explains why each count has the formula it has. It then asks which piles hold a square number of balls, and how many pairs of balls touch. Along the way it finds a few things we did not see in the sources: a complete rule for which piles with an oblong base can hold a square number of balls (it works for 7 in 12 lengths of the short side, and never for a side of 2 or 3), formulas for the touching pairs, and a computer search that rules out any other square pile below $10^"10,000"$ layers.

*How to read this.* Sections #ref(<sec-flat>, supplement: none) to #ref(<sec-puzzles>, supplement: none) tell the story with pictures and arithmetic, and a parent and child can read them together. Sections #ref(<sec-proofs>, supplement: none) to #ref(<sec-touch>, supplement: none) prove the claims. Sections #ref(<sec-relatives>, supplement: none) to #ref(<sec-check>, supplement: none) set the piles beside their relatives, list open questions and say how everything was checked. Boxes marked #text(size: 8.5pt, weight: "bold", fill: orange.darken(35%), tracking: 0.04em)[IDEA] explain a tool the first time it is used. Boxes marked #text(size: 8.5pt, weight: "bold", fill: slate.darken(35%), tracking: 0.04em)[TRY IT] are puzzles to try before reading on, with answers in @app-answers. It is fine to skip a proof, or to stop anywhere.

#figure(
  {
    let tet = tetra-pile(4)
    let sq = square-pile(4)
    let g = grid(columns: 2, column-gutter: 2.4em, align: bottom, pile3d(sq, width: 4.2cm), pile3d(tet, width: 3.6cm, turn: 12deg))
    visb(fitw(g))
  },
  caption: [A square pile of four layers (30 balls) and a tetrahedral pile of four layers (20 balls). Every ball above the bottom rests in a hollow between balls of the layer below.],
) <fig-two-piles>

= Balls on the ground <sec-flat>

Before stacking, lay the balls flat. A *triangle* of side 4 has rows of 1, 2, 3 and 4 balls, like the ten pins in bowling. A *square* of side 4 has four rows of four.

#figure(
  visb(fitw(grid(columns: 4, column-gutter: 1.6em, align: bottom,
    ..range(1, 5).map(n => flat(tri(n)))))),
  caption: [Triangles of side 1 to 4 hold 1, 3, 6 and 10 balls. These are the _triangular numbers_.],
) <fig-triangles>

#tryit[How many balls are in a triangle of side 10? Try to find the answer without adding ten numbers one by one.]

#beside(right: 46mm)[
#mnote(inline: true)[Two triangles make a box. Then halve it!]Here is a quick way. Slide the rows of the triangle to the left, so that it becomes a staircase. Take a second staircase of the same size, turn it upside down and push it against the first. The two make a rectangle of $n$ rows with $n + 1$ balls in each row. So one triangle holds half of $n times (n + 1)$. For side 10 that is half of 110, which is 55.
][
#figure(
  {
    let n = 4
    let balls = ()
    for r in range(n) { for k in range(n + 1) {
      balls.push(((2 * k, -2 * r), if k <= r { blue } else { green }))
    } }
    flat(balls)
  },
  caption: [A blue triangle of side 4 and a green one turned over fill a rectangle of 4 rows of 5. So the triangle holds half of $4 times 5$, which is 10.],
) <fig-pairing>
]

#idea[a formula][A _formula_ is a recipe that works for every size at once. The letter $n$ stands for the side of the triangle, whatever it is. The recipe "multiply $n$ by $n + 1$, then halve" gives the triangle for every $n$: write $T(n) = n(n+1) slash 2$, and read $T(n)$ as "T of n". The picture is the reason the recipe works, not only an example of it.]

#beside(right: 46mm)[
A square splits into two triangles the same way. Cut a square of side 5 along the line right above its diagonal: the lower part is a triangle of side 5 and the upper part a triangle of side 4, so $25 = 15 + 10$.
][
#figure(
  {
    let n = 5
    let balls = ()
    for r in range(n) { for k in range(n) {
      balls.push(((2 * k, -2 * r), if k <= r { blue } else { green }))
    } }
    flat(balls)
  },
  caption: [A square of side 5 is a triangle of side 5 (blue) and one of side 4 (green): $25 = 15 + 10$.],
) <fig-square-split>
]

= Stacking them up <sec-piles>

A triangle can carry a smaller triangle on top, each ball resting in a hollow between three balls below. Keep going and the pile comes to a point: that is a *tetrahedral pile*, named for the tetrahedron, the shape with four triangular faces. A square can carry a smaller square, each ball resting on four, and that pile is the *square pile*. @fig-layers takes both piles apart.

#figure(
  {
    let a = pile3d(square-pile(4, lift: 1.6), width: 3.6cm)
    let b = pile3d(tetra-pile(4, lift: 1.6), width: 3.3cm, turn: 12deg)
    visb(fitw(grid(columns: 2, column-gutter: 2.6em, align: bottom, a, b)))
  },
  caption: [The two piles of @fig-two-piles with their layers lifted apart. The square pile has layers of 16, 9, 4 and 1 balls; the tetrahedral pile has layers of 10, 6, 3 and 1.],
) <fig-layers>

#tryit[A square pile has four layers. How many balls does it hold? And a tetrahedral pile of three layers?]

Adding up the layers gives the counts in @tab-counts. A square pile of four layers holds $1 + 4 + 9 + 16 = 30$ balls, and a tetrahedral pile of three layers holds $1 + 3 + 6 = 10$, the same as a triangle of side 4.

#figure(
  placement: auto,
  caption: [How many balls each shape holds, for sides or layers from 1 to 10. Exact, by adding up the layers and by building each pile ball by ball in a program.],
  table(
    columns: 5,
    align: (center, right, right, right, right),
    table.hline(stroke: 0.6pt),
    [Side or\ layers], [Triangle], [Square], [Tetrahedral\ pile], [Square\ pile],
    table.hline(stroke: 0.4pt),
    ..R.table.map(row => row.map(v => [#fmt(v)])).flatten(),
    table.hline(stroke: 0.6pt),
  ),
) <tab-counts>

#block(breakable: false, width: 100%)[
A tetrahedral pile of $n$ layers holds
$ "Te"(n) = (n (n + 1) (n + 2)) / 6 $
balls, and a square pile of $n$ layers holds
$ P(n) = (n (n + 1) (2n + 1)) / 6. $
]
Both have the same shape: three numbers multiplied, then divided by 6. @sec-why explains where the 6 comes from. For a pile of four layers, $4 times 5 times 9 slash 6 = 30$.

= Why the formulas look like that <sec-why>

#mnote(inline: true)[Three piles fill a box, more or less. So one pile is about a third of the box.]*The square pile is about a third of a box.* Slide each layer of a square pile sideways until one corner of every layer sits above the same corner of the bottom layer. The number of balls does not change, and the pile now looks like a pyramid whose tip is above a corner of its base. A solid pyramid of that shape, as tall as its base is wide, is exactly a third of a cube: three of them fill the cube (@fig-cube). So a square pile of $n$ layers holds roughly $n^3 slash 3$ balls. The formula agrees: $P(n) = n^3 slash 3 + n^2 slash 2 + n slash 6$. The rest of the formula counts the steps on the sides, where balls stick out of the smooth pyramid.

#let cube-pyramids(apart) = {
  let turn = 35deg
  let tilt = 22deg
  let view = (-calc.sin(turn) * calc.cos(tilt), -calc.cos(turn) * calc.cos(tilt), calc.sin(tilt))
  let dot3(p, q) = p.at(0) * q.at(0) + p.at(1) * q.at(1) + p.at(2) * q.at(2)
  let sub(p, q) = (p.at(0) - q.at(0), p.at(1) - q.at(1), p.at(2) - q.at(2))
  let cross(p, q) = (p.at(1) * q.at(2) - p.at(2) * q.at(1), p.at(2) * q.at(0) - p.at(0) * q.at(2), p.at(0) * q.at(1) - p.at(1) * q.at(0))
  let apex = (0, 0, 2)
  // each pyramid: color, base (a face of the cube away from the apex), direction to pull it
  let pyramids = (
    (blue, ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)), (0, 0, -1)),
    (green, ((2, 0, 0), (2, 2, 0), (2, 2, 2), (2, 0, 2)), (1, 0, 0)),
    (purple, ((0, 2, 0), (2, 2, 0), (2, 2, 2), (0, 2, 2)), (0, 1, 0)),
  )
  let faces = ()
  for (col, base, dir) in pyramids {
    let mv(p) = (p.at(0) + apart * dir.at(0), p.at(1) + apart * dir.at(1), p.at(2) + apart * dir.at(2))
    let pts = base + (apex,)
    let mid = (pts.map(p => p.at(0)).sum() / 5, pts.map(p => p.at(1)).sum() / 5, pts.map(p => p.at(2)).sum() / 5)
    for f in (base,) + range(4).map(i => (apex, base.at(i), base.at(calc.rem(i + 1, 4)))) {
      let nrm = cross(sub(f.at(1), f.at(0)), sub(f.at(2), f.at(0)))
      if dot3(nrm, sub(f.at(0), mid)) < 0 { nrm = nrm.map(c => -c) }   // point outward
      if dot3(nrm, view) > 0.001 {
        let g = f.map(mv)
        faces.push((g, col, g.map(p => proj(p, turn, tilt).at(2)).sum() / g.len()))
      }
    }
  }
  vis(cetz.canvas(length: 0.95cm, {
    import cetz.draw: *
    for (f, col, d) in faces.sorted(key: x => x.at(2)) {
      line(..f.map(p => { let (u, v, d) = proj(p, turn, tilt); (u, v) }), close: true,
        fill: col.lighten(55%), stroke: (paint: col.darken(25%), thickness: 0.7pt, join: "round"))
    }
  }))
}

#figure(
  visb(fitw(grid(columns: 2, column-gutter: 3em, align: horizon, cube-pyramids(0), cube-pyramids(1.1)))),
  caption: [Three pyramids fill a cube (left). Each has a face of the cube as its base and its tip at the cube's top front corner. On the right they are pulled apart.],
) <fig-cube>

*The exact count, with three triangles.* Write the layers of a square pile of $n$ layers as rows of numbers: one 1, two 2s, three 3s, up to $n$ copies of $n$. Row $k$ adds up to $k times k$, so the whole triangle of numbers adds up to $P(n)$. Now turn the triangle a third of a turn, and again, and lay the three copies on top of each other (@fig-three). At every position the three numbers add up to $2n + 1$. There are $T(n)$ positions, so $3 P(n) = (2n + 1) T(n)$, and dividing by 3 gives the formula. The 6 is the 3 from the three copies times the 2 from the triangle.

#figure(
  {
    let n = 4
    let lab(f) = (r, k) => text(fill: white, str(f(r, k)))
    let t1 = tri(n, col: (r, k) => blue, label: lab((r, k) => r + 1))
    let t2 = tri(n, col: (r, k) => green, dx: 10, label: lab((r, k) => n - k))
    let t3 = tri(n, col: (r, k) => purple, dx: 20, label: lab((r, k) => k + n - r))
    let t4 = tri(n, col: (r, k) => slate, dx: 30, label: lab((r, k) => 2 * n + 1))
    visb(fitw(cetz.canvas(length: 0.3cm, {
      import cetz.draw: *
      for b in t1 + t2 + t3 + t4 {
        circle(b.at(0), radius: 1, fill: shade(b.at(1)), stroke: 0.3pt + b.at(1).darken(60%))
        content(b.at(0), text(size: 7.5pt, weight: "bold", b.at(2)))
      }
      for (x, s) in ((5, $+$), (15, $+$), (25, $=$)) { content((x, -3), text(size: 13pt, s)) }
    })))
  },
  caption: [The layers of a square pile of four layers, written as a triangle of numbers (blue), and the same triangle turned a third of a turn (green) and two thirds (purple). Read any position across the three: the numbers always add up to 9, which is $2 times 4 + 1$. Ten positions with 9 at each make 90, three times 30.],
) <fig-three>

*The tetrahedral pile is a way of choosing.* A ball in a tetrahedral pile of $n$ layers can be named by three steps from the top: how many balls along each of three edges, which gives three whole numbers $i$, $j$, $k$ with $i + j + k$ at most $n - 1$. The three numbers $i + 1$, $i + j + 2$ and $i + j + k + 3$ go up, and all lie between 1 and $n + 2$. Every way of choosing three different numbers from 1 to $n + 2$ arises from exactly one ball. So the pile holds as many balls as there are ways to choose three things out of $n + 2$, and that number is $(n + 2)(n + 1) n slash 6$. Here the 6 counts the orders in which three chosen things could be listed: $3 times 2 times 1$.

#idea[choosing][The number of ways to choose 3 things out of 5 is 10: list them as 123, 124, 125, 134, 135, 145, 234, 235, 245, 345. Picking them in order gives $5 times 4 times 3 = 60$ lists, but each choice turns up in $3 times 2 times 1 = 6$ orders, so there are $60 slash 6 = 10$ choices. A tetrahedral pile of three layers holds 10 balls, one for each choice.]

*A square pile is two tetrahedral piles.* Cut each layer of a square pile along the line next to its diagonal, as in @fig-square-split. Each layer splits into two triangles, of sides $k$ and $k - 1$, so the whole pile holds a tetrahedral pile's worth of $n$ layers and another of $n - 1$: $P(n) = "Te"(n) + "Te"(n - 1)$. For four layers, $30 = 20 + 10$ (@fig-diagonal). A short calculation gives a second link: a tetrahedral pile of $2n$ layers holds exactly four times a square pile of $n$ layers.

#figure(
  {
    let layer(k) = {
      let balls = ()
      for r in range(k) { for c in range(k) { balls.push(((2 * c, -2 * r), if c <= r { blue } else { green })) } }
      flat(balls, unit: 0.26cm)
    }
    visb(fitw(grid(columns: 4, column-gutter: 1.4em, align: bottom, ..range(1, 5).map(layer))))
  },
  caption: [The four layers of a square pile of four layers, each cut next to its diagonal. The blue triangles, of sides 1 to 4, add up to a tetrahedral pile of four layers (20 balls); the green ones, of sides 0 to 3, to one of three layers (10).],
) <fig-diagonal>

= Oblong piles <sec-oblong>

Piles were also built on a rectangle. Such a pile starts from a rectangle of $a$ rows of $b$ balls, with $a$ at most $b$. The next layer is $a - 1$ by $b - 1$, and so on, until the top is a single row of $b - a + 1$ balls, a ridge like the top of a roof. Call this an *oblong pile*.

#figure(
  pile3d(oblong-pile(3, 6), width: 5cm),
  caption: [An oblong pile on a base of 3 rows of 6: layers of 18, 10 and 4 balls, 32 in all. The top is a ridge of four balls.],
) <fig-oblong>

#mnote(inline: true)[A longer pile is the same pile plus one more slice.]Making an oblong pile one ball longer adds a slice on its end, and that slice is a triangle of side $a$. A square pile is the oblong pile with $b = a$, so the count is the square pile plus $b - a$ triangles:
$ (a (a+1)(2a+1)) / 6 + (b - a) (a (a+1)) / 2 = (a (a + 1) (3 b - a + 1)) / 6. $

#tryit[Find an oblong pile with exactly 100 balls. (There are two, apart from a single row of 100.)]

= The cannonball problem <sec-lucas>

A square pile of 24 layers holds 4,900 balls. That is $70 times 70$, so the same balls could be laid out as a flat square of side 70 (@fig-4900). Do other square piles do this?

#figure(
  {
    let pile = pile3d(square-pile(24, lit: (h, i, j) => h == 23 or i == 0 or j == 0 or i == 23 - h or j == 23 - h), width: 6.2cm)
    let dots = vis(box(width: 4.2cm, height: 4.2cm, fill: tiling(size: (0.06cm, 0.06cm), place(circle(radius: 0.026cm, fill: slate)))))
    visb(fitw(grid(columns: 2, column-gutter: 1.6em, align: horizon, pile, dots)))
  },
  caption: [4,900 balls two ways: a square pile of 24 layers, and a flat square of 70 rows of 70.],
) <fig-4900>

The French mathematician Édouard Lucas posed this in 1875 as question 1180 of the _Nouvelles Annales de Mathématiques_. He asserted that a square pile of balls holds a square number "que lorsqu'elle en contient 24 sur le côté de la base": only when its base has 24 on a side. His statement leaves out the pile of a single ball, which is a square too, a $1 times 1$ square. Lucas could not prove his claim. A proof by Moret-Blanc (1876) and one by Lucas (1877) both had gaps. The first complete proof came from G. N. Watson in 1918.

#theorem(title: [Watson, 1918])[
  A square pile of $n$ layers holds a square number of balls only when $n = 1$ or $n = 24$.
] <thm-watson>

The theorem is about every $n$, and no computer can try every $n$. What a computer can do is try a great many. @fig-gaps shows where each square pile up to 60 layers lands between two squares. Only two land on one.

#figure(
  placement: auto,
  {
    let rows = csv("gaps.csv", row-type: dictionary)
    let xs = rows.map(r => int(r.layers))
    let ys = rows.map(r => float(r.share))
    let hit = rows.filter(r => int(r.balls) == int(r.below))
    lqd(width: 10.5cm, height: 4.2cm,
      xlabel: [layers in the square pile], ylabel: [place between squares],
      xlim: (0, 61), ylim: (-0.05, 1.05), yaxis: (ticks: (0, 1)),
      lq.scatter(xs, ys, size: 6pt, color: slate, stroke: none),
      lq.scatter(hit.map(r => int(r.layers)), hit.map(r => 0), size: 10pt, color: red, stroke: none),
    )
  },
  caption: [For each square pile of 1 to 60 layers, where its number of balls falls between the square below it (0) and the next square (1). Red dots land exactly on a square: 1 layer and 24 layers. Exact; written by `report.py` to `gaps.csv`.],
) <fig-gaps>

#idea[a search is not a proof][A computer that tries every pile up to some size can say what happens up to that size, and nothing beyond it. The pattern "no square after 24" could break at a size nobody has tried. A _proof_ is an argument that covers every size at once. So a search tells us what to believe, and a proof tells us why it must be so.]

Going further takes a better idea than trying every pile one by one. @sec-why-24 sorts every possible solution into one of six kinds, rules four out with remainders, and turns each of the other two into a Pell equation. That makes a fast search possible: it rules out every square pile other than 1 and 24 below $10^"10,000"$ layers, a 1 followed by ten thousand zeros, after checking #fmt(R.pell_walked.at("611") + R.pell_walked.at("123")) candidates. Watson's proof covers the rest.

= More puzzles <sec-puzzles>

Answers are in @app-answers.

#puzzle[*1.* A box holds the balls of a triangle of side 8. How many balls is that? Use the pairing trick.]
#puzzle[*2.* In a square pile of two layers, the top ball touches the four below it, and the four bottom balls touch their neighbors in a ring. How many touching pairs is that? How many in a tetrahedral pile of two layers?]
#puzzle[*3.* An oblong pile two balls wide and $b$ balls long holds $3b - 1$ balls: 5, 8, 11, 14, and so on. Can it ever be a square? Look at what is left over when you divide the counts, and the squares 1, 4, 9, 16, 25, by 3.]
#puzzle[*4.* Cut the top off a square pile and keep two layers, of sides 3 and 4: $9 + 16 = 25$, a square. Find two more layers next to each other whose balls add up to a square.]
#puzzle[*5.* A tetrahedral pile of 48 layers holds how many times as many balls as a square pile of 24 layers? Is the number a square?]

= The formulas, proved <sec-proofs>

From here on the note proves its claims. Write $T(n)$, $"Te"(n)$ and $P(n)$ for the balls in a triangle of side $n$, a tetrahedral pile of $n$ layers and a square pile of $n$ layers, and $B(a, b)$ for an oblong pile on an $a$ by $b$ base, with $a <= b$. Each statement says how it is known. Every formula was also checked by a program that builds the pile ball by ball, for up to 12 layers.

#proposition(title: [triangles])[
  For every whole number $n >= 1$, a triangle of side $n$ holds $T(n) = n(n + 1) slash 2$ balls, and $T(n) + T(n - 1) = n^2$.
] <prop-tri>

#proof[
  Two copies of a triangle of side $n$, one turned over, fill a rectangle of $n$ rows of $n + 1$ (@fig-pairing), so $2 T(n) = n(n+1)$. The second claim is the cut of @fig-square-split.
]

#proposition(title: [square piles])[
  For every whole number $n >= 1$, the square pile satisfies $3 P(n) = (2n + 1) T(n)$, so $P(n) = n(n+1)(2n+1) slash 6$.
] <prop-pyramid>

#proof[
  Label the positions of a triangle of side $n$ by three whole numbers $(a, b, c)$ that add up to $n - 1$: $a$ counts the rows up from the bottom edge, and $b$ and $c$ do the same from the other two edges. Put the number $n - a$ at each position. The row with $a = n - k$ holds $k$ positions, each carrying $k$, so the numbers add up to $1^2 + dots.c + n^2 = P(n)$. The triangle turned a third of a turn carries $n - b$ at the same position, and turned twice $n - c$. The three add up to $3n - (a + b + c) = 2n + 1$ at each of the $T(n)$ positions.
]

#proposition(title: [tetrahedral piles])[
  For every whole number $n >= 1$, the tetrahedral pile holds $"Te"(n) = binom(n + 2, 3)$ balls, which is $n(n+1)(n+2) slash 6$.
] <prop-tetra>

#proof[
  The balls of a tetrahedral pile of $n$ layers are the triples $(i, j, k)$ of whole numbers at least 0 with $i + j + k <= n - 1$: walk $i$ steps along one edge of the top, $j$ along a second, $k$ along the third. The map $(i, j, k) arrow.r.bar (i + 1, i + j + 2, i + j + k + 3)$ gives three numbers that go up and lie between 1 and $n + 2$. Every such triple $p < q < r$ comes from exactly one ball, namely $i = p - 1$, $j = q - p - 1$, $k = r - q - 1$. So the balls match the choices of three numbers from $n + 2$, and there are $binom(n + 2, 3)$ choices.
]

#proposition(title: [links between the piles])[
  For every whole number $n >= 1$, the piles satisfy $P(n) = "Te"(n) + "Te"(n - 1)$ and $"Te"(2n) = 4 P(n)$.
] <prop-links>

#proof[
  The layer of side $k$ of a square pile splits into triangles of sides $k$ and $k - 1$ (@prop-tri). Adding up over the layers gives the first identity. For the second, $"Te"(2n) = 2n (2n + 1)(2n + 2) slash 6$, which is $4 dot n(n+1)(2n+1) slash 6$.
]

#proposition(title: [oblong piles])[
  Let $1 <= a <= b$. An oblong pile on an $a$ by $b$ base holds $B(a, b) = a(a+1)(3b - a + 1) slash 6$ balls, and $B(a, b + 1) - B(a, b) = T(a)$.
] <prop-oblong>

#proof[
  Layer $h$ from the bottom ($h = 0, dots, a - 1$) has $(a - h)(b - h)$ balls. Raising $b$ by one adds $a - h$ balls to layer $h$, in all $a + (a - 1) + dots.c + 1 = T(a)$. Since $B(a, a) = P(a)$, we get $B(a, b) = P(a) + (b - a) T(a)$, which simplifies to the formula.
]

= Why only 1 and 24 <sec-why-24>

This section explains how the proofs of @thm-watson go, proves the first steps, and describes the search. The last step, that two particular sequences of numbers never meet again, is where the real difficulty lies. That step is taken from the published proofs and not reproduced.

*Step 1: three numbers that share nothing.* Suppose a square pile of $n$ layers holds $m^2$ balls. Then $n(n+1)(2n+1) = 6 m^2$.

#idea[sharing no factor][Two whole numbers _share no factor_ when no number bigger than 1 divides both. Neighbors such as 24 and 25 share no factor: anything dividing both would divide their difference, 1. If numbers that share no factor multiply to a square, each one is a square on its own. For example $4 times 9 = 36$, and 4 and 9 are squares; but $2 times 8 = 16$ does not count, because 2 and 8 share the factor 2.]

#block(breakable: false, width: 100%)[
#lemma(title: [six cases])[
  If $n(n+1)(2n+1) = 6 m^2$ for whole numbers $n >= 1$ and $m$, then there are whole numbers $x$, $y$, $z$ and a case $(s_1, s_2, s_3)$ from the list
  $ (1, 2, 3), (2, 1, 3), (1, 6, 1), (6, 1, 1), (2, 3, 1), (3, 2, 1) $
  with $n = s_1 x^2$, $n + 1 = s_2 y^2$ and $2n + 1 = s_3 z^2$.
] <lem-cases>
]

#proof[
  Any number dividing two of $n$, $n + 1$, $2n + 1$ divides 1, because $(n + 1) - n = 1$ and $(2n + 1) - 2n = 1$ and $2(n + 1) - (2n + 1) = 1$. So each prime other than 2 and 3 that divides the product divides only one of the three, and does so an even number of times, as in $6 m^2$. The prime 2 divides $6 m^2$ an odd number of times, all of them in $n$ or in $n + 1$, since $2n + 1$ is odd; likewise 3 divides it an odd number of times, all in one of the three. Take the single extra factor 2 and the single extra factor 3 out of the numbers they divide; three squares are left. The 2 can go in either of the first two numbers and the 3 in any of the three, which gives the six cases.
]

#idea[remainders][Divide a number by 3 and keep the remainder. Squares leave only 0 or 1: $1, 4, 9, 16, 25, 36$ leave $1, 1, 0, 1, 1, 0$. A number that leaves 2 can never be a square. We write $a equiv b space (mod 3)$, read "$a$ leaves the same remainder as $b$ on division by 3."]

#proposition(title: [four cases are impossible])[
  The cases $(2, 1, 3)$, $(1, 6, 1)$, $(2, 3, 1)$ and $(3, 2, 1)$ of @lem-cases have no solution with $n >= 1$.
] <prop-four>

#proof[
  Each case gives an equation between two of the three numbers that fails on division by 3.
  Case $(2, 1, 3)$: $2n + 1 = 4x^2 + 1$ must be $3z^2$, a multiple of 3; but $4 x^2 + 1 equiv x^2 + 1$, which leaves 1 or 2.
  Case $(1, 6, 1)$: $n + 1 = x^2 + 1$ must be $6 y^2$, a multiple of 3; again $x^2 + 1$ leaves 1 or 2.
  Case $(3, 2, 1)$: $n + 1 = 3x^2 + 1$ must be $2y^2$; the left side leaves 1, while $2y^2$ leaves 0 or 2.
  Case $(2, 3, 1)$: $2n + 1 = 4x^2 + 1$ must be $z^2$, so $(z - 2x)(z + 2x) = 1$, which forces $x = 0$ and $n = 0$.
]

The two cases left are the ones the two piles fall in: $n = 1$ is case $(1, 2, 3)$, with $1 = 1^2$, $2 = 2 dot 1^2$, $3 = 3 dot 1^2$; and $n = 24$ is case $(6, 1, 1)$, with $24 = 6 dot 2^2$, $25 = 5^2$, $49 = 7^2$. A computer check over all remainders confirms that no divisor up to 200 rules either of them out: remainders alone cannot finish the proof.

*Step 2: Pell equations.* In case $(6, 1, 1)$, the first two conditions say $y^2 - 6x^2 = 1$. In case $(1, 2, 3)$ they say $x^2 - 2y^2 = -1$.

#idea[Pell equations][An equation such as $y^2 - 6 x^2 = 1$ is called a _Pell equation_. Its whole-number solutions form an endless chain, and each one gives the next by a fixed rule: from $(y, x) = (5, 2)$ the rule $(y, x) arrow.r.bar (5y + 12x, 2y + 5x)$ gives $(49, 20)$, then $(485, 198)$. A classical theorem says that the chain contains every solution. For $x^2 - 2y^2 = -1$ the chain starts $(1, 1), (7, 5), (41, 29)$, with the rule $(x, y) arrow.r.bar (3x + 4y, 2x + 3y)$.]

So in each case the candidates for $n$ come one after another along a chain, and only the third condition is left to test: is $12 x^2 + 1$ a square in case $(6, 1, 1)$, and is $(2x^2 + 1) slash 3$ a square in case $(1, 2, 3)$? @tab-chains shows the first few links. The candidates grow about ten times longer at each step, so a few thousand steps reach numbers of ten thousand digits.

#figure(
  placement: auto,
  caption: [The first four candidates in each case that remainders cannot rule out. The last column says whether the third condition holds. Exact.],
  table(
    columns: 5,
    align: (left, right, right, right, center),
    table.hline(stroke: 0.6pt),
    [Case], [$x$], [layers $n$], [must be a square], [square?],
    table.hline(stroke: 0.4pt),
    ..R.pell_first.at("611").map(r => ([$(6, 1, 1)$], [#fmt(r.at(0))], [#fmt(r.at(1))], [#fmt(r.at(2))], if r.at(3) [*yes*: $7^2$] else [no])).flatten(),
    table.hline(stroke: 0.3pt),
    ..R.pell_first.at("123").map(r => ([$(1, 2, 3)$], [#fmt(r.at(0))], [#fmt(r.at(1))], [#fmt(r.at(2))], if r.at(3) [*yes*: $1^2$] else [no])).flatten(),
    table.hline(stroke: 0.6pt),
  ),
) <tab-chains>

#figure(
  kind: raw,
  placement: auto,
  caption: [The walk along the chain for case $(6, 1, 1)$, from `cannonballs/lucas.py` (shortened).],
  wide(w: 5.4in)[```python
y, x = 5, 2                     # the first solution of y^2 - 6 x^2 = 1
while 6 * x * x < 10**10_000:
    n = 6 * x * x               # then n + 1 = y^2 is a square
    if is_square(2 * n + 1):    # the third condition
        found.append(n)
    y, x = 5 * y + 12 * x, 2 * y + 5 * x
```]
) <lst-walk>

*The search.* Walking both chains below $10^"10,000"$ layers takes a few seconds: #fmt(R.pell_walked.at("611")) links in case $(6, 1, 1)$ and #fmt(R.pell_walked.at("123")) in case $(1, 2, 3)$. Only the known solutions 24 and 1 pass the third test. Since every solution of each Pell equation lies on its chain, this rules out every other $n$ below that bound. A plain search that adds up squares one at a time agrees for every $n$ up to #fmt(R.bounds.brute), and a test compares the two methods on every candidate up to $10^6$.

*Step 3: the chains never meet again.* Showing that no later link passes the third test is the hard part. Watson (1918) did most cases by elementary means and the last with elliptic functions. Ljunggren (1952) gave a proof by algebra in a larger number system. Ma (1985) and Anglin (1990) gave proofs that use only Pell equations and remainders. Roughly, such proofs divide the terms of one chain by terms of related chains and use the rules for which remainders a square can leave to show that no later term passes. Anglin's proof fits in five journal pages. We have not checked these proofs and rely on them for every $n$ beyond our search.

The equation $6 m^2 = n(n+1)(2n+1)$ also describes a curve, of the kind called elliptic, and @thm-watson says which points on it have whole-number coordinates. Bennett (2002) solved the wider family $x(x+1)(x+2) = d y^2$ for many $d$ and recovered Lucas's problem as the case $d = 6$.

= Oblong piles that are squares <sec-oblong-sq>

An oblong pile two balls wide never holds a square number (puzzle 3), and the same holds for three wide. Four wide, a pile 11 long holds 100. Which widths allow a square?

#theorem(title: [oblong piles])[
  Let $a >= 1$. Some oblong pile on an $a$ by $b$ base, with $b >= a$, holds a square number of balls exactly when one of the following holds:
  + $a$ leaves remainder 1 on division by 3; or
  + the one of $a$ and $a + 1$ that is a multiple of 3, divided by 3, has the form $9^k (3j + 2)$ for whole numbers $k, j >= 0$.
  When one holds, infinitely many lengths $b$ work.
] <thm-oblong>

So widths 2 and 3 never work, since $3 slash 3 = 1$, and widths 8 and 9 never work, since $9 slash 3 = 3$. Width 5 works, since $6 slash 3 = 2$, and so does width 53, since $54 slash 3 = 9 times 2$.

#proof[
  Write $c = 3b - a + 1$. By @prop-oblong the pile holds $N = T(a) c slash 3$ balls. As $b$ runs over the whole numbers from $a$ up, $c$ runs over the numbers from $2a + 1$ up that leave the same remainder as $1 - a$ on division by 3.

  _If $a$ leaves remainder 1,_ then $c$ can be any multiple of 3 from $2a + 1$ up. Take $c = 3 T(a) s^2$ for a large $s$; then $N = (T(a) s)^2$.

  _Otherwise_ 3 divides $a$ or $a + 1$, and $c$ is not a multiple of 3. Write $T(a) = 3^v w$ with $v >= 1$ and $w$ not a multiple of 3. Then $N = 3^(v - 1) w c$ with $w c$ not a multiple of 3. If $N$ is a square, $v - 1$ is even and $w c$ is a square not divisible by 3, so $w c$ leaves remainder 1, and therefore $c$ leaves the same remainder as $w$. Since $c$ also leaves the same remainder as $1 - a$, we need $w equiv 1 - a space (mod 3)$. Conversely, suppose $v$ is odd and $w equiv 1 - a$. Choose $c = w s^2$, with $s$ large and not a multiple of 3. This $c$ is allowed, and $N = (3^((v - 1) slash 2) w s)^2$.

  It remains to restate the two conditions. Let $M$ be the one of $a$ and $a + 1$ divisible by 3, and $M = 3^v u$ with $u$ not a multiple of 3; the other one of $a, a + 1$ is not divisible by 3, so $v$ is the same as before. Halving leaves the same remainder as doubling, since $2 times 2 = 4$ leaves 1. If $M = a$, then $a + 1$ leaves 1 and $w equiv 2 u$, while $1 - a$ leaves 1, so the condition is $u equiv 2$. If $M = a + 1$, then $a$ leaves 2, $w equiv 2 dot 2 dot u equiv u$, and $1 - a$ leaves 2, so again $u equiv 2$. Finally, "$v$ odd and $u equiv 2$" says $M slash 3 = 3^(v - 1) u$ with $v - 1$ even, which is $9^k (3j + 2)$.
]

#corollary(title: [seven in twelve])[
  Among the widths $1, 2, dots, A$, the share that allow a square tends to $7 slash 12$ as $A$ grows.
] <cor-share>

#proof[
  A third of all widths leave remainder 1. For the other two thirds, $M slash 3$ runs over all whole numbers as $a$ does, each one twice, and the share of whole numbers of the form $9^k (3j + 2)$ is $1/3 (1 + 1/9 + 1/81 + dots.c) = 3/8$. In all, $1/3 + 2/3 dot 3/8 = 7/12$.
]

Up to width #fmt(R.bounds.oblong_rule), exactly #fmt(R.oblong_with_square) widths work. @thm-oblong was also checked against a direct method for every width up to #R.bounds.oblong_residues: whether some square leaves the remainder that the pile counts leave on division by $T(a)$. That method needs no theory beyond @prop-oblong, since the counts for longer and longer piles go up in steps of $T(a)$. For widths up to 40 a third method, trying every length up to 3,000 longer than the width, agrees again. @tab-oblong lists the shortest pile for small widths.

#figure(
  placement: auto,
  caption: [For each width $a$ from 1 to 15, the shortest oblong pile holding a square number of balls, or a dash if none does. Exact.],
  {
    let rows = range(1, 16).map(a => {
      let r = R.oblong_smallest.at(str(a))
      if r == none { ([#a], [--], [--]) } else { ([#a], [#r.at(0)], [$#fmt(r.at(1) * r.at(1)) = #r.at(1)^2$]) }
    })
    table(
      columns: 3, align: (right, right, left),
      table.hline(stroke: 0.6pt),
      [Width $a$], [Length $b$], [Balls],
      table.hline(stroke: 0.4pt),
      ..rows.flatten(),
      table.hline(stroke: 0.6pt),
    )
  },
) <tab-oblong>

Width 24 gives Lucas's pile again, with $b = 24$.

= Other shapes <sec-other>

The same question can be asked for every pair of shapes: which tetrahedral piles are squares, which square piles are triangles, and so on. All the pairs among the piles and the flat shapes have been settled. @tab-other lists the answers, the proofs, and our own searches, which agree with each of them.

#figure(
  placement: auto,
  caption: [Piles that hold the number of balls of another shape. "Layers" lists every pile that does. Each row is proved in the source named; our search tried every pile up to #fmt(R.bounds.coincide) layers, and found exactly the piles listed.],
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    table.hline(stroke: 0.6pt),
    [Pile, also a...], [Layers], [Balls; proved by],
    table.hline(stroke: 0.4pt),
    [square pile, square], [1, 24], [1, 4,900; Watson (1918)],
    [square pile, triangle], [1, 5, 6, 85], [1, 55, 91, 208,335; Finkelstein and London (1972)],
    [tetrahedral pile, square], [1, 2, 48], [1, 4, 19,600; Meyl (1878)],
    [tetrahedral pile, triangle], [1, 3, 8, 20, 34], [1, 10, 120, 1,540, 7,140; Avanesov (1967)],
    [tetrahedral pile, square pile], [1], [1; Beukers and Top (1988)],
    table.hline(stroke: 0.6pt),
  ),
) <tab-other>

The row for tetrahedral piles and squares hides Lucas's problem. By @prop-links a tetrahedral pile of $2n$ layers holds $4 P(n)$ balls, which is a square exactly when $P(n)$ is. So the even-layered tetrahedral piles that are squares are the doubles of the square piles that are squares: 2 and 48 layers, with $19,600 = 4 times 4,900 = 140^2$. Lucas asked about tetrahedral piles in 1876, as question 1194 of the same journal, and Meyl answered it in 1878. The link settles the even numbers of layers once Lucas's problem is solved; odd numbers need an argument of their own.

#mnote(inline: true)[Ten balls make a bowling triangle and a pile of three layers.]The number 10 is both a triangle of side 4 and a tetrahedral pile of three layers, and 120 is both a triangle of side 15 and a tetrahedral pile of eight layers. Avanesov proved that 1,540 and 7,140 are the last two numbers of this kind.

*Cubes.* No square pile of more than one layer up to $10^18$ layers holds a cube number of balls. The search splits into the same six cases as @lem-cases, with cubes in place of squares, and walks through each in about a million steps. Remainders rule out only two of the six cases for cubes, so this search is not a proof; we did not find the question settled in the sources we read. For tetrahedral piles, the only cube is the single ball; a case split like @lem-cases reduces this to equations $|a x^3 - b y^3| = 1$, which a theorem of Bennett (2001) settles.

*Piles with the top cut off.* A square pile without its top few layers is a run of consecutive squares, such as $9 + 16 = 25$. Which numbers of layers allow a square? @tab-truncated answers this for up to 30 layers. Our list agrees with the sequence A001032 in the OEIS. Beeckmans (1994) gives conditions on the number of layers and a method that finds all such piles for a given number.

#figure(
  placement: auto,
  caption: [Piles with the top cut off, by number of layers $L$: the side $t >= 1$ of the smallest top layer for which the $L$ layers of sides $t$ to $t + L - 1$ hold a square number of balls. For the other $L$ up to 30 no top works. Every one except 25 is ruled out by a divisor (listed in `results.json`); 25 is @prop-25.],
  {
    let found = R.truncated.filter(r => r.at(1) == "found")
    table(
      columns: 4, align: (right, right, left, left),
      table.hline(stroke: 0.6pt),
      [Layers $L$], [Top $t$], [Layer sides], [Balls],
      table.hline(stroke: 0.4pt),
      ..found.map(r => {
        let (L, t) = (r.at(0), r.at(2))
        let total = range(t, t + L).map(k => k * k).sum()
        let root = calc.round(calc.sqrt(total))
        ([#L], [#t], if L == 1 [#t] else [#t to #(t + L - 1)], [$#fmt(total) = #root^2$])
      }).flatten(),
      table.hline(stroke: 0.6pt),
    )
  },
) <tab-truncated>

#proposition(title: [25 layers])[
  No run of 25 consecutive squares starting from $t^2$ with $t >= 1$ adds up to a square. With $t = 0$, the run $0^2 + 1^2 + dots.c + 24^2 = 70^2$ does.
] <prop-25>

#proof[
  With $u = t + 12$ the run adds up to $25 (u^2 + 52)$, a square exactly when $u^2 + 52 = w^2$ for a whole number $w$, that is when $(w - u)(w + u) = 52$. The two factors have the same parity, so they are 2 and 26, giving $u = 12$ and $t = 0$.
]

So the only square run of 25 squares is Lucas's pile with an empty layer on top.

= Touching balls <sec-touch>

#mnote(inline: true)[A ball deep inside is hugged by twelve others.]In a big pile, a ball in the middle touches four balls in its own layer, four above and four below: twelve in all (@fig-neighbors). In a tetrahedral pile it is six in its layer, three above and three below, again twelve. That is no coincidence: the two piles are cut from the same arrangement of balls, as @sec-relatives explains.

#figure(
  {
    let pos = ()
    // the middle layer, 3 by 3, with the ball in the middle in red and its four neighbors in blue
    for i in range(3) { for j in range(3) {
      let c = if i == 1 and j == 1 { red } else if calc.abs(i - 1) + calc.abs(j - 1) == 1 { blue } else { steel }
      pos.push(((2 * i, 2 * j), c))
    } }
    vis(cetz.canvas(length: 0.55cm, {
      import cetz.draw: *
      for (p, c) in pos { circle(p, radius: 1, fill: shade(c), stroke: 0.3pt + c.darken(60%)) }
      // the four above, drawn as outlines over the hollows next to the red ball
      for (x, y) in ((1, 1), (3, 1), (1, 3), (3, 3)) {
        circle((x, y), radius: 1, stroke: 1.1pt + blue)
      }
      content((2, -1.9), text(size: 8pt)[one layer, from above])
    }))
  },
  caption: [A ball (red) in the middle of a layer of a square pile, seen from above. It touches the four blue balls in its layer and the four balls drawn as blue rings, which rest on it from the layer above. Four more under it touch it from below, in the hollows on the other side.],
) <fig-neighbors>

#tryit[How many pairs of balls touch in a square pile of three layers? Count the pairs inside each layer and the pairs between layers.]

#proposition(title: [touching pairs])[
  A square pile of $n$ layers has $2 n^2 (n - 1)$ touching pairs, a tetrahedral pile of $n$ layers has $(n - 1) n (n + 1)$, and an oblong pile on an $a$ by $b$ base has $2 a^2 (a - 1) + (b - a) a (3a - 2)$.
] <prop-touch>

#proof[
  In a layer of $p$ rows of $q$, $p (q - 1)$ pairs touch along the rows and $q (p - 1)$ along the columns. Each ball of the layer above touches four below. Adding up over the layers of a square pile, $sum_(k=1)^n 2 k (k - 1) + sum_(k=1)^(n - 1) 4 k^2$, which equals $2n^2 (n - 1)$. In a triangle of side $k$, $3 T(k - 1)$ pairs touch, and each ball of the layer above touches three below, so a tetrahedral pile has $sum_(k=1)^n 3 T(k - 1) + sum_(k=1)^(n-1) 3 T(k) = 6 "Te"(n - 1)$. The oblong case is the same sum with layers $(a - h)$ by $(b - h)$.
]

The square pile of 24 layers has #fmt(R.pile24.touching_pairs) touching pairs. A pile can only use so many: a ball can touch at most 12 others of the same size, the _kissing number_ in three dimensions (@sec-relatives), so a pile has fewer than 6 touching pairs per ball. The square pile has $12 n (n - 1) slash ((n + 1)(2n + 1))$ pairs per ball, which climbs toward 6.

#corollary(title: [square numbers of touching pairs])[
  A square pile of $n$ layers has a square number of touching pairs exactly when $2(n - 1)$ is a square: $n = 1, 3, 9, 19, 33, dots$ A tetrahedral pile of two or more layers never does.
] <cor-touch-square>

#proof[
  $2 n^2 (n - 1)$ is $n^2$ times $2(n - 1)$. For the tetrahedral pile, $n$ and $n^2 - 1$ share no factor, so if their product were a square each would be one; but $n^2 - 1$ lies strictly between the squares $(n - 1)^2$ and $n^2$ when $n >= 2$.
]

The square pile of three layers has $2 times 9 times 2 = 36$ touching pairs, which is $6^2$.

#proposition(title: [balls that touch twelve])[
  In a square pile of $n >= 4$ layers, $P(n - 3)$ balls touch twelve others. In a tetrahedral pile of $n >= 5$ layers, $"Te"(n - 4)$ balls do.
] <prop-twelve>

#proof[
  A ball of a square pile touches twelve exactly when it has four neighbors in its layer, four above and four below. That rules out the bottom layer and the edge of each layer; every other ball has all twelve. The layer of side $k$ keeps $(k - 2)^2$ balls, for $k = 3, dots, n - 1$, and these add up to $P(n - 3)$. For the tetrahedral pile, give each ball four numbers, its distances in layers from the four faces of the pile; they add up to $n - 1$. The twelve possible neighbors change one of the four numbers up by one and another down by one, so all twelve are in the pile exactly when all four numbers are at least 1. Taking 1 from each leaves four numbers at least 0 that add up to $n - 5$, and there are $binom(n - 2, 3) = "Te"(n - 4)$ ways to choose them.
]

In the pile of 24 layers, #fmt(R.pile24.balls_by_neighbors.at("12")) balls touch twelve others; they form a square pile of 21 layers inside it.

#figure(
  placement: auto,
  pile3d(square-pile(5, col: (h, i, j) => if j == 0 { blue } else { steel }), width: 4.6cm, turn: 20deg),
  caption: [A square pile of five layers with the balls of one sloping side in blue. On the slope they sit in rows of triangles, as on the sides of a tetrahedral pile.],
) <fig-slope>

@fig-slope shows why the two piles are related. A sloping side of a square pile is packed in triangles, like the floor of a tetrahedral pile. Tip a square pile over onto one of its sloping sides and it rests on a triangle pattern. Both piles are pieces of one endless arrangement, the _face-centered cubic packing_.

= Relatives <sec-relatives>

*Kepler's conjecture.* In 1611 Kepler described the cannonball arrangement in a short book on snowflakes and called it "the tightest possible". The balls fill $pi slash sqrt(18)$ of the space, about 74%. Whether any arrangement of equal balls fills more stayed open for almost four hundred years. Hales announced a proof in 1998 and published it in 2005; the proof leans on long computer calculations, and a team led by Hales later checked every step with proof-checking programs (Hales et al., 2017). Hales's article "Cannonballs and Honeycombs" (2000) tells the story for a general reader. His account of the history describes a chart made in 1591 for counting stored cannonballs in piles "with triangular, square, or oblong base", the three shapes of this note.

*How many balls can touch one.* Twelve balls can touch a thirteenth of the same size, as in a pile. Whether a thirteenth could be squeezed in was argued about as early as 1694. Schütte and van der Waerden (1953) proved that it cannot.

*The Leech lattice.* The number 4,900 turns up in a much larger packing. In 24 dimensions the densest known arrangement of balls is the Leech lattice. Conway and Sloane (1982) showed that it can be built from the 26-dimensional vector $(0, 1, 2, dots, 24 | 70)$. In their geometry the last coordinate counts with a minus sign, so the vector's squared length is $0^2 + 1^2 + dots.c + 24^2 - 70^2 = 0$, which is Lucas's pile.

*Runs of squares.* @tab-truncated is the start of a larger story: Beeckmans (1994) showed that when a number of layers $L$ that is not a square allows a square run, infinitely many runs of $L$ squares add up to a square, and gave necessary conditions on $L$.

= Open questions <sec-open>

+ *Square piles that are cubes.* Is the single ball the only square pile that holds a cube number of balls? The search above found no other up to $10^18$ layers. Four of the six cases for cubes survive every divisor up to 200, so a proof would need another idea, perhaps a bound on equations of the form $|a x^3 - b y^3| = 1$ as for tetrahedral piles.
+ *A short proof for the tetrahedral pile of 34 layers.* Avanesov's proof that 7,140 is the last number that is both a triangle and a tetrahedral pile uses heavy machinery. Is there a proof by Pell equations and remainders, like Anglin's for Lucas's problem?
+ *Oblong piles that are triangles or cubes.* @thm-oblong settles squares. Which widths allow an oblong pile with a triangular number of balls? Because the counts go up in steps of $T(a)$, the same remainder method applies, but we have not worked it out.
+ *Other packings.* The piles here are pieces of the face-centered cubic packing. A pile can also be built with each new layer turned the other way, as in the hexagonal close packing, which is equally dense. Which pile shapes stay stable then, and how do their counts and touching pairs change?

= How this was checked <sec-check>

Every claim in this note is known in one of four ways (@tab-status).

#figure(
  caption: [How each claim is known.],
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [], [Claims],
    table.hline(stroke: 0.4pt),
    [Proved here, by hand], [@prop-tri to @prop-oblong, @lem-cases, @prop-four, @thm-oblong, @cor-share, @prop-25, @prop-touch, @cor-touch-square, @prop-twelve],
    [Computed exactly], [@tab-counts, @tab-chains, @tab-oblong, @tab-truncated, @fig-gaps; every search, up to its stated bound],
    [Cited], [@thm-watson beyond the search; the rows of @tab-other; tetrahedral piles that are cubes; Kepler's conjecture, the kissing number, the Leech lattice],
    [Open], [the questions of @sec-open],
    table.hline(stroke: 0.6pt),
  ),
) <tab-status>

*Two methods for each count.* Each formula is compared with a program that places the balls one by one at points of the face-centered cubic lattice and counts them, for up to 12 layers. The same program finds touching pairs in two ways, by looking up each ball's twelve possible neighbors and by measuring every pair, and checks that no two balls overlap and that every ball above the bottom rests on at least three below.

*Two methods for Lucas's problem.* The walk along the Pell chains agrees up to #fmt(R.bounds.brute) layers with a plain search that adds squares one at a time, and the walk alone goes on to $10^"10,000"$. A test checks that the chains contain every candidate up to a million.

*Three methods for oblong piles.* The rule of @thm-oblong, the remainder method, and trying every length agree on every width checked (@sec-oblong-sq).

*Published lists.* Our searches reproduce every row of @tab-other up to #fmt(R.bounds.coincide) layers and the first terms of OEIS A001032 (@tab-truncated). We found no disagreement with a published result.

*Code.* The `cannonballs` folder of the code that accompanies this note has `piles.py` (counts, piles and touching pairs), `lucas.py` (the searches), `report.py`, which writes every number quoted here to `results.json` in about half a minute, and `figures.py`. A test checks that the committed results file is what the code writes. Only the Python standard library is needed.

*Typesetting.* Typst, with the packages cetz, lilaq and codly.

// ---------- appendices ----------
#show figure.where(kind: table): set text(size: 10pt)
#counter(heading).update(0)
#set heading(hanging-indent: 0pt)
#set heading(numbering: (..n) => "Appendix " + numbering("A.", ..n.pos()))

= Answers to the puzzles <app-answers>

*Triangle of side 10 (@sec-flat).* Half of $10 times 11$, which is 55.

*Piles (@sec-piles).* A square pile of four layers holds $1 + 4 + 9 + 16 = 30$ balls, and a tetrahedral pile of three layers $1 + 3 + 6 = 10$.

*An oblong pile of 100 (@sec-oblong).* Four rows of 11 at the bottom, $44 + 30 + 18 + 8 = 100$; or five rows of 8, $40 + 28 + 18 + 10 + 4 = 100$.

*Touching pairs (@sec-touch).* In a square pile of three layers the layers have 12, 4 and 0 pairs inside them; four balls rest on the bottom layer and one on the middle one, each touching four, which adds $16 + 4$. In all $12 + 4 + 16 + 4 = 36$.

#[
#set par(justify: false)
*More puzzles (@sec-puzzles).*
+ Half of $8 times 9$, which is 36.
+ In the square pile, 4 pairs in the ring and 4 with the top ball: 8. In the tetrahedral pile all four balls touch each other: 6 pairs.
+ Never. The counts $3b - 1$ leave remainder 2 on division by 3, and squares leave only 0 or 1.
+ $20^2 + 21^2 = 400 + 441 = 841 = 29^2$. Another is $119^2 + 120^2 = 169^2$.
+ Four times as many: $"Te"(48) = 4 P(24) = 19,600$, which is $140^2$, a square.
]

= Words and symbols <app-words>

#figure(
  caption: [Words and symbols used in this note.],
  table(inset: (x: 6pt, y: 3pt),
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [Word or symbol], [Meaning],
    table.hline(stroke: 0.4pt),
    [triangle, square], [A flat layer: rows of $1, 2, dots, n$ balls; or $n$ rows of $n$.],
    [tetrahedral pile], [Triangles of sides $1, dots, n$ stacked, each ball in a hollow of three.],
    [square pile], [Squares of sides $1, dots, n$ stacked, each ball in a hollow of four.],
    [oblong pile], [Layers $a$ by $b$, $a - 1$ by $b - 1$, ..., ending in a ridge.],
    [$T(n)$, $"Te"(n)$, $P(n)$], [Balls in a triangle, a tetrahedral pile, a square pile.],
    [$B(a, b)$], [Balls in an oblong pile on an $a$ by $b$ base.],
    [$binom(n, k)$], [“$n$ choose $k$”: the number of ways to choose $k$ things out of $n$.],
    [$a equiv b space (mod 3)$], [“$a$ leaves the same remainder as $b$ on division by 3.”],
    [$sum_(k=1)^n$], [“The sum, for $k$ from 1 to $n$, of ...”],
    [$x arrow.r.bar y$], [“$x$ becomes $y$.”],
    [$|x|$], [The size of $x$, ignoring its sign.],
    [share no factor], [No whole number bigger than 1 divides both.],
    [Pell equation], [An equation such as $y^2 - 6x^2 = 1$, to be solved in whole numbers.],
    [face-centered cubic packing], [The endless arrangement that every pile here is cut from.],
    [kissing number], [The most balls that can touch one ball of the same size: 12.],
    [lemma, theorem], [A stepping stone; a main result. A proposition is a single fact.],
    [$qed$], [The end of a proof.],
    table.hline(stroke: 0.6pt),
  ),
)

#block(breakable: phone)[
#heading(numbering: none, level: 1)[Sources]
#set text(size: 9pt)
- É. Lucas, #link("https://www.numdam.org/item?id=NAM_1875_2_14__336_0")[Question 1180], _Nouvelles Annales de Mathématiques_ (2) 14 (1875), 336. Moret-Blanc, solution, ibid. 15 (1876), 46--48; É. Lucas, solution, ibid. 16 (1877), 429--432.
- G. N. Watson, The problem of the square pyramid, _Messenger of Mathematics_ 48 (1918), 1--22.
- W. Ljunggren, New solution of a problem proposed by E. Lucas, _Norsk Matematisk Tidsskrift_ 34 (1952), 65--72.
- D. G. Ma, An elementary proof of the solutions to the Diophantine equation $6y^2 = x(x+1)(2x+1)$, _Sichuan Daxue Xuebao_ 4 (1985), 107--116.
- W. S. Anglin, #link("https://doi.org/10.2307/2323911")[The square pyramid puzzle], _American Mathematical Monthly_ 97 (1990), 120--124.
- M. A. Bennett, #link("https://doi.org/10.4064/aa105-4-3")[Lucas' square pyramid problem revisited], _Acta Arithmetica_ 105 (2002), no. 4; and Rational approximation to algebraic numbers of small height: the Diophantine equation $|a x^n - b y^n| = 1$, _J. reine angew. Math._ 535 (2001), 1--49.
- A.-J.-J. Meyl, #link("https://www.numdam.org/item/NAM_1878_2_17__464_1.pdf")[Solution de la question 1194], _Nouvelles Annales de Mathématiques_ (2) 17 (1878), 464--467.
- E. T. Avanesov, #link("https://doi.org/10.4064/aa-12-4-409-420")[Solution of a problem on figurate numbers] (in Russian), _Acta Arithmetica_ 12 (1966/67), 409--420.
- R. Finkelstein and H. London, On triangular numbers which are sums of consecutive squares, _Journal of Number Theory_ 4 (1972), 455--462.
- F. Beukers and J. Top, On oranges and integral points on certain plane cubic curves, _Nieuw Archief voor Wiskunde_ (4) 6 (1988), 203--210.
- L. Beeckmans, #link("https://doi.org/10.1080/00029890.1994.11996969")[Squares expressible as sum of consecutive squares], _American Mathematical Monthly_ 101 (1994), 437--442; and #link("https://oeis.org/A001032")[OEIS A001032].
- J. Kepler, _Strena seu de nive sexangula_ (1611); English translation _The Six-Cornered Snowflake_ (Oxford, 1966).
- T. C. Hales, #link("https://www.ams.org/notices/200004/fea-hales.pdf")[Cannonballs and honeycombs], _Notices AMS_ 47 (2000), 440--449; A proof of the Kepler conjecture, _Annals of Mathematics_ 162 (2005), 1065--1185, with the history in the #link("https://annals.math.princeton.edu/supplemental/2005/162-3/p01/proof_kepler.pdf")[unabridged version]; T. Hales et al., #link("https://doi.org/10.1017/fmp.2017.1")[A formal proof of the Kepler conjecture], _Forum of Mathematics, Pi_ 5 (2017), e2.
- K. Schütte and B. L. van der Waerden, Das Problem der dreizehn Kugeln, _Mathematische Annalen_ 125 (1953), 325--334.
- J. H. Conway and N. J. A. Sloane, #link("https://neilsloane.com/doc/Me91.pdf")[Lorentzian forms for the Leech lattice], _Bulletin AMS_ 6 (1982), 215--217.
]
