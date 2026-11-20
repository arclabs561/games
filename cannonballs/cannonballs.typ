#import "@preview/cetz:0.4.2"
#import "@preview/lilaq:0.6.0" as lq
#import "@preview/codly:1.3.0": codly, codly-init

// Every number in this note comes from results.json and gaps.csv, which report.py writes, or
// is checked against them by tests/test_cannonballs.py. The tables read results.json directly.
// The rules, the table of results, the checks and the open questions are the README's shared
// sections (from docs/, filled by tools/sync_docs.py), so the note and the README say the same.
#import "/tools/shared_docs.typ": shared, fmt
#let R = json("results.json")
#let readme = read("README.md")

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
// a displayed formula with words for a screen reader (PDF /Alt on the Formula tag)
#let eqalt(alt, eq) = math.equation(block: true, alt: alt, eq.body)
#let ialt(alt, eq) = math.equation(block: false, alt: alt, eq.body)
// $...$ in the shared Markdown fragments: GitHub reads it as TeX, Typst as its own math. The
// fragments keep to what both read alike; \le becomes <=, and a slash stays a slash.
#let md-math(src, block: false) = math.equation(block: block,
  eval(src.replace(regex("\\\\le\\b"), "<=").replace("/", "\\/"), mode: "math"))
#let sh(name, ..args) = shared(readme, name, math: md-math, ..args)

#let html-css = " :root { color-scheme: light; } body { max-width: 42rem; margin: 0 auto; padding: 1.2rem 1.1rem 4rem; font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, Georgia, serif; font-size: 1.12rem; line-height: 1.55; color: #161616; background: #fff; } h1 { font-size: 2rem; line-height: 1.2; text-align: center; margin: 1.2rem 0 0.4rem; } p.subtitle { text-align: center; font-style: italic; margin: 0; } p.date { text-align: center; color: #666; font-size: 0.9rem; margin: 0.3rem 0 1.6rem; } h2 { font-size: 1.45rem; margin: 2.2rem 0 0.6rem; line-height: 1.25; } h3 { font-size: 1.15rem; margin: 1.6rem 0 0.5rem; } p { margin: 0.7rem 0; } a { color: #0072B2; text-decoration: none; } a:hover { text-decoration: underline; } svg { max-width: 100%; height: auto !important; vertical-align: middle; } svg[style*=\"display: block\"] { margin: 0.8rem auto; } figure { margin: 1.6rem 0; text-align: center; overflow-x: auto; } figure.thm, figure:has(div.thm) { text-align: left; } figcaption { font-size: 0.9rem; color: #444; margin-top: 0.5rem; text-align: center; } table { border-collapse: collapse; margin: 0.4rem auto; font-size: 0.9rem; line-height: 1.35; border-top: 1.5px solid #222; border-bottom: 1.5px solid #222; } td, th { padding: 0.25rem 0.55rem; text-align: left; vertical-align: top; } tr:first-child td, tr:first-child th { font-weight: 600; border-bottom: 1px solid #222; } aside.box { border-left: 4px solid; padding: 0.6rem 0.95rem; margin: 1.3rem 0; border-radius: 0 5px 5px 0; font-size: 1rem; line-height: 1.45; } aside.idea { background: #fdf6e7; border-color: #E69F00; } aside.tryit { background: #f0f1f3; border-color: #5b6770; } aside.box p { margin: 0.35rem 0; } .boxlabel { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; margin-right: 0.45em; } aside.idea .boxlabel { color: #8a5a00; } aside.tryit .boxlabel { color: #3d464d; } pre { text-align: left; overflow-x: auto; font-size: 0.78rem; line-height: 1.45; background: #fafafa; border: 1px solid #ddd; border-radius: 4px; padding: 0.7rem 0.8rem; } code { font-size: 0.9em; overflow-wrap: anywhere; } pre code { overflow-wrap: normal; } math { font-size: 1.06em; font-family: \"STIX Two Math\", \"Latin Modern Math\", \"Cambria Math\", math; } math[display=\"block\"] { margin: 0.9rem 0; overflow-x: auto; overflow-y: hidden; } ul, ol { padding-left: 1.4rem; } li { margin: 0.25rem 0; } .thm { margin: 1.2rem 0; } blockquote.epigraph { margin: 0.2rem 0 1.4rem auto; max-width: 30rem; text-align: right; font-style: italic; font-size: 0.95rem; } p.mnote { font-size: 0.88rem; font-style: italic; color: #555; margin-left: 1.2rem; } blockquote.epigraph footer { font-style: normal; font-size: 0.85rem; color: #555; } "
#context if target() == "html" { html.elem("style", html-css) }
#show align: it => context if target() == "html" { it.body } else { it }

// ---------- look ----------
#set document(title: "How Many Cannonballs?", date: none)  // no date: every build is identical
#set page(..if phone { (width: 4.4in, height: 7.6in, margin: (x: 0.32in, top: 0.42in, bottom: 0.55in)) }
  else { (paper: "us-letter", margin: (x: 1.55in, y: 1.1in)) }, numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)
// ragged right on the phone, as in the Erase paper: justifying a line of about 50 characters
// opens wide gaps
#set par(justify: not phone, leading: 0.6em, spacing: 1em)
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
#show figure.caption: set par(justify: not phone)
#show math.equation.where(block: true): it => if phone { fitw(it) } else { it }
// the shared table of results (tools/shared_docs.typ) fills the width it gets and breaks across pages
// tables labeled <phone-breaks> (the glossary) are set at the phone's own width, as in Erase
#show table: it => if phone and it.at("label", default: none) not in (<shared-table>, <phone-breaks>) {
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

This note counts the balls in piles of every common shape and explains why each count has the formula it has. It then asks which piles hold a square number of balls, and how many pairs of balls touch. Most of the answers are classical, and the note says where each comes from. It writes out a short proof, from a theorem of Euler, that the single ball is also the only square pile that holds a cube number of balls. It proves an elementary rule for which widths of an oblong pile allow a square number of balls: 7 in 12 of them in the long run, and never 2 or 3. We did not find that rule in the sources we read. For oblong piles of a fixed shape it lists the squares exactly, with a computer algebra system, for the bases $a$ by $a + c$ with $c$ up to 20. The counts of touching pairs turn out to be sequences already in the OEIS, the online encyclopedia of integer sequences. A computer search below #ialt("10 to the power 10,000", $10^"10,000"$) layers finds no square pile other than those of 1 and 24 layers that holds a square number of balls. It adds nothing to Watson's proof; it is there to check the code and the method.

*How to read this.* Sections #ref(<sec-flat>, supplement: none) to #ref(<sec-puzzles>, supplement: none) tell the story with pictures and arithmetic, and a parent and child can read them together. Sections #ref(<sec-proofs>, supplement: none) to #ref(<sec-touch>, supplement: none) prove the claims. Sections #ref(<sec-relatives>, supplement: none) to #ref(<sec-check>, supplement: none) set the piles beside their relatives, list open questions and say how everything was checked. Boxes marked #text(size: 8.5pt, weight: "bold", fill: orange.darken(35%), tracking: 0.04em)[IDEA] explain a tool the first time it is used. Boxes marked #text(size: 8.5pt, weight: "bold", fill: slate.darken(35%), tracking: 0.04em)[TRY IT] are puzzles to try before reading on, with answers in @app-answers. It is fine to skip a proof, or to stop anywhere.

#figure(
  {
    let tet = tetra-pile(4)
    let sq = square-pile(4)
    let g = grid(columns: 2, column-gutter: 2.4em, align: bottom, pile3d(sq, width: 4.2cm), pile3d(tet, width: 3.6cm, turn: 12deg))
    visb(fitw(g))
  },
  alt: "Two piles of shaded balls: a square pile of four layers, with a square base of 16 and one ball on top, and a tetrahedral pile of four layers, with a triangular base of 10 and one ball on top.",
  caption: [A square pile of four layers (30 balls) and a tetrahedral pile of four layers (20 balls). Every ball above the bottom rests in a hollow between balls of the layer below.],
) <fig-two-piles>

= Balls on the ground <sec-flat>

Before stacking, lay the balls flat. A *triangle* of side 4 has rows of 1, 2, 3 and 4 balls, like the ten pins in bowling. A *square* of side 4 has four rows of four.

#figure(
  visb(fitw(grid(columns: 4, column-gutter: 1.6em, align: bottom,
    ..range(1, 5).map(n => flat(tri(n)))))),
  alt: "Four triangles of balls, of side 1, 2, 3 and 4, holding 1, 3, 6 and 10 balls.",
  caption: [Triangles of side 1 to 4 hold 1, 3, 6 and 10 balls. These are the _triangular numbers_.],
) <fig-triangles>

#tryit[How many balls are in a triangle of side 10? Try to find the answer without adding ten numbers one by one.]

#beside(right: 46mm)[
#mnote(inline: true)[Two triangles make a box. Then halve it!]Here is a quick way. Slide the rows of the triangle to the left, so that it becomes a staircase. Take a second staircase of the same size, turn it upside down and push it against the first. For side 10 the two make a rectangle of 10 rows with 11 balls in each row. So one triangle holds half of $10 times 11$, which is half of 110, or 55.
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
  alt: "A rectangle of 4 rows of 5 balls: a blue staircase of 1, 2, 3 and 4 balls on the left, and a green staircase of 4, 3, 2 and 1 balls fitted against it.",
  caption: [A blue triangle of side 4 and a green one turned over fill a rectangle of 4 rows of 5. So the triangle holds half of $4 times 5$, which is 10.],
) <fig-pairing>
]

#idea[a formula][A _formula_ is a recipe that works for every size at once. The letter $n$ stands for the side of the triangle, whatever it is. The recipe "multiply $n$ by $n + 1$, then halve" gives the triangle for every $n$: write $T(n) = n(n+1) slash 2$, and read $T(n)$ as "T of n." The picture is the reason the recipe works, not only an example of it.]

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
  alt: "A square of 5 rows of 5 balls, cut along a staircase: the blue balls on and below the diagonal make a triangle of side 5, and the green balls above it a triangle of side 4.",
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
  alt: "The square pile and the tetrahedral pile of four layers, with their layers lifted apart so each layer can be seen.",
  caption: [The two piles of @fig-two-piles with their layers lifted apart. The square pile has layers of 16, 9, 4 and 1 balls; the tetrahedral pile has layers of 10, 6, 3 and 1.],
) <fig-layers>

#tryit[A square pile has five layers. How many balls does it hold? And a tetrahedral pile of five layers?]

Adding up the layers gives the counts in @tab-counts. A square pile of four layers holds $1 + 4 + 9 + 16 = 30$ balls, and a tetrahedral pile of three layers holds $1 + 3 + 6 = 10$, the same as a triangle of side 4.

#figure(
  caption: [How many balls each shape holds, for sides or layers from 1 to 10. Exact, by adding up the layers and by building each pile ball by ball in a program.],
  table(
    columns: 5,
    align: (center, right, right, right, right),
    table.hline(stroke: 0.6pt),
    table.header([Side or\ layers], [Triangle], [Square], [Tetrahedral\ pile], [Square\ pile]),
    table.hline(stroke: 0.4pt),
    ..R.table.map(row => row.map(v => [#fmt(v)])).flatten(),
    table.hline(stroke: 0.6pt),
  ),
) <tab-counts>

#block(breakable: false, width: 100%)[
A tetrahedral pile of $n$ layers holds
#eqalt("T e of n equals n times n plus 1 times n plus 2, over 6.", $ "Te"(n) = (n (n + 1) (n + 2)) / 6 $)
balls, and a square pile of $n$ layers holds
#eqalt("P of n equals n times n plus 1 times 2 n plus 1, over 6.", $ P(n) = (n (n + 1) (2n + 1)) / 6. $)
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

// a square pile with each layer slid so that one corner of every layer sits over the same
// corner of the bottom layer; the layers are lifted apart a little and shaded in turn, so
// that each one can be seen
#let slid-pile(n) = {
  let out = ()
  for h in range(n) { for i in range(n - h) { for j in range(n - h) {
    out.push(((2 * i, 2 * j, 2.9 * h), if calc.even(h) { steel } else { steel.lighten(45%) }))
  } } }
  out
}

#figure(
  {
    let slid = pile3d(slid-pile(4), width: 3.4cm, turn: 215deg, tilt: 22deg)
    let cubes = grid(columns: 2, column-gutter: 2.4em, align: horizon, cube-pyramids(0), cube-pyramids(1.1))
    visb(fitw(if phone { stack(spacing: 1.2em, align(center, slid), cubes) }
      else { grid(columns: 2, column-gutter: 2.4em, align: horizon, slid, cubes) }))
  },
  alt: "Left, a square pile of four layers with every layer slid into its back corner, so that it forms a stepped pyramid with its tip over a corner of the base. Middle, three colored pyramids that together fill a cube. Right, the same three pyramids pulled apart.",
  caption: [Left: a square pile of four layers with each layer slid into the back corner, and the layers lifted apart a little. It is a stepped pyramid with its tip over a corner of its base. Middle: three solid pyramids of that shape fill a cube. Each has a face of the cube as its base and its tip at the cube's top front corner. Right: the three pulled apart.],
) <fig-cube>

*The exact count, with three triangles.* Write the layers of a square pile of $n$ layers as rows of numbers: one 1, two 2s, three 3s, up to $n$ copies of $n$. Row $k$ adds up to $k times k$, so the whole triangle of numbers adds up to $P(n)$. Now turn the triangle a third of a turn, and again, and lay the three copies on top of each other (@fig-three). At every position the three numbers add up to $2n + 1$. There are $T(n)$ positions, so $3 P(n) = (2n + 1) T(n)$, and dividing by 3 gives the formula. The 6 is the 3 from the three copies times the 2 from the triangle.

#figure(
  {
    let n = 4
    let lab(f) = (r, k) => str(f(r, k))
    // on a phone the four triangles go two by two, so that the numbers stay readable
    let (d2, d3, d4) = if phone { ((10, 0), (4, -9), (14, -9)) } else { ((10, 0), (20, 0), (30, 0)) }
    let signs = if phone { ((5, -3, $+$), (-1.5, -12, $+$), (9, -12, $=$)) } else { ((5, -3, $+$), (15, -3, $+$), (25, -3, $=$)) }
    let shift(t, d) = t.map(b => ((b.at(0).at(0), b.at(0).at(1) + d.at(1)), b.at(1), b.at(2)))
    let t1 = tri(n, col: (r, k) => blue, label: lab((r, k) => r + 1))
    let t2 = shift(tri(n, col: (r, k) => green, dx: d2.at(0), label: lab((r, k) => n - k)), d2)
    let t3 = shift(tri(n, col: (r, k) => purple, dx: d3.at(0), label: lab((r, k) => k + n - r)), d3)
    let t4 = shift(tri(n, col: (r, k) => slate, dx: d4.at(0), label: lab((r, k) => 2 * n + 1)), d4)
    visb(fitw(cetz.canvas(length: if phone { 0.42cm } else { 0.3cm }, {
      import cetz.draw: *
      for b in t1 + t2 + t3 + t4 {
        // a light disk with a dark rim, so that the black number reads at any size
        circle(b.at(0), radius: 1, fill: b.at(1).lighten(72%), stroke: 0.9pt + b.at(1))
        content(b.at(0), text(size: 7.5pt, weight: "bold", b.at(2)))
      }
      for (x, y, s) in signs { content((x, y), text(size: 13pt, s)) }
    })))
  },
  alt: "Three triangles of numbers for a square pile of four layers, and their sum. The first has rows 1; 2 2; 3 3 3; 4 4 4 4. The second and third are the same triangle turned a third of a turn and two thirds. Adding the three position by position gives a triangle with 9 at every one of its ten positions.",
  caption: [The layers of a square pile of four layers, written as a triangle of numbers (blue), and the same triangle turned a third of a turn (green) and two thirds (purple). Read any position across the three: the numbers always add up to 9, which is $2 times 4 + 1$. Ten positions with 9 at each make 90, three times 30.],
) <fig-three>

*The tetrahedral pile is a way of choosing.* A ball in a tetrahedral pile of $n$ layers can be named by three steps from the top: how many balls along each of three edges, which gives three whole numbers $i$, $j$, $k$ with $i + j + k$ at most $n - 1$. The three numbers $i + 1$, $i + j + 2$ and $i + j + k + 3$ go up, and all lie between 1 and $n + 2$. Every way of choosing three different numbers from 1 to $n + 2$ arises from exactly one ball. So the pile holds as many balls as there are ways to choose three things out of $n + 2$, and that number is $(n + 2)(n + 1) n slash 6$. Here the 6 counts the orders in which three chosen things could be listed: $3 times 2 times 1$.

#beside(right: 58mm,
  idea[choosing][The number of ways to choose 3 things out of 5 is 10: list them as 123, 124, 125, 134, 135, 145, 234, 235, 245, 345. Picking them in order gives $5 times 4 times 3 = 60$ lists, but each choice turns up in $3 times 2 times 1 = 6$ orders, so there are $60 slash 6 = 10$ choices. A tetrahedral pile of three layers holds 10 balls, one for each choice, as @fig-choose shows.],
  [#figure(
    {
      let r3 = calc.sqrt(3)
      let layers = range(3).map(d => tri(d + 1, dx: (0, 5, 12).at(d),
        label: (r, k) => str(r - k + 1) + str(r + 2) + str(d + 3)))
      visb(fitw(cetz.canvas(length: 0.4cm, {
        import cetz.draw: *
        for (d, layer) in layers.enumerate() {
          for b in layer {
            circle(b.at(0), radius: 1, fill: steel.lighten(65%), stroke: 0.8pt + steel.darken(25%))
            content(b.at(0), text(size: 6.5pt, weight: "bold", b.at(2)))
          }
          content(((0, 5, 12).at(d), -2 * r3 - 2.3), text(size: 8pt, ("top", "middle", "bottom").at(d)))
        }
      })))
    },
    alt: "The three layers of a tetrahedral pile of three layers, side by side. The top ball is labeled 123. The middle layer of three is labeled 124, 234 and 134. The bottom layer of six is labeled 125, 235, 135, 345, 245 and 145.",
    caption: [The tetrahedral pile of three layers, taken apart. Each ball carries the three numbers from 1 to 5 that it chooses. In a layer the largest number is the same for every ball, and in a row the middle one is.],
  ) <fig-choose>],
)

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
  alt: "The four layers of a square pile of four layers, squares of side 1, 2, 3 and 4, each split by a staircase into a blue triangle on and below the diagonal and a green triangle above it.",
  caption: [The four layers of a square pile of four layers, each cut next to its diagonal. The blue triangles, of sides 1 to 4, add up to a tetrahedral pile of four layers (20 balls); the green ones, of sides 0 to 3, to one of three layers (10).],
) <fig-diagonal>

= Oblong piles <sec-oblong>

Piles were also built on a rectangle. Such a pile starts from a rectangle of $a$ rows of $b$ balls, with $a$ at most $b$. The next layer is $a - 1$ by $b - 1$, and so on, until the top is a single row of $b - a + 1$ balls, a ridge like the top of a roof. Call this an *oblong pile*.

#figure(
  pile3d(oblong-pile(3, 6), width: 5cm),
  alt: "An oblong pile of shaded balls on a base of 3 rows of 6, rising to a ridge of four balls.",
  caption: [An oblong pile on a base of 3 rows of 6: layers of 18, 10 and 4 balls, 32 in all. The top is a ridge of four balls.],
) <fig-oblong>

#mnote(inline: true)[A longer pile is the same pile plus one more slice.]Making an oblong pile one ball longer adds a slice on its end, and that slice is a triangle of side $a$. A square pile is the oblong pile with $b = a$, so the count is the square pile plus $b - a$ triangles:
#eqalt("a times a plus 1 times 2 a plus 1, over 6, plus b minus a times a times a plus 1 over 2, equals a times a plus 1 times 3 b minus a plus 1, over 6.", $ (a (a+1)(2a+1)) / 6 + (b - a) (a (a+1)) / 2 = (a (a + 1) (3 b - a + 1)) / 6. $)

#tryit[Find an oblong pile with exactly 100 balls. (There are two, apart from a single row of 100.)]

= The cannonball problem <sec-lucas>

A square pile of 24 layers holds 4,900 balls. That is $70 times 70$, so the same balls could be laid out as a flat square of side 70 (@fig-4900). Do other square piles do this?

#figure(
  {
    let pile = pile3d(square-pile(24, lit: (h, i, j) => h == 23 or i == 0 or j == 0 or i == 23 - h or j == 23 - h), width: 6.2cm)
    let dots = vis(box(width: 4.2cm, height: 4.2cm, fill: tiling(size: (0.06cm, 0.06cm), place(circle(radius: 0.026cm, fill: slate)))))
    visb(fitw(grid(columns: 2, column-gutter: 1.6em, align: horizon, pile, dots)))
  },
  alt: "Left, a square pile of 24 layers of shaded balls. Right, a flat square of 70 rows of 70 small dots.",
  caption: [4,900 balls two ways: a square pile of 24 layers, and a flat square of 70 rows of 70.],
) <fig-4900>

The French mathematician Édouard Lucas posed this in 1875 as question 1180 of the _Nouvelles Annales de Mathématiques_. He asserted that a square pile of balls holds a square number "que lorsqu'elle en contient 24 sur le côté de la base": only when its base has 24 on a side. His statement leaves out the pile of a single ball, which is a square too, a $1 times 1$ square. Lucas could not prove his claim. A proof by Moret-Blanc (1876) and one by Lucas (1877) both had gaps. The first complete proof came from G. N. Watson in 1918.

#theorem(title: [Watson, 1918])[
  A square pile of $n$ layers holds a square number of balls only when $n = 1$ or $n = 24$.
] <thm-watson>

The theorem is about every $n$, and no computer can try every $n$. What a computer can do is try a great many. @fig-gaps shows how close each square pile up to 60 layers comes to a square. Only two land on one.

#let gap-rows = csv("gaps.csv", row-type: dictionary)
#let near = gap-rows.filter(r => int(r.distance) == 1)
#figure(
  placement: auto,
  {
    let xs = gap-rows.map(r => int(r.layers))
    // a log scale has no place for 0, so the exact squares get a row of their own below it
    let y(r) = if int(r.distance) == 0 { 0.4 } else { float(r.distance) }
    let hit = gap-rows.filter(r => int(r.distance) == 0)
    let miss = gap-rows.filter(r => int(r.distance) > 0)
    lqd(width: 10.5cm, height: 4.6cm,
      xlabel: [layers in the square pile], ylabel: [balls from a square],
      xlim: (0, 61), ylim: (0.28, 400), yscale: "log",
      yaxis: (ticks: ((0.4, [0]), (1, [1]), (10, [10]), (100, [100])), subticks: none),
      lq.plot((0, 61), (0.65, 0.65), stroke: (paint: luma(60%), thickness: 0.6pt, dash: "dashed"), mark: none),
      lq.scatter(miss.map(r => int(r.layers)), miss.map(y), size: 6pt, color: slate, stroke: none),
      lq.scatter(hit.map(r => int(r.layers)), hit.map(y), size: 10pt, color: red, stroke: none),
    )
  },
  alt: "A dot plot over square piles of 1 to 60 layers of how many balls each is from the nearest square, on a log scale from 1 to a few hundred. Below a dashed line, on a row of its own for 0, are the two exact squares, 1 and 24 layers. Every other pile is at least one ball away; 2, 47 and 48 layers are exactly one ball away.",
  caption: [For each square pile of 1 to 60 layers, how many balls it is from the nearest square, on a log scale. The orange dots below the dashed line are exact squares: 1 layer and 24 layers. Three piles miss by a single ball: #near.map(r => {
    let (n, m, root) = (int(r.layers), int(r.balls), int(r.root))
    [#n #if n == 1 [layer] else [layers], with #fmt(m) = #if m > root * root [$#root^2 + 1$] else [$#root^2 - 1$]]
  }).join([, ], last: [ and ]). Exact; written by `report.py` to `gaps.csv`.],
) <fig-gaps>

#idea[a search is not a proof][A computer that tries every pile up to some size can say what happens up to that size, and nothing beyond it. The pattern "no square after 24" could break at a size nobody has tried. A _proof_ is an argument that covers every size at once. So a search tells us what to believe, and a proof tells us why it must be so.]

Going further takes a better idea than trying every pile one by one. @sec-why-24 sorts every possible solution into one of six kinds, rules four out with remainders, and turns each of the other two into a Pell equation, an equation whose whole-number solutions come in a chain, each one giving the next. That makes a fast search possible: it rules out every square pile other than 1 and 24 below #ialt("10 to the power 10,000", $10^"10,000"$) layers, a 1 followed by ten thousand zeros, after checking #fmt(R.pell_walked.at("611") + R.pell_walked.at("123")) candidates. The search proves nothing new, since Watson's proof covers every size; it checks that the program and the method agree with the theorem.

= More puzzles <sec-puzzles>

Answers are in @app-answers.

#puzzle[*1.* A box holds the balls of a triangle of side 8. How many balls is that? Use the pairing trick.]
#puzzle[*2.* In a square pile of two layers, the top ball touches the four below it, and the four bottom balls touch their neighbors in a ring. How many touching pairs is that? How many in a tetrahedral pile of two layers?]
#puzzle[*3.* An oblong pile two balls wide and $b$ balls long holds $3b - 1$ balls: 5, 8, 11, 14, and so on. Can it ever be a square? Look at what is left over when you divide the counts by 3 (the remainder: 14 leaves 2, since $14 = 4 times 3 + 2$), and what is left over when you divide the squares 1, 4, 9, 16, 25 by 3.]
#puzzle[*4.* Cut the top off a square pile and keep two layers, of sides 3 and 4: $9 + 16 = 25$, a square. Find two more layers next to each other whose balls add up to a square.]
#puzzle[*5.* A tetrahedral pile of 48 layers holds how many times as many balls as a square pile of 24 layers? Is the number a square?]

= The formulas, proved <sec-proofs>

From here on the note proves its claims, about the piles described below.

#sh("rules")

Write $T(n)$, $"Te"(n)$ and $P(n)$ for the balls in a triangle of side $n$, a tetrahedral pile of $n$ layers and a square pile of $n$ layers, and $B(a, b)$ for an oblong pile on an $a$ by $b$ base, with $a <= b$. Each statement says how it is known. Every formula was also checked by a program that builds the pile ball by ball, for up to 12 layers.

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
  Label the positions of a triangle of side $n$ by three whole numbers $(u, v, w)$ that add up to $n - 1$: $u$ counts the rows up from the bottom edge, and $v$ and $w$ do the same from the other two edges. Put the number $n - u$ at each position. The row with $u = n - k$ holds $k$ positions, each carrying $k$, so the numbers add up to $1^2 + dots.c + n^2 = P(n)$. The triangle turned a third of a turn carries $n - v$ at the same position, and turned twice $n - w$. The three add up to $3n - (u + v + w) = 2n + 1$ at each of the $T(n)$ positions.
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
  #eqalt("The cases 1 2 3; 2 1 3; 1 6 1; 6 1 1; 2 3 1; and 3 2 1.", $ (1, 2, 3), (2, 1, 3), (1, 6, 1), (6, 1, 1), (2, 3, 1), (3, 2, 1) $)
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
  Three of the cases give an equation between two of the three numbers that fails on division by 3; the fourth factors.
  Case $(2, 1, 3)$: $2n + 1 = 4x^2 + 1$ must be $3z^2$, a multiple of 3; but $4 x^2 + 1 equiv x^2 + 1$, which leaves 1 or 2.
  Case $(1, 6, 1)$: $n + 1 = x^2 + 1$ must be $6 y^2$, a multiple of 3; again $x^2 + 1$ leaves 1 or 2.
  Case $(3, 2, 1)$: $n + 1 = 3x^2 + 1$ must be $2y^2$; the left side leaves 1, while $2y^2$ leaves 0 or 2.
  Case $(2, 3, 1)$: $2n + 1 = 4x^2 + 1$ must be $z^2$, so $(z - 2x)(z + 2x) = 1$, which forces $x = 0$ and $n = 0$.
]

The two cases left are the ones the two piles fall in: $n = 1$ is case $(1, 2, 3)$, with $1 = 1^2$, $2 = 2 dot 1^2$, $3 = 3 dot 1^2$; and $n = 24$ is case $(6, 1, 1)$, with $24 = 6 dot 2^2$, $25 = 5^2$, $49 = 7^2$. So no remainder can rule out either case, since each holds a real solution. What is left is to show that each case holds no other solution, and that needs a different idea.

*Step 2: Pell equations.* In case $(6, 1, 1)$, the first two conditions say $y^2 - 6x^2 = 1$. In case $(1, 2, 3)$ they say $x^2 - 2y^2 = -1$.

#idea[Pell equations][An equation such as $y^2 - 6 x^2 = 1$ is called a _Pell equation_. Its whole-number solutions form an endless chain, and each one gives the next by a fixed rule: from $(y, x) = (5, 2)$ the rule $(y, x) arrow.r.bar (5y + 12x, 2y + 5x)$ gives $(49, 20)$, then $(485, 198)$. A classical theorem says that the chain contains every solution. For $x^2 - 2y^2 = -1$ the chain starts $(1, 1), (7, 5), (41, 29)$, with the rule $(x, y) arrow.r.bar (3x + 4y, 2x + 3y)$.]

So in each case the candidates for $n$ come one after another along a chain, and only the third condition is left to test: is $12 x^2 + 1$ a square in case $(6, 1, 1)$, and is $(2x^2 + 1) slash 3$ a square in case $(1, 2, 3)$? @tab-chains shows the first few links. Each link multiplies $n$ by about 98 in case $(6, 1, 1)$, which is $(5 + 2 sqrt(6))^2$, and by about 34 in case $(1, 2, 3)$, which is $(3 + 2 sqrt(2))^2$. So a link adds about two digits to $n$ in the first case and one and a half in the second, and a few thousand links reach numbers of ten thousand digits.

#figure(
  placement: auto,
  caption: [The first four solutions of each Pell equation, with the number the third condition needs to be a square. The last column says whether it is one. Exact.],
  table(
    columns: 5,
    align: (left, right, right, right, center),
    table.hline(stroke: 0.6pt),
    table.header([Case], [$x$], [layers $n$], [must be a square], [square?]),
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
  // comments on lines of their own, so that the listing fits a phone page at full size
  [```python
# the first solution of y^2 - 6 x^2 = 1
y, x = 5, 2
while 6 * x * x < 10**10_000:
    # then n + 1 = y^2 is a square
    n = 6 * x * x
    # the third condition
    if is_square(2 * n + 1):
        found.append(n)
    y, x = 5 * y + 12 * x, 2 * y + 5 * x
```]
) <lst-walk>

*The search.* Walking both chains below $10^"10,000"$ layers takes a few seconds: #fmt(R.pell_walked.at("611")) links in case $(6, 1, 1)$ and #fmt(R.pell_walked.at("123")) in case $(1, 2, 3)$. Only the known solutions 24 and 1 pass the third test. Since every solution of each Pell equation lies on its chain, this rules out every other $n$ below that bound. A plain search that adds up squares one at a time agrees for every $n$ up to #fmt(R.bounds.brute), and a test compares the two methods on every candidate up to $10^6$.

*Step 3: the chains never meet again.* Showing that no later link passes the third test is the hard part. Watson (1918) did most cases by elementary means and the last with elliptic functions. Ljunggren (1952) gave a proof by algebra in a larger number system. Ma (1985) and Anglin (1990) gave proofs that use only Pell equations and remainders. Roughly, such proofs divide the terms of one chain by terms of related chains and use the rules for which remainders a square can leave to show that no later term passes. Anglin's proof fits in five journal pages. We have not checked these proofs and rely on them for every $n$ beyond our search.

The equation $6 m^2 = n(n+1)(2n+1)$ also describes a curve, of the kind called elliptic, and @thm-watson says which points on it have whole-number coordinates. Bennett (2002) bounded the number of solutions of $x(x+1)(x+2) = d y^2$ for every $d$ with no square factor, and the bound settles Lucas's problem as the case $d = 6$, with $x = 2n$.

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
      table.header([Width $a$], [Length $b$], [Balls]),
      table.hline(stroke: 0.4pt),
      ..rows.flatten(),
      table.hline(stroke: 0.6pt),
    )
  },
) <tab-oblong>

Width 24 gives Lucas's pile again, with $b = 24$.

= Other shapes <sec-other>

The same question can be asked for every pair of shapes: which tetrahedral piles are squares, which square piles are triangles, and so on. Every pair among the square and tetrahedral piles, the triangles and the squares has been settled. @tab-other lists the answers, the proofs, and our own searches, which agree with each of them. Oblong piles and cubes are treated after it.

#figure(
  placement: auto,
  caption: [Piles that hold the number of balls of another shape. "Layers" lists every pile that does, and "Balls" the numbers of balls, separated by semicolons. Each row is proved in the source named; our search tried every pile up to #fmt(R.bounds.coincide) layers, and found exactly the piles listed. The second and fourth rows are OEIS A053611 and A224421.],
  {
    let row(kind, who, layers, balls) = ([#kind \ #text(size: 0.85em, fill: luma(30%), who)], layers, balls)
    table(
      columns: (1fr, auto, auto),
      align: (left, left, left),
      table.hline(stroke: 0.6pt),
      table.header([Pile, also a...; proved by], [Layers], [Balls]),
      table.hline(stroke: 0.4pt),
      ..row[square pile, square][Watson (1918)][1; 24][1; 4,900],
      ..row[square pile, triangle][Finkelstein and London (1972)][1; 5; 6; 85][1; 55; 91; 208,335],
      ..row[tetrahedral pile, square][Meyl (1878)][1; 2; 48][1; 4; 19,600],
      ..row[tetrahedral pile, triangle][Avanesov (1967)][1; 3; 8; 20; 34][1; 10; 120; 1,540; 7,140],
      ..row[tetrahedral pile, square pile][Beukers and Top (1988)][1][1],
      table.hline(stroke: 0.6pt),
    )
  },
) <tab-other>

The row for tetrahedral piles and squares hides Lucas's problem. By @prop-links a tetrahedral pile of $2n$ layers holds $4 P(n)$ balls, which is a square exactly when $P(n)$ is. So the even-layered tetrahedral piles that are squares are the doubles of the square piles that are squares: 2 and 48 layers, with $"19,600" = 4 times "4,900" = 140^2$. Lucas asked about tetrahedral piles in 1876, as question 1194 of the same journal, and Meyl answered it in 1878. The link settles the even numbers of layers once Lucas's problem is solved; odd numbers need an argument of their own.

#mnote(inline: true)[Ten balls make a bowling triangle and a pile of three layers.]The number 10 is both a triangle of side 4 and a tetrahedral pile of three layers, and 120 is both a triangle of side 15 and a tetrahedral pile of eight layers. Avanesov proved that 1,540 and 7,140 are the last two numbers of this kind.

*Cubes.* Which square piles hold a cube number of balls? The single ball does, since $1 = 1^3$. A search of every pile up to #ialt("10 to the 18th", $10^18$) layers finds no other, and a short argument shows there is none. It needs one more fact about remainders, this time on division by 7: a cube leaves only 0, 1 or 6, as $0^3, 1^3, dots, 6^3$ leave $0, 1, 1, 6, 1, 6, 6$.

#proposition(title: [cubes])[
  A square pile of $n >= 1$ layers holds a cube number of balls only when $n = 1$.
] <prop-cubes>

#proof(title: [Proof, using a theorem of Euler])[
  Suppose that $n(n+1)(2n+1) = 6 m^3$. As in @lem-cases, the three numbers share no factor, so each is a cube times its share of the 6, and the shares give the same six cases: $n = s_1 x^3$, $n + 1 = s_2 y^3$ and $2n + 1 = s_3 z^3$.

  Two cases fail on division by 7. In case $(2, 3, 1)$, $n = 2x^3$ leaves 0, 2 or 5, and $n = 3y^3 - 1$ leaves 6, 2 or 3, so $n$ leaves 2; then $2n + 1$ leaves 5, which is not the remainder of a cube. In case $(3, 2, 1)$, $n = 3x^3$ leaves 0, 3 or 4, and $n = 2y^3 - 1$ leaves 6, 1 or 4, so $n$ leaves 4; then $2n + 1$ leaves 2, again not the remainder of a cube.

  Each of the other four cases gives the equation $u^3 + 1 = 2 v^3$ in whole numbers, from two of the differences $(n + 1) - n = 1$, $(2n + 1) - 2n = 1$ and $2(n + 1) - (2n + 1) = 1$:
  - case $(1, 2, 3)$: $x^3 + 1 = 2y^3$;
  - case $(6, 1, 1)$: $z^3 + 1 = 2y^3$;
  - case $(1, 6, 1)$: $z^3 - 1 = 2x^3$, that is $(-z)^3 + 1 = 2(-x)^3$;
  - case $(2, 1, 3)$: $y^3 - 1 = 2x^3$, that is $(-y)^3 + 1 = 2(-x)^3$.
  Euler proved in 1770 that the sum of two cubes is twice a cube, $u^3 + w^3 = 2v^3$, only when $u = w$ or $u = -w$. With $w = 1$, the only solutions are $u = 1$, $v = 1$ and $u = -1$, $v = 0$. Going back through the four cases, $n$ is one of $-2$, $-1$, $0$ and $1$, and the only one that is a number of layers is $n = 1$.
]

The proof uses Euler's theorem only for $u^3 + 1 = 2v^3$, and a search confirms that no whole number $u$ from $-#fmt(R.bounds.euler)$ to #fmt(R.bounds.euler) other than #R.euler_cube_solutions.map(p => str(p.at(0)).replace("-", "−")).join(" and ") solves it; a program also confirms the two cases ruled out on division by 7. A second route is a theorem of Bennett (2001): an equation $|c u^3 - d v^3| = 1$, for fixed positive whole numbers $c$ and $d$, has at most one solution in positive whole numbers. The four equations above all have the form $|u^3 - 2 v^3| = 1$, whose one positive solution is $u = v = 1$. The same argument appears in the MathPages note "One In The Chamber." For tetrahedral piles the only cube is again the single ball; a case split like @lem-cases leads to equations $|c u^3 - d v^3| = 1$, which Bennett's theorem settles.

#let CV = json("curves.json")
*Oblong piles of a fixed shape.* @thm-oblong fixes the width and lets the length grow, so the count grows in equal steps and the question is one about remainders. Lucas's question instead fixes the shape and lets the pile grow. Take the base $a$ by $a + c$ for a fixed $c$. By @prop-oblong the pile holds $a(a+1)(2a + k) slash 6$ balls, with $k = 3c + 1$, and with $X = 12 a$ and $Y = 72 m$ it holds $m^2$ balls exactly when
#eqalt("Y squared equals X times X plus 12 times X plus 6 k.", $ Y^2 = X (X + 12)(X + 6k). $)
This is an elliptic curve, and each pile that holds a square gives a point on it with whole-number coordinates. By a theorem of Siegel there are only finitely many such points, and the computer algebra system Sage lists them all: it finds a basis of the curve's rational points with the program mwrank, which proves the basis complete, and then bounds the size of a whole-number point by elliptic logarithms. For $c = 0$ the curve is Lucas's, and Sage finds #CV.by_longer_side.at("0").layers.map(str).join(" and ") layers again, a check of @thm-watson by a method independent of ours. @tab-shapes gives the answer for every $c$ up to 20. We label these results computed exactly, resting on Sage; a search of every $a$ up to #fmt(R.bounds.shapes) agrees with each row.

#figure(
  placement: auto,
  caption: [Oblong piles on an $a$ by $a + c$ base that hold a square number of balls. The rank counts the independent points of infinite order on the curve. "Layers" lists every $a >= 2$; a single row ($a = 1$) of $c + 1$ balls is a square when $c + 1$ is. Computed exactly with Sage (passagemath #CV.sage; `curves.py`, which writes `curves.json`).],
  {
    let rows = range(0, 21).map(c => {
      let r = CV.by_longer_side.at(str(c))
      let a = r.layers.filter(x => x >= 2)
      ([#c], [#r.rank], if a.len() == 0 [--] else [#a.map(fmt).join(", ")])
    })
    let half = 11
    let first = rows.slice(0, half)
    let second = rows.slice(half) + (([], [], []),)
    table(
      columns: 7, align: (right, right, left, center, right, right, left),
      column-gutter: (0pt, 0pt, 0pt, 1.2em, 0pt, 0pt),
      table.hline(stroke: 0.6pt),
      table.header([$c$], [Rank], [Layers], [], [$c$], [Rank], [Layers]),
      table.hline(stroke: 0.4pt),
      ..range(half).map(i => first.at(i) + ([],) + second.at(i)).flatten(),
      table.hline(stroke: 0.6pt),
    )
  },
) <tab-shapes>

Another shape is the base $a$ by $2a - 1$, the oblong pile whose top ridge is as long as the base is wide. It holds $a(a+1)(5a - 2) slash 6$ balls, and with $X = 30 a$ and $Y = 180 m$ the curve is $Y^2 = X(X + 30)(X - 12)$. An answer to question 99084 on Puzzling Stack Exchange found its whole-number points with Sage: the squares come only at #CV.ridge.layers.map(str).join(", ", last: " and ") layers, with $1 = 1^2$, $196 = 14^2$ and $"99,225" = 315^2$. Our own run of Sage gives the same list, and so does a search up to #fmt(R.bounds.shapes) layers. Whether there is a rule for every $c$ is among the open questions (@sec-open).

*Piles with the top cut off.* A square pile without its top few layers is a run of consecutive squares, such as $9 + 16 = 25$. Which numbers of layers allow a square? @tab-truncated answers this for up to 30 layers. Our list agrees with the sequence A001032 in the OEIS. Beeckmans (1994) gives conditions on the number of layers and a method that finds all such piles for a given number.

#figure(
  placement: auto,
  caption: [Piles with the top cut off, by number of layers $L$: the side $t >= 1$ of the smallest top layer for which the $L$ layers of sides $t$ to $t + L - 1$ hold a square number of balls. For the other $L$ up to 30 no top works. Every one except 25 is ruled out by its remainders on division by some number (listed in `results.json`); 25 is @prop-25.],
  {
    let found = R.truncated.filter(r => r.at(1) == "found")
    table(
      columns: 4, align: (right, right, left, left),
      table.hline(stroke: 0.6pt),
      table.header([Layers $L$], [Top $t$], [Layer sides], [Balls]),
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
  No run of 25 consecutive squares starting from $t^2$ with $t >= 1$ adds up to a square. With $t = 0$, the run #ialt("0 squared plus 1 squared plus, and so on, plus 24 squared, equals 70 squared", $0^2 + 1^2 + dots.c + 24^2 = 70^2$) does.
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
  alt: "A three by three layer of balls seen from above. The middle ball is orange, the four balls beside it are blue, and four blue rings over the hollows around it mark the balls resting on it from the layer above.",
  caption: [A ball (orange, in the middle) of a layer of a square pile, seen from above. It touches the four blue balls in its layer and the four balls drawn as blue rings, which rest on it from the layer above. Four more under it touch it from below, in the hollows on the other side.],
) <fig-neighbors>

#tryit[How many pairs of balls touch in a square pile of three layers? Count the pairs inside each layer and the pairs between layers.]

#proposition(title: [touching pairs])[
  A square pile of $n$ layers has $2 n^2 (n - 1)$ touching pairs, a tetrahedral pile of $n$ layers has $(n - 1) n (n + 1)$, and an oblong pile on an $a$ by $b$ base has $2 a^2 (a - 1) + (b - a) a (3a - 2)$.
] <prop-touch>

The same numbers appear in the OEIS under other names. The count for a square pile, $2n^2(n - 1)$, is A035006, the number of moves a rook can make on an $n$ by $n$ chessboard. The count for a tetrahedral pile, $(n - 1) n (n + 1)$, is A007531 shifted by one, and a comment there from 2013 counts the contacts between balls in a tetrahedral pile.

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
  alt: "A square pile of five layers with the balls on one sloping side colored blue; on the slope they sit in a triangle pattern.",
  caption: [A square pile of five layers with the balls of one sloping side in blue. On the slope they sit in rows of triangles, as on the sides of a tetrahedral pile.],
) <fig-slope>

@fig-slope shows why the two piles are related. A sloping side of a square pile is packed in triangles, like the floor of a tetrahedral pile. Tip a square pile over onto one of its sloping sides and it rests on a triangle pattern. Both piles are pieces of one endless arrangement, the _face-centered cubic packing_.

= Relatives <sec-relatives>

*Kepler's conjecture.* In 1611 Kepler described the cannonball arrangement in a short book on snowflakes and called it "the tightest possible." The balls fill $pi slash sqrt(18)$ of the space, about 74%. Whether any arrangement of equal balls fills more stayed open for almost four hundred years. Hales announced a proof in 1998 and published it in 2005; the proof leans on long computer calculations, and a team led by Hales later checked every step with proof-checking programs (Hales et al., 2017). Hales's article "Cannonballs and Honeycombs" (2000) tells the story for a general reader. His account of the history describes a chart made in 1591 for counting stored cannonballs in piles "with triangular, square, or oblong base," the three shapes of this note.

*How many balls can touch one.* Twelve balls can touch a thirteenth of the same size, as in a pile. Whether a thirteenth could be squeezed in was argued about as early as 1694. Schütte and van der Waerden (1953) proved that it cannot.

*The Leech lattice.* The number 4,900 turns up in a much larger packing. In 24 dimensions the densest arrangement of balls is the Leech lattice, as Cohn, Kumar, Miller, Radchenko and Viazovska (2017) proved. Conway and Sloane (1982) showed that it can be built from the 26-dimensional vector $(0, 1, 2, dots, 24 | 70)$. In their geometry the last coordinate counts with a minus sign, so the vector's squared length is #ialt("0 squared plus 1 squared plus, and so on, plus 24 squared, minus 70 squared, equals 0", $0^2 + 1^2 + dots.c + 24^2 - 70^2 = 0$), which is Lucas's pile.

*Runs of squares.* @tab-truncated is the start of a larger story: Beeckmans (1994) showed that when a number of layers $L$ that is not a square allows a square run, infinitely many runs of $L$ squares add up to a square, and gave necessary conditions on $L$.

= Open questions <sec-open>

#sh("open")

= How this was checked <sec-check>

@tab-results lists each result and how it is known: proved here by hand, computed exactly, or cited. The facts quoted in @sec-relatives are cited, and the questions of @sec-open are open. The checks behind @tab-results are listed after it.

#[
// the table is longer than a page on a phone, and nearly a page in print, so it may break
// across pages in both editions (its header repeats)
#show figure: set block(breakable: true)
#figure(
  placement: none,
  kind: table,
  caption: [The results and how each is known. Exact unless marked.],
  text(size: if phone { 9pt } else { 10pt }, sh("results", columns: (1fr, 1.15fr, 1.15fr))),
) <tab-results>
]

#sh("checks")

*Code.* The `cannonballs` folder of the code that accompanies this note has `piles.py` (counts, piles and touching pairs), `lucas.py` (the searches), `report.py`, which writes the results file in about half a minute, and `figures.py`. They need only the Python standard library. `curves.py` lists the whole-number points of the elliptic curves for oblong piles with Sage, and writes `curves.json`.

*Typesetting.* Typst, with the packages cetz, lilaq and codly.

// ---------- appendices ----------
#show figure.where(kind: table): set text(size: 10pt)
#counter(heading).update(0)
#set heading(hanging-indent: 0pt)
#set heading(numbering: (..n) => "Appendix " + numbering("A.", ..n.pos()))

= Answers to the puzzles <app-answers>

*Triangle of side 10 (@sec-flat).* Half of $10 times 11$, which is 55.

*Piles (@sec-piles).* A square pile of five layers holds $1 + 4 + 9 + 16 + 25 = 55$ balls, and a tetrahedral pile of five layers $1 + 3 + 6 + 10 + 15 = 35$.

*An oblong pile of 100 (@sec-oblong).* Four rows of 11 at the bottom, $44 + 30 + 18 + 8 = 100$; or five rows of 8, $40 + 28 + 18 + 10 + 4 = 100$.

*Touching pairs (@sec-touch).* In a square pile of three layers the layers have 12, 4 and 0 pairs inside them; four balls rest on the bottom layer and one on the middle one, each touching four, which adds $16 + 4$. In all $12 + 4 + 16 + 4 = 36$.

#[
#set par(justify: false)
*More puzzles (@sec-puzzles).*
+ Half of $8 times 9$, which is 36.
+ In the square pile, 4 pairs in the ring and 4 with the top ball: 8. In the tetrahedral pile all four balls touch each other: 6 pairs.
+ Never. The counts $3b - 1$ leave remainder 2 on division by 3, and squares leave only 0 or 1.
+ #ialt("20 squared plus 21 squared equals 400 plus 441, which is 841, or 29 squared.", $20^2 + 21^2 = 400 + 441 = 841 = 29^2$). Another is #ialt("119 squared plus 120 squared equals 169 squared.", $119^2 + 120^2 = 169^2$).
+ Four times as many: $"Te"(48) = 4 P(24) = "19,600"$, which is $140^2$, a square.
]

= Words and symbols <app-words>

#[
// on a phone the glossary is set at the page's own width, and may break across pages
#show figure: set block(breakable: phone)
#figure(
  caption: [Words and symbols used in this note.],
  [#table(inset: (x: 6pt, y: 3pt),
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    table.header([Word or symbol], [Meaning]),
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
    [elliptic curve], [A curve of the form $y^2 = $ a cubic in $x$; it has only finitely many whole-number points.],
    [OEIS], [The On-Line Encyclopedia of Integer Sequences, a public catalog of number sequences.],
    [lemma, theorem], [A stepping stone; a main result. A proposition is a single fact.],
    [$qed$], [The end of a proof.],
    table.hline(stroke: 0.6pt),
  ) <phone-breaks>],
)
]

#block(breakable: phone)[
#heading(numbering: none, level: 1)[Sources]
#set text(size: 9pt)
- É. Lucas, #link("https://www.numdam.org/item?id=NAM_1875_2_14__336_0")[Question 1180], _Nouvelles Annales de Mathématiques_ (2) 14 (1875), 336. Moret-Blanc, solution, ibid. 15 (1876), 46--48; É. Lucas, solution, ibid. 16 (1877), 429--432.
- G. N. Watson, The problem of the square pyramid, _Messenger of Mathematics_ 48 (1918), 1--22.
- W. Ljunggren, New solution of a problem proposed by E. Lucas, _Norsk Matematisk Tidsskrift_ 34 (1952), 65--72.
- D. G. Ma, An elementary proof of the solutions to the Diophantine equation $6y^2 = x(x+1)(2x+1)$, _Sichuan Daxue Xuebao_ 4 (1985), 107--116.
- W. S. Anglin, #link("https://doi.org/10.2307/2323911")[The square pyramid puzzle], _American Mathematical Monthly_ 97 (1990), 120--124.
- M. A. Bennett, #link("https://doi.org/10.4064/aa105-4-3")[Lucas' square pyramid problem revisited], _Acta Arithmetica_ 105 (2002), 341--347.
- M. A. Bennett, Rational approximation to algebraic numbers of small height: the Diophantine equation $|a x^n - b y^n| = 1$, _J. reine angew. Math._ 535 (2001), 1--49.
- L. Euler, #link("https://www.deutschestextarchiv.de/book/view/euler_algebra02_1770?p=522")[_Vollständige Anleitung zur Algebra_], vol. 2 (St. Petersburg, 1770), section 2, §247: $x^3 + y^3 = 2z^3$ only when $x = y$.
- MathPages, #link("https://www.mathpages.com/home/kmath445.htm")[One In The Chamber], on square piles that hold a cube.
- Puzzling Stack Exchange, #link("https://puzzling.stackexchange.com/questions/99084")[question 99084, "Cannonballs packing"] (2020), answers on oblong piles on an $a$ by $2a - 1$ base.
- A.-J.-J. Meyl, #link("https://www.numdam.org/item/NAM_1878_2_17__464_1.pdf")[Solution de la question 1194], _Nouvelles Annales de Mathématiques_ (2) 17 (1878), 464--467.
- E. T. Avanesov, #link("https://doi.org/10.4064/aa-12-4-409-420")[Solution of a problem on figurate numbers] (in Russian), _Acta Arithmetica_ 12 (1966/67), 409--420.
- R. Finkelstein and H. London, On triangular numbers which are sums of consecutive squares, _Journal of Number Theory_ 4 (1972), 455--462.
- F. Beukers and J. Top, On oranges and integral points on certain plane cubic curves, _Nieuw Archief voor Wiskunde_ (4) 6 (1988), 203--210.
- L. Beeckmans, #link("https://doi.org/10.1080/00029890.1994.11996969")[Squares expressible as sum of consecutive squares], _American Mathematical Monthly_ 101 (1994), 437--442.
- The OEIS, sequences #link("https://oeis.org/A001032")[A001032] (runs of squares), #link("https://oeis.org/A007531")[A007531] and #link("https://oeis.org/A035006")[A035006] (touching pairs), #link("https://oeis.org/A053611")[A053611] and #link("https://oeis.org/A224421")[A224421] (coincidences).
- J. Kepler, _Strena seu de nive sexangula_ (1611); English translation _The Six-Cornered Snowflake_ (Oxford, 1966).
- T. C. Hales, #link("https://www.ams.org/notices/200004/fea-hales.pdf")[Cannonballs and honeycombs], _Notices AMS_ 47 (2000), 440--449; A proof of the Kepler conjecture, _Annals of Mathematics_ 162 (2005), 1065--1185; its #link("https://annals.math.princeton.edu/supplemental/2005/162-3/p01/proof_kepler.pdf")[unabridged version] holds the history.
- T. Hales et al., #link("https://doi.org/10.1017/fmp.2017.1")[A formal proof of the Kepler conjecture], _Forum of Mathematics, Pi_ 5 (2017), e2.
- K. Schütte and B. L. van der Waerden, Das Problem der dreizehn Kugeln, _Mathematische Annalen_ 125 (1953), 325--334.
- H. Cohn, A. Kumar, S. D. Miller, D. Radchenko and M. Viazovska, #link("https://doi.org/10.4007/annals.2017.185.3.8")[The sphere packing problem in dimension 24], _Annals of Mathematics_ 185 (2017), 1017--1033.
- J. H. Conway and N. J. A. Sloane, #link("https://neilsloane.com/doc/Me91.pdf")[Lorentzian forms for the Leech lattice], _Bulletin AMS_ 6 (1982), 215--217.
]
