#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "@preview/lilaq:0.6.0" as lq
#import "@preview/codly:1.3.0": codly, codly-init

// ---------- editions ----------
// print (default): US letter. phone: a narrow page that a phone shows at reading size,
// compiled with sys input edition=phone. html: compiled with format "html"; drawings
// become inline SVG through html.frame, everything else becomes plain HTML.
#let edition = sys.inputs.at("edition", default: "print")
#let phone = edition == "phone"
#let vis(x) = context if target() == "html" { box(html.frame(x)) } else { x }    // inline drawing
#let visb(x) = context if target() == "html" { html.frame(x) } else { x }         // block drawing
// shrink something too wide for the text column (phone pages only)
#let fitw(x) = layout(size => {
  let w = measure(x).width
  if w > size.width { scale(size.width / w * 100%, reflow: true, x) } else { x }
})
// phone pages: lay something out as if the column were w wide, then scale it to fit
#let wide(x, w: 4.8in) = if not phone { x } else { layout(size => {
  let m = measure(x, width: w)
  if m.width > size.width { scale(size.width / m.width * 100%, reflow: true, block(width: m.width, x)) } else { x }
}) }
// a floating block in print, an ordinary one in HTML
#let floated(body) = context if target() == "html" { body } else { place(auto, float: true, clearance: 1.2em, body) }
// two things side by side in print, one above the other on a phone or in HTML
#let beside(a, b, right: 64mm) = context if target() == "html" { a; b } else if phone { stack(spacing: 1em, align(left, a), align(center, b)) } else {
  grid(columns: (1fr, right), column-gutter: 10pt, align: (left + horizon, center + horizon), a, b)
}
#let diagram(..args) = visb(fitw(fletcher.diagram(..args)))
#let lqd(..args) = visb(fitw(lq.diagram(..args)))


// HTML edition: one style sheet, inlined, so the page is a single self-contained file
#let html-css = " :root { color-scheme: light; } body { max-width: 42rem; margin: 0 auto; padding: 1.2rem 1.1rem 4rem; font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, Georgia, serif; font-size: 1.12rem; line-height: 1.55; color: #161616; background: #fff; } h1 { font-size: 2rem; line-height: 1.2; text-align: center; margin: 1.2rem 0 0.4rem; } p.subtitle { text-align: center; font-style: italic; margin: 0; } p.date { text-align: center; color: #666; font-size: 0.9rem; margin: 0.3rem 0 1.6rem; } h2 { font-size: 1.45rem; margin: 2.2rem 0 0.6rem; line-height: 1.25; } h3 { font-size: 1.15rem; margin: 1.6rem 0 0.5rem; } p { margin: 0.7rem 0; } a { color: #0072B2; text-decoration: none; } a:hover { text-decoration: underline; } svg { max-width: 100%; height: auto !important; vertical-align: middle; } svg[style*=\"display: block\"] { margin: 0.8rem auto; } figure { margin: 1.6rem 0; text-align: center; overflow-x: auto; } figure.thm, figure:has(div.thm) { text-align: left; } figcaption { font-size: 0.9rem; color: #444; margin-top: 0.5rem; text-align: center; } table { border-collapse: collapse; margin: 0.4rem auto; font-size: 0.9rem; line-height: 1.35; border-top: 1.5px solid #222; border-bottom: 1.5px solid #222; } td, th { padding: 0.25rem 0.55rem; text-align: left; vertical-align: top; } tr:first-child td, tr:first-child th { font-weight: 600; border-bottom: 1px solid #222; } aside.box { border-left: 4px solid; padding: 0.6rem 0.95rem; margin: 1.3rem 0; border-radius: 0 5px 5px 0; font-size: 1rem; line-height: 1.45; } aside.idea { background: #fdf6e7; border-color: #E69F00; } aside.tryit { background: #f0f1f3; border-color: #5b6770; } aside.box p { margin: 0.35rem 0; } .boxlabel { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; margin-right: 0.45em; } aside.idea .boxlabel { color: #8a5a00; } aside.tryit .boxlabel { color: #3d464d; } pre { text-align: left; overflow-x: auto; font-size: 0.78rem; line-height: 1.45; background: #fafafa; border: 1px solid #ddd; border-radius: 4px; padding: 0.7rem 0.8rem; } code { font-size: 0.9em; overflow-wrap: anywhere; } pre code { overflow-wrap: normal; } math { font-size: 1.06em; font-family: \"STIX Two Math\", \"Latin Modern Math\", \"Cambria Math\", math; } math[display=\"block\"] { margin: 0.9rem 0; overflow-x: auto; overflow-y: hidden; } ul, ol { padding-left: 1.4rem; } li { margin: 0.25rem 0; } .thm { margin: 1.2rem 0; } blockquote.epigraph { margin: 0.2rem 0 1.4rem auto; max-width: 30rem; text-align: right; font-style: italic; font-size: 0.95rem; } p.mnote { font-size: 0.88rem; font-style: italic; color: #555; margin-left: 1.2rem; } blockquote.epigraph footer { font-style: normal; font-size: 0.85rem; color: #555; } "
#context if target() == "html" { html.elem("style", html-css) }
// HTML has no align: keep the content, drop the alignment (CSS centers figures and drawings)
#show align: it => context if target() == "html" { it.body } else { it }

// ---------- look ----------
#set document(title: "Can You Always Make Ten?")  // no author: the owner's name stays out
#set page(..if phone { (width: 4.4in, height: 7.6in, margin: (x: 0.32in, top: 0.42in, bottom: 0.55in)) }
  else { (paper: "us-letter", margin: (x: 1.55in, y: 1.1in)) }, numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)  // about 72 characters per line
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)  // Regular, to match the text weight (the default is Book)
#set par(justify: true, leading: 0.6em, spacing: 1em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set block(above: 1.6em, below: 0.8em)
#show heading.where(level: 2): set block(above: 1.2em, below: 0.6em)
// appendix headings read "Appendix A. ..."; references to them read "Appendix A"
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading and type(el.numbering) == function {
    box(link(el.location(), [Appendix #numbering("A", ..counter(heading).at(el.location()))]))  // box: never break "Appendix B"
  } else { it }
}
#show link: set text(fill: rgb("#0072B2"))
#show raw.where(block: true): set text(size: 8.5pt)
#show figure.caption: set text(size: 10pt)
// one-line captions centred, longer ones set as a justified paragraph
#show figure.caption: it => context if target() == "html" { it } else { layout(size => {
  if measure(it).width <= size.width { align(center, it) } else { set par(justify: not phone); align(left, it) }
}) }
#show math.equation.where(block: true): it => if phone { fitw(it) } else { it }
// tables with a 1fr column fill the width they get, so lay those out at 4.8in; tables
// of auto columns keep their natural width (up to 7in) and are scaled down whole
#show table: it => if phone {
  let fr = type(it.columns) == array and it.columns.any(c => type(c) == fraction)
  wide(it, w: if fr { 4.8in } else { 7in })
} else { it }
#show figure.caption: set par(justify: not phone)
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

#let red = rgb("#D55E00")   // Okabe-Ito palette: colour-blind safe
#let blue = rgb("#0072B2")
#let green = rgb("#009E73")
#let amber = rgb("#a15c00")
#let orange = rgb("#E69F00")
#let purple = rgb("#AA4499")   // case-checked by computer
#let slate = rgb("#5b6770")    // neutral: puzzle boxes

#let tag(label, col) = vis(box(
  inset: (x: 3pt, y: 1.5pt), radius: 2pt, baseline: 1.5pt,
  fill: col.lighten(88%), stroke: 0.4pt + col,
  text(size: 7.5pt, fill: col, weight: "medium", label),
))
#let proved = tag("proved", green)
#let computed = tag("computed", amber)
#let certified = tag("certified", blue)
#let usesbig = tag([uses #ref(<thm-big>)], blue)

// theorem-like environments share one counter and can be referenced with @label
// statements may break across pages; wrap a short one in block(breakable: false) if a display strands
#show figure.where(kind: "thm"): set block(breakable: true)
#show figure.where(kind: "thm"): it => block(
  width: 100%, breakable: true, above: 1.2em, below: 1.1em,
  align(left)[*#it.supplement #it.counter.display(it.numbering)*#it.body],
)
// statements upright: long italic passages tire young readers
#let env(kind, title: none, italic: false, body) = figure(
  kind: "thm", supplement: kind, numbering: "1", outlined: false,
  [#if title != none [ (#title)]*.* #if italic { text(style: "italic", body) } else { body }],
)
#let definition = env.with([Definition], italic: false)
#let lemma = env.with([Lemma])
#let theorem = env.with([Theorem])
#let proposition = env.with([Proposition])
#let corollary = env.with([Corollary])
#let proof(title: [Proof], body) = block(above: 0.7em, below: 1.2em)[_#title._ #body#h(0.6em)#h(1fr)$qed$]

// playing cards
#let card(v, middle: false, loud: false) = box(
  width: 1.55em, height: 2.15em, radius: 2.5pt, baseline: 0.65em,
  stroke: 0.7pt + (if middle { blue } else { luma(70) }),
  fill: if middle { blue.lighten(90%) } else { white },
  align(center + horizon, text(weight: "bold", fill: if loud { red } else { black }, str(v))),
)
#let cards(..vs, middle: false) = vs.pos().map(v => card(v, middle: middle)).join(h(2.5pt))
// a deal: hand and center side by side, each named underneath (no bare divider glyph)
#let deal(hand, ctr) = visb(box(grid(
  columns: 2, column-gutter: 16pt, row-gutter: 4pt, align: center,
  cards(..hand), cards(..ctr, middle: true),
  text(7.5pt, fill: luma(100), tracking: 0.06em)[HAND], text(7.5pt, fill: blue.darken(10%), tracking: 0.06em)[CENTER],
)))
// labeled small puzzles, two to a row
#let minis(narrow: false, ..sets) = align(center, visb(grid(columns: if narrow and phone { 2 } else { 4 },
  column-gutter: if narrow and phone { 5pt } else { (5pt, 2.2em, 5pt) }, row-gutter: 0.6em, align: if narrow and phone { left + horizon } else { horizon },
  ..sets.pos().enumerate().map(((i, s)) => (text(size: 10pt)[(#"abcdefgh".at(i))], box(cards(..s)))).flatten())))

// diagram helpers, in the style of the fletcher gallery: pastel fill, same-hue stroke
#let blob(pos, label, tint: luma(50%), ..args) = node(pos, align(center, { set par(justify: false); label }),
  fill: tint.lighten(88%), stroke: 0.8pt + tint, corner-radius: 4pt, inset: 6pt, ..args)
#let op(pos, sym, name) = node(pos, sym, shape: circle, fill: luma(96%), stroke: 0.6pt + luma(40%), radius: 3.2mm, name: name)
#let cardnode(pos, v, name, loud: false) = node(pos, text(weight: "bold", fill: if loud { red } else { black }, str(v)),
  shape: rect, corner-radius: 2.5pt, width: 6mm, height: 8mm, fill: white,
  stroke: 0.7pt + (if loud { red } else { luma(40%) }), name: name)
#let res(c, body) = text(8.5pt, fill: c, body)


#let ms(..xs) = "{" + xs.pos().map(str).join(", ") + "}"

#let hour(pos, v, c, name) = node(pos, text(weight: "bold", str(v)), shape: circle, radius: 4.2mm,
  fill: c.lighten(85%), stroke: 0.9pt + c, name: name)
#let note(pos, body, c: luma(30%)) = node(pos, text(9pt, fill: c, body), stroke: none)

// a margin note, after the graffiti in Concrete Mathematics: a short aside beside the text.
// Print: set in the right margin. Phone and HTML have no margin, so it becomes a small indented line.
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
  // inline: anchored to the first line of the paragraph it opens, so it follows that paragraph across pages
  // and its first baseline matches the text's. Otherwise it sits just before the block that follows.
  if inline { box(place(top + left, dx: 5.56in, dy: -5.5pt, note)) }
  else { place(right, dx: 1.36in, dy: 2.7pt, note) }
}

// a short quotation under a section heading, set to the right
#let epigraph(quote, source) = context if target() == "html" {
  html.elem("blockquote", attrs: (class: "epigraph"), { quote; html.elem("footer", source) })
} else {
  block(sticky: true, width: 100%, above: 0.2em, below: 1.3em, align(right, block(width: if phone { 88% } else { 68% }, {
    set par(justify: false, leading: 0.5em)
    set text(size: 10pt)
    align(right, emph(quote))
    v(-0.35em)
    align(right, text(size: 9pt, fill: luma(35%), source))
  })))
}

// ---------- boxes for the young reader ----------
// One quiet style, two labels: "Idea" explains a tool where it is first used; "Try it" invites the reader in.
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
// a puzzle in the puzzles section: plain, no box
#let puzzle(body) = context if target() == "html" { body; parbreak() } else {
  block(breakable: false, width: 100%, above: 1.2em, below: 1.2em, { set par(justify: false); body }) }
// six boxes for the eleven card values: seven cards cannot all sit alone
#let pigeon(hand) = visb(fitw(grid(columns: 6, column-gutter: 5pt, row-gutter: 3pt, align: center,
  ..range(6).map(k => box(width: 3.9em, height: 2.9em, radius: 3pt, fill: luma(97.5%),
    stroke: (paint: luma(55%), dash: "dashed", thickness: 0.6pt),
    align(center + horizon, cards(..hand.filter(v => calc.quo(v, 2) == k))))),
  ..range(6).map(k => text(7.5pt, fill: luma(100), if k == 5 [10] else [#(2 * k) or #(2 * k + 1)])),
)))
// which card values each family allows: solid = any number, dashed = at most one in all, faded = none
#let vcard(v, kind) = box(width: 1.45em, height: 1.9em, radius: 2.5pt, baseline: 0.6em,
  stroke: if kind == "in" { 0.9pt + black } else if kind == "one" { (paint: luma(40%), dash: "dashed", thickness: 0.7pt) } else { 0.5pt + luma(86%) },
  fill: if kind == "in" { luma(93%) } else { white },
  align(center + horizon, text(weight: "bold", fill: if kind == "out" { luma(80%) } else { black }, str(v))))
#let familypic = visb(fitw(grid(columns: 2, column-gutter: 10pt, row-gutter: 6pt, align: (right + horizon, left + horizon),
  text(9pt)[3-family], range(11).map(v => vcard(v, if calc.rem(v, 3) == 0 { "in" } else { "out" })).join(h(2pt)),
  text(9pt)[4-family], range(11).map(v => vcard(v, if calc.rem(v, 4) == 0 { "in" } else if calc.odd(v) { "one" } else { "out" })).join(h(2pt)),
)))

// a group of cards erased together, with the arithmetic underneath
#let grp(hand, ctr: none, expr) = vis(box(
  stroke: 0.6pt + luma(65%), radius: 4pt, inset: (x: 5pt, top: 5pt, bottom: 4pt),
  stack(spacing: 4pt,
    [#cards(..hand)#if ctr != none [#h(3pt)#card(ctr, middle: true)]],
    align(center, text(9.5pt, expr))),
))

// a small wastebasket, drawn rather than borrowed
#let bin(size: 11pt) = vis(box(baseline: 15%, width: size * 0.95, height: size, {
  place(polygon(fill: luma(93%), stroke: 0.6pt + luma(35%),
    (0pt, 0.12 * size), (0.95 * size, 0.12 * size), (0.8 * size, size), (0.15 * size, size)))
  place(line(start: (-0.06 * size, 0.12 * size), end: (1.01 * size, 0.12 * size), stroke: 0.9pt + luma(35%)))
  for x in (0.33, 0.475, 0.62) {
    place(line(start: (x * size, 0.3 * size), end: (x * size, 0.86 * size), stroke: 0.4pt + luma(55%)))
  }
}))

// ---------- title ----------
#context if target() == "html" {
  html.elem("h1")[Can You Always Make Ten?]
  html.elem("p", attrs: (class: "subtitle"))[A card-game mystery: silent cards, a wastebasket, and one deal in 204,722]
  html.elem("p", attrs: (class: "date"))[November 2026]
} else [
  #align(center)[
    #set par(justify: false)
    #text(size: 21pt, weight: "bold")[Can You Always Make Ten?]
    #v(2pt)
    #text(size: 11.5pt, style: "italic")[A card-game mystery: silent cards, a wastebasket, and one deal in 204,722]
    #v(4pt)
    #text(size: 9.5pt, fill: luma(90))[November 2026]
  ]
]

#v(4pt)
#epigraph[“Can you do Addition?” the White Queen asked. “What's one and one and one and one and one and one and one and one and one and one?”][Lewis Carroll, _Through the Looking-Glass_]

Can every deal of the card game Erase be cleared once players may add, subtract and multiply as many of their cards as they like? The short answer is _almost_. One deal in 204,722 is stuck, and a stuck deal is stuck completely: not one card can be erased. Every other deal can be cleared, and always with one calculation that uses the whole hand and at most one center card. These numbers are for the deck the rules ask for, cards 0 to 10. With cards 1 to 10 a stuck deal is far rarer, about 1 in 14.7 million (@prop-oneten).

Three ideas explain this. *Silent cards:* looking only at remainders turns most cards into zeros. *The wastebasket:* a real 0 lets you throw away any cards you don't need. *Big hands:* seven cards with no 0 among them almost always make 10 using every one of them. The first two ideas are proved by hand. The third is a finite fact that a computer checks case by case, writing down a checkable answer, called a certificate, for each case, and the proofs that rely on it say so. A few numbers come from exhaustive computer counts or from simulation alone; they are marked where they appear.

*How to read this.* The note has three parts. Sections #ref(<sec-game>, supplement: none) to #ref(<sec-puzzles>, supplement: none) tell the whole story, with advice for the table and more puzzles, and need only arithmetic. Sections #ref(<sec-plan>, supplement: none) to #ref(<sec-other>, supplement: none) prove every claim. Sections #ref(<sec-relatives>, supplement: none) and #ref(<sec-check>, supplement: none) set the game beside its relatives and say how everything was checked. Boxes marked #text(size: 8.5pt, weight: "bold", fill: orange.darken(35%), tracking: 0.04em)[IDEA] explain a tool the first time it is used. Boxes marked #text(size: 8.5pt, weight: "bold", fill: slate.darken(35%), tracking: 0.04em)[TRY IT] are puzzles to try before reading on, with answers in @app-answers. In every puzzle you may add, subtract and multiply, but not divide. Each card is used at most once, and brackets are allowed. It is fine to skip a proof, or to stop anywhere, and come back later.

= The game <sec-game>

Erase is Game 9 in _Math Fact Fluency_ by Jennifer Bay-Williams and Gina Kling, a classroom game for practicing the pairs that make 10.

- *Deck.* The sheet asks for a "0-10 deck of cards": four of each value from 0 to 10, 44 cards. Some classrooms use cards 1 to 10 instead; @prop-oneten covers that deck.
- *Players.* Two to four.
- *Deal.* Each player gets seven cards. Four more go face up in the middle as _center cards_, shared by everyone.
- *Erase.* Two of your own cards that make 10 are both erased. One of your cards and a center card that make 10 erase only your card. The center card stays, and anyone may use it again.
- *Score.* The number of cards you have left. All cards are reshuffled and dealt again each round, and after five rounds the lowest total wins.

Here is the instruction sheet's example deal, with the center cards in blue:

#align(center, deal((4, 4, 6, 9, 5, 8, 5), (7, 5, 1, 8)))

#mnote(inline: true)[The 8 wants a 2. The 4 wants a 6. Nobody has one!]Erase $4+6$ and $5+5$, then the 9 with the center 1. The 4 and the 8 are left, so the score is 2.

The sheet lists a variation: "Players may use more than 2 cards to make a 10." It does not say which operations are allowed. The sheet files Erase under addition and subtraction facts, and the authors' teacher page calls it a strategy game for addition, so adding (and perhaps subtracting) is the likely intent. This note mainly studies a more generous reading that allows $+$, $-$ and $times$, and @tab-readings compares the others. A move is then: choose some of your cards, with or without one center card, that make 10 when combined with $+$, $-$ and $times$. Your cards in it are erased, and the center card stays. Such a set of cards is called a _group_. Under that reading the same hand clears completely, in four groups that reuse the center 1 and 5:

#align(center, block(breakable: false)[
  #grp((6, 8, 4), $6 + 8 - 4$) #h(6pt)
  #grp((4, 5), ctr: 1, $4 + 5 + 1$) #h(6pt)
  #grp((9,), ctr: 1, $9 + 1$) #h(6pt)
  #grp((5,), ctr: 5, $5 + 5$)
])

It can even be cleared in one breath, with a single calculation that uses all seven cards (@fig-tree).

The scoring raises a question. Erase is scored by the cards left over, so if _every_ deal could be cleared, every score would be zero and the scoring would decide nothing. Is there a deal where perfect play still leaves cards?

= The answer at a glance

#epigraph[Never tell me the odds!][Han Solo, in _The Empire Strikes Back_]

The answer depends on how generously "more than 2 cards" is read. @tab-readings gives the numbers.

#figure(
  placement: if phone { auto } else { none },
  caption: [How often a deal cannot be cleared, even with perfect play, for the sheet's deck and for a deck without 0s. Entries with $plus.minus$ are simulated (30,000 deals, 95% interval). All others are exact: see @prop-pairs, @prop-wild, @prop-oneten and @thm-any. With $+$ and $-$, parity alone blocks exactly 3.914% and 2.650% (@prop-parity). The bold row is the subject of this note.],
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    table.hline(stroke: 0.6pt),
    [Reading of the rules], [Cards 0 to 10], [Cards 1 to 10],
    table.hline(stroke: 0.4pt),
    [Pairs only, as printed], [96.02%], [96.90%],
    [Addition only], [70.8% $plus.minus$ 0.5%], [80.7% $plus.minus$ 0.4%],
    [$+$ and $-$], [3.8% $plus.minus$ 0.2%], [2.6% $plus.minus$ 0.2%],
    [*$bold(+)$, $bold(-)$, $bold(times)$*], [*1 in 204,722*], [*1 in 14.7 million*],
    [$+$, $-$, $times$, with four wild kings], [1 in 603,146], [1 in 48.8 million],
    [$+$, $-$, $times$, any number of center cards per group], [1 in 1,345,498], [1 in 193 million],
    [$+$, $-$, $times$, $div$], [1 in 91 million], [none],
    [$+$, $-$, $times$, $div$, any number of center cards per group], [none], [none],
    table.hline(stroke: 0.6pt),
  ),
) <tab-readings>

#idea[exact and simulated numbers][An _exact_ number comes from a proof, or from counting every case, such as all 2,530,881,913,560 possible deals. A _simulated_ number comes from dealing 30,000 random deals and counting what happens, the way you might estimate how often a coin lands heads by tossing it many times. The $plus.minus$ after a simulated number says how far luck could plausibly have moved it.]

#mnote(inline: true)[Nobody can take your cards. It is you against the deal.]Players never compete for cards. Center cards are reused, and nobody touches anyone else's hand. So each player is effectively playing alone against a random deal of seven cards plus four center cards, and the table is about one such deal.

So the worry from @sec-game, that every deal might clear, is exactly right at the bottom of the table and wrong at the top. With addition only, the most likely intent, Erase is very much a game. With full arithmetic it becomes a different game, in which the answer is almost always "zero" and the challenge is finding it. The analysis also gives a practical warning. Grabbing the easy pairs first, as the basic game trains you to, strands a card in about a quarter of hands (@sec-play).

How rare is one deal in 204,722? If you dealt yourself a hand every minute, day and night, you would wait about twenty weeks, on average, to see a stuck one.

= A hand you cannot erase <sec-stuck>

#epigraph[“I can't do either!” said Pooh. “Oh, help _and_ bother!”][A. A. Milne, _Winnie-the-Pooh_, stuck in Rabbit's front door]

#block(breakable: false)[
  #align(center, deal((4, 4, 4, 4, 8, 8, 8), (3, 7, 8, 9)))

  #tryit[Make 10 from some of the white cards, with or without one blue card, using $+$, $-$ and $times$ but not $div$. Give it two minutes before reading on.]
]

Most attempts land near 10: $8 + 8 - 4$ gives 12, $4 times 4 - 8$ gives 8, $8 + 3$ gives 11, and $9 + 4 - 4$ gives 9. You can make 9 and you can make 11, but never 10, as long as you only add, subtract and multiply. Dividing changes that: $8 + 8 div 4 = 10$.

This deal has no 0 in it, so it is stuck whether the deck starts at 0 or at 1. The center 3 and 7 make 10 together, but a group may use only one center card.

#idea[what a proof is][Trying calculations can only show that we have not found 10 _yet_. A proof shows that 10 can _never_ come out, however the cards are combined. The proof below does not try calculations one by one. It finds a property that every calculation keeps and that 10 does not have.]

#mnote(inline: true)[Tame numbers: 7, 8, 9, then 11, 12, 13. Ten is missing!]The proof needs one new word. Call a whole number _tame_ if it is odd or a multiple of 4. Negative numbers count too, so $-3$ and $-4$ are tame. Ten is not tame.

#proposition(title: [a dead deal])[
  In this deal, using $+$, $-$ and $times$, no set of your cards, with or without one center card, makes 10. Not a single card can be erased.
] <prop-dead>

#proof[
  A set of cards here holds only 4s and 8s from the hand, plus perhaps one center card, which is an 8 or odd. Any part of the calculation that avoids the odd card is a multiple of 4, because multiples of 4 stay multiples of 4 under $+$, $-$ and $times$. Now follow the odd card up through the calculation. Adding or subtracting a multiple of 4 keeps it odd. Multiplying it by a multiple of 4 gives a multiple of 4, which then stays one. So every value such a set can make is tame, and none is 10.
]

#floated(beside(
  idea[remainders and clocks][Divide a whole number by 4 and keep the remainder: 10 is $2 times 4 + 2$, so it leaves 2; 13 leaves 1; 8 leaves 0. The remainders go round like the hours on a four-hour clock. We write $10 equiv 2 space (mod 4)$ and read it as “10 leaves the same remainder as 2 when divided by 4.” Negative numbers fit too: $-1$ is one hour before 0, the same place as 3. Mod 7 the clock has seven hours, and so on.],
  [#figure(
    diagram(
    spacing: (8mm, 6.5mm), edge-stroke: 0.7pt + luma(65%),
    hour((1, 0), 0, green, <h0>), hour((2, 1), 1, green, <h1>),
    hour((1, 2), 2, red, <h2>), hour((0, 1), 3, green, <h3>),
    edge(<h0>, <h1>, bend: 35deg), edge(<h1>, <h2>, bend: 35deg),
    edge(<h2>, <h3>, bend: 35deg), edge(<h3>, <h0>, bend: 35deg),
    note((1, -0.62), [multiples of 4]),
    note((1, 2.62), [10 lives here], c: red),
  ),
    caption: [A four-hour clock. Multiples of 4 sit at 0, odd numbers at 1 or 3. Every value the cards of @prop-dead can make lands on a green hour; 10 lands on 2.],
  ) <fig-clock>],
))

The proof never asks what a calculation _does_ with the cards, only what kind of number can come out. The mutilated chessboard puzzle is solved the same way. You stop trying tilings and count colors instead.

= Playing it for real <sec-play>

The numbers in @tab-readings are about perfect play on one hand. A few practical points change how it feels at the table.

#mnote(inline: true)[Save a pair for later. The last card needs a friend.]*Grabbing pairs first is a trap.* The basic game trains players to erase every pair that makes 10, and with full arithmetic that habit backfires. Suppose a player first erases all the pairs the printed rules allow, then tries arithmetic on whatever is left. In 30,000 simulated deals that could be cleared, this strands at least one card 23.8% of the time ($plus.minus$ 0.5%).

#block(breakable: false, width: 100%)[
  Here is one such deal.
  #align(center, deal((3, 3, 4, 7, 8, 8, 8), (1, 2, 6, 6)))
]

The pairs $3 + 7$, $8 + 2$ (three times) and $4 + 6$ leave a lone 3, which no center card can rescue. Yet the whole hand clears at once: $3 + 3 + 4 + 7 times 8 times (8 - 8) = 10$. The lesson: keep two equal cards, or a 0, in reserve until the end. Multiplying by $8 - 8$, or by a 0, throws away the cards the last group cannot use (@sec-waste).

*Small groups are nearly always enough.* The theorems clear a hand with one long calculation, often by multiplying spare cards by 0. One long calculation is a way to prove things, not the way to play. In 30,000 simulated deals, 97.5% ($plus.minus$ 0.2%) could be cleared using only groups of at most three cards, counting the center card, and 99.8% with at most four. With cards 1 to 10 the figures are 99.8% and all but 6 of the 30,000. Brackets are hardly needed either: in the same deals, from each deck, every deal could be cleared with each group worked as a running total, starting from one card and adding, subtracting or multiplying by one more card at a time.

*Spotting a stuck deal.* Counting odd cards is a scan for the $+$ and $-$ game. With $times$ allowed, two scans do the same job. First: is every card on the table, yours and the center's, a 0, 3, 6 or 9? Second: are your cards all 0s, 4s and 8s, apart from at most one odd card, with no 2, 6 or 10 in the center, and no odd center card if you hold an odd one? If either answer is yes, nothing can be erased, and you can say so before trying a single sum (@def-families). With cards 1 to 10 these two scans find every stuck deal (@prop-oneten). With cards 0 to 10 they find about 95% of them (@tab-shapes); the rest are hands of several 0s with too little material.

*If your table plays differently.* House rules vary, and the sheet leaves several points open. @tab-house collects what each choice changes.

#figure(
  placement: auto,
  caption: [House rules and what they change, with $+$, $-$ and $times$ unless the row says otherwise. Counts are exact; the simulated rows each used 30,000 deals per deck.],
  table(
    columns: (0.62fr, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [If your table...], [...then],
    table.hline(stroke: 0.4pt),
    [uses cards 1 to 10], [a stuck deal is 1 in 14.7 million, not 1 in 204,722 (@prop-oneten)],
    [allows $div$], [cards 0 to 10: 1 in 91 million; cards 1 to 10: every deal clears. Requiring every division to come out whole changes neither figure],
    [has not met brackets], [no simulated deal needed them: each group can be worked as one running total, a card at a time],
    [keeps to groups of two or three cards], [97.5% of deals still clear; 99.8% with cards 1 to 10 (simulated)],
    [never lets a total drop below 0], [nothing changes with cards 0 to 10 (@sec-model)],
    [lets a 10 be erased by itself], [nothing changes with $+$, $-$, $times$. Under the printed pairs-only rules with cards 1 to 10, 92.10% of deals are stuck instead of 96.90%],
    [lets a group use several center cards], [1 in 1,345,498; 1 in 193 million with cards 1 to 10 (@thm-any)],
    [deals fewer than seven cards], [stuck deals become more common, and some are only partly stuck (@tab-sizes)],
    [adds wild cards], [1 in 603,146; 1 in 48.8 million with cards 1 to 10 (@prop-wild)],
    table.hline(stroke: 0.6pt),
  ),
) <tab-house>

*Keeping it a game.* With full arithmetic nearly every hand scores 0, so the scoring stops distinguishing players. For a puzzle, a score of 0 every time is fine. The authors' teacher page says that scoring is de-emphasized and that thinking strategies come first. To keep a scored game, restrict the arithmetic. With addition only, about 29% of hands clear, and with $+$ and $-$, about 96%.

= More puzzles <sec-puzzles>

#epigraph[Looks like we've got another mystery on our hands.][Fred, in _Scooby-Doo_]

In these puzzles every answer combines cards more than once: make a number, then use it again. Use $+$, $-$ and $times$ only, not $div$. The answers are in @app-answers; a computer checked each one.

#puzzle[
  *1. Three cards.* Make 10 with all three cards, using each once. No two of them make 10 on their own, so every answer takes two steps.
  #minis((2, 3, 4), (1, 2, 4), (3, 6, 8), (2, 4, 9))
]

#puzzle[
  *2. Four cards.* Make 10 with all four cards, using each once. This time no two or three of them make 10 on their own.
  #minis((2, 2, 2, 2), (1, 1, 1, 4), (3, 3, 8, 9), (4, 7, 9, 9))
]

#puzzle[
  *3. Three odd cards.*
  #align(center, deal((1, 2, 3, 4, 6, 7, 8), (2, 4, 6, 8)))
  (a) With only $+$ and $-$, why must one card be left over? Hint: count the odd cards. (b) Now allow $times$ as well, and erase all seven cards.
]

#puzzle[
  *4. No pairs.* No hand card makes 10 with only one other card, even with $times$, so every group needs three cards or more. Erase the whole hand.
  #align(center, deal((1, 1, 1, 3, 3, 6, 6), (3, 5, 6, 8)))
]

#puzzle[
  *5. Homemade.* Make 10 with all the cards of each hand. Two equal cards make a homemade 0, as $7 - 7$, and two cards that differ by 1 make a homemade 1, as $9 - 8$. One of them will tidy away the cards you do not need.
  #minis(narrow: true, (2, 5, 8, 9), (3, 5, 5, 7, 8, 9, 9))
]

#puzzle[
  *6. Challenge.* Every hand card is odd and every center card is even. Again no hand card makes 10 with only one other card. Erase the whole hand.
  #align(center, deal((1, 1, 3, 3, 3, 3, 5), (4, 6, 8, 8)))
]

= The plan <sec-plan>

The proof of @prop-dead used one trick. The next five sections turn it into a complete answer, using three ideas.

+ *Silent cards* (@sec-silent). Looking at remainders mod $m$, every card divisible by $m$ behaves like 0. If at most one or two cards are not silent, the possible results are few, and 10 may not be among them. The proof of @prop-dead is the case of the four-hour clock.
+ *The wastebasket* #bin() (@sec-waste). A real 0 can swallow any cards you don't need. Multiply them by 0 and add. So a hand containing a 0 can be cleared if _some_ of its cards, perhaps with one center card, make 10, and otherwise not at all.
+ *Big hands* (@sec-big). Seven or more nonzero cards almost always make 10 using every card. Big hands are the one step proved by checking cases, and the computer supplies a certificate for each case.

Only the second idea needs a 0 card. In a deck of cards 1 to 10 it drops out, and the other two still give the whole answer (@prop-oneten). @fig-map, in @sec-check, shows which result rests on which.


= The model <sec-model>

#epigraph[Simon says touch your nose. Now touch your toes. Aha! I didn't say “Simon says.”][playground game]

To prove things we need the game in precise terms.

#definition(title: [deals])[
  The deck has four cards of each value $0, 1, dots, 10$. A _deal_ $(H, C)$ is a hand $H$ of seven cards and a center $C$ of four further cards. Treating the 44 cards as distinct, there are $binom(44, 7) binom(37, 4) = "2,530,881,913,560"$ deals, all equally likely.
]

#idea[bags of cards, and counting them][A _multiset_ is a bag in which repeats count: ${4, 4, 8}$ holds two 4s and an 8, and order does not matter. Curly brackets list what is in the bag. The number $binom(44, 7)$, read “44 choose 7”, counts the ways to pick seven cards out of 44 when order does not matter; it is 38,320,568. There are $binom(37, 4) = "66,045"$ ways to pick the four center cards from the 37 left, and multiplying the two gives the number of deals.]

#mnote(inline: true)[A 4 of hearts and a 4 of spades are different cards. Here both count as a 4.]Only card values matter, so deals with the same values form one _value pattern_. There are 17,237,220 value patterns, and every count in this note adds up how many deals each pattern stands for (@app-count).

#definition(title: [expressions])[
  An _expression_ over a multiset $M$ of numbers combines all of them, each exactly once, using $+$, $-$, $times$ and brackets. Formally it is a binary tree with the elements of $M$ at the leaves and operations at the inner nodes. We write $V(M)$ for the set of values of all such expressions, and say that $M$ _makes 10_ if $10 in V(M)$. A _part_ of $M$ is a nonempty sub-multiset.
]

For example, $V({2, 5}) = {7, 3, -3, 10}$, so ${2, 5}$ makes 10. Making 10 always means using _every_ card; when only some cards are needed, we say that a part of $M$ makes 10. @fig-tree shows an expression over all seven cards of the sheet's hand, drawn as a tree.

#figure(
  diagram(
    spacing: (6.5mm, 7.5mm), edge-stroke: 0.7pt + luma(45%), mark-scale: 70%,
    node((2.2, -0.6), text(9.5pt)[$= 10$], stroke: none),
    op((2.2, 0), $+$, <t0>),
    op((0.5, 1), $+$, <tl>), cardnode((0, 2), 4, <t4a>), cardnode((1, 2), 6, <t6>),
    op((3.9, 1), $times$, <tr>),
    op((2.8, 2), $-$, <tm>), cardnode((2.3, 3), 5, <t5a>), cardnode((3.3, 3), 5, <t5b>),
    op((5, 2), $times$, <tx>), cardnode((4.4, 3), 4, <t4b>),
    op((5.6, 3), $times$, <ty>), cardnode((5.1, 4), 8, <t8>), cardnode((6.1, 4), 9, <t9>),
    edge(<t4a>, <tl>, res(luma(40%))[4], label-side: center), edge(<t6>, <tl>, res(luma(40%))[6], label-side: center),
    edge(<tl>, <t0>, res(green)[*10*], label-side: center, stroke: 1pt + green),
    edge(<t5a>, <tm>, res(luma(40%))[5], label-side: center), edge(<t5b>, <tm>, res(luma(40%))[5], label-side: center),
    edge(<tm>, <tr>, res(luma(25%))[*0*], label-side: center, stroke: 1pt + luma(35%)),
    edge(<t8>, <ty>, res(luma(40%))[8], label-side: center), edge(<t9>, <ty>, res(luma(40%))[9], label-side: center),
    edge(<ty>, <tx>, res(luma(40%))[72], label-side: center), edge(<t4b>, <tx>, res(luma(40%))[4], label-side: center),
    edge(<tx>, <tr>, res(luma(40%))[288], label-side: center),
    edge(<tr>, <t0>, res(luma(25%))[*0*], label-side: center, stroke: 1pt + luma(35%)),
    node(enclose: (<tr>, <tm>, <tx>, <ty>, <t5a>, <t5b>, <t4b>, <t8>, <t9>),
      stroke: (paint: luma(55%), dash: "dashed", thickness: 0.6pt), fill: luma(97.5%), inset: 6pt, corner-radius: 6pt, name: <tbin>),
    node((7.6, 0.4), align(left, text(8.5pt, luma(30%))[#bin() a homemade 0: \ $5 - 5$ swallows \ the 4, 8 and 9]), stroke: none, name: <tnote>),
    edge(<tnote>, (6.55, 1.2), "-|>", bend: -25deg, stroke: 0.7pt + luma(45%)),
  ),
  caption: [The sheet's hand as one expression, $(4 + 6) + (5 - 5) times 4 times 8 times 9 = 10$. The cards sit at the bottom. Each circle combines the two values below it, and the numbers on the lines are the values passed up.],
) <fig-tree>

#definition(title: [groups])[
  A _group_ of the deal $(H, C)$ is a multiset of two or more cards that makes 10: a part of the hand, with or without one center card. Playing a group erases its hand cards; its center card, if any, stays. The hand is _cleared_ when it has been split into groups, reusing center cards as often as needed. A deal whose hand cannot be cleared is _stuck_. A deal is _dead_ if it has no group at all, so that not a single card can be erased.
]

Dead deals are stuck. @thm-single shows that the converse also holds: every stuck deal is dead. Intermediate values may be negative, as in $4 - 8$; forbidding that changes no count here, and neither does letting a lone 10 count as a group.

*Assumptions.* The deck has four cards of each value from 0 to 10, or from 1 to 10 in @prop-oneten. A deck with a different mix changes the counts, and it changes the large-moduli step of @thm-five and the cases of @thm-big, but not the lemmas. Center cards may be reused by any number of groups, as the sheet says.

= Silent cards <sec-silent>

The proof of @prop-dead looked only at remainders mod 4. Here is the same idea for any modulus.

#mnote(inline: true)[On the four-hour clock, 4 and 8 land on 0. They make no sound.]Fix a modulus $m >= 2$, the number of hours on the clock of @fig-clock. A card is _silent_ if $m$ divides its value and _loud_ otherwise. Mod $m$, a silent card sits at 0, so it behaves like a zero.

#lemma(title: [one loud card])[
  Let $m >= 2$, and suppose that every element of $M$ is silent, that is, divisible by $m$, except possibly one element $x$. Then every value in $V(M)$ is congruent mod $m$ to $x$, to $-x$ or to 0.
] <lem-one>

#idea[proof by induction on trees][Every expression is built by joining two smaller expressions, down to single cards at the bottom, as in @fig-tree. To show that every value has some property, check it for single cards, then check that joining two pieces that have it gives a value that has it too. The property then climbs the tree from the bottom to the top, one join at a time. @fig-trees shows it climbing.]

#proof[
  We use induction on the expression tree. A leaf is either $x$ or silent. Consider an inner node whose branches have values $v_1$ and $v_2$. Each card is used once, so $x$ lies in at most one branch; say it is not in the second, so that $v_2 equiv 0$. The node's value is $v_1 + v_2$ or $v_1 - v_2$, both congruent to $v_1$; or $v_2 - v_1$, congruent to $-v_1$; or $v_1 v_2$, congruent to 0. By induction $v_1$ is congruent to $x$, $-x$ or 0, so the node's value is too.
]

#figure(
  diagram(
    spacing: (7mm, 8mm), edge-stroke: 0.7pt + luma(45%), mark-scale: 70%,
    node((1, -0.8), text(9pt)[$(8+3)-4 = 7 equiv 3$], stroke: none),
    op((1, 0), $-$, <a>), op((0, 1), $+$, <ap>), cardnode((2, 1), 4, <a4>),
    cardnode((-0.5, 2), 8, <a8>), cardnode((0.5, 2), 3, <a3>, loud: true),
    edge(<a8>, <ap>, res(luma(40%))[0], label-side: center),
    edge(<a3>, <ap>, res(red)[3], label-side: center, stroke: 1pt + red),
    edge(<ap>, <a>, res(red)[3], label-side: center, stroke: 1pt + red),
    edge(<a4>, <a>, res(luma(40%))[0], label-side: center),
    node((5, -0.8), text(9pt)[$(8+3) times 4 = 44 equiv 0$], stroke: none),
    op((5, 0), $times$, <b>), op((4, 1), $+$, <bp>), cardnode((6, 1), 4, <b4>),
    cardnode((3.5, 2), 8, <b8>), cardnode((4.5, 2), 3, <b3>, loud: true),
    edge(<b8>, <bp>, res(luma(40%))[0], label-side: center),
    edge(<b3>, <bp>, res(red)[3], label-side: center, stroke: 1pt + red),
    edge(<bp>, <b>, res(red)[3], label-side: center, stroke: 1pt + red),
    edge(<b4>, <b>, res(luma(40%))[0], label-side: center),
  ),
  caption: [@lem-one at work, mod 4. The loud 3 keeps its remainder as it rises through $+$ and $-$ (left), until a multiplication by a silent value wipes it out (right). It can never become 2, the remainder of 10.],
) <fig-trees>

With two loud cards there are a few more possibilities, but still not many. Below, $plus.minus$ is read “plus or minus”.

#block(breakable: false)[
For two numbers $x$ and $y$, write
$ S(x, y) = {0, plus.minus x, plus.minus y, plus.minus x plus.minus y, plus.minus x y}. $
The set $S(x, y)$ contains the negative of each of its elements.

#lemma(title: [two loud cards])[
  Let $m >= 2$, and suppose that every element of $M$ is silent except possibly two elements $x$ and $y$. Then every value in $V(M)$ is congruent mod $m$ to an element of $S(x, y)$.
] <lem-two>
]

#proof[
  Again we use induction. Consider an inner node whose branches have values $v_1$ and $v_2$. If $x$ and $y$ lie in the same branch, say the first, then $v_1$ is congruent to an element of $S$ by induction and $v_2 equiv 0$. The node's value is then congruent to $0$, $v_1$ or $-v_1$, all in $S$ because $S$ is closed under negation. Otherwise each branch has at most one loud card, and @lem-one gives $v_1 equiv 0$ or $plus.minus x$, and $v_2 equiv 0$ or $plus.minus y$. Every sum, difference and product of these lies in $S$.
]

For example, mod 7 the cards 1 and 6 are worth $+1$ and $-1$, so $S(1, 6) = {0, plus.minus 1, plus.minus 2}$ mod 7. This set never includes 3, the remainder of 10 mod 7. Applying this to every possible group gives a test for dead deals.

#theorem(title: [congruence test])[
  Let $(H, C)$ be a deal and $m >= 2$. Suppose that at most one hand card is loud. Let $x$ be that card, or let $x = 0$ if every hand card is silent. If, for every center card $c$, no element of $S(x, c)$ is congruent to 10 mod $m$, then the deal is dead. In that case we say that the test _catches_ the deal at $m$.
] <thm-test>

#proof[
  A group consists of hand cards, among which only $x$ can be loud, and at most one center card $c$. If it has a center card, @lem-two shows that its value is congruent to an element of $S(x, c)$. If it has none, @lem-one shows that its value is congruent to 0 or $plus.minus x$, which also lie in $S(x, c)$. Either way the value is not congruent to 10, so it is not 10.
]

Which moduli need trying? Five suffice, and each of the five is needed.

#theorem(title: [five moduli suffice])[
  If the congruence test catches a deal at some modulus $m >= 2$, then it also catches the deal at one of $m = 3, 4, 6, 7, 9$. None of these five can be dropped, because each catches a deal that the other four miss.
] <thm-five>

#figure(
  diagram(
    spacing: (10mm, 9mm), edge-stroke: 0.8pt, mark-scale: 70%,
    blob((0, 0), [every modulus $m >= 2$], width: 46mm, name: <s0>),
    blob((0, 1), [$3, 4, 6, 7, 8, 9, 11, 12, dots$], width: 46mm, name: <s1>),
    blob((0, 2), [$3, 4, 6, 7, 8, 9$], width: 46mm, name: <s2>),
    blob((0, 3), [*$3, 4, 6, 7, 9$* \ #text(8pt)[each one needed]], tint: green, width: 46mm, name: <s3>),
    edge(<s0>, <s1>, "-|>"), edge(<s1>, <s2>, "-|>"), edge(<s2>, <s3>, "-|>"),
    ..range(3).map(k => (
      blob((1.6, k + 0.5), ([$2, 5, 10$], [$11, 12, dots$], [$8$]).at(k), width: 20mm),
      node((2.75, k + 0.5), box(width: 46mm, align(left, { set par(justify: false); text(8.5pt, hyphenate: false, (
        [divide 10, so $10 equiv 0$ is always reachable],
        [only 0 is silent, and a hand has $>= 3$ nonzero cards],
        [whatever it catches, 4 already catches]).at(k)) })), stroke: none),
      edge((0, k + 0.5), (1.6, k + 0.5), "-|>", stroke: luma(55%)),
    )).flatten(),
  ),
  caption: [The proof of @thm-five. Read down the left column: each box on the right removes some moduli, for the reason beside it, until five are left.],
) <fig-funnel>

#proof[
  _Divisors of 10._ If $m$ divides 10, then 10 is congruent to 0, which always lies in $S$. The test never catches anything.

  _Moduli of 11 or more._ The only silent card is then 0, and the deck has only four 0s. A seven-card hand therefore has at least three loud cards, while the test allows at most one.

  _The modulus 8._ Here 10 is congruent to 2. Suppose that the test catches a deal at 8, and let $x$ be the loud card in the hand. We show that it also catches the deal at 4.
  - The set $S(x, c)$ contains $plus.minus x$ and $plus.minus c$ but nothing congruent to 2, while the cards 2, 6 and 10 are congruent to $plus.minus 2$ mod 8. So neither $x$ nor any center card is a 2, 6 or 10.
  - When $x$ and a center card $c$ are both odd, $x + c$ and $x - c$ are even and differ by $2c$, which is 2 mod 4. One of them is then 2 mod 4, that is, $plus.minus 2$ mod 8, and both lie in $S(x, c)$ with their negatives, so $S(x, c)$ contains 2 and the test does not catch the deal. In a deal that the test catches, therefore, $x$ and the center cards are not both odd.
  - Now work mod 4, where 10 is congruent to 2. Every hand card other than $x$ is 0 or 8, which is silent. If $x$ is odd, every center card is 0, 4 or 8, also silent, and $S(x, c)$ reduces to ${0, plus.minus x}$. If $x$ is even, it is 4 or absent, the whole hand is silent, and $S(0, c) = {0, plus.minus c}$ with $c$ not a 2, 6 or 10. Either way $S$ holds only 0 and odd remainders, never 2, so the test catches the deal at 4.

  #block(breakable: false)[That leaves 3, 4, 6, 7 and 9. Each catches a deal the other four miss, for the reason in the last column:
  #align(center, table(
    columns: (auto, auto, auto, 1fr), align: (center, left, left, left), inset: (x: 6pt, y: 2.5pt),
    [$m$], [Hand], [Center], [Why the other four miss],
    [3], [#ms(0,0,0,0,3,3,3)], [#ms(3,6,6,6)], [the three 3s are loud at 4, 6, 7, 9],
    [4], [#ms(0,0,0,0,1,4,4)], [#ms(4,4,8,8)], [1, 4, 4 are loud at 3, 6, 7, 9],
    [6], [#ms(0,0,0,0,6,6,6)], [#ms(1,1,1,1)], [at 3, a center 1 is already $equiv 10$; the 6s are loud at 4, 7, 9],
    [7], [#ms(0,0,0,0,1,7,7)], [#ms(1,1,1,6)], [1, 7, 7 are loud at 3, 4, 6, 9],
    [9], [#ms(0,0,0,0,2,9,9)], [#ms(2,2,2,7)], [at 3, $2 + 2 equiv 10$; 2, 9, 9 are loud at 4, 6, 7],
  ))]
  At its own modulus each hand has at most one loud card, and the test catches the deal.
]

Spelled out modulus by modulus, the test catches eight concrete shapes. For each modulus, either the hand has no loud card or it has exactly one. The shares are fractions of the 12,362,536 stuck deals.

#figure(
  caption: [The shapes caught by the congruence test.],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, left, left, right),
    table.hline(stroke: 0.6pt),
    [$m$], [10 mod $m$], [Hand], [Center cards], [Share],
    table.hline(stroke: 0.4pt),
    [4], [2], [all in ${0, 4, 8}$], [no 2, 6 or 10], [81.0%],
    [3], [1], [all in ${0, 3, 6, 9}$], [all in ${0, 3, 6, 9}$], [11.7%],
    [4], [2], [${0, 4, 8}$ plus one odd card], [all in ${0, 4, 8}$], [2.2%],
    [7], [3], [${0, 7}$ plus one of 1, 6, 8], [all in ${0, 1, 6, 7, 8}$], [1.9%],
    [7], [3], [all in ${0, 7}$], [no 3, 4 or 10], [0.8%],
    [9], [1], [all in ${0, 9}$], [no 1, 8 or 10], [0.8%],
    [6], [4], [all in ${0, 6}$], [no 2, 4, 8 or 10], [0.4%],
    [9], [1], [${0, 9}$ plus one of 2, 7], [all in ${0, 2, 7, 9}$], [0.2%],
    table.hline(stroke: 0.4pt),
    [], [], [], [total], [99.1%],
    table.hline(stroke: 0.6pt),
  ),
) <tab-shapes>

The mod 3 and mod 4 shapes will matter again, so they get names.

#definition(title: [the two families])[
  A multiset of cards is in the _3-family_ if every card is a multiple of 3, and in the _4-family_ if it has no 2, 6 or 10 and at most one odd card. Put another way, a multiset is in the 3-family if no card is loud mod 3, and in the 4-family if at most one card is loud mod 4 and no card is congruent to 2 mod 4.
] <def-families>

#figure(
  familypic,
  caption: [The card values each family allows. A multiset may hold any number of the solid values, at most one card in all from the dashed values, and none of the faded ones. Neither row can make 10.],
) <fig-families>

Taking cards away keeps a multiset in its family, and so does adding a 0. Nonzero cards in both families could only be 3s and 9s, with at most one odd card among them, so two or more nonzero cards are never in both families. No multiset in either family makes 10, by @lem-one: mod 3 every value is 0, and mod 4 every value is 0 or odd, while 10 is 1 mod 3 and 2 mod 4.

*Why one loud card is so weak.* Because each card is used once, an expression is _affine_ in each card, which means that its value has the form $A x + B$, where $A$ and $B$ are built from the other cards. Computer scientists call such expressions #link("https://www.sciencedirect.com/science/article/pii/S0304397517307478")[read-once formulas]. If every other card is silent, then $B equiv 0$, and $A$ is $plus.minus 1$ if nothing multiplied $x$, or congruent to 0 if something did. So the value is congruent to $plus.minus x$ or 0. This argument proves @lem-one in one line.

= A zero is a wastebasket <sec-waste>

#epigraph[I was made for soup, salad, maybe chili, and then the trash.][Forky, in _Toy Story 4_]

#mnote(inline: true)[0 times a million is still 0. That is the whole trick.]The congruence test misses 0.9% of stuck deals. A typical one has hand $#ms(0,0,0,0,6,7,7)$ and center cards drawn from 1, 5, 6, 7, 8. All of these deals have a 0 in the hand, in fact at least two, and a real zero can do something no remainder can: it throws cards away.

#tryit[Using some of the cards 1, 6, 7 and 7, each at most once, and $+$, $-$ and $times$, get as close to 10 as you can.]

#figure(
  diagram(
    spacing: (7mm, 9mm), edge-stroke: 0.8pt, mark-scale: 70%,
    op((1, 0), $+$, <r>),
    blob((0, 1), [$E$ \ #text(8pt)[makes 10]], tint: green, name: <e>),
    op((2, 1), $times$, <m>),
    blob((1.6, 2), [$P$ \ #text(8pt)[every other card]], tint: luma(45%), name: <p>),
    cardnode((2.5, 2), 0, <z>),
    edge(<e>, <r>, res(green)[10], label-side: center),
    edge(<m>, <r>, res(luma(40%))[0], label-side: center),
    edge(<p>, <m>), edge(<z>, <m>),
    node(enclose: (<m>, <p>, <z>), stroke: (paint: luma(55%), dash: "dashed", thickness: 0.6pt),
      fill: luma(97%), inset: 7pt, corner-radius: 6pt),
    node((3.35, 1.5), text(8.5pt, luma(35%))[#bin(size: 14pt) \ the \ wastebasket], stroke: none, name: <wnote>),
  ),
  caption: [The wastebasket: @fig-tree with letters in place of cards. If some cards make 10, put every other card into $P$; multiplying by the 0 throws them away.],
) <fig-waste>

For example, the hand ${0, 2, 3, 5, 7, 8, 9}$ contains $3 + 7$, so the whole hand makes 10: $(3 + 7) + (2 times 5 times 8 times 9) times 0$. The next lemma says exactly what a multiset with a 0 can make.

#lemma(title: [wastebasket])[
  Let $M$ contain at least one 0, and let $N$ be its nonzero elements. Then $M$ can make 0, and apart from that it can make exactly the values that some part $N'$ of $N$ can make, together with their negatives:
  $ V(M) = {0} union union.big_(N' "a part of" N) plus.minus V(N'). $
] <lem-waste>

#proof[
  _Every listed value can be made._ Take an expression $E$ over $N'$. The rest of $M$ contains a 0. If the rest is only that 0, use $E + 0$ and $0 - E$. Otherwise let $P$ be any expression over the rest with one 0 removed, and use $E + P times 0$ and $P times 0 - E$. To get 0 itself, multiply anything by the 0.

  _Nothing else can be made._ We show by induction that every subexpression has value 0, or $plus.minus w$ where $w$ can be made by some part of the subexpression's nonzero cards. A leaf is 0 or a nonzero card, which is a part by itself. At a node, combining 0 with a value $w$ gives 0 or $plus.minus w$. Combining $plus.minus u$ and $plus.minus w$, where $u$ comes from a part $N'_1$ of the first branch and $w$ from a part $N'_2$ of the second, gives $plus.minus (u + w)$, $plus.minus (u - w)$ or $plus.minus u w$. Each of these is $plus.minus$ a value of the part made of $N'_1$ and $N'_2$ together.
]

The wastebasket can hand back $-E$ as well as $E$. The sign does no harm, because whenever a set of cards can make $-10$, it can also make $10$.

#lemma(title: [negatives flip])[
  If $M$ consists of nonnegative numbers and $v in V(M)$ is negative, then $-v in V(M)$.
] <lem-flip>

#proof[
  We use induction on the expression; a leaf is never negative. At a node whose branches have values $v_1$ and $v_2$, there are three cases. If $v = v_1 - v_2$, then $-v = v_2 - v_1$. If $v = v_1 + v_2 < 0$, one of them, say $v_1$, is negative; by induction $-v_1$ can be made from the same cards, and $-v = (-v_1) - v_2$. If $v = v_1 v_2 < 0$, exactly one factor is negative, say $v_1$, and $-v = (-v_1) v_2$.
]

#idea[if and only if][“A if and only if B” means that A and B are true together or false together: each one guarantees the other. “A exactly when B” says the same thing. So the next theorem is a complete test. It says when a deal with a 0 is dead, and also when it is not.]

#theorem(title: [hands with a zero])[
  Suppose that the hand contains a 0, and let $N$ be its nonzero cards.
  + The deal is dead if and only if no part of $N$, with or without one center card, makes 10, and no center card is a 10.
  + If the deal is not dead, the whole hand, with at most one center card, makes 10. So a single group erases all seven cards.
] <thm-zero>

#proof[
  Suppose that some part of $N$, perhaps with a center card $c$, makes 10, or that a center card $c$ is a 10 on its own. The whole hand, together with $c$ if it was used, contains those cards and a 0, so it makes 10 by @lem-waste, and it is a group. This argument proves (2) and shows that such a deal is not dead.

  Conversely, suppose that the deal has a group. If the group contains no 0, its hand cards form a part of $N$ that makes 10 with the group's center card, if any. If it contains a 0, @lem-waste gives a part of its nonzero cards that makes 10 or $-10$, and @lem-flip turns $-10$ into 10. That part is a part of $N$ with at most one center card, or else a center card alone, which must then be a 10.
]

About half of all hands (51.3%) contain a 0, and they account for 98.3% of stuck deals. For them, the question "can this deal be cleared?" is the puzzle "can some of these cards make 10?" on a handful of numbers. In the deals the congruence test misses, the hand turns out to have at least two 0s, so at most six cards are ever in play; @app-lists lists all 23 such hands. For the typical one, hand $#ms(0,0,0,0,6,7,7)$ with center 1, no part of ${1, 6, 7, 7}$ makes 10. (The full set appears on the published list of impossible Make 10 puzzles as 1677.)


The proof for the smallest stuck hand borrows a classic argument.

#tryit[With the cards 1, 1, 1 and 3, each used at most once, and $+$, $-$ and $times$, what is the biggest number you can make?]

#lemma(title: [subtraction never helps you reach far])[
  Take an expression whose leaves are nonnegative, let $v$ be its value, and let $v^+$ be the value of the same expression with every $-$ replaced by $+$. Then $|v| <= v^+$.
] <lem-size>

#proof[
  We use induction on the expression. At a leaf, $|v| = v = v^+$. At a node whose branches have values $a$ and $b$, induction gives $|a| <= a^+$ and $|b| <= b^+$, so
  $ |a plus.minus b| <= |a| + |b| <= a^+ + b^+ quad "and" quad |a b| = |a| |b| <= a^+ b^+. $
  The right-hand sides are the node's value with every $-$ replaced by $+$.
]

#proposition(title: [three ones])[
  The hand $#ms(0,0,0,0,1,1,1)$ is dead whenever every center card is 1, 2 or 3.
] <prop-ones>

#proof[
  The center card $c$ is not a 10, so by @thm-zero it is enough to show that no part of ${1, 1, 1, c}$ makes 10. Take an expression over such a part and replace $c$ by $1 + dots.c + 1$, with $c$ ones. The value is unchanged, and the expression now uses at most six 1s. By @lem-size, the absolute value of its value is at most what the same expression gives with every $-$ replaced by $+$, which leaves only $+$ and $times$. So it is enough to know that six 1s, combined with $+$ and $times$, reach at most $(1+1+1) times (1+1+1) = 9$. This bound is a small case of one due to Selfridge: $k$ ones make at most $3^(k\/3)$.
]

= Big hands always make 10 <sec-big>

#epigraph[Many hands make light work.][proverb]

Everything so far is proved by hand. One more ingredient finishes the job, and it concerns hands with no 0.

Why should seven nonzero cards make 10 _using every card_? Because ten-makers are everywhere ($a + (10 - a)$, $2 times 5$, $3 times 3 + 1$, …), and leftover cards are easy to dispose of even without a real 0. Two equal cards make a homemade zero, $x - x$, which works as a wastebasket. Two cards that differ by 1 make a homemade 1, which multiplies anything harmlessly. The sheet's example hand clears in one breath, as in @fig-tree:

$ (4 + 6) + (5 - 5) times 4 times 8 times 9 = 10. $

*Every hand has a homemade 0 or a homemade 1.* Sort the eleven card values into six boxes: 0 and 1, 2 and 3, 4 and 5, 6 and 7, 8 and 9, and 10 alone. Seven cards cannot all go into different boxes, so some box holds two of your cards. Those two are equal, which gives a homemade 0, or they differ by 1, which gives a homemade 1. By exact count, 86% of hands hold two equal cards and 97% hold two cards that differ by 1. A homemade 0 or 1 can only tidy cards away. It cannot make a 10 that is not there, which is why the hand of @prop-dead, full of equal cards, is still stuck.

#figure(
  pigeon((1, 2, 3, 4, 5, 6, 7)),
  caption: [Seven cards, six boxes: some box must hold two cards. Look at any box with two cards in it. They are equal or differ by 1, so they make a homemade 0 or a homemade 1; here $3 - 2 = 1$.],
) <fig-boxes>

Even a hand with no repeated card manages.

#tryit[Make 10 from 1, 2, 3, 4, 5, 6 and 7, using every card exactly once, with $+$, $-$ and $times$.]

The two families from @sec-silent can never make 10, and the next theorem says that for big hands they are essentially the only obstacle. It stops at eleven cards because a deal has eleven cards in all.

#idea[certificates][Finding an expression that makes 10 can take a long search, but checking one takes a minute with pencil and paper. A _certificate_ is such a checkable answer, written down. The computer found 265,148 of them, one for each case of the next theorem. Nobody has to trust the program that found them, because any one can be checked by hand, and a short, separate program checks them all.]

#theorem(title: [big hands; checked by computer])[
  Let $M$ be a multiset of seven to eleven nonzero cards, with at most four of each value, that is in neither family. Then $M$ makes 10, using every one of its cards, with exactly one exception, $M = #ms(1,6,6,7,7,7,7)$. That multiset has a part that makes 10, namely $1 + 7 + 7 + 7 - 6 - 6$, but all seven cards together cannot.
] <thm-big>

#proof(title: [Verification])[There are 265,148 such multisets. The search wrote down one expression for each, and a separate 40-line program (@sec-check) checks them all in about ten seconds. It lists the cases itself, so none can be skipped. For each one it confirms that the expression uses exactly those cards, with only $+$, $-$ and $times$, and equals 10. @app-certs shows a sample.]

#mnote(inline: true)[One naughty hand out of 265,148.]The exception is a curiosity. All seven of its cards together can make every whole number from 0 to 20 _except_ 10. A third, independent search confirms this.

*Why seven?* Call a multiset _ten-free_ if no part of it with two or more cards makes 10. @fig-seven counts the nonzero ten-free multisets that are in neither family.

#figure(
  placement: auto,
  lqd(width: 8cm, height: 3.6cm,
    xlabel: [nonzero cards], ylabel: [ten-free multisets],
    xaxis: (ticks: range(1, 8)), ylim: (0, 80),
    lq.bar((1, 2, 3, 4, 5, 6, 7), (2, 29, 70, 34, 25, 10, 0), fill: luma(82%), stroke: luma(40%)),
    lq.place(7, 5, text(8pt)[*0*]),
  ),
  caption: [How many ten-free multisets of nonzero cards lie outside both families, by number of cards. Look at the last bar: at seven cards none is left. Exhaustive count.],
) <fig-seven>

Nine of the ten survivors at six cards (listed in @app-lists) rest on four silent 7s or four silent 9s. Four 7s with two cards from 1, 6, 8, all worth $plus.minus 1$ mod 7, reach only $0, plus.minus 1, plus.minus 2$ and never 3 (@lem-two). Four 9s with two cards from 2 and 7, worth $plus.minus 2$ mod 9, never reach 1. The deck has no fifth 7 or 9, so a seventh card would be a third loud card, and @lem-two no longer applies. The tenth survivor, three 6s with three 7s, has no such explanation. Mod 7 the 6s are three loud cards, and $7 - 6 - 6 - 6 = -11$ does land on 3. It was checked directly. At seven cards nothing survives, and seven is exactly the number of cards Erase deals. We know no short proof of this last step, and small patterns cannot give one: in ${6, 6, 6, 7, 7, 7, 7}$ the only part that makes 10 is all seven cards, $7 + 7 + 7 + 7 - 6 - 6 - 6$.

Now we can draw the consequences.

#mnote[All or nothing!]
#theorem(title: [single group])[
  In every deal, either no card can be erased, or the whole hand, together with at most one center card, makes 10.
] <thm-single>

#proof(title: [Proof, using @thm-big])[
  If the hand has a 0, this is @thm-zero. Otherwise the hand is seven nonzero cards.

  _The hand is in neither family._ If it is not the exception, @thm-big says that the whole hand makes 10. If it is the exception $#ms(1,6,6,7,7,7,7)$, add any center card $c$. If $c$ is 0, the wastebasket turns $1 + 7 + 7 + 7 - 6 - 6$ into an expression over all eight cards. If $c$ is not 0, the eight cards contain a 1 and a 6, so they are in neither family, and @thm-big applies.

  _The hand is in a family._ Take a group, if the deal has one. A group makes 10 and a multiset in a family never does, so the group is in neither family. Its hand cards are in the hand's family, because taking cards away keeps a multiset in its family. So the group has a center card $c$, and $c$ is not 0, because adding a 0 also keeps a multiset in its family. Now look at the hand together with $c$, eight nonzero cards. They contain the group, so they are not in the hand's family. Seven nonzero cards are never in both families, so the hand is not in the other family, and neither are the eight cards. By @thm-big the eight cards make 10, and they form a single group that clears the hand.
]

For example, $#ms(3,3,3,3,6,6,6)$ is in the 3-family, so it needs help. With a center 1, the eight cards give $(3 times 3 + 1) + (6 - 6) times 3 times 3 times 6 = 10$.

#theorem(title: [classification])[
  A deal is stuck if and only if it is dead. A deal is dead exactly when one of the following holds:
  - the hand contains a 0, no part of its nonzero cards makes 10 with or without one center card, and no center card is a 10; or
  - the hand contains no 0, and for every center card $c$, the hand together with $c$ is in the 3-family or the 4-family.
] <thm-class>

#proof(title: [Proof, using @thm-big])[
  A dead deal is stuck, and by @thm-single an stuck deal is dead. The first bullet is @thm-zero. Now take a hand with no 0.

  If the hand together with each center card is in a family, then so is every group, since a group is a part of one of these multisets. Family multisets never make 10, so the deal is dead.

  Conversely, suppose that the deal is dead. @thm-big says that seven to eleven nonzero cards make 10 unless they are in a family or are the exception. In a dead deal the hand does not make 10, and neither does the hand together with a center card, since each would be a group. The hand is not the exception either, because part of the exception makes 10 and would be a group. So the hand is in a family. With a center card $c$ that is not 0, the hand and $c$ are eight nonzero cards that do not make 10, and eight cards are not the exception, so they are in a family too. A center 0 keeps the hand in its family.
]

The second bullet is the congruence test at $m = 3$ or $m = 4$, written out: the first three rows of @tab-shapes. The first bullet covers 12,153,144 deals and the second 209,392. @fig-status shows how they split.

#figure(
  diagram(
    spacing: (5mm, 10mm), edge-stroke: 0.8pt, mark-scale: 70%,
    blob((1, 0), [*12,362,536* stuck deals], name: <all>),
    blob((0.5, 1), [hand has a 0: *12,153,144* (98.3%) \ #text(8.5pt)[exactly characterized by @thm-zero]], tint: green, width: 54mm, name: <z>),
    blob((1.85, 1), [no 0 in hand: *209,392* (1.7%) \ #text(8.5pt)[exactly the test at $m = 3, 4$]], tint: purple, width: 46mm, name: <nz>),
    blob((0, 2), [also caught by the \ congruence test: *12,044,776*], tint: green, width: 40mm, name: <zt>),
    blob((1, 2), [too little material: *108,368* \ #text(8.5pt)[small Make 10 checks]], tint: green, width: 40mm, name: <zl>),
    edge(<all>, <z>, "-|>", stroke: green), edge(<all>, <nz>, "-|>", stroke: purple),
    edge(<z>, <zt>, "-|>", stroke: green), edge(<z>, <zl>, "-|>", stroke: green),
  ),
  caption: [Where the stuck deals live. Green is proved by hand; purple also uses @thm-big.],
) <fig-status>

*Cards 1 to 10.* The sheet asks for a 0 to 10 deck, but some classrooms play with cards 1 to 10, four of each: 40 cards, or playing cards without the queens. How much of the argument needs a 0? Only @sec-waste, where a real 0 throws cards away. The silent cards of @sec-silent and the big hands of @sec-big do not: @thm-big is about nonzero cards, and most of its expressions make their own 0s and 1s from pairs such as $7 - 7$ and $9 - 8$. So with no 0 in the deck the wastebasket section drops out, every deal that is not stuck still clears in a single group (@thm-single), and only the two families can block a deal.

#mnote[No 0 in the deck? Then watch for 4s and 8s, or 3s, 6s and 9s.]
#proposition(title: [cards 1 to 10])[
  With four cards of each value from 1 to 10, and $+$, $-$ and $times$, a deal is stuck exactly when
  + all eleven cards, hand and center, are 3s, 6s and 9s; or
  + the hand is seven of the eight 4s and 8s, and every center card is odd or is the last 4 or 8.
  There are 51,840 such deals among the 762,894,475,200, about 1 in 14.7 million.
] <prop-oneten>

#proof(title: [Proof, using @thm-big])[
  The hand has no 0, so by @thm-class the deal is stuck exactly when, for every center card $c$, the hand together with $c$ is in the 3-family or the 4-family. Taking $c$ away keeps a multiset in its family, and seven nonzero cards are never in both families, so the hand is in one family and every center card keeps it there. In the 3-family every card is a 3, 6 or 9. There are twelve of these, so the hand takes seven of them and the center four of the other five: $binom(12, 7) binom(5, 4) = "3,960"$ deals. In the 4-family the nonzero cards are 4s and 8s, plus at most one odd card. If the hand had an odd card, every center card would have to be a 4 or an 8, and only two would be left. So the hand is seven of the eight 4s and 8s, in $binom(8, 7) = 8$ ways, and each center card is one of the 20 odd cards or the last 4 or 8, in $binom(21, 4) = "5,985"$ ways. The count is $8 times "5,985" = "47,880"$ deals, and $"3,960" + "47,880" = "51,840"$ out of $binom(40, 7) binom(33, 4) = "762,894,475,200"$. The brute-force counter, which uses none of the theorems, finds the same 51,840.
]

The deal of @prop-dead is of the second kind.

*Smaller hands behave differently.* With fewer cards a hand can get _partly stuck_, which means that some cards can be erased, but not all. Take the three cards 2, 2, 5 with four 7s in the center: $2 times 5$ erases two of them, and the other 2 is stranded. Keeping four center cards and shrinking the hand, partly stuck deals become rarer with every card, and at seven they are gone, the same size at which @fig-seven reaches zero.

#figure(
  caption: [Hand size against stuck deals, with four center cards throughout. Exhaustive at every size.],
  table(
    columns: (auto, auto, auto),
    align: (center, right, right),
    table.hline(stroke: 0.6pt),
    [Hand size], [Stuck deals], [Of those, partly stuck],
    table.hline(stroke: 0.4pt),
    [2], [1 in 6], [33%],
    [3], [1 in 51], [16%],
    [4], [1 in 427], [1.8%],
    [5], [1 in 3,462], [0.13%],
    [6], [1 in 27,103], [0.005%],
    [7], [1 in 204,722], [none],
    [8], [1 in 1.65 million], [none],
    table.hline(stroke: 0.6pt),
  ),
) <tab-sizes>

*A closed form.* If a group may borrow any number of center cards, the hand and the center play the same role, and the answer depends only on the eleven cards on the table. The count then has a short formula.

#theorem(title: [any number of center cards])[
  If groups may use any number of center cards, a deal is stuck exactly when its eleven cards, hand and center together, are in the 3-family or in the 4-family.
] <thm-any>

#proof(title: [Proof, using @thm-big])[
  If the eleven cards are in a family, so is every group, and no group makes 10. Otherwise there are two cases. If no card is 0, the eleven cards make 10 together by @thm-big, and the whole table is one group. If some card is 0, there are at least seven nonzero cards, because the deck has only four 0s, and they are in neither family. So they include a card that is not a multiple of 3, and either a 2, 6 or 10 or two odd cards. Choose seven nonzero cards that include these, so that the seven are in neither family. By @thm-big some part of the seven makes 10, and the wastebasket absorbs every other card on the table.
]

Counting the eleven-card sets in each family: the 3-family draws from the sixteen 0s, 3s, 6s and 9s; the 4-family takes all eleven cards from the twelve 0s, 4s and 8s, or ten of them plus one of the twenty odd cards. No eleven cards are in both families, since they would be 0s with at most one 3 or 9, and there are only four 0s. So

$ P("stuck") = (binom(16, 11) + binom(12, 11) + 20 binom(12, 10)) / binom(44, 11) = 5700 / "7,669,339,132" approx 1 / "1,345,498". $

= The other readings <sec-other>

*Plus and minus only.* Without $times$, the modulus 2, useless above, does most of the work.

#proposition(title: [parity])[
  With only $+$ and $-$, if the hand holds an odd number of odd cards and no center card is odd, at least one card is left.
] <prop-parity>

#proof[
  Subtracting a number changes parity exactly as adding it does, so a group's value has the same parity as the sum of its cards. Ten is even, so every group holds an even number of odd cards. Without an odd center card, the odd cards in the hand, an odd number of them, cannot all be covered.
]

#idea[count the odd cards][With only $+$ and $-$, odd cards can only leave in twos: odd and odd make even, and 10 is even. So count your odd cards. If the number is odd and no center card is odd, one of them is staying.]

The deck has 20 odd and 24 even cards. Choose $k$ odd cards and $7 - k$ even cards for the hand, then all four center cards from the $17 + k$ even cards that remain. Adding up over the odd values of $k$, the obstruction occurs in

$ (sum_(k "odd") binom(20, k) binom(24, 7-k) binom(17+k, 4)) / (binom(44, 7) binom(37, 4)) = "99,053,234,280" / "2,530,881,913,560" approx 3.914% $

of deals. In the 30,000 simulated deals of @tab-readings, 1,106 of the 1,126 failures are of this kind, and all but one of the failures leave exactly one card.

#mnote(inline: true)[Find each card a partner. A card with no partner stays.]*Pairs only, as printed.* Here the answer is a matter of counting.

#proposition(title: [pairs only])[
  Under the printed rules, pair off the hand as far as possible: each card $a$ with a card $10 - a$, and 5s with each other. The hand can be cleared exactly when every card left over has its complement in the center. Concretely, for each value $a$ from 0 to 4: if the hand has more $a$'s than $(10 - a)$'s, the center must contain a $10 - a$, and if it has fewer, the center must contain an $a$. If the hand has an odd number of 5s, the center must contain a 5.
] <prop-pairs>

#proof[
  A card $a$ can be erased only together with a card worth $10 - a$, from the hand or from the center. Within the hand, $a$'s and $(10 - a)$'s pair off until one kind runs out, and 5s pair with each other. Each card left over can only be erased by a center card, which is reusable, so one center card of the right value is both necessary and enough.
]

Counting the hands that meet this condition, value by value (add the cases, then subtract what was counted twice), shows that a deal can be cleared in 100,621,356,600 of the 2,530,881,913,560 deals, about 3.976%.

*Addition only.* The hand must split into piles that each sum to 10, with at most one center card per pile. Finding such a split is a small packing problem. One obstruction is quick to see. A 10 can only be erased together with 0s, since every other card would push the sum past 10. In 30,000 simulated deals, about 29.2% pack perfectly (@tab-readings).

*With division.* Fractions are allowed along the way, as in the Make 10 puzzle 1 1 9 9: $(1 + 1 div 9) times 9 = 10$. Division breaks the silent-card lemma, because dividing by a silent card means dividing by 0 mod $m$: $8 + 8 div 4 = 10$ uses only multiples of 4, and $(3 times 9 + 3) div 3 = 10$ only multiples of 3. The wastebasket survives, since $0 div w = 0$ and the sign flip still works. With $div$ and one center card per group, an exhaustive count finds 27,712 deals that stay stuck, about 1 in 91.3 million. They form 106 value patterns, and all but three are four 0s plus three cards, so by the wastebasket each comes down to a published impossible Make 10 puzzle such as 6677, 1677 or 7788. With $div$ and any number of center cards, the same count finds that every deal can be cleared.

*Wild cards.* The sheet's second variation puts "wildcards" in the deck, and the teacher page suggests a king, a card the 0 to 10 deck does not otherwise use. Take the 44 cards plus the four kings as wild cards. We let a wild card be worth any value from 0 to 10, chosen by the player each time it is used.

#proposition(title: [wild cards])[
  With $+$, $-$ and $times$ and four wild kings in the deck, every deal with a wild card in it can be cleared in a single group. So a deal is stuck exactly when it has no wild card and is stuck in the ordinary game. The stuck deals number 12,362,536 of the 7,456,416,121,440, about 1 in 603,146.
] <prop-wild>

#proof[
  Pick a wild card $w$ in the deal, and give every other wild card in the hand the value 0. Sort the hand cards other than $w$ as $a_1 <= a_2 <= dots.c <= a_k$, and take their alternating sum from the top, $t = a_k - a_(k-1) + a_(k-2) - dots.c$. Grouping the terms in pairs from the top, $(a_k - a_(k-1)) + (a_(k-2) - a_(k-3)) + dots.c$, shows that $t >= 0$. Grouping them after the first term, $a_k - (a_(k-1) - a_(k-2)) - dots.c$, shows that $t <= a_k <= 10$. Give $w$ the value $10 - t$. Then these cards and $w$ make 10, using only $+$ and $-$. If $w$ is in the hand, they are the whole hand; if $w$ is in the center, they are the whole hand and one center card. Either way a single group erases all seven cards. Deals with no wild card are exactly the deals of the ordinary game, so they contribute the same 12,362,536 stuck deals, now out of $binom(48, 7) binom(41, 4)$.
]

Each wild card takes a single value in this proof, so nothing changes if the rules make a wild card keep its value for the whole round.

*Cards 1 to 10 under the other readings.* @tab-readings has a column for the deck without 0s. With division every deal clears: only the 51,840 deals of @prop-oneten could be stuck, and the computer finds a way to clear each one, for example with $9 + 4 div 4$ or $3 + 8 - 8 div 8$. With any number of center cards per group only the first kind is left, since eleven cards with at most one odd card cannot all come from the eight 4s and 8s: 3,960 deals, about 1 in 193 million (@thm-any). The argument of @prop-wild does not need a 0 card (a wild card may still be worth 0), so with four wild kings the stuck deals are the same 51,840, about 1 in 48.8 million. With only $+$ and $-$, parity blocks exactly 2.650% of deals. Under the printed rules 96.90% of deals are stuck, more than with a 0, partly because a 10 can now never be erased: its only partner was a 0.

= Relatives <sec-relatives>

#epigraph[Ohana means family. Family means nobody gets left behind or forgotten.][Lilo, in _Lilo & Stitch_]

Erase with full arithmetic belongs to a well-studied family of puzzles, in which you make a target from given numbers, using each once.

#figure(
  caption: [Neighboring games. All four allow $div$.],
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [Game], [What is known],
    table.hline(stroke: 0.4pt),
    [#link("https://www.4nums.com/24_game/facts/")[24 game]], [1,362 of the 1,820 four-card hands (cards 1 to 13) can make 24],
    [#link("https://ja.wikipedia.org/wiki/%E3%83%86%E3%83%B3%E3%83%91%E3%82%BA%E3%83%AB")[Make 10]], [552 of the 715 sets of four digits can make 10; played in Japan on ticket and license-plate numbers],
    [#link("https://mkaz.blog/math/krypto-game-analysis/")[Krypto]], [61,910 of 2,968,875 games (2.09%) have no solution],
    [#link("https://arxiv.org/abs/1502.05450")[Countdown]], [91.2% of the 11.9 million number rounds can be hit exactly],
    table.hline(stroke: 0.6pt),
  ),
)

Erase is far kinder than these because it gives you seven cards, a free choice of how to group them, and center cards that never run out. The others demand a single expression, and all but Countdown must use every number.

Four results from further away bear on Erase.

- *Hard in general, but not here.* Deciding whether $n$ numbers can make a target is #link("https://arxiv.org/abs/2011.11767")[NP-complete] for every set of operations from $+$, $-$, $times$, $div$ except $+$ alone and $times$ alone. Roughly, no known method avoids trying a huge number of possibilities as $n$ grows. For $+$, $-$ and $times$ the hardness is of the _weak_ kind: it needs the numbers themselves to be large. Cards stop at 10, so the result says nothing about Erase, and @thm-big points the other way. With small cards, more cards make 10 easier to reach, not harder.
- *Integer complexity.* @prop-ones is the Mahler–Popken problem in miniature: the #link("https://en.wikipedia.org/wiki/Integer_complexity")[integer complexity] of 10, the least number of 1s that make 10 with $+$ and $times$, is 7, one more 1 than the smallest stuck hand can muster.
- *Fractions are rarely needed.* Of the 552 Make 10 sets that work, only four need a fraction along the way (1158, 1199, 1337 and 3478), and of the 1,362 solvable 24-game hands only 16 do. Erase behaves the same way, since requiring every division to come out whole changes none of its counts (@tab-house). A short computation confirms all three figures.
- *Different digits help.* Any four different digits from 1 to 9 make 10 with $+$, $-$, $times$ and $div$; a computer check of all 126 sets confirms it. @thm-big is the Erase version of that fact. Seven nonzero cards outside the two families make 10, and they do it without division.

= How this was checked <sec-check>

#epigraph[They call me the Count because I love to count things!][Count von Count, in _Sesame Street_]

Every claim in this note is known in one of four ways: by a proof written out here, by a computer check of every case, by an exhaustive count, or by simulation. @tab-status says which is which, and @fig-map shows which result rests on which. The rest of the section describes the checks.

#figure(
  caption: [How each claim is known.],
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [], [Claims],
    table.hline(stroke: 0.4pt),
    [By hand], [@prop-dead; silent cards, the congruence test and the five moduli (@thm-test, @thm-five); the wastebasket and hands with a zero (@thm-zero); three ones (@prop-ones); parity and pairs only, with exact counts; wild cards (@prop-wild)],
    [By computer, every case], [@thm-big, and through it @thm-single, @thm-class, @thm-any and @prop-oneten],
    [By exhaustive count], [the count of 12,362,536 and the other exact rows of @tab-readings; the hand-size table; the counts with division; @fig-seven and @app-lists],
    [By simulation], [the simulated entries of @tab-readings and @tab-house, and the numbers in @sec-play],
    [Open], [A proof of @thm-big without checking cases; in particular, a short reason why no ten-free multiset of seven nonzero cards survives outside the families.],
    table.hline(stroke: 0.6pt),
  ),
) <tab-status>

#figure(
  diagram(
    spacing: (5mm, 8.5mm), edge-stroke: 0.9pt, mark-scale: 70%, edge-corner-radius: 8pt,
    blob((0, 0), [silent cards \ #text(8pt)[@lem-one, @lem-two]], tint: green, name: <sil>),
    blob((1.5, 0), [negatives flip \ #text(8pt)[@lem-flip]], tint: green, name: <flip>),
    blob((2.8, 0), [wastebasket \ #text(8pt)[@lem-waste]], tint: green, name: <waste>),
    blob((4.2, 0), [big hands \ #text(8pt)[@thm-big]], tint: purple, name: <big>),
    blob((-0.4, 1), [congruence test \ #text(8pt)[@thm-test]], tint: green, name: <test>),
    blob((1.0, 1), [two families \ #text(8pt)[@def-families]], tint: green, name: <fam>),
    blob((2.5, 1), [hands with a 0 \ #text(8pt)[@thm-zero]], tint: green, name: <zero>),
    blob((-0.4, 2), [five moduli \ #text(8pt)[@thm-five]], tint: green, name: <five>),
    blob((3.4, 2), [*single group* \ #text(8pt)[@thm-single]], tint: purple, name: <single>),
    blob((2.0, 3), [*classification* \ #text(8pt)[@thm-class]], tint: purple, name: <class>),
    {
      let carry(..args, paint) = edge(..args, "-|>", stroke: paint)
      carry(<sil>, <test>, green)
      carry(<test>, <five>, green)
      carry(<sil>, <fam>, green)
      carry(<fam>, <single>, green)
      carry(<fam>, <class>, green)
      carry(<flip>, <zero>, luma(45%))
      carry(<waste>, <zero>, luma(45%))
      carry(<zero>, <single>, luma(45%))
      carry(<big>, <single>, purple)
      carry(<single>, <class>, purple)
    },
  ),
  caption: [The map of the proof. Green boxes are proved by hand; purple boxes use the case-checked @thm-big. Arrows carry silent cards and the families (green), the wastebasket (grey) or big hands (purple). @thm-any, for groups that may use any number of center cards, rests on the families, the wastebasket and @thm-big.],
) <fig-map>

*Certificates for @thm-big.* The search emits one expression per case. The checker shares no code with it. It lists the cases itself with `itertools`, parses each expression with Python's own parser, rejects anything but $+$, $-$, $times$ and card values, and compares the cards used with the multiset. @lst-check shows it, apart from the imports, reading the file and printing the count; `multisets` is `itertools.combinations_with_replacement`.

#figure(
  placement: auto,
  kind: raw,
  caption: [The independent checker for @thm-big.],
wide(w: 5.4in)[```python
OPS = {ast.Add: lambda a, b: a + b,
       ast.Sub: lambda a, b: a - b,
       ast.Mult: lambda a, b: a * b}

def value(expr, leaves):
    def ev(n):
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if (isinstance(n, ast.Constant) and type(n.value) is int
                and 1 <= n.value <= 10):
            leaves.append(n.value)
            return n.value
        raise ValueError("illegal expression")
    return ev(ast.parse(expr, mode="eval").body)

def in_families(m):
    three = all(v % 3 == 0 for v in m)
    four = not {2, 6, 10} & set(m) and sum(v % 2 for v in m) <= 1
    return three or four

def check(m, cert):
    leaves = []
    assert value(cert["expr"], leaves) == 10, m
    if cert["uses"] == "all":
        assert Counter(leaves) == Counter(m), m
    else:                             # the one exception
        assert m == (1, 6, 6, 7, 7, 7, 7)
        assert not Counter(leaves) - Counter(m)

for n in range(7, 12):
    for m in multisets(range(1, 11), n):
        if max(Counter(m).values()) > 4 or in_families(m):
            continue
        check(m, certs[m])            # a missing case raises KeyError
# all 265,148 cases certified
```]
) <lst-check>

#idea[what z3 does][z3 is a program that hunts for counterexamples. Given a claim about whole numbers, such as “if $a$ and $b$ are even, then $a + b$ is even,” it looks for numbers that make the claim false, by reasoning about the claim rather than by trying numbers one at a time. When it reports that there are none, the claim holds for _all_ whole numbers, including ones far too big to try.]

*Machine checks of the hand proofs.* Each induction step was stated for all integers and given to the #link("https://github.com/Z3Prover/z3")[z3] solver, which found no counterexample. Congruences are written as $a - b = m k$ with an explicit witness $k$, so no check has to range over “for all” or “there exists”. The solver checks the steps; the induction that strings them together is the usual one on expression trees.

#figure(
  placement: auto,
  caption: [Machine-checked steps.],
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [Result], [What z3 confirmed for all integers],
    table.hline(stroke: 0.4pt),
    [@lem-one], [$+$ and $-$ keep $plus.minus x$; reversed $-$ flips its sign; $times$ by a silent value erases it],
    [@lem-two], [sums, differences and products of $plus.minus x$ and $plus.minus y$ land in $S(x, y)$, which is closed under negation],
    [@thm-five], [the mod-8 test implies the mod-4 test, for all 121 pairs of hand and center card],
    [@lem-waste], [signed sums, differences and products stay in $plus.minus V$],
    [@lem-flip], [the three sign identities used in the step],
    [@lem-size], [$|a plus.minus b| <= |a| + |b|$ and $|a b| = |a| |b|$],
    table.hline(stroke: 0.6pt),
  ),
)

*Independent checks of the impossibilities.* A second encoding, as straight-line programs (pick two unused values, combine them, repeat), confirms with z3 that the ten six-card survivors of @fig-seven have no part making 10. Bit-vectors (binary numbers of a fixed width) of 26 bits are exact here, because six cards of value at most 10 never exceed $11^6 < 2^25$ in absolute value. By @lem-size it is enough to bound the value with every $-$ replaced by $+$; that value plus 1 is at most the product of $c + 1$ over the cards $c$, because $(a + b) + 1$ and $a b + 1$ are both at most $(a + 1)(b + 1)$. A third engine, the classic 24-game search over states, confirms that $#ms(1,6,6,7,7,7,7)$ cannot make 10 with all its cards.

#mnote(inline: true)[Three people do the sum on their own. If all three agree, believe it.]*Search engines.* Three separately written searches agree on the count of 12,362,536; the one published with this note checks every value pattern by brute force in about 35 seconds. The engine reproduces the published counts for the 24 game (1,362 of 1,820) and Make 10 (552 of 715). @thm-single was compared with the original solver on 20,000 random deals, and @thm-zero on all 7,302,660 value patterns with a 0 in the hand, with no disagreements.

*Code.* The count, the certificates and the machine checks can be rerun from the `erase` folder of the code that accompanies this note, with the Python standard library, except the z3 checks, which need the `z3-solver` package. `erase.py` solves a deal, `count.py` counts dead deals (with `--no-zero` for cards 1 to 10), `make_certs.py` writes the certificates in about two and a half minutes and `verify_certs.py` checks them, `simulate.py` reruns every simulation in this note with fixed seeds, and `smt/` holds the z3 scripts and the state-space engine.

*Typesetting.* Typst, with the packages fletcher, lilaq and codly.

// ---------- appendices ----------
#show figure.where(kind: table): set text(size: 10pt)
#counter(heading).update(0)
#set heading(hanging-indent: 0pt)
#set heading(numbering: (..n) => "Appendix " + numbering("A.", ..n.pos()))

= Sample certificates <app-certs>

A few cases of @thm-big and its consequences, each a single expression using every card listed. Most follow the pattern of @sec-big: a ten-maker, plus a homemade 0 or 1 to absorb the rest.

#figure(
  caption: [Single expressions that clear a whole hand.],
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [Cards], [Expression],
    table.hline(stroke: 0.4pt),
    [4 4 5 5 6 8 9 (the sheet's hand)], [$(4 + 6) + (5 - 5) times 4 times 8 times 9$],
    [1 2 3 4 5 6 7], [$(4 + 6) + (1 + 2 - 3) times 5 times 7$],
    [1 1 1 1 2 2 2], [$2 times (2 + 2 + 1) times 1 times 1 times 1$],
    [7 7 7 8 8 8 9], [$(9 + 8 - 7) + (7 - 7) times 8 times 8$],
    [9 9 9 9 10 10 10], [$10 times (10 - 9) + (9 - 9) times 9 times 10$],
    [3 3 3 3 6 6 6 with center 1], [$(3 times 3 + 1) + (6 - 6) times 3 times 3 times 6$],
    [0 2 3 5 7 8 9], [$(3 + 7) + (2 times 5 times 8 times 9) times 0$],
    [1 6 6 7 7 7 7 with center 2], [$(1 + 2 + 7) + (6 - 6) times 7 times 7 times 7$],
    table.hline(stroke: 0.6pt),
  ),
)

= The stuck hands, listed <app-lists>

*The ten survivors of @fig-seven.* These are the ten-free multisets of six nonzero cards outside both families.

#figure(
  caption: [Ten-free sets of six nonzero cards.],
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.hline(stroke: 0.6pt),
    [Cards], [Why],
    table.hline(stroke: 0.4pt),
    [1 1 7 7 7 7 · 1 6 7 7 7 7 · 1 7 7 7 7 8 \ 6 6 7 7 7 7 · 6 7 7 7 7 8 · 7 7 7 7 8 8], [four silent 7s and two cards worth $plus.minus 1$ mod 7; remainder 3 is out of reach (@lem-two)],
    [2 2 9 9 9 9 · 2 7 9 9 9 9 · 7 7 9 9 9 9], [four silent 9s and two cards worth $plus.minus 2$ mod 9; remainder 1 is out of reach],
    [6 6 6 7 7 7], [no remainder rules it out ($7 - 6 - 6 - 6 equiv 3$ mod 7); checked by z3],
    table.hline(stroke: 0.6pt),
  ),
)

*The 23 hands the congruence test misses.* Each hand is dead exactly when every center card has one of the values listed beside it. Every one of them has at least two 0s. The last column counts deals.

#figure(
  caption: [Too little material.],
  text(size: 9.5pt, table(
    columns: (auto, auto, auto, 2.5em, auto, auto, auto),
    align: (left, left, right, center, left, left, right),
    table.hline(stroke: 0.6pt),
    [Hand], [Center from], [Deals], [], [Hand], [Center from], [Deals],
    table.hline(stroke: 0.4pt),
    [0 0 0 0 6 7 7], [1 5 6 7 8], [39,960], [], [0 0 0 0 1 1 1], [1 2 3], [504],
    [0 0 0 6 6 7 7], [0 1 6 7], [18,144], [], [0 0 0 1 6 6 7], [0 7], [384],
    [0 0 0 0 6 6 7], [1 6 7 8], [17,160], [], [0 0 0 0 1 6 6], [6 7], [360],
    [0 0 0 0 1 6 7], [1 6 7], [8,064], [], [0 0 0 0 5 7 7], [6 7], [360],
    [0 0 0 1 6 7 7], [0 6 7], [5,760], [], [0 0 0 6 6 6 7], [0 6 7], [320],
    [0 0 0 0 5 8 8], [4 8 9], [4,680], [], [0 0 6 6 6 7 7], [0 7], [144],
    [0 0 0 0 8 8 9], [4 5 8], [4,680], [], [0 0 0 0 1 1 2], [1 2], [120],
    [0 0 0 0 7 8 8], [4 7 8], [2,664], [], [0 0 0 0 7 7 9], [7 9], [120],
    [0 0 0 0 3 3 3], [3 5 6 9], [2,356], [], [0 0 0 0 1 1 6], [7], [24],
    [0 0 0 0 6 7 8], [6 7], [960], [], [0 0 0 0 6 6 8], [7], [24],
    [0 0 0 0 1 1 7], [6 7], [840], [], [0 0 0 1 1 1 1], [0 2], [20],
    [0 0 6 6 7 7 7], [0 6 7], [720], [], [], [], [],
    table.hline(stroke: 0.6pt),
  )),
)

= Counting deals <app-count>

All counts treat the 44 cards as distinct. Take a value pattern, and let $h_v$ and $c_v$ count the cards of value $v$ in the hand and in the center. The pattern stands for
$ product_(v = 0)^(10) binom(4, h_v) binom(4 - h_v, c_v) $
deals: choose which of the four cards of value $v$ go to the hand, then which of the remaining ones go to the center. Summing this over the 6,861 stuck value patterns gives 12,362,536, and summing over all 17,237,220 patterns gives $binom(44, 7) binom(37, 4)$. The 18,722 possible hands, as value patterns, are what the counting program walks through.

= Answers to the puzzles <app-answers>

*The stuck hand (@sec-stuck).* With $+$, $-$ and $times$ there is no answer: @prop-dead proves it. With $div$ there is one, $8 + 8 div 4 = 10$, which erases two 8s and a 4. The numbers next to 10 that can be made are 9, as $9 + 4 - 4$, and 11, as $8 + 3$.

*1, 6, 7 and 7 (@sec-waste).* The closest is 9, as $7 + 7 - 6 + 1$. Neither 10 nor 11 can be made, so above 10 the nearest is 12, as $6 + 7 - 1$.

*1, 1, 1 and 3 (@sec-waste).* The biggest is 9, as $(1 + 1 + 1) times 3$. @prop-ones explains why nothing bigger is possible.

*1 to 7 (@sec-big).* One answer is $(4 + 6) + (1 + 2 - 3) times 5 times 7 = 10$. The bracket $1 + 2 - 3$ is a homemade 0, which swallows the 5 and the 7. There are many others.

#[
#set par(justify: false)
*More puzzles (@sec-puzzles).*
+ (a)~$2 times 3 + 4$. (b)~$(1 + 4) times 2$. (c)~$3 times 6 - 8$. (d)~$(9 - 4) times 2$.
+ (a)~$2 times 2 times 2 + 2$. (b)~$(1 + 1) times (1 + 4)$. (c)~$3 times 3 + 9 - 8$. (d)~$4 times 7 - 9 - 9$. With $div$, three of the cards in (c) would already do: $9 + 3 div 3 = 10$.
+ (a) The hand has three odd cards, 1, 3 and 7, and no center card is odd. With $+$ and $-$, every group holds an even number of odd cards (@prop-parity), so one odd card is always left over. (b)~$7 times 1 + 3$, then $2 + 8$, then $4 + 6$. Multiplying by 1 changes nothing, so the 1 can join the 7 and the 3 without changing their total.
+ $(1 + 1) times 5$ with the center 5, $3 times 3 + 1$ with the center 3, and $6 + 6 + 3 - 5$ with the center 5 again. Center cards stay, so using the 5 twice is allowed. There are other answers.
+ (a)~$2 times 5 times (9 - 8)$, with a homemade 1. Plain $8 + 9 - 5 - 2$ works too. (b)~$(3 + 7) + (5 - 5) times 8 times 9 times 9$, with a homemade 0.
+ $1 + 1 + 8$ with the center 8, then $3 + 3 + 4$ and $3 times 4 + 3 - 5$, both with the center 4. Without $times$ it cannot be done: the hand has seven odd cards and the center has none (@prop-parity).
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
    [$a equiv b space (mod m)$], [“$a$ leaves the same remainder as $b$ when divided by $m$.” The clock of @fig-clock.],
    [$plus.minus$], [“plus or minus”: either sign.],
    [${4, 4, 8}$], [A bag of cards, a _multiset_: repeats count, order does not.],
    [$x in A$], [“$x$ is in $A$.”],
    [$A union B$], [“$A$ together with $B$.”],
    [$|v|$], [The size of $v$, ignoring its sign: $|-3| = 3$.],
    [$binom(n, k)$], [“$n$ choose $k$”: the number of ways to pick $k$ things from $n$.],
    [$V(M)$], [Every number the cards of $M$ can make, each card used once.],
    [$S(x, y)$], [What two loud cards $x$ and $y$ can make at most, mod $m$: 0, $plus.minus x$, $plus.minus y$, $plus.minus x plus.minus y$ and $plus.minus x y$.],
    [modulus], [The number of hours on the clock: the $m$ in “mod $m$”.],
    [deal], [Seven hand cards and four center cards.],
    [value pattern], [A deal with the suits forgotten: only the card values.],
    [expression], [A calculation that uses each of its cards exactly once.],
    [part], [Some of the cards, at least one.],
    [group], [Hand cards and at most one center card that make 10 and are erased together.],
    [stuck], [The hand cannot be cleared, however the cards are grouped.],
    [dead], [No card at all can be erased.],
    [homemade 0, homemade 1], [A 0 made from two equal cards, as $7 - 7$; a 1 made from two cards that differ by 1, as $9 - 8$.],
    [silent, loud], [A card is silent mod $m$ if $m$ divides it, loud otherwise.],
    [3-family, 4-family], [All multiples of 3; or no 2, 6 or 10 and at most one odd card.],
    [ten-free], [No part of two or more cards makes 10.],
    [certificate], [A written answer that anyone can check.],
    [lemma, theorem], [A stepping stone; a main result. A proposition is a single fact.],
    [$qed$], [The end of a proof.],
    table.hline(stroke: 0.6pt),
  ),
)

#block(breakable: phone)[
#heading(numbering: none, level: 1)[Sources]
#set text(size: 9pt)
- #link("https://kcm.nku.edu/mathfactfluency/docs/Games/en-new/Game9-InstructionPage-Erase.pdf")[Erase instruction page], _Math Fact Fluency_ companion resource (ASCD, 2019).
- #link("https://kcm.nku.edu/mathfactfluency/docs/Games/en/MathFactFluency-Game%209-Erase-teacher.pdf")[Erase teacher page], with the original rules and the deck.
- L. Alcock et al., #link("https://arxiv.org/abs/2011.11767")[Arithmetic Expression Construction], ISAAC 2020.
- J.-M. Alliot, #link("https://arxiv.org/abs/1502.05450")[The (Final) Countdown], 2015.
- M. Mahajan and A. Tawari, #link("https://www.sciencedirect.com/science/article/pii/S0304397517307478")[Sums of read-once formulas: how many summands are necessary?]
- #link("https://en.wikipedia.org/wiki/Integer_complexity")[Integer complexity] (Wikipedia) and #link("https://oeis.org/A005245")[OEIS A005245].
- #link("https://www.4nums.com/24_game/facts/")[Facts about the 24 game], 4nums.com.
- #link("https://ja.wikipedia.org/wiki/%E3%83%86%E3%83%B3%E3%83%91%E3%82%BA%E3%83%AB")[テンパズル] (Japanese Wikipedia).
- #link("https://mkaz.blog/math/krypto-game-analysis/")[Krypto game analysis], mkaz.blog.
- On the writing: D. E. Knuth, T. Larrabee and P. M. Roberts, #link("https://cs.stanford.edu/~knuth/papers/cs1193.pdf")[_Mathematical Writing_] (1989); P. R. Halmos, _How to Write Mathematics_ (1970); S. Strogatz, Writing about math for the perplexed and the traumatized, _Notices AMS_ 61 (2014); R. E. Schwartz, Communicating mathematics to children, _Notices AMS_ 63 (2016).
]
