#import "@preview/cetz:0.4.2"
#import "@preview/lilaq:0.6.0" as lq

// Every number in this note that a script computes is read from results.json, which report.py
// writes from peggle.py, stopping.py and variations.py. The plots read better.csv, priors.csv,
// walk.csv and formats.csv, which those scripts write. The rules, the table of results, the
// checks and the open questions are the README's shared sections (from docs/, filled by
// tools/sync_docs.py). Numbers typed by hand are inputs (p, N, e, seeds), steps of hand
// arithmetic in proofs and answers, and figures quoted from cited sources.
#import "/tools/shared_docs.typ": shared, fmt
#let R = json("results.json")
#let readme = read("README.md")
#let O = R.outcomes
#let PR = R.priors
#let F = R.formats

// x rounded to d decimals, with trailing zeros: fix(0.12, 3) is "0.120". report.py stores
// floats unrounded, and tests/test_peggle.py checks that none sits on a rounding tie at the
// places printed, where Typst (which rounds the shortest decimal form) and Python (the binary
// value) could differ.
#let fix(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}
#let pct(x, d) = fix(x * 100, d) + "%"
#let words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")

// ---------- editions (as in the Erase paper) ----------
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

// HTML edition: one style sheet, inlined, so the page is a single self-contained file
#let html-css = " :root { color-scheme: light; } body { max-width: 42rem; margin: 0 auto; padding: 1.2rem 1.1rem 4rem; font-family: \"Iowan Old Style\", \"Palatino Linotype\", Palatino, Georgia, serif; font-size: 1.12rem; line-height: 1.55; color: #161616; background: #fff; } h1 { font-size: 2rem; line-height: 1.2; text-align: center; margin: 1.2rem 0 0.4rem; } p.subtitle { text-align: center; font-style: italic; margin: 0; } p.date { text-align: center; color: #666; font-size: 0.9rem; margin: 0.3rem 0 1.6rem; } h2 { font-size: 1.45rem; margin: 2.2rem 0 0.6rem; line-height: 1.25; } h3 { font-size: 1.15rem; margin: 1.6rem 0 0.5rem; } p { margin: 0.7rem 0; } a { color: #0072B2; text-decoration: none; } a:hover { text-decoration: underline; } svg { max-width: 100%; height: auto !important; vertical-align: middle; } svg[style*=\"display: block\"] { margin: 0.8rem auto; } figure { margin: 1.6rem 0; text-align: center; overflow-x: auto; } figure.thm, figure:has(div.thm) { text-align: left; } figcaption { font-size: 0.9rem; color: #444; margin-top: 0.5rem; text-align: center; } table { border-collapse: collapse; margin: 0.4rem auto; font-size: 0.9rem; line-height: 1.35; border-top: 1.5px solid #222; border-bottom: 1.5px solid #222; } td, th { padding: 0.25rem 0.55rem; text-align: left; vertical-align: top; } tr:first-child td, tr:first-child th { font-weight: 600; border-bottom: 1px solid #222; } aside.box { border-left: 4px solid; padding: 0.6rem 0.95rem; margin: 1.3rem 0; border-radius: 0 5px 5px 0; font-size: 1rem; line-height: 1.45; } aside.idea { background: #fdf6e7; border-color: #E69F00; } aside.tryit { background: #f0f1f3; border-color: #5b6770; } aside.box p { margin: 0.35rem 0; } .boxlabel { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; margin-right: 0.45em; } aside.idea .boxlabel { color: #8a5a00; } aside.tryit .boxlabel { color: #3d464d; } pre { text-align: left; overflow-x: auto; font-size: 0.78rem; line-height: 1.45; background: #fafafa; border: 1px solid #ddd; border-radius: 4px; padding: 0.7rem 0.8rem; } code { font-size: 0.9em; overflow-wrap: anywhere; } pre code { overflow-wrap: normal; } math { font-size: 1.06em; font-family: \"STIX Two Math\", \"Latin Modern Math\", \"Cambria Math\", math; } math[display=\"block\"] { margin: 0.9rem 0; overflow-x: auto; overflow-y: hidden; } ul, ol { padding-left: 1.4rem; } li { margin: 0.25rem 0; } .thm { margin: 1.2rem 0; } blockquote.epigraph { margin: 0.2rem 0 1.4rem auto; max-width: 30rem; text-align: right; font-style: italic; font-size: 0.95rem; } p.mnote { font-size: 0.88rem; font-style: italic; color: #555; margin-left: 1.2rem; } blockquote.epigraph footer { font-style: normal; font-size: 0.85rem; color: #555; } "
// Erase's tables have no table.header, so its style sheet treats the first row as the header.
// Here a table.header becomes <thead>; keep the first body row plain and rule off the header.
// Long URLs in the sources may break, so that no line is wider than a phone's screen.
#let html-css = html-css + " thead th { font-weight: 600; } thead tr:last-child th { border-bottom: 1px solid #222; } thead + tbody tr:first-child td { font-weight: normal; border-bottom: none; } body { overflow-wrap: break-word; } "
#context if target() == "html" { html.elem("style", html-css) }
// HTML has no align: keep the content, drop the alignment (CSS centers figures and drawings)
#show align: it => context if target() == "html" { it.body } else { it }

// ---------- look ----------
#set document(title: "Who Is the Better Player?", date: none)  // no date: every build is identical
#set page(..if phone { (width: 4.4in, height: 7.6in, margin: (x: 0.32in, top: 0.42in, bottom: 0.55in)) }
  else { (paper: "us-letter", margin: (x: 1.55in, y: 1.1in)) }, numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)
#show math.equation.where(block: false): box
#set par(justify: true, leading: 0.6em, spacing: 1em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set block(above: 1.6em, below: 0.8em)
#show heading: set text(hyphenate: false)
// appendix headings read "Appendix A. ..."; references to them read "Appendix A"
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading and type(el.numbering) == function {
    box(link(el.location(), [Appendix #numbering("A", ..counter(heading).at(el.location()))]))
  } else { it }
}
#show link: set text(fill: rgb("#0072B2"))
#show figure.caption: set text(size: 10pt)
// one-line captions centred, longer ones set as a justified paragraph
#show figure.caption: it => context if target() == "html" { it } else { layout(size => {
  if measure(it).width <= size.width { align(center, it) } else { set par(justify: not phone); align(left, it) }
}) }
#show math.equation.where(block: true): it => if phone { fitw(it) } else { it }
// tables with a 1fr column fill the width they get, so lay those out at 4.8in; tables
// of auto columns keep their natural width (up to 7in) and are scaled down whole
// the shared table of results (tools/shared_docs.typ) fills the width it gets
#show table: it => if phone and it.at("label", default: none) != <shared-table> {
  let fr = type(it.columns) == array and it.columns.any(c => type(c) == fraction)
  wide(it, w: if fr { 4.8in } else { 7in })
} else { it }
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set block(breakable: false)
#set table(stroke: none, inset: (x: 6pt, y: 3.5pt))
#show table: set par(justify: false)
#show table.cell.where(y: 0): set text(weight: "semibold")

// One meaning per color, in every figure and table:
#let blue = rgb("#0072B2")    // the uniform prior, and the 95% rule built on it
#let green = rgb("#009E73")   // the Bayes factor rule
#let red = rgb("#D55E00")     // lead rules: Wald's test, and win by two
#let purple = rgb("#AA4499")  // the skeptical prior
#let orange = rgb("#E69F00")  // Idea boxes only
#let slate = rgb("#5b6770")   // Try it boxes only
// black dashed: the Jeffreys prior; grey: neutral (sample paths, other formats, other points)

// theorem-like statements share one counter and can be referenced with @label
#show figure.where(kind: "thm"): it => block(
  width: 100%, breakable: true, above: 1.2em, below: 1.1em,
  align(left)[*#it.supplement #it.counter.display(it.numbering)*#it.body],
)
#let env(kind, title: none, body) = figure(
  kind: "thm", supplement: kind, numbering: "1", outlined: false,
  [#if title != none [ (#title)]*.* #body],
)
#let proposition = env.with([Proposition])
#let theorem = env.with([Theorem])
#let corollary = env.with([Corollary])
#let proof(title: [Proof], body) = block(above: 0.7em, below: 1.2em)[_#title._ #body#h(0.6em)#h(1fr)$qed$]

// boxes: "Idea" explains a tool where it is first used; "Try it" invites the reader in
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
// a Try-it puzzle with its rating on Knuth's scale, such as "10" or "20"; best: the one most worth doing
#let tryit(rating, best: false, body) = sidebox("Try it", slate, [[#rating]#if best [ $triangle.filled.r$]], body)
#let tag(label, col) = text(size: 8.5pt, weight: "bold", fill: col.darken(35%), tracking: 0.04em, label)

// a column heading in two words, set on two lines on a phone page so that the table stays wide enough to read
#let hd(a, b) = if phone [#a\ #b] else [#a #b]

// a blank CSV cell is a gap in a plotted line
#let num(s) = if s == "" { float.nan } else { float(s) }

// ---------- title ----------
#context if target() == "html" {
  html.elem("h1")[Who Is the Better Player?]
  html.elem("p", attrs: (class: "subtitle"))[Peggle duels, coin flips, and when to stop playing]
  html.elem("p", attrs: (class: "date"))[November 2026]
} else [
  #align(center)[
    #set par(justify: false)
    #text(size: 21pt, weight: "bold")[Who Is the Better Player?]
    #v(2pt)
    #text(size: 11.5pt, style: "italic")[Peggle duels, coin flips, and when to stop playing]
    #v(4pt)
    #text(size: 9.5pt, fill: luma(90))[November 2026]
  ]
  #v(6pt)
]

#shared(readme, "summary")

*How to read this.* Sections #ref(<sec-game>, supplement: none) to #ref(<sec-bayes>, supplement: none) set up the game and give the coin-flip answer, and need only arithmetic, as do the match formats and ratings of Sections #ref(<sec-formats>, supplement: none) and #ref(<sec-elo>, supplement: none). Sections #ref(<sec-billiard>, supplement: none) to #ref(<sec-wald>, supplement: none) prove why the answers hold and what happens when the friends stop whenever they like. @sec-model asks what the model leaves out, and Sections #ref(<sec-relatives>, supplement: none) to #ref(<sec-open>, supplement: none) give relatives, checks and open questions. Boxes marked #tag("IDEA", orange) explain a tool the first time it is used. Boxes marked #tag("TRY IT", slate) are puzzles to try before reading on, with answers and the source of each puzzle in @app-answers. Each puzzle is rated on the scale of Knuth's _Art of Computer Programming_ @knuth: 00 is immediate, 10 takes a minute, 20 about a quarter of an hour and 30 a few hours, and $triangle.filled.r$ marks the one most worth doing. It is fine to skip any proof, or to stop anywhere. Words and symbols, with how to read them aloud, are listed in @app-words. In every figure, blue marks the uniform prior and its 95% rule, green the Bayes factor rule, red a rule of the form "play until someone leads by $c$", purple a skeptical prior and black dashes the Jeffreys prior; grey is neutral.

Every number is exact, computed with fractions or whole numbers, unless it is marked as floating point or simulated.

= The duel <sec-game>

In Peggle a ball is shot from the top of a board of pegs, and the aim is to clear the orange ones. In the duel mode of Peggle Deluxe @readme, "two players take turns shooting the ball" on the same board. Each starts with six balls. A shot that hits no orange peg costs the shooter 25% of their score, and a ball caught in the Free Ball Bucket earns points instead of a free ball. The duel ends when someone clears the last orange peg, which sets off a bonus round called Extreme Fever, or when both players run out of balls. "The player with the most points at the end of the game wins." Hitting a green peg gives the shooter's Peggle Master a special power; in a duel only one green peg is on the board at a time @wiki. Peggle 2 lets the host choose the number of rounds and green pegs @windowscentral. The rules sheet does not say who shoots first, or what happens if the scores are equal; this note ignores ties.

So a duel always has a winner, as do chess without draws and most racket sports. The model for the rest of the note is the simplest one.

*The model.* The other player wins a game with chance $1 - p$, so the first player is better when $p > 1 - p$. These are the rules as modeled:

#shared(readme, "rules")

The games are flips of a coin with bias $p$, and the share of games won, $hat(p) = k slash n$ (read "p hat"), estimates $p$.

The model may not fit Peggle well. The player who shoots first may have more orange pegs to aim at, and the one who clears the last peg gets Extreme Fever, so the order of turns may matter; players also improve. @sec-model measures what these effects do to the answers.

= How many games pin down the chance?

Concentration bounds say how far $hat(p)$ is likely to stray from $p$. One Chernoff-type bound is
$ P(|p - hat(p)| >= delta) <= 2 exp(-(delta^2)/(delta + 2) n), $
so $n >= (delta + 2) / delta^2 log(2 / gamma)$ games make $hat(p)$ fall within $delta$ of $p$ with probability at least $1 - gamma$. Hoeffding's inequality gives the sharper $2 exp(-2 n delta^2)$ and so needs only $n >= log(2 slash gamma) slash (2 delta^2)$. @tab-bounds gives both.

#figure(
  table(
    columns: 3, align: (left, right, right),
    table.hline(),
    [within], [Chernoff-type bound], [Hoeffding],
    table.hline(stroke: 0.5pt),
    [$delta = 0.1$], [#fmt(R.coins.bounds.d10.at(0)) games], [#fmt(R.coins.bounds.d10.at(1)) games],
    [$delta = 0.05$], [#fmt(R.coins.bounds.d05.at(0)) games], [#fmt(R.coins.bounds.d05.at(1)) games],
    table.hline(),
  ),
  caption: [Games needed for $hat(p)$ to be within $delta$ of $p$ with probability at least 95%.],
) <tab-bounds>

Either way, a close match between nearly equal players takes hundreds of games to measure. But the question was not how big $p$ is, only whether it exceeds one half, and that has a direct answer.


= The Bayesian answer <sec-bayes>

#idea[prior and posterior][A _prior_ says what we believe about $p$ before any game is played; a _posterior_ says what we believe after seeing the record. Bayes's rule turns one into the other: each possible value of $p$ is weighted by how likely it makes the record, and the weights are rescaled to add up to 1. If six wins in ten are likely for some value of $p$, that value gains weight; if unlikely, it loses weight.]

Give $p$ a _uniform prior_ on $[0, 1]$, so that no value is favored before any game is played. After $k$ wins in $n$ games, Bayes's rule gives the posterior density
$ f(p | k, n) = (binom(n, k) p^k (1 - p)^(n - k)) / (integral_0^1 binom(n, k) q^k (1 - q)^(n - k) dif q) = (p^k (1 - p)^(n - k)) / (B(k + 1, n - k + 1)), $
the Beta$(k + 1, n - k + 1)$ density. The chance of being the better player is its mass above one half:
$ P(p > 1/2 | k, n) = 1 / B(k + 1, n - k + 1) integral_(1 slash 2)^1 q^k (1 - q)^(n - k) dif q. $

#idea[the Beta distribution][A _Beta$(a, b)$ distribution_ is a way of spreading belief over the chances between 0 and 1, with density proportional to $p^(a - 1) (1 - p)^(b - 1)$. Its mean is $a slash (a + b)$, and it gets narrower as $a + b$ grows. Beta(1, 1) is flat: the uniform prior. After $k$ wins and $n - k$ losses the uniform prior becomes Beta$(k + 1, n - k + 1)$, so the numbers in the brackets count wins and losses, each plus one. The number $B(a, b)$ in the formula above, the _beta function_, only rescales the density so that its total is 1.]

#block(breakable: false)[
That integral has a plain-language value, proved in @sec-billiard.

*The chance of being the better player after $k$ wins in $n$ games equals the chance of at most $k$ heads in $n + 1$ flips of a fair coin:*
$ P(p > 1/2 | k, n) = 1 / 2^(n + 1) sum_(j = 0)^k binom(n + 1, j). $
]

#let (six-num, six-den) = R.coins.six_of_ten
For six wins in ten games the formula gives $#(2 * six-num) slash #(2 * six-den) = #six-num slash #six-den approx #fix(R.coins.six_of_ten_value, 3)$. The claim of being better is more likely right than wrong, but about one time in four a player with that record is in fact the weaker one. `peggle.py` checks the identity against the Beta integral, computed exactly with fractions, for every $k$ and every $n$ up to 12.

@tab-wins shows how many wins it takes to be 95% sure.

#figure(
  table(
    columns: 7, align: (left, ..(right,) * 6),
    table.hline(),
    [games $n$], [10], [20], [50], [100], [200], [1,000],
    table.hline(stroke: 0.5pt),
    ..{
      let ns = (10, 20, 50, 100, 200, 1000)
      let w = ns.map(n => R.coins.wins_needed.at(str(n)))
      (
        [wins needed], ..w.map(((k, _)) => [#k]),
        [share], ..ns.zip(w).map(((n, (k, _))) => [#pct(k / n, 1)]),
        [chance of being better], ..w.map(((_, b)) => [#fix(b, 3)]),
      )
    },
    table.hline(),
  ),
  caption: [The fewest wins in $n$ games that make $P(p > 1 slash 2) >= 0.95$, uniform prior. Exact.],
) <tab-wins>

The share of wins needed falls toward one half, roughly like $1 slash 2 + 0.82 slash sqrt(n)$, the normal approximation to the coin flips.

*A one-line approximation.* The curves in @fig-better look like normal distribution functions. Near $k = n slash 2$ the posterior has mean $(k + 1) slash (n + 2)$ and standard deviation $1 slash (2 sqrt(n + 3))$, so $P(p > 1 slash 2)$ is close to the normal distribution function $Phi(z)$ (read "phi of z") at $z = 2 (k - n slash 2) sqrt(n + 3) slash (n + 2)$. The logistic curve $1 slash (1 + e^(-1.702 z))$ stays within 0.01 of $Phi(z)$, which gives
$ P(p > 1 slash 2 | k, n) approx 1 / (1 + e^(-a(n) (k - n slash 2))), #if phone { linebreak() } else { $quad$ } a(n) = (3.404 sqrt(n + 3)) / (n + 2) approx 3.4 / sqrt(n). $
An earlier version of this note used the fitted slope $3 n^(-0.48)$. Over every $k$ and every $n <= 300$, the largest difference from the exact chance is #fix(R.coins.logistic_error.posterior.at(0), 3) with $a(n)$ and #fix(R.coins.logistic_error.fitted.at(0), 3) with the fitted slope; for $30 <= n <= 300$ it is #fix(R.coins.logistic_error.posterior.at(1), 3) and #fix(R.coins.logistic_error.fitted.at(1), 3). `peggle.py` computes all four exactly.

#figure(
  {
    let rows = csv("better.csv", row-type: dictionary)
    let curve(n, col) = {
      let rs = rows.filter(r => int(r.n) == n)
      (rs.map(r => float(r.share)), rs.map(r => float(r.at(col))))
    }
    let shades = (luma(70%), luma(45%), luma(15%))
    let marks = ("o", "s", "d")
    visb(fitw(lq.diagram(
      width: 10cm, height: 3.6cm,
      xlabel: [share of games won, $k slash n$], ylabel: [$P(p > 1 slash 2)$],
      xlim: (0.2, 0.8), ylim: (0, 1),
      legend: (position: left + top),
      ..((10, 50, 250).zip(shades, marks).map(((n, c, m)) => {
        let (xs, ys) = curve(n, "exact")
        let (_, ls) = curve(n, "logistic")
        (
          lq.plot(xs, ys, mark: m, mark-size: 2.6pt, stroke: none, color: c, label: [$n = #n$]),
          lq.plot(xs, ls, mark: none, stroke: 0.7pt + c),
        )
      }).flatten()),
    )))
  },
  caption: [Exact chance of being better (marks) and the approximation with slope $a(n)$ (lines), for 10, 50 and 250 games. The more games, the steeper the step at one half.],
) <fig-better>

= Why coin flips: Bayes's billiard table <sec-billiard>

The coin-flip identity looks like an accident of algebra. It is not. Bayes @bayes imagined balls rolled across a level table, and Stigler @stigler shows how that picture carries his argument. Here is the argument in the form we need, as a proof by counting points.

#proposition(title: [coin flips])[
  Give $p$ the uniform prior, and suppose the first player has won $k$ of $n$ games. Then the chance that $p > 1 slash 2$ equals the chance of at most $k$ heads in $n + 1$ flips of a fair coin.
] <prop-coins>

#proof[
  Drop $n + 1$ points at random on the line from 0 to 1, each uniformly and independently. Call the first one $p$, and let each of the other $n$ points be a game, won when it lands left of $p$. Given $p$, each game is won with chance $p$, independently, which is the model with a uniform prior. The number of wins is the number of other points left of $p$, so "$k$ wins" says that $p$ is the #box[$(k + 1)$th] point from the left. All $n + 1$ points are alike, so given that, $p$ is distributed as the #box[$(k + 1)$th] smallest of $n + 1$ uniform points (which is the Beta$(k + 1, n - k + 1)$ posterior above). That point lies right of $1 slash 2$ exactly when at most $k$ of the $n + 1$ points lie left of $1 slash 2$, and each point does so with chance one half, independently: $n + 1$ fair coin flips.
]

#figure(
  visb(fitw(cetz.canvas(length: 1cm, {
    import cetz.draw: *
    line((0, 0), (10, 0), stroke: 0.6pt)
    for (x, t) in ((0, $0$), (5, $1 slash 2$), (10, $1$)) {
      line((x, -0.12), (x, 0.12), stroke: 0.6pt)
      content((x, -0.45), t)
    }
    line((5, -0.12), (5, 1.0), stroke: (paint: luma(50%), dash: "dashed", thickness: 0.6pt))
    let others = (0.6, 1.4, 2.3, 3.1, 3.8, 5.6, 6.9, 7.7, 8.4, 9.1)
    for x in others { circle((x, 0), radius: 0.11, fill: luma(55%), stroke: none) }
    circle((6.3, 0), radius: 0.15, fill: blue, stroke: none)
    content((6.3, 0.5), text(10pt, fill: blue)[$p$])
    content((2.5, 0.75), text(9pt)[6 games won: left of $p$])
    content((8.2, 0.75), text(9pt)[4 lost])
  }))),
  caption: [Eleven random points on $[0, 1]$. The blue one is $p$; the other ten are games, won when they fall left of $p$. Six wins put $p$ seventh from the left, so $p > 1 slash 2$ exactly when at most six of all eleven points fall left of the dashed line.],
) <fig-billiard>

The proof is a bijection in disguise: the same eleven points are read once as a prior and a record, and once as eleven coin flips. Nothing was computed. The integral of @sec-bayes is the probability of one event, and the coin flips are the same event seen from the dashed line.

#tryit("10")[Use the identity to find the chance of being the better player after two wins in two games, and after one win in two games.]

*Laplace's rule and Pólya's urn.* The same points answer a second question. After $k$ wins in $n$ games, what is the chance of winning the next one? A new point lands left of $p$ when it falls in one of the $k + 1$ gaps left of $p$ among the $n + 2$ gaps made by the $n + 1$ points, so the chance is $(k + 1) slash (n + 2)$, Laplace's rule of succession. Those chances are the draws from an urn that starts with one ball of each color and gets an extra ball of the drawn color after every draw, the urn of Eggenberger and Pólya @polya. Friends whose $p$ comes from the uniform prior look, game by game, like that urn.

*More points for more caution.* A Beta$(a, b)$ prior with whole numbers $a$ and $b$ is the distribution of the #box[$a$th] smallest of $a + b - 1$ uniform points. Adding them to the table proves a little more.

#corollary(title: [coin flips for other priors])[
  Give $p$ the prior Beta$(a, b)$ with whole numbers $a, b >= 1$, and suppose the first player has won $k$ of $n$ games. Then the chance that $p > 1 slash 2$ equals the chance of at most $k + a - 1$ heads in $n + a + b - 1$ flips of a fair coin.
] <cor-coins>

#proof[
  The prior Beta$(a, b)$ is the uniform prior after $a - 1$ wins and $b - 1$ losses, since both have density proportional to $p^(a - 1) (1 - p)^(b - 1)$. So the record of $k$ wins in $n$ games, under this prior, gives the same posterior as $k + a - 1$ wins in $n + a + b - 2$ games under the uniform prior. @prop-coins finishes the proof.
]

In words, a Beta$(a, a)$ prior is a record that starts at $a - 1$ to $a - 1$ instead of 0 to 0. `variations.py` checks @cor-coins against the Beta integral, exactly, for $a = 1, 2, 3, 12$ and all records of up to eight games.

= Which prior? <sec-prior>

The uniform prior treats a 90% player as being as likely as a 50% one. Friends who play each other a lot are usually close, so it is fair to ask how much the verdict depends on this choice. Three priors, each symmetric about one half, are compared here.

- The *uniform prior*, Beta(1, 1).
- The *Jeffreys prior*, Beta(1/2, 1/2), which piles a little more belief near 0 and 1 @jeffreys. Its answers do not depend on whether the chance is measured as $p$ or on another scale, such as the log-odds of @sec-elo.
- A *skeptical prior*, Beta(12, 12), with mean 1/2 and standard deviation exactly 1/10. By @cor-coins it is a record that starts at 11 wins to 11. Under it, a player better than 60% is a long shot: Beta(12, 12) puts chance #fix(PR.skeptic_above, 2) above 0.6.

#figure(
  {
    let rows = csv("priors.csv", row-type: dictionary)
    let xs = rows.map(r => float(r.x))
    let col(c) = rows.map(r => float(r.at(c)))
    let ys = col("uniform")
    let above = rows.filter(r => float(r.x) >= 0.5)
    visb(fitw(lq.diagram(
      width: 10cm, height: 3.8cm,
      xlabel: [chance $p$ that the first player wins a game], ylabel: [posterior density],
      xlim: (0, 1), ylim: (0, 5),
      legend: (position: left + top),
      lq.fill-between(above.map(r => float(r.x)), above.map(r => float(r.uniform)), fill: blue.lighten(80%)),
      lq.plot(xs, ys, mark: none, stroke: 1pt + blue, label: [uniform]),
      lq.plot(xs, col("jeffreys"), mark: none, stroke: (paint: black, thickness: 0.8pt, dash: "dashed"), label: [Jeffreys]),
      lq.plot(xs, col("skeptical"), mark: none, stroke: 1pt + purple, label: [skeptical]),
      lq.vlines(0.5, stroke: (paint: luma(50%), thickness: 0.5pt, dash: "dotted")),
    )))
  },
  placement: auto,
  caption: [Posterior densities after six wins in ten games. The shaded area, #fix(PR.six_of_ten.uniform, 3), is the uniform prior's chance that the first player is better. The skeptical prior keeps $p$ near one half and gives #fix(PR.six_of_ten.skeptical, 3).],
) <fig-priors>

After six wins in ten games the three chances of being better are #fix(PR.six_of_ten.uniform, 3), #fix(PR.six_of_ten.jeffreys, 3) and #fix(PR.six_of_ten.skeptical, 3) (@fig-priors). The Jeffreys value is not a fraction: its density involves square roots, and the chance comes out as $1 slash 2 + #PR.jeffreys_part.at(0) slash (#PR.jeffreys_part.at(1) pi)$, exactly. `variations.py` computes the fraction next to $pi$ by stepping through the beta function's recurrence from Beta(1/2, 1/2), whose mass above one half is exactly 1/2, and checks it against numerical integration to 12 decimal places for all records of up to 14 games.

#figure(
  table(
    columns: 8, align: (left, ..(right,) * 7),
    table.hline(),
    [prior], [6 of 10], [$n = 10$], [20], [50], [100], [200], [1,000],
    table.hline(stroke: 0.5pt),
    ..(("uniform", text(fill: blue)[uniform]), ("jeffreys", [Jeffreys]), ("skeptical", text(fill: purple)[skeptical])).map(((key, name)) => (
      name, [#fix(PR.six_of_ten.at(key), 3)], ..PR.fewest.at(key).map(k => [#k]),
    )).flatten(),
    table.hline(),
  ),
  caption: [The chance of being better after six wins in ten games, and the fewest wins in $n$ games that make each prior 95% sure. Exact; the Jeffreys chance is $1 slash 2 + #PR.jeffreys_part.at(0) slash (#PR.jeffreys_part.at(1) pi)$.],
) <tab-priors>

@tab-priors shows how far the verdict moves. With a hundred games or more the three priors nearly agree: they ask for #PR.fewest.uniform.at(3) or #PR.fewest.skeptical.at(3) wins in 100 and #PR.fewest.uniform.at(5) in 1,000. With a handful of games they disagree, and the prior decides. The skeptic needs #words.at(PR.fewest.skeptical.at(0)) wins in ten games before being 95% sure; the Jeffreys prior is satisfied by #words.at(PR.shortest.jeffreys) in #words.at(PR.shortest.jeffreys), and the uniform prior by #words.at(PR.shortest.uniform) in #words.at(PR.shortest.uniform). The difference matters most when the friends stop as soon as they are sure, as the next section shows.

= Playing until someone is sure <sec-stop>

#idea[stopping rules][A _stopping rule_ decides after each game whether to play another, looking only at the games so far. "Play exactly 100 games" is one. "Play until someone is 95% sure, or until 100 games" is another, and it is the one most friends use without saying so. The cap of $N$ games keeps the duel from going on forever.]

Suppose the friends play until one of them is 95% sure of being better, or until $N$ games. The rule looks only at $k$ and $n$, so the number of win-loss sequences that first stop at each $(k, n)$ does not depend on $p$; a dynamic program counts them, and each has chance $p^k (1 - p)^(n - k)$. `stopping.py` sums these exactly (@tab-stop), agrees with seeded simulation, and with a binomial test in place of the posterior reproduces Table 1 of Armitage, McPherson and Rowe @armitage.

#tryit("12")[Playing until someone is 95% sure under the uniform prior, what is the shortest possible duel? Use @prop-coins.]

#idea[error rates][Fix a pair of players, so that $p$ is one number, and imagine them replaying the whole match many times. The _error rate_ is the share of those matches that end with the wrong player declared better. For two equal players every verdict is an error. An error rate is a promise about one pair; a posterior is a statement about one record.]

#figure(
  table(
    columns: 6, align: (left, right, right, right, right, right),
    table.hline(),
    [$p$], [$N$], hd[better][declared], hd[worse][declared], hd[no][verdict], hd[mean][games],
    table.hline(stroke: 0.5pt),
    ..(("0.5", "p50", 100), ("0.5", "p50", 1000), ("0.55", "p55", 100), ("0.55", "p55", 1000),
       ("0.6", "p60", 1000), ("0.7", "p70", 1000)).map(((p, key, cap)) => {
      let (first, second, neither, games) = O.posterior.at(key).at(str(cap))
      let equal = key == "p50"
      ([#p], [#fmt(cap)], [#fix(if equal { first + second } else { first }, 3)],
       if equal [] else [#fix(second, 3)], [#fix(neither, 3)], [#fix(games, 1)])
    }).flatten(),
    table.hline(),
  ),
  caption: [Play until one player is 95% sure, or $N$ games. For $p = 0.5$ the first column is the chance that either player is declared better. Exact.],
) <tab-stop>

Between equal players someone is declared better #pct(R.equal_verdict.posterior.at("100"), 0) of the time with $N = 100$ and #pct(R.equal_verdict.posterior.at("1000"), 0) with $N = 1000$, against #pct(R.one_look.at("100"), 1) and #pct(R.one_look.at("1000"), 1) for one look after $N$ games. The threshold on $k - n slash 2$ grows like $0.82 sqrt(n)$, and by the law of the iterated logarithm a fair random walk passes any multiple of $sqrt(n)$, so the chance tends to one as $N$ grows. The error rate of the 95% rule is not 5%, and stopping early is why.

*Yet each verdict means what it says.* Draw $p$ from the uniform prior and play by the same rule: #pct(R.uniform_prior.at("100").at(2), 1) of verdicts are wrong with $N = 100$ and #pct(R.uniform_prior.at("1000").at(2), 1) with $N = 1000$, exactly. The reason is that the stopping rule cannot change the posterior.

#proposition(title: [stopping does not change the posterior])[
  Give $p$ any prior, and let the friends stop by any rule that looks only at the games played so far. If play stops after a particular sequence with $k$ wins in $n$ games, the posterior for $p$ is the same as if exactly $n$ games had been planned from the start.
] <prop-honest>

#proof[
  Given $p$, a particular sequence of $n$ results has chance $p^k (1 - p)^(n - k)$. Whether play stopped at the end of it, and not before, is decided by the sequence itself, so "play followed this sequence and stopped" is the same event as "the first $n$ games went this way", for each sequence on which the rule stops at game $n$. Its chance is still $p^k (1 - p)^(n - k)$. Bayes's rule weighs each value of $p$ by that chance, so the posterior is the one for $n$ planned games.
]

Edwards, Lindman and Savage put it as "the rules governing when data collection stops are irrelevant to data interpretation" @els. So each verdict of the 95% rule has posterior chance at least 0.95 of being right, and averaged over the prior at most 5% of verdicts are wrong, whatever the rule and the cap. @tab-calib splits the verdicts of the rule with $N = 1000$ by when they came. Early verdicts are a little safer than 5%, because the posterior jumps past 0.95 rather than landing on it, and late ones come close to 5%.

#figure(
  table(
    columns: 3, align: (left, right, right),
    table.hline(),
    [verdict reached at game], [chance], [share wrong],
    table.hline(stroke: 0.5pt),
    ..([1 to 10], [11 to 100], [101 to 1,000]).zip(R.calibration).map(((range, (chance, wrong))) => (
      range, [#fix(chance, 3)], [#pct(wrong, 1)],
    )).flatten(),
    table.hline(),
  ),
  caption: [The 95% rule with $N = 1000$, with $p$ drawn from the uniform prior: the chance of a verdict within each range of games, and the share of those verdicts that name the worse player. Exact.],
) <tab-calib>

The two numbers, #pct(R.equal_verdict.posterior.at("100"), 0) and #pct(R.uniform_prior.at("100").at(2), 1), answer different questions. The error rate asks about one pair of players replaying the same match; for an equal pair, every verdict is wrong. The posterior asks about all the pairs a uniform prior could produce, and almost none of them is exactly equal. The 5% is an average over the prior, not a promise for each $p$; at $p = 0.55$ the worse player is declared better #pct(O.posterior.p55.at("1000").at(1), 0) of the time.

Laplace's rule of succession is the same calculation read one game at a time: after $k$ wins in $n$ games the next is won with chance $(k + 1) slash (n + 2)$, the posterior mean. Multiplying these along a sequence gives $k! (n - k)! slash (n + 1)!$, the prior average of $p^k (1 - p)^(n - k)$ and the weight the Bayesian check gives each sequence. Under these chances the posterior is a martingale: when play stops, by any rule, its average is still 1/2.

*The prior matters more when you stop early.* Each prior of @sec-prior gives its own 95% rule. @tab-stop-priors runs all three with the same dynamic program. The Jeffreys rule can stop after three straight wins and declares equal players unequal more often; the skeptical rule waits longer and errs less at $p = 0.55$.

#figure(
  placement: auto,
  table(
    columns: 7, align: (left, right, right, right, right, right, right),
    table.hline(),
    table.header(
      table.cell(rowspan: 2, align: bottom)[prior], table.cell(rowspan: 2, align: bottom + right)[$N$],
      table.cell(colspan: 2, align: center)[equal players], table.cell(colspan: 3, align: center)[$p = 0.55$],
      table.hline(start: 2, end: 4, stroke: 0.4pt), table.hline(start: 4, end: 7, stroke: 0.4pt),
      [verdict], [games], [better], [worse], [games],
    ),
    table.hline(stroke: 0.5pt),
    ..(for cap in ("100", "1000") {
      for (name, rows) in ((text(fill: blue)[uniform], O.posterior), ([Jeffreys], PR.outcomes.jeffreys),
                           (text(fill: purple)[skeptical], PR.outcomes.skeptical)) {
        let e = rows.p50.at(cap)
        let g = rows.p55.at(cap)
        ((name, [#fmt(int(cap))], [#fix(e.at(0) + e.at(1), 3)], [#fix(e.at(3), 1)],
          [#fix(g.at(0), 3)], [#fix(g.at(1), 3)], [#fix(g.at(3), 1)]),)
      }
    }).flatten(),
    table.hline(),
  ),
  caption: [Play until the posterior under each prior is 95% sure for either player, or $N$ games. Exact.],
) <tab-stop-priors>

Which prior is right is a question about the friends, not about the games. If the friends' $p$ really comes from the skeptical prior, the uniform 95% rule is overconfident: #pct(PR.wrong_if_skeptical.uniform.at("100"), 1) of its verdicts are wrong with $N = 100$ and #pct(PR.wrong_if_skeptical.uniform.at("1000"), 1) with $N = 1000$, exactly, while the skeptic's own rule is wrong #pct(PR.wrong_if_skeptical.skeptical.at("100"), 1) and #pct(PR.wrong_if_skeptical.skeptical.at("1000"), 1) of the time. @prop-honest protects the prior you use, not one you do not believe.

= A verdict that is safe for equal players <sec-bayes-factor>

The 95% rule keeps its promise on average over the prior, and the uniform prior gives $p = 1 slash 2$ no weight at all. To protect equal players we need a number whose behavior is known when $p = 1 slash 2$ exactly. The *Bayes factor* compares how likely the record is under the two views. After $k$ wins in $n$ games, the uniform prior gives the record the chance $integral_0^1 p^k (1 - p)^(n - k) dif p = 1 slash ((n + 1) binom(n, k))$ (the `prior_path_chance` of `stopping.py`), and equal players give it $2^(-n)$.

#block(breakable: false, width: 100%)[
  Their ratio is
  $ B_n = 2^n / ((n + 1) binom(n, k)). $
]
Six wins in ten games give $B_10 = #R.bayes_factor.six_of_ten_fraction.at(0) slash #R.bayes_factor.six_of_ten_fraction.at(1) approx #fix(R.bayes_factor.six_of_ten, 2)$: that record is more likely between equal players. The *Bayes factor rule* plays until $B_n >= 20$, or until $N$ games, and then declares better whoever has won more games. It takes #R.bayes_factor.wins_needed.at("100") wins in 100 games, or #R.bayes_factor.wins_needed.at("1000") in 1,000, against #R.coins.wins_needed.at("100").at(0) and #R.coins.wins_needed.at("1000").at(0) for the 95% rule.

#idea[fair games][A _fair game_, or _martingale_, is a running total whose average after the next step equals its value now, whatever has happened so far. A gambler's fortune at even odds on fair coin flips is one. The key fact about fair games is that no rule for when to quit can make one favorable on average.]

*The Bayes factor is a fair bet against equal players.* Suppose $p = 1 slash 2$. The next game is won or lost with chance one half each, and the average of the two new factors is the old one:
$ 1/2 B_(n + 1)(k + 1) + 1/2 B_(n + 1)(k) = B_n (k). $
(Both sides equal the prior average of $(2p)^k (2 - 2p)^(n - k)$, and the next game multiplies that by $2p$ or $2 - 2p$, which average to 1.) So $B_n$ is a *test martingale*: a nonnegative fair game that starts at 1. It is the wealth of a gambler who starts with one unit and bets at even odds, spreading the stake over every $p$ by the uniform prior @shafer2011 @savi. A fair game cannot be made favorable by choosing when to stop. Stopped at a verdict or at $N$ games, $B$ still averages exactly 1; `stopping.py` checks this with exact fractions for several caps up to 200.

#proposition(title: [Ville's inequality, here])[
  Suppose $p = 1 slash 2$. However long the friends play, the Bayes factor rule declares someone better with chance at most $1 slash 20$.
] <prop-ville>

#proof[
  Fix a cap $N$. When play stops, $B >= 20$ if there is a verdict and $B >= 0$ otherwise, and $B$ averages 1. So $1 >= 20 P("verdict")$. The bound does not depend on $N$, so it holds for unlimited play too.
]

Ville proved this for every test martingale @ville @savi, and Shafer and Vovk @shafervovk build probability itself on such betting games. The exact chances are #fix(R.equal_verdict.at("bayes factor").at("100"), 4) for $N = 100$ and #fix(R.equal_verdict.at("bayes factor").at("1000"), 4) for $N = 1000$, against #fix(R.equal_verdict.posterior.at("100"), 3) and #fix(R.equal_verdict.posterior.at("1000"), 3) for the 95% rule (@tab-rules). The chance grows with $N$, to #fix(R.bayes_factor.far.at("10000"), 4) at $N = 10000$ (computed in floating point), but it never reaches 0.05. Between equal players $B_n$ drifts to 0 on almost every path that never reaches 20 (by the law of the iterated logarithm it is at most about $(log n)^(1 + epsilon) slash sqrt(n)$), so in unlimited play the chance of a verdict is exactly 1 divided by the average factor at the verdict. That factor jumps past 20 rather than landing on it: the first possible verdict, eight wins in eight games, has $B_8 = #R.bayes_factor.first_verdict.at(0) slash #R.bayes_factor.first_verdict.at(1) approx #fix(R.bayes_factor.first_verdict_value, 1)$.

The two rules answer different questions. The 95% rule says that, of all the pairs of players a uniform prior could produce, about 95% of those declared better are better; it says nothing about a pair who are exactly equal. The Bayes factor rule promises that a pair of equal players is declared unequal at most one time in twenty, whenever the friends stop. It is also a Bayesian rule, with a prior that takes equality seriously: give "equal" chance one half, and spread the other half uniformly over $p$. The posterior chance of "equal" is then $1 slash (1 + B_n)$, and $B_n >= 20$ means that it has fallen to $1 slash 21$ or less. The two priors disagree about records such as #R.coins.wins_needed.at("1000").at(0) wins in 1,000 games, where the uniform prior is #pct(R.coins.wins_needed.at("1000").at(1), 1) sure while $B_1000 approx #fix(R.bayes_factor.at_527, 2)$ favors equality. This disagreement is Lindley's paradox @lindley.

The safety has a price. #let bf55 = O.at("bayes factor").p55.at("1000")
At $p = 0.55$ the Bayes factor rule wrongly favors the worse player only #pct(bf55.at(1), 1) of the time, but within 1,000 games it reaches a verdict only #pct(bf55.at(0), 0) of the time, after #fix(bf55.at(3), 0) games on average.

The Bayes factor rule is safe for every cap at once. When the cap is fixed in advance, the 95% rule can be made safe for equal players by asking for more certainty. The chance of a verdict between equal players falls as the threshold rises, and it changes only at thresholds equal to a posterior that occurs. So bisection, with the dynamic program at each step, finds the least strict threshold that keeps the chance at or below 5%. #let S1 = R.strict.at("100")
#let S2 = R.strict.at("1000")
The posterior must reach #S1.threshold with $N = 100$ and #S2.threshold with $N = 1000$; a threshold of 0.997037 or 0.9987002, a little lower, lets the chance pass 5%. Equal players are then declared unequal with chance #fix(S1.equal.at(0), 4) for both caps, exactly. These rules need #S1.wins_needed wins in 100 games, or #S2.wins_needed in 1,000. They spend the whole 5% within the cap, so at $p = 0.55$ they declare the better player better with chance #fix(S1.p55.at(0), 3) and #fix(S2.p55.at(0), 3), against #fix(O.at("bayes factor").p55.at("100").at(0), 3) and #fix(O.at("bayes factor").p55.at("1000").at(0), 3) for the Bayes factor rule, after #fix(S1.p55.at(2), 1) and #fix(S2.p55.at(2), 1) games on average (@tab-rules).

#figure(
  placement: top,
  text(10pt, table(
    columns: 7, align: (left, right, right, right, right, right, right),
    inset: (x: 4.5pt, y: 3.5pt),
    table.hline(),
    table.header(
      table.cell(rowspan: 2, align: bottom)[rule], table.cell(rowspan: 2, align: bottom + right)[$N$],
      table.cell(colspan: 2, align: center)[equal players], table.cell(colspan: 3, align: center)[$p = 0.55$],
      table.hline(start: 2, end: 4, stroke: 0.4pt), table.hline(start: 4, end: 7, stroke: 0.4pt),
      [verdict], [games], [better], [worse], [games],
    ),
    table.hline(stroke: 0.5pt),
    ..{
      let row(name, cap, rows) = {
        let (e1, e2, _, eg) = rows.p50.at(cap)
        let (g1, g2, _, gg) = rows.p55.at(cap)
        (name, [#fmt(int(cap))], [#fix(e1 + e2, 3)], [#fix(eg, 1)], [#fix(g1, 3)], [#fix(g2, 3)], [#fix(gg, 1)])
      }
      let strict(cap) = {
        let t = R.strict.at(cap)
        ([posterior #t.threshold], [#fmt(int(cap))], [#fix(t.equal.at(0), 3)], [#fix(t.equal.at(1), 1)],
         [#fix(t.p55.at(0), 3)], [#fix(t.p55.at(1), 3)], [#fix(t.p55.at(2), 1)])
      }
      (
        ..row([posterior 95%], "100", O.posterior), ..row([posterior 95%], "1000", O.posterior),
        ..row([Bayes factor 20], "100", O.at("bayes factor")), ..row([Bayes factor 20], "1000", O.at("bayes factor")),
        ..strict("100"), ..strict("1000"),
        ..row([lead of #R.wald.lead], "1000", O.lead),
        [lead of #R.wald.lead], [none], [1], [#R.wald.equal_games], [#fix(1 - R.wald.worse, 3)], [#fix(R.wald.worse, 3)], [#fix(R.wald.games, 1)],
      )
    },
    table.hline(),
  )),
  caption: [Four stopping rules: the posterior rule at 95% and at the thresholds that keep equal players' chance of a verdict at 5%, the Bayes factor rule and Wald's test. For equal players, the chance that someone is declared better and the mean number of games; for $p = 0.55$, the chances that the better or the worse player is declared better, and the mean number of games. Exact, by the dynamic program; the last row, with no cap, by the gambler's ruin.],
) <tab-rules>

= Wald's test: play until someone leads by 15 <sec-wald>

Suppose the friends only want to tell $p = 0.45$ from $p = 0.55$, a 55–45 edge for one of them. After $k$ wins in $n$ games the likelihood ratio of the two is $(11 slash 9)^(2k - n)$, which depends only on the lead $2k - n$. Wald's sequential probability ratio test @wald stops when the ratio leaves a fixed interval, so it is *play until one player leads by $c$ games*.

#idea[random walks][Keep score as a single number, the first player's lead: up one after a win, down one after a loss. The lead wanders like a person taking steps left or right at random, a _random walk_. When $p = 1 slash 2$ the steps are fair and the walk has no drift; when $p = 0.55$ it drifts upward by $2p - 1 = 0.1$ per game on average, slowly enough that luck dominates for dozens of games.]

#figure(
  {
    let rows = csv("walk.csv", row-type: dictionary)
    let xs = rows.map(r => float(r.n))
    let seeds = ("seed1", "seed2", "seed3", "seed4", "seed5")
    let even = rows.filter(r => calc.even(int(r.n)) and r.posterior != "")
    let ex = even.map(r => float(r.n))
    let ep = even.map(r => float(r.posterior))
    let bf = rows.filter(r => calc.even(int(r.n)) and r.bayes != "")
    visb(fitw(lq.diagram(
      width: 11cm, height: 5.2cm,
      xlabel: [games played], ylabel: [first player's lead],
      xlim: (0, 260), ylim: (-34, 34),
      ..seeds.map(s => lq.plot(xs, rows.map(r => num(r.at(s))), mark: none, stroke: 0.6pt + luma(45%))),
      lq.hlines(15, -15, stroke: 1.1pt + red),
      lq.plot(ex, ep, mark: none, stroke: 0.9pt + blue),
      lq.plot(ex, ep.map(y => -y), mark: none, stroke: 0.9pt + blue),
      lq.plot(bf.map(r => float(r.n)), bf.map(r => float(r.bayes)), mark: none, stroke: 0.9pt + green),
      lq.plot(bf.map(r => float(r.n)), bf.map(r => -float(r.bayes)), mark: none, stroke: 0.9pt + green),
    )))
  },
  caption: [Five simulated duels at $p = 0.55$ (grey; seeds 1 to 5 of `variations.py`, not chosen). Wald's test stops at the red walls, a lead of 15 either way. The 95% rule stops at the blue curves, which start at a lead of 4 and widen like $sqrt(n)$; two of the five duels touch them at game 4, one in each direction. The Bayes factor rule stops at the green curves.],
) <fig-walk>

The lead moves one step up or down after each game until it hits a wall at $+c$ or $-c$: the gambler's ruin. @fig-walk shows five such walks.

#proposition(title: [the gambler's ruin])[
  Suppose that each game is won by the first player with chance $p$, independently, and the friends play until one of them leads by $c$ games. Let $r = (1 - p) slash p$. If $p != 1 slash 2$, the first player reaches $+c$ first with chance $1 slash (1 + r^c)$, and play lasts $c (1 - r^c) slash ((1 + r^c)(2p - 1))$ games on average. If $p = 1 slash 2$, each player reaches the wall first with chance one half, and play lasts $c^2$ games on average.
] <prop-ruin>

#proof[
  Let $h(x)$ be the chance of reaching $+c$ before $-c$ when the lead is $x$. The next game moves the lead to $x + 1$ with chance $p$ or to $x - 1$ with chance $q = 1 - p$, so $h(x) = p h(x + 1) + q h(x - 1)$, with $h(c) = 1$ and $h(-c) = 0$. Write the gaps as $d(x) = h(x + 1) - h(x)$. The equation says $p d(x) = q d(x - 1)$, so each gap is $r$ times the one before. The $2c$ gaps from $-c$ to $c$ add up to 1, and the $c$ gaps from $-c$ to 0 add up to $h(0)$, so
  $ h(0) = (1 + r + dots.c + r^(c - 1)) / (1 + r + dots.c + r^(2c - 1)) = 1 / (1 + r^c). $
  For the length, the lead minus $(2p - 1)$ times the number of games played is a fair game, because each game adds $2p - 1$ to the lead on average. Play stops after a number of games $T$ with a finite average, so at the end the lead averages $(2p - 1)$ times the average of $T$. The lead at the end is $+c$ with chance $h(0)$ and $-c$ otherwise, so the average of $T$ is $c (2 h(0) - 1) slash (2p - 1)$, which is the formula. When $p = 1 slash 2$, $h(0) = 1 slash 2$ by symmetry, and the lead squared minus the number of games is a fair game, since each fair step adds 1 to the square on average; at the end the square is $c^2$.
]

At $p = 0.55$, $r = 9 slash 11$, the worse player is declared better with chance $9^15 slash (9^15 + 11^15) = #fix(R.wald.worse, 4)$, after #fix(R.wald.games, 1) games on average. A lead of 14 would err #pct(R.wald.lead_14_worse, 1) of the time, so #R.wald.lead is the smallest lead that errs at most 5%. Because the lead moves one game at a time, it lands exactly on $plus.minus 15$, and Wald's approximations, which ignore overshoot, are exact here. `variations.py` checks @prop-ruin a second way that shares nothing with the formulas: it writes the equations for $h(x)$ and the mean length at every lead from $-14$ to 14 and solves them by elimination in fractions, for several $p$ and $c$. The dynamic program with a cap of 1,000 games agrees too (@tab-rules): play beyond the cap can only add to each column.

The test is the best of its kind, in a precise sense.

#theorem(title: [Wald and Wolfowitz @waldwolfowitz])[
  Suppose that the games are independent with a common chance $p$, and that $p$ is either 0.45 or 0.55. Consider any rule for stopping and declaring a winner whose mean number of games is finite at both values, which declares the worse player better with chance at most #fix(R.wald.worse, 4) when $p = 0.45$ and at most #fix(R.wald.worse, 4) when $p = 0.55$. Then its mean number of games is at least #fix(R.wald.games, 1) at $p = 0.45$, and at least #fix(R.wald.games, 1) at $p = 0.55$.
] <thm-ww>

The 95% rule with $N = 1000$ also needs about #fix(O.posterior.p55.at("1000").at(3), 0) games at $p = 0.55$, but errs #pct(O.posterior.p55.at("1000").at(1), 1) of the time. @thm-ww says nothing about other values of $p$, and Wald's test is not a test of equality: at $p = 1 slash 2$ it declares someone better after #R.wald.equal_games games on average, as it must, since it only chooses between 0.45 and 0.55. Rules that keep the mean length small at every $p$, and so near $p = 1 slash 2$ too, solve the problem of Kiefer and Weiss @kieferweiss, and Weiss @weiss gives a way to compute them for exactly this binomial case. This note does not compute them.

= Match formats <sec-formats>

Real matches use simpler rules: best of three, best of seven, win by two. Each is a stopping rule with a fixed chance of crowning the better player. How much do they amplify a 55% edge?

A *best-of-$n$* match, with $n$ odd, is won by whoever first wins $(n + 1) slash 2$ games. It crowns the same player as playing all $n$ games and counting, so the better player wins it with chance $sum_(j >= (n + 1) slash 2) binom(n, j) p^j (1 - p)^(n - j)$. *Win by two* means playing until someone leads by two: the lead rule with $c = 2$.

#tryit("20", best: true)[Two players start win-by-two at 0–0, and the first wins each game with chance $p$. Explain why the first player wins the match with chance $p^2 slash (p^2 + (1 - p)^2)$, and work it out for $p = 0.55$.]

#figure(
  {
    let rows = csv("formats.csv", row-type: dictionary)
    let xs = rows.map(r => float(r.p))
    let col(c) = rows.map(r => float(r.at(c)))
    visb(fitw(lq.diagram(
      width: 10cm, height: 4.4cm,
      xlabel: [chance $p$ of winning one game], ylabel: [chance of winning the match],
      xlim: (0.5, 0.75), ylim: (0.5, 1),
      legend: (position: right + bottom),
      lq.plot(xs, col("single"), mark: none, stroke: (paint: luma(70%), thickness: 0.8pt, dash: "dotted"), label: [one game]),
      lq.plot(xs, col("best3"), mark: none, stroke: 0.8pt + luma(60%), label: [best of 3]),
      lq.plot(xs, col("best7"), mark: none, stroke: 0.8pt + luma(25%), label: [best of 7]),
      lq.plot(xs, col("lead2"), mark: none, stroke: (paint: red, thickness: 0.9pt, dash: "dashed"), label: [win by 2]),
      lq.plot(xs, col("lead15"), mark: none, stroke: 1.1pt + red, label: [win by 15]),
      lq.vlines(0.55, stroke: (paint: luma(50%), thickness: 0.5pt, dash: "dotted")),
    )))
  },
  caption: [How match formats amplify an edge in single games. Read up from the dotted line at $p = 0.55$: best of seven gives #fix(F.at("best of 7").at(0), 3), win by two #fix(F.at("win by 2").at(0), 3), win by 15 #fix(F.at("win by 15").at(0), 3). Exact formulas, plotted in floating point.],
) <fig-formats>

#figure(
  table(
    columns: 4, align: (left, right, right, right),
    table.hline(),
    [format], [better player wins], [mean games], [slope at $p = 1 slash 2$],
    table.hline(stroke: 0.5pt),
    ..{
      let slope((a, b)) = if b == 1 { str(a) } else { str(a) + "/" + str(b) }
      let row(name, key) = {
        let (win, games, s) = F.at(key)
        (name, [#fix(win, 3)], [#fix(games, 2)], [#slope(s)])
      }
      let (win, games, s) = F.at("best of fewest")
      (
        ..row([one game], "best of 1"), ..row([best of 3], "best of 3"), ..row([best of 5], "best of 5"),
        ..row([best of 7], "best of 7"), ..row(text(fill: red)[win by 2], "win by 2"),
        ..row(text(fill: red)[win by 15], "win by 15"),
        [best of #F.fewest_best_of], [#fix(win, 3)], [#fix(games, 2)], [#fix(s, 1)],
      )
    },
    table.hline(),
  ),
  caption: [Match formats at $p = 0.55$: the chance that the better player wins, the mean number of games when play stops as soon as the match is decided, and how steeply the first column rises near even players. Exact; the last slope is rounded.],
) <tab-formats>

@tab-formats gives the numbers. Best of seven turns a 55% edge into #pct(F.at("best of 7").at(0), 1), and win by two into #pct(F.at("win by 2").at(0), 1), in fewer games. The last column measures amplification near even players: a best-of-$n$ match has slope $n binom(n - 1, (n - 1) slash 2) slash 2^(n - 1)$, which grows like $sqrt(2n slash pi)$, while win by $c$ has slope exactly $c$ and lasts $c^2$ games on average between equal players. The dynamic program of @sec-stop, run with the rule "stop at $(n + 1) slash 2$ wins", reproduces the first two columns exactly for every best-of-$n$ up to 13.

What this says about tournaments is sobering. A best-of-seven final between a 55% player and a 45% player goes to the weaker one #pct(1 - F.at("best of 7").at(0), 1) of the time. For the better player to win 95% of the time, a fixed format needs best of #F.fewest_best_of, and even stopping as soon as it is decided that takes #fix(F.at("best of fewest").at(1), 1) games on average. Win by 15 does as well in #fix(R.wald.games, 1): a lead rule is a best-of-$n$ match that does not waste games on a lopsided score, and @thm-ww says no rule does better. Short matches are good at entertaining and poor at measuring.

= Ratings: Elo and Bradley–Terry <sec-elo>

Chess and many online games turn records into ratings. In the Elo system @elo, a player rated $D$ points above another is expected to win with chance
$ p = 1 / (1 + 10^(-D slash 400)). $
The formula is the Bradley–Terry model @bradleyterry: give each player a strength, and let each win with chance proportional to their strength, with strength $10^("rating" slash 400)$. On this scale a 55% edge is $400 log_10(11 slash 9) approx #fix(R.elo.gap_55, 1)$ points, and 400 points is $p = 10 slash 11$: ten wins for every loss.

#tryit("05")[What chance of winning does a 200-point gap give? Write $10^(-1 slash 2) approx 0.316$.]

The odds $p slash (1 - p)$ are $10^(D slash 400)$, so the $r$ of @prop-ruin is $10^(-D slash 400)$, and a lead of $c$ errs with chance at most 5% exactly when $r^c <= 1 slash 19$, that is, when
$ c >= (400 log_10 19) / D approx #fix(R.elo.lead_constant, 1) / D. $
The lead needed is inversely proportional to the rating gap (@tab-elo). Two players 400 points apart are separated by win by two: #let G4 = R.elo.gap_400
the weaker one takes the match with chance $#G4.worse.at(0) slash #G4.worse.at(1)$, after $#G4.games.at(0) slash #G4.games.at(1) approx #fix(G4.games_value, 2)$ games on average.

#figure(
  table(
    columns: 6, align: (right, right, right, right, right, right),
    table.hline(),
    hd[rating][gap], [$p$], hd[lead][needed], hd[its][error], hd[mean][games], hd[best][of],
    table.hline(stroke: 0.5pt),
    ..R.elo.rows.map(((gap, p, lead, worse, games, best)) => (
      [#if gap == calc.round(gap) { str(int(gap)) } else { fix(gap, 1) }], [#fix(p, 3)], [#lead],
      [#fix(worse, 3)], [#fix(games, 1)], [#best],
    )).flatten(),
    table.hline(),
  ),
  caption: [Rating gaps on the Elo scale: the chance $p$ of the stronger player, the smallest lead that errs at most 5%, its error, the mean number of games, and the shortest best-of-$n$ match that the stronger player wins with chance at least 95%. Floating point, except the rows for 0.55 and 10/11, which are exact.],
) <tab-elo>

The 400 in the formula is a convention, and real games do not follow it exactly. Glickman and Jones @glickmanjones fitted 225,621 rated chess games and found that the higher-rated player won less often than the formula says, and suggested "replacing 400 in the winning expectancy formula with 561". #let (p561, w561, g561) = R.elo.scale_561
On that scale a 400-point gap means $p approx #fix(p561, 3)$; win by two still suffices, erring #pct(w561, 1) of the time in #fix(g561, 2) games on average. Chess also has draws, which Peggle does not.

= When the chances are not constant <sec-model>

The model of @sec-game says that every game has the same chance $p$, independently. Two ways this can fail matter for Peggle duels.

*Who shoots first.* Turn order may matter in a duel: the first shooter may have more orange pegs to aim at, and the last orange peg brings Extreme Fever. The rules sheet does not say how the first shooter is chosen. Suppose, for a pair of equal players, that whoever shoots first wins with chance $e$, so that the first player wins a game with chance $e$ or $1 - e$ depending on who starts. No one is better; the question is how often a rule says otherwise. Three house rules for who starts each game after the first are compared, with the first player starting game 1.

#tryit("15")[If the winner of each duel shoots first in the next one, will a long record look more streaky than coin flips, or less?]

- *Alternate.* The players take turns to start. The games are independent but not alike: the first player's chance alternates between $e$ and $1 - e$.
- *Winner shoots first.* Each game repeats the last winner with chance $e$, so wins come in streaks.
- *Loser shoots first.* Each game repeats the last winner with chance $1 - e$, so streaks are broken up.

A dynamic program over wins, games and the last winner gives exact chances (@tab-shooter); `variations.py` checks it against all sequences of up to ten games.

#figure(
  placement: auto,
  table(
    columns: 5, align: (left, right, right, right, right),
    table.hline(),
    hd[who][starts], [$e$], hd[one look,][100 games], hd[stop,][$N = 100$], hd[stop,][$N = 1000$],
    table.hline(stroke: 0.5pt),
    ..(("no edge", "0.5", "none"), ("alternate", "0.6", "alternate e60"), ("winner", "0.6", "winner e60"),
       ("loser", "0.6", "loser e60"), ("alternate", "0.7", "alternate e70"), ("winner", "0.7", "winner e70"),
       ("loser", "0.7", "loser e70")).map(((who, e, key)) => (
      [#who], [#e], ..R.changing.shooter.at(key).map(x => [#fix(x, 4)]),
    )).flatten(),
    table.hline(),
  ),
  caption: [Equal players when whoever shoots first wins with chance $e$: the chance that someone is declared better by one look at the uniform posterior after 100 games, and by the 95% rule with a cap of $N$ games. Exact.],
) <tab-shooter>

Alternating is harmless, and even helps a little. For one look, that is a theorem.

#proposition(title: [Hoeffding @hoeffding, for one look])[
  Suppose that the games are independent and that the first player wins game $i$ with chance $p_i$, where the average of $p_1, dots, p_n$ is 1/2. Then one look at the posterior after $n$ games declares someone better no more often than it does when every $p_i$ is 1/2.
] <prop-hoeffding>

#proof(title: [Proof, using Hoeffding's theorem])[
  The look declares no one exactly when the number of wins $S$ lies in an interval of whole numbers $b <= S <= c$ around $n slash 2$, so $b <= n slash 2 <= c$. Hoeffding's Theorem 5 says that, among independent trials whose chances average $1 slash 2$, the chance of $b <= S <= c$ is smallest when every chance is $1 slash 2$. So the chance of a verdict, one minus that, is largest then.
]

Hoeffding's paper shows more: with the chances fixed in advance, the number of wins is never more spread out than for coin flips with the same average. So for players who alternate, a one-look verdict treats them as a single pair with $p$ equal to the average of $p_"first"$ and $p_"second"$, and the error rates computed for that $p$ are, if anything, too pessimistic. For stopping rules there is no such theorem, but the exact chances in @tab-shooter move the same way. Letting the winner shoot first is different: the games are no longer independent, streaks are longer than coin flips allow, and equal players are declared unequal far more often, #pct(R.changing.shooter.at("winner e60").at(1), 0) of the time within 100 games when $e = 0.6$. A pair who play by that house rule should alternate instead, or use a much stricter rule.

*Players who improve.* Over a season, $p$ drifts. Take the simplest drift: the first player wins with chance 0.45 in the first 500 games and 0.55 in the next 500, so the average is one half. #let (D1, D2) = (R.changing.drift.at("45to55"), R.changing.drift.at("40to60"))
One look after 1,000 games declares someone better with chance #fix(D1.one_look, 3), below the #fix(R.one_look.at("1000"), 3) of a constant $p = 1 slash 2$, as @prop-hoeffding promises. The 95% rule with $N = 1000$ declares the improving player better with chance #fix(D1.stop.at(0), 3) and the other player with chance #fix(D1.stop.at(1), 3), after #fix(D1.stop.at(2), 1) games on average; with 0.4 and then 0.6 the chances are #fix(D2.stop.at(0), 3) and #fix(D2.stop.at(1), 3), after #fix(D2.stop.at(2), 1) games. Both are exact.

These numbers do not say that the rule is wrong; they say what the verdict is about. The rule usually stops early, while the other player really is better. A constant-$p$ verdict summarizes the games actually played, weighted as they were played. It is not a statement about who is better now, nor a forecast. When skill drifts, the natural fix is a rating that forgets old games, such as the dynamic paired-comparison models of Glickman @glickman, and this note does not analyze those.

= Relatives <sec-relatives>

The questions here have a long history in statistics.

- *Repeated tests.* Armitage, McPherson and Rowe @armitage computed how often repeated significance tests reject a true null hypothesis, the frequentist side of @tab-stop; their Table 1 is reproduced by `stopping.py` (@sec-check).
- *The likelihood principle.* Edwards, Lindman and Savage @els argue that stopping rules do not affect the interpretation of data, which is @prop-honest; the same paper notes that a sharp null hypothesis can be supported by data that a classical test rejects, the disagreement Lindley @lindley made famous.
- *Testing by betting.* Ville's inequality @ville, test martingales @shafer2011 and safe anytime-valid inference @savi give the guarantee of @prop-ville for any test martingale, not only the Bayes factor.
- *Sequential tests.* Wald's test @wald and its optimality @waldwolfowitz are @sec-wald; Kiefer and Weiss @kieferweiss and Weiss @weiss study tests that are efficient at every $p$.
- *Paired comparisons.* The Bradley–Terry model @bradleyterry underlies Elo ratings @elo @glickmanjones; Davidson and Beaver @davidsonbeaver extend it to an order effect, such as an edge for whoever moves first, the effect of @tab-shooter; Glickman @glickman lets strengths drift over time.
- *Bayes's table and Pólya's urn.* The billiard argument of @prop-coins is Bayes's own @bayes @stigler, and Laplace's rule is the urn of Eggenberger and Pólya @polya.
- *Unequal trials.* Hoeffding @hoeffding shows that independent trials with unequal chances are less spread out than coin flips with the same average, which is @prop-hoeffding.

= How this was checked <sec-check>

Every claim in this note is known in one of four ways (@tab-status).

#figure(
  placement: auto,
  table(
    columns: (auto, 1fr), align: (left, left),
    table.hline(),
    [how it is known], [claims],
    table.hline(stroke: 0.5pt),
    [proved here], [the coin-flip identity and its version for Beta$(a, b)$ priors (@prop-coins, @cor-coins); stopping does not change the posterior (@prop-honest); Ville's inequality for the Bayes factor rule (@prop-ville); the gambler's ruin (@prop-ruin); the lead needed for a rating gap; win by two],
    [proved elsewhere], [Wald and Wolfowitz (@thm-ww); Hoeffding's theorem behind @prop-hoeffding; the law of the iterated logarithm],
    [exact computation], [every chance and mean in Tables #ref(<tab-wins>, supplement: none), #ref(<tab-priors>, supplement: none) to #ref(<tab-formats>, supplement: none) and #ref(<tab-shooter>, supplement: none), with fractions and whole numbers; the strict thresholds #R.strict.at("100").threshold and #R.strict.at("1000").threshold; the drift chances; the Jeffreys chance as a fraction over $pi$],
    [floating point], [the Bayes factor rule at $N = 10000$ (#fix(R.bayes_factor.far.at("10000"), 4)); the rows of @tab-elo other than 0.55 and 10/11; the logistic approximation and its errors; the curves in the figures],
    [simulation], [only as checks, and the five paths of @fig-walk],
    table.hline(),
  ),
  caption: [How each claim is known.],
) <tab-status>

@tab-results lists the main results and how each is known. The checks behind them are these.

#shared(readme, "checks")

#[
// on a phone the table may break across pages, and stays where it is introduced
#show figure: set block(breakable: phone)
#figure(
  placement: if phone { none } else { auto },
  kind: table,
  caption: [The main results and how each is known. Exact unless marked.],
  [
    #set table.cell(breakable: false)
    #text(size: if phone { 9pt } else { 10pt }, shared(readme, "results", columns: (1.1fr, 1fr, 1fr)))
  ],
) <tab-results>
]

*Code.* `peggle.py` computes the coin-flip chances and the logistic approximation; `stopping.py` the stopping rules, the Bayes factor, the strict thresholds and Wald's test; `variations.py` the priors, the chain check, the match formats, the ratings and the changing chances. All three use only the Python standard library, and run in about half a minute together. `report.py` writes every number this note quotes to a results file, which the note reads.

*Typesetting.* Typst, with the packages cetz and lilaq.

= Questions left open <sec-open>

#shared(readme, "open")

// ---------- appendices ----------
#pagebreak(weak: true)
#show figure.where(kind: table): set text(size: 10pt)
#counter(heading).update(0)
#set heading(hanging-indent: 0pt)
#set heading(numbering: (..n) => "Appendix " + numbering("A.", ..n.pos()))

= Answers to the puzzles <app-answers>

Each answer gives the puzzle's rating and its source; "new" marks a puzzle made for this note.

*Two games (@sec-billiard), rated 10.* Two wins in two games: at most two heads in three flips, $7 slash 8$. One win in two games: at most one head in three flips, $4 slash 8 = 1 slash 2$, as symmetry demands. _Source:_ new, on Bayes's identity @bayes.

*The shortest duel (@sec-stop), rated 12.* #upper(words.at(PR.shortest.uniform).first())#words.at(PR.shortest.uniform).slice(1) games. Three straight wins give at most three heads in four flips, $15 slash 16 = 0.9375$, short of 0.95; four straight wins give $31 slash 32 approx 0.969$. Under the Jeffreys prior #words.at(PR.shortest.jeffreys) straight wins are enough (#fix(PR.jeffreys_straight, 3)), and under the skeptical prior it takes #words.at(PR.shortest.skeptical). _Source:_ new.

*Win by two (@sec-formats), rated 20.* Look at the games in pairs. A pair won twice by the first player ends the match for them, with chance $p^2$; a pair won twice by the other player ends it for them, with chance $(1 - p)^2$; a split pair returns the score to level, and play starts over. So the first player wins with chance $p^2 slash (p^2 + (1 - p)^2)$. For $p = 0.55$ it is $0.3025 slash 0.505 = #F.at("win by 2 fraction").at(0) slash #F.at("win by 2 fraction").at(1) approx #fix(F.at("win by 2").at(0), 3)$. _Source:_ folklore; it is the chance of winning from deuce in tennis.

*A 200-point gap (@sec-elo), rated 05.* $1 slash (1 + 0.316) approx #fix(R.elo.rows.at(2).at(1), 3)$. _Source:_ new, on Elo's formula @elo.

*Who shoots first (@sec-model), rated 15.* More streaky. With an edge $e$ above one half for the first shooter, each duel repeats the last winner with chance $e$, while fair coin flips repeat with chance one half. Longer streaks make one player look better than they are, which is why the "winner" rows of @tab-shooter are so high. _Source:_ new.

= Words and symbols <app-words>

#figure(
  table(inset: (x: 6pt, y: if phone { 2pt } else { 3pt }),
    columns: (auto, 1fr), align: (left, left),
    table.hline(),
    [word or symbol], [meaning, and how to read it aloud],
    table.hline(stroke: 0.5pt),
    [$p$], [The chance that the first player wins a game.],
    [$hat(p)$], [“p hat”: the share of games won, $k slash n$, an estimate of $p$.],
    [$k$, $n$, $N$], [Wins, games played, and the cap on games.],
    [$P(A | B)$], [“the chance of $A$ given $B$”.],
    [$binom(n, k)$], [“$n$ choose $k$”: the number of ways to pick $k$ games out of $n$.],
    [$integral_a^b f(q) dif q$], [“the integral of $f$ from $a$ to $b$”: the area under the curve $f$ between $a$ and $b$.],
    [Beta$(a, b)$], [“beta $a$ $b$”: a spread of belief over chances, with density proportional to $p^(a - 1) (1 - p)^(b - 1)$.],
    [$B(a, b)$], [The beta function, which rescales that density to total 1.],
    [$B_n$], [The Bayes factor after $n$ games: how many times likelier the record is under the uniform prior than with $p = 1 slash 2$.],
    [$Phi(z)$], [“phi of $z$”: the chance that a standard normal number is at most $z$.],
    [$alpha$, $gamma$, $delta$, $epsilon$], [“alpha”, “gamma”, “delta”, “epsilon”: an error chance, a miss chance, a distance and a small positive number.],
    [$r$, $c$, $e$], [The odds $(1 - p) slash p$; the lead that ends Wald's test; the first shooter's chance of winning.],
    [$approx$, $>=$, $plus.minus$], [“is about”, “is at least”, “plus or minus”.],
    [$log$, $log_10$], [The natural logarithm; the logarithm to base 10.],
    [prior, posterior], [Belief about $p$ before, and after, seeing the games.],
    [verdict], [A declaration that one player is better.],
    [stopping rule], [A rule for when to stop playing, using only the games so far.],
    [error rate], [For one fixed pair of players, the chance of a wrong verdict.],
    [martingale, fair game], [A running total whose next value averages to its present one.],
    [random walk], [A lead that moves one step up or down after each game.],
    [lead], [Wins minus losses, $2k - n$.],
    [best of $n$], [The first to win $(n + 1) slash 2$ games wins the match.],
    [Elo rating], [A number per player; a gap of $D$ points means a chance $1 slash (1 + 10^(-D slash 400))$.],
    [exact], [Computed with fractions or whole numbers, with no rounding until printing.],
    [$qed$], [The end of a proof.],
    table.hline(),
  ),
  caption: [Words and symbols used in this note.],
)

#bibliography("peggle.yml", style: "ieee")
