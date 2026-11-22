#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "@preview/lilaq:0.6.0" as lq

// Every number in this note that a script computes is read from results.json, which report.py
// writes from tenzi.py, optimal.py and maximum.py; the plot reads turns.csv, which tenzi.py
// writes. The rules, the table of results, the checks and the open questions are the README's
// shared sections (from docs/, filled by tools/sync_docs.py), so the note and the README say
// the same. Numbers typed by hand are inputs, steps of hand arithmetic, or the slides' figures.
#import "/tools/shared_docs.typ": shared, fmt
#let R = json("results.json")
#let readme = read("README.md")
#let H = R.harmonic
#let S = R.strategies

// x rounded to d decimals, with trailing zeros: fix(0.12, 3) is "0.120". report.py stores
// floats unrounded, and tests/test_tenzi.py checks that none sits on a rounding tie, where
// Typst (which rounds the shortest decimal form) and Python (the binary value) could differ.
#let fix(x, d) = {
  let s = str(calc.round(x, digits: d))
  if d == 0 { return s }
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  parts.at(0) + "." + frac + "0" * (d - frac.len())
}
#let pct(x, d) = fix(x * 100, d) + "%"
// x as $m times 10^e$, with d decimals in m
#let sci(x, d) = {
  let e = calc.floor(calc.log(calc.abs(x), base: 10))
  let m = calc.round(x / calc.pow(10.0, e), digits: d)
  if calc.abs(m) >= 10 { m = m / 10; e = e + 1 }
  $#fix(m, d) times 10^(#str(e))$
}
// "16.564848860842" as 16.564 848 860 842, in groups of three
#let spaced(s) = {
  let (whole, frac) = s.split(".")
  $#(whole + "." + frac.slice(0, 3)) thin #frac.slice(3, 6) thin #frac.slice(6, 9) thin #frac.slice(9, 12)$
}
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
#set document(title: "How Many Turns Does Tenzi Take?", date: none)
#set page(..if phone { (width: 4.4in, height: 7.6in, margin: (x: 0.32in, top: 0.42in, bottom: 0.55in)) }
  else { (paper: "us-letter", margin: (x: 1.55in, y: 1.1in)) }, numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)
#show math.equation.where(block: false): box
#set par(justify: true, leading: 0.6em, spacing: 1em)
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 0.7em)
#show heading: set text(hyphenate: false)
#show figure.caption: set text(size: 10pt)
#show figure.caption: set par(justify: not phone)
#show link: set text(fill: rgb("#0072B2"))
#show math.equation.where(block: true): it => if phone { fitw(it) } else { it }
// tables with a 1fr column fill the width they get, so lay those out at 4.8in; tables
// of auto columns keep their natural width (up to 7in) and are scaled down whole
#set table(stroke: none)
// the shared table of results (tools/shared_docs.typ) fills the width it gets
#show table: it => if phone and it.at("label", default: none) != <shared-table> {
  let fr = type(it.columns) == array and it.columns.any(c => type(c) == fraction)
  wide(it, w: if fr { 4.8in } else { 7in })
} else { it }

// a table of two long rows; on a phone it is turned into two columns, which fit the page.
// In HTML it scrolls sideways on a narrow screen instead of widening the page.
#let tworow(inset: (x: 4pt, y: 4pt), a, b) = if not phone {
  let t = table(columns: a.len(), align: right, stroke: none, inset: inset,
    table.hline(), ..a, table.hline(stroke: 0.5pt), ..b, table.hline())
  context if target() == "html" { html.elem("div", attrs: (style: "overflow-x: auto"), t) } else { t }
} else {
  block(breakable: false, table(columns: 2, align: right, stroke: none, inset: (x: 10pt, y: 3pt),
    table.hline(), a.first(), b.first(), table.hline(stroke: 0.5pt),
    ..a.slice(1).zip(b.slice(1)).flatten(), table.hline()))
}

// a lead-in sentence and its display, kept on one page (phone pages only)
#let together(body) = if phone { block(breakable: false, body) } else { body }

// ---------- title ----------
#context if target() == "html" {
  html.elem("h1")[How Many Turns Does Tenzi Take?]
} else [
  #align(center, text(17pt, weight: "bold")[How Many Turns Does Tenzi Take?])
  #v(0.8em)
]

In *Tenzi* each player has ten dice. Everyone rolls at once; then each player keeps the dice that show one face and rerolls the rest, again and again, until all ten dice show that face. The first player done wins. How many rolls does that take, and is there a better way to choose the face?

= Two strategies

The obvious plan is to keep whichever face came up most often. Call the number of dice showing the chosen face the *matched* count, $m$. The note compares the following ways to play.

#shared(readme, "rules")

The first two are the *fixed-face* and the *switching* strategy. The number of turns is random; we want its distribution.

= The first roll

The first roll fixes the starting count: the largest number of dice that show the same face. Ten dice on six faces always show some face at least twice. Counting the $6^10$ equally likely rolls exactly gives:

#figure(
  kind: table,  // the table is built inside a context, so its kind must be given
  text(10.5pt, tworow(inset: (x: 4pt, y: 4pt),
    ([count], [2], [3], [4], [5], [6], [7], [8], [9, 10]),
    ([chance], ..range(2, 9).map(m => [#pct(R.first_roll.at(str(m)), 2)]), [\<0.001%]),
  )),
  caption: [The largest count after the first roll. Three of a kind is the usual start. Exact.],
)

Finishing on the first roll needs all ten dice to agree. For one face that happens with chance $1 slash 6^10$, and there are six faces, so the chance is $6 slash 6^10 = 1 slash 6^9$, about one in ten million.

= The fixed-face strategy is a Markov chain

With $m$ dice matched, a turn rerolls the other $10 - m$, and each of them shows the chosen face with chance $1 slash 6$. So the count goes up by a binomial number, $"Bin"(10 - m, 1 slash 6)$, and nothing else about the past matters. The chance that a turn adds nothing is $(5 slash 6)^(10 - m)$, which grows as the game goes on: with nine dice matched, five turns in six are wasted.

#figure(
  visb(fitw(fletcher.diagram(
    spacing: if phone { (2.2mm, 5mm) } else { (5.5mm, 5mm) },
    node-stroke: 0.6pt,
    for m in range(2, 11) {
      node((m - 2, 0), [#m], shape: circle, width: 6.5mm, name: label("m" + str(m)))
    },
    for m in range(2, 10) {
      edge(label("m" + str(m)), label("m" + str(m + 1)), "-|>")
      edge(label("m" + str(m)), label("m" + str(m)), "-|>", bend: 130deg, loop-angle: 90deg,
        label: text(7.5pt)[#calc.round(calc.pow(5 / 6, 10 - m) * 100)%], label-side: center)
    },
    for m in range(2, 11) {
      let left = R.turns_left_fixed.at(str(m))
      node((m - 2, 0.6), text(9pt, if m == 10 { "0" } else { fix(left, 1) }), stroke: none)
    },
    node((-1.25, 0), text(9pt)[matched], stroke: none),
    node((-1.25, 0.6), text(9pt)[turns left], stroke: none),
  ))),
  caption: [The fixed-face strategy as a chain on the matched count. Each loop is the chance that a turn adds no match, $(5 slash 6)^(10 - m)$. The arrows to the right stand for any gain, since one turn can add several matches. Below each count, the expected number of turns still to come. Exact.],
) <fig-chain>

The expected turns still to come, $E_m$, satisfy $E_10 = 0$ and
$ E_m = 1 + sum_(m' > m) P(m -> m') E_(m') + P(m -> m) E_m, $
which can be solved from $m = 9$ downward. At $m = 9$ only one die is rerolled, and it matches with chance $1 slash 6$, so $E_9 = 6$: on average *the last die alone takes six turns*, about two fifths of the whole game.

= How long a game takes

Following the chain turn by turn gives the exact distribution of the number of turns (@fig-turns). The mean is #fix(S.at("fixed face").mean, 2) turns, the median #S.at("fixed face").median, and the most likely number #S.at("fixed face").mode. The tail is long: #pct(S.at("fixed face").over_30, 1) of games take more than 30 turns.

#figure(
  placement: if phone { auto } else { none },
  {
    let rows = csv("turns.csv", row-type: dictionary)
    let xs = rows.map(r => int(r.turns))
    let ys = rows.map(r => float(r.fixed) * 100)
    visb(fitw(lq.diagram(
      width: 11cm, height: 4.5cm,
      xlabel: [turns], ylabel: [chance (%)],
      xlim: (0, 46), ylim: (0, 8),
      lq.bar(xs.slice(0, 45), ys.slice(0, 45), fill: luma(80%), stroke: luma(40%)),
    )))
  },
  caption: [Chance that the fixed-face strategy finishes on each turn, up to 45 turns. The peak is at #S.at("fixed face").mode, the mean at #fix(S.at("fixed face").mean, 2). Exact.],
) <fig-turns>

= Does switching help? <sec-switch>

The switching strategy is also a chain on the matched count, with a different step: after a reroll, the new count is the larger of the kept face's count and the largest count of any other face among the rerolled dice. Counting all outcomes of each reroll exactly gives a mean of #fix(S.switching.mean, 2) turns, against #fix(S.at("fixed face").mean, 2) for the fixed face. Switching saves about a twentieth of a turn per game. The median (#S.switching.median) and the most likely number (#S.switching.mode) are the same.

Switching rarely matters because it needs the rerolled dice to beat the kept ones. With three kept, seven rerolled dice must show four of some other face, and later in the game the kept group is too large to beat.

= Naming the face first <sec-named>

Some players name the face before the first roll. Then each die is on its own: it needs a geometric number of rolls, with chance $1 slash 6$ each time, and the game lasts as long as the slowest of the ten. With $q = 5 slash 6$,
$ P(T > k) = 1 - (1 - q^k)^10 = sum_(j=1)^10 (-1)^(j+1) binom(10, j) q^(j k), $
and summing over $k$ gives the mean,
$ E[T] = sum_(k >= 0) P(T > k) = sum_(j=1)^10 (-1)^(j+1) binom(10, j) / (1 - q^j) = #fix(S.at("target first").mean, 3) "turns," $
with standard deviation #fix(S.at("target first").sd, 3). The chain started from no matched dice gives the same mean exactly. Choosing the face after the first roll is worth #fix(S.at("target first").mean - S.at("fixed face").mean, 1) turns.

Bacinski and Pennings @slides play this way and report a mean of 15.565 with standard deviation 6.834. Their mean is the exact one less one: their table of finishing times puts, beside each roll $k$, the exact chance of finishing on roll $k + 1$, to within 0.0006. Their state vectors do count the first roll: $S_4$ gives #fix(R.target_first_shifted.done_by_4, 4) for ten matched dice, which is $P(T <= 4)$, and $S_20 - S_19 = #fix(R.target_first_shifted.on_20, 4) = P(T = 20)$. A shift changes the mean but not the spread, so their standard deviation is exact.

= A formula good to ten decimals <sec-harmonic>

The alternating sum hides a much simpler answer. Let $H_n$ be the $n$th harmonic number, $1 + 1 slash 2 + dots.c + 1 slash n$, and let $lambda = ln(6 slash 5) approx #fix(H.rate, 4)$. Then
$ H_10 / lambda + 1 / 2 = #spaced(H.approx_10), $
while the exact mean is #spaced(H.exact_10). The two differ by #sci(H.gap_10, 2). With one die the same formula gives #fix(H.one_die, 3) against the exact 6, and with twenty dice it is off by #sci(H.table.at("20").at(1), 1). Why do ten alternating terms, most of them in the hundreds, collapse to a harmonic number, and what is the small remainder?

*Each die is a clock rounded up.* Give a die a continuous waiting time $X$ with $P(X > x) = e^(-lambda x) = (5 slash 6)^x$, an exponential time. Rounding it up to the next whole number, $ceil(X)$, gives $P(ceil(X) > k) = P(X > k) = (5 slash 6)^k$, the law of the die's geometric wait. So we may let each die be matched on roll $ceil(X)$. Rounding up and taking the largest commute, so the game lasts $T = ceil(M)$, where $M$ is the largest of the ten exponential times.

*The largest time averages $H_n slash lambda$.* Of $n$ independent exponential times, the first ends after an exponential time with rate $n lambda$. The other $n - 1$ forget how long they have run, so the next gap is exponential with rate $(n - 1) lambda$, and so on down to rate $lambda$ (Rényi's representation). The gaps are independent, and their means add up to $E[M] = H_n slash lambda$.

*Rounding up adds about a half.* Rounding up adds $1 - {M}$, where ${M} = M - floor(M)$ is the fractional part of $M$. If ${M}$ were spread evenly over $[0, 1)$, rounding would add exactly $1 slash 2$ on average. The remainder is therefore $1 slash 2 - E[{M}]$, which measures how far the fractional part of $M$ is from even.

For a real number $s$, write $phi(s) = E[e^(i s M)]$ for the characteristic function of $M$. Fourier series turn the unevenness of ${M}$ into values of $phi$.

#block(width: 100%, inset: (y: 2pt))[
  *Theorem.* Let $T$ be the largest of $n >= 1$ independent waits, each geometric on $1, 2, dots$ with success chance $p$, and let $lambda = -ln(1 - p)$. Then
  $ E[T] = H_n / lambda + 1 / 2 + sum_(k != 0) phi(2 pi k) / (2 pi i k), quad phi(s) = product_(j = 1)^n (j lambda) / (j lambda - i s). $
]

_Proof._ A single exponential gap with rate $j lambda$ has characteristic function $j lambda slash (j lambda - i s)$, and the gaps are independent, so their product is $phi$. The sawtooth ${x} - 1 slash 2$ has the Fourier series $-sum_(k != 0) e^(2 pi i k x) slash (2 pi i k)$. Its symmetric partial sums are bounded, and they converge to the sawtooth at every $x$ that is not a whole number. The time $M$ has a density, so it is a whole number with chance zero. Taking expectations term by term (bounded convergence) gives $E[{M}] = 1 slash 2 - sum_(k != 0) phi(2 pi k) slash (2 pi i k)$. Then $E[T] = E[M] + 1 - E[{M}]$ gives the formula. #h(1fr)$qed$

The terms for $k$ and $-k$ are complex conjugates. Together they have size at most
$ 2 / (lambda y_k) product_(j = 1)^n (1 + y_k^2 / j^2)^(-1 slash 2), quad y_k = (2 pi k) / lambda approx #fix(H.y, 2) k. $
Each die multiplies the bound by $j slash sqrt(j^2 + y_k^2)$, which is at most #fix(H.largest_factor, 2) when $j <= 10$. Each exponential gap smooths the distribution of $M$, and ten of them leave a fractional part that is even to ten decimals. For ten dice the terms with $k = plus.minus 1$ come to #sci(H.size_k1, 3), nearly all of the remainder, and those with $k = plus.minus 2$ come to #sci(H.size_k2, 1). `maximum.py` adds the first 200 pairs in 60-digit arithmetic. For 10, 20 and 50 dice the sum agrees with the exact remainder to 30 digits or more (@tab-harmonic).

#figure(
  text(10.5pt, table(
    columns: 4, align: (right, right, right, right), stroke: none, inset: (x: 6pt, y: 3.5pt),
    table.hline(),
    table.header([dice $n$], [exact mean], [exact $-$ formula], [terms $k = plus.minus 1$]),
    table.hline(stroke: 0.5pt),
    ..(1, 2, 3, 5, 10, 20, 100).map(n => {
      let (mean, gap, term) = H.table.at(str(n))
      ([#n], [#fix(mean, 6)], sci(gap, 1), sci(term, 1))
    }).flatten(),
    table.hline(),
  )),
  caption: [The exact mean with the face named first, its difference from $H_n slash lambda + 1 slash 2$, and the two Fourier terms with $k = plus.minus 1$. The sign changes because the angle of $phi(2 pi)$ turns as dice are added. Exact, and 60-digit arithmetic.],
) <tab-harmonic>

The remainder does not shrink forever. As $n$ grows, the product of the factors $(1 + y^2 slash j^2)^(-1 slash 2)$ converges to $sqrt(pi y slash sinh(pi y))$, by Euler's product for the sine at an imaginary point. So the size of the terms with $k = plus.minus 1$ tends to $(2 slash lambda) |Gamma(i y_1)| approx #sci(H.wobble, 2)$. Their angle keeps turning with $log n$, and the remainder settles into a wobble that repeats each time $n$ is multiplied by $6 slash 5$. Szpankowski and Rego found this wobble, with an error term of order $1 slash n$, as quoted by Eisenberg @eisenberg. He points out that for moderate $n$ that error term, not the wobble, sets the size of the remainder, and he studies it with the same Fourier analysis of ${M}$ used here. The terms here reach within a factor of 2 of the wobble's size only from #H.dice_for_wobble dice on. At ten dice the remainder is $10^#calc.floor(calc.log(H.gap_over_wobble, base: 10))$ times larger than the wobble. Rice's method, which writes the alternating sum as a contour integral, gives the same identity: the pole at 0 gives $H_n slash lambda + 1 slash 2$, and the poles at $2 pi i k slash lambda$ give the Fourier terms.

= The race

With $n$ players who roll together, the game ends on the first turn on which someone finishes. If one player is still going after turn $t$ with chance $S(t)$, the first finish comes on turn $t$ with chance $S(t-1)^n - S(t)^n$. Taking away the chance that exactly one player finishes then gives the chance of a tie. The chain also gives the chance of nine matched dice after each turn, and from it the expected number of players left one die short when the game ends (@tab-race).

#figure(
  placement: auto,
  text(10.5pt, table(
    columns: 7, align: right, stroke: none, inset: (x: 5pt, y: 3.5pt),
    table.hline(),
    table.header([players], [winning turn], [slides], [P(tie)], [one die left], [slides], [switching]),
    table.hline(stroke: 0.5pt),
    // the slides' columns are their figures, cited; every other column is from results.json
    ..((2, "13.0", "0.53"), (3, "11.4", "0.91"), (4, "10.6", "1.21"), (6, "9.6", "1.71"),
       (10, "8.5", "2.37"), (20, "7.4", "3.58"), (30, "6.8", "4.36")).map(((n, turn, left)) => {
      let (win, tie, one) = R.race.at("target first").at(str(n))
      ([#n], [#fix(win, 2)], [#turn], [#fix(tie, 3)], [#fix(one, 2)], [#left], [#fix(R.race.switching.at(str(n)).at(0), 2)])
    }).flatten(),
    table.hline(),
  )),
  caption: [A race in which the face is named first, exact, against the slides @slides, and the mean winning turn when every player switches, which comes about one turn sooner. The slides label each row with the number of other players. Their race means count the first roll, unlike their single-player mean, and their share of players with one die left divides by everyone, winner included. The sums stop at #R.race_turns turns, where the chance of a longer game is below $10^(-14)$.],
) <tab-race>

#let RS = R.race_simulated.at("target first")
#let R4 = R.race.at("target first").at("4")
The slides' figures differ from the exact ones by up to 0.1 turns and 0.05 players, as a simulation would. A seeded simulation of #fmt(R.race_simulated.races) races of four players gives $#fix(RS.turn.at(0), 3) plus.minus #fix(RS.turn.at(1), 3)$, $#fix(RS.tie.at(0), 3) plus.minus #fix(RS.tie.at(1), 3)$ and $#fix(RS.one_left.at(0), 3) plus.minus #fix(RS.one_left.at(1), 3)$, against the exact #fix(R4.at(0), 3), #fix(R4.at(1), 3) and #fix(R4.at(2), 3).

= The best strategy <sec-best>

Is there anything better than switching? After a roll, all that matters is how many dice show each face, such as $(4, 3, 2, 1)$. The player keeps some dice and rerolls the rest, and each roll costs a turn. This is a Markov decision process with #R.optimal.kinds_of_roll kinds of roll, small enough to solve with exact fractions. Policy iteration starts from switching and stops at once: no keep improves on it. That holds when the kept dice must share one face, as the rules require, and also when any dice may be kept. So #fix(R.optimal.mean, 4) turns is the best possible mean.

#together[
After keeping $k$ dice of one face, the expected turns still to come are

#align(center, text(10.5pt, tworow(inset: (x: 4pt, y: 3.5pt),
  ([$k$], [0, 1], [2], [3], [4], [5], [6], [7], [8], [9]),
  ([turns], ..range(1, 10).map(k => [#fix(R.optimal.after_keeping.at(str(k)), 2)])),
)))
]

These fall as $k$ grows, so keeping more is always better, and every other keep loses at least #fix(R.optimal.margin.at("any dice"), 2) turns. One kept die is worth nothing: since faces are interchangeable, it is as good as rolling it. On small games, up to four dice on three faces, the method agrees with value iteration over every ordered roll and every subset of dice to keep.

= Is switching also best in a race? <sec-deadline>

A race rewards finishing before the others, not finishing in few turns on average. A strategy with a slightly larger mean could still finish by turn 10 more often. So fix a deadline $t$ and ask for the largest chance of being done by turn $t$. It solves a finite-horizon program. Let $V_0("roll")$ be 1 if all ten dice match and 0 otherwise, and let $V_s ("roll")$ be the largest, over the keeps allowed after the roll, of $sum P("keep" -> "next") V_(s - 1)("next")$. The best chance of being done by turn $t$ is then the average of $V_(t - 1)$ over the first roll. The values are whole numbers divided by $6^(10 s)$, so `optimal.py` computes them exactly with whole numbers, for every $s$ and every one of the #R.optimal.kinds_of_roll kinds of roll.

*From the start, switching gives the best chance of finishing by any turn through #R.deadline.last*, both when the kept dice must share one face and when any dice may be kept. Turn #R.deadline.last is where the chance of being done reaches $1 - 10^(-12)$, so after it no strategy can gain more than $10^(-12)$. The best chance of being done by turn 5, 10, 14, 20 and 30 is #("5", "10", "14", "20", "30").map(t => fix(R.deadline.chances.at(t), 4)).join(", ", last: " and "), and these are the switching strategy's own chances. On small games, up to four dice on three faces, the program agrees with value iteration over every ordered roll and every subset of dice to keep. In every game of two to six dice on two to six faces, switching is again best through the last deadline checked for that game.

To win a lockstep race outright, you have to finish before the first opponent does. Suppose the opponents roll independently of you and make their choices without using your dice. Fix their rolls and choices. If the first opponent finishes on turn $tau$, your deadline is $tau - 1$. If a tie counts as a win, your deadline is $tau$. In either case, switching is optimal for deadlines through #R.deadline.last. For any later deadline, its chance of still rolling is already below $10^(-12)$. Averaging over the opponents' finishing times gives the same bound on what any other strategy could gain, even one that watches the opponents or counts a tie as half a win.

= Checks

`tenzi.py` and `optimal.py` compute everything above with exact fractions, and `tenzi.py` also plays #fmt(R.simulated.games) seeded games with each strategy (@tab-checks).

#figure(
  placement: auto,
  table(
    columns: 3, align: (left, right, right), stroke: none,
    table.hline(),
    [strategy], [exact mean], [simulated mean (95% interval)],
    table.hline(stroke: 0.5pt),
    ..(("face named first", "target first"), ("fixed face", "fixed face"), ("switching", "switching")).map(((label, name)) => {
      let (m, h) = R.simulated.at(name)
      ([#label], [#fix(S.at(name).mean, 3)], [#fix(m, 3) ± #fix(h, 3)])
    }).flatten(),
    table.hline(),
  ),
  caption: [Exact means and simulated ones; each exact mean lies inside its simulated interval. Each simulation draws its games from its own generator, seeded with #("target first", "fixed face", "switching").map(n => str(R.simulated.seeds.at(n))).join(", ", last: " and "), so the three share no random numbers and are independent checks.],
) <tab-checks>

@tab-results lists each result and how it is known. The checks behind it are these.

#shared(readme, "checks")

The tests also check that each distribution sums to 1 within $10^(-12)$, and the Fourier identity of @sec-harmonic to 30 digits for 10, 20 and 50 dice.

#[
// on a phone the table may break across pages, and stays where it is introduced
#show figure: set block(breakable: phone)
#figure(
  placement: if phone { none } else { auto },
  kind: table,
  caption: [The results and how each is known. Exact unless marked.],
  [
    #show table.cell.where(y: 0): set text(weight: "semibold")
    #show table: set par(justify: false)
    #set table.cell(breakable: false)
    #set table(inset: (x: 5pt, y: 3.5pt))
    #text(size: if phone { 9pt } else { 10pt }, shared(readme, "results", columns: (1fr, 1.1fr, 1.2fr)))
  ],
) <tab-results>
]

= Questions left open

#shared(readme, "open")

#bibliography("tenzi.yml", style: "ieee")
