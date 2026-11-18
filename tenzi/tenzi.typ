#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "@preview/lilaq:0.6.0" as lq

// Every number in this note is printed by tenzi.py or optimal.py; the plot reads turns.csv, which tenzi.py writes.
#set document(title: "How Many Turns Does Tenzi Take?", date: none)
#set page(paper: "us-letter", margin: (x: 1.55in, y: 1.1in), numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)
#show math.equation.where(block: false): box
#set par(justify: true, leading: 0.6em, spacing: 1em)
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 0.7em)
#show figure.caption: set text(size: 10pt)
#show figure.caption: set par(justify: true)
#show link: set text(fill: rgb("#0072B2"))
#let fitw(x) = layout(size => {
  let w = measure(x).width
  if w > size.width { scale(size.width / w * 100%, reflow: true, x) } else { x }
})

#align(center, text(17pt, weight: "bold")[How Many Turns Does Tenzi Take?])
#v(0.8em)

In *Tenzi* each player has ten dice. Everyone rolls at once; then each player keeps the dice that show one face and rerolls the rest, again and again, until all ten dice show that face. The first player done wins. How many rolls does that take, and is there a better way to choose the face?

= Two strategies

The obvious plan is to keep whichever face came up most often. Call the number of dice showing the chosen face the *matched* count, $m$.

- The *fixed-face* strategy chooses the most common face after the first roll and never changes it. If two faces tie, either will do: the count is what matters.
- The *switching* strategy changes face whenever the rerolled dice show more of another face than are already kept.

A *turn* is one roll, so the first roll is turn 1. The number of turns is random; we want its distribution.

= The first roll

The first roll fixes the starting count: the largest number of dice that show the same face. Ten dice on six faces always show some face at least twice. Counting the $6^10$ equally likely rolls exactly gives:

#figure(
  text(10.5pt, table(
    columns: 9, align: right, stroke: none, inset: (x: 4pt, y: 4pt),
    table.hline(),
    [count], [2], [3], [4], [5], [6], [7], [8], [9, 10],
    table.hline(stroke: 0.5pt),
    [chance], [6.75%], [52.93%], [31.05%], [7.81%], [1.30%], [0.15%], [0.01%], [\<0.001%],
    table.hline(),
  )),
  caption: [The largest count after the first roll. Three of a kind is the usual start. Exact.],
)

Finishing on the first roll needs all ten dice to agree. For one face that happens with chance $1 slash 6^10$, and there are six faces, so the chance is $6 slash 6^10 = 1 slash 6^9$, about one in ten million.

= The fixed-face strategy is a Markov chain

With $m$ dice matched, a turn rerolls the other $10 - m$, and each of them shows the chosen face with chance $1 slash 6$. So the count goes up by a binomial number, $"Bin"(10 - m, 1 slash 6)$, and nothing else about the past matters. The chance that a turn adds nothing is $(5 slash 6)^(10 - m)$, which grows as the game goes on: with nine dice matched, five turns in six are wasted.

#figure(
  fitw(fletcher.diagram(
    spacing: (5.5mm, 5mm),
    node-stroke: 0.6pt,
    for m in range(2, 11) {
      node((m - 2, 0), [#m], shape: circle, width: 6.5mm, name: label("m" + str(m)))
    },
    for m in range(2, 10) {
      edge(label("m" + str(m)), label("m" + str(m + 1)), "-|>")
      edge(label("m" + str(m)), label("m" + str(m)), "-|>", bend: 130deg, loop-angle: 90deg,
        label: text(7.5pt)[#calc.round(calc.pow(5 / 6, 10 - m) * 100)%], label-side: center)
    },
    for (m, left) in ((2, "15.4"), (3, "14.7"), (4, "13.9"), (5, "13.0"), (6, "11.9"), (7, "10.6"), (8, "8.7"), (9, "6.0"), (10, "0")) {
      node((m - 2, 0.6), text(9pt, left), stroke: none)
    },
    node((-1.25, 0), text(9pt)[matched], stroke: none),
    node((-1.25, 0.6), text(9pt)[turns left], stroke: none),
  )),
  caption: [The fixed-face strategy as a chain on the matched count. Each loop is the chance that a turn adds no match, $(5 slash 6)^(10 - m)$. The arrows to the right stand for any gain, since one turn can add several matches. Below each count, the expected number of turns still to come. Exact.],
) <fig-chain>

The expected turns still to come, $E_m$, satisfy $E_10 = 0$ and
$ E_m = 1 + sum_(m' > m) P(m -> m') E_(m') + P(m -> m) E_m, $
which can be solved from $m = 9$ downward. At $m = 9$ only one die is rerolled, and it matches with chance $1 slash 6$, so $E_9 = 6$: on average *the last die alone takes six turns*, about two fifths of the whole game.

= How long a game takes

Following the chain turn by turn gives the exact distribution of the number of turns (@fig-turns). The mean is 15.35 turns, the median 14, and the most likely number 12. The tail is long: one game in 30 takes more than 30 turns.

#figure(
  {
    let rows = csv("turns.csv", row-type: dictionary)
    let xs = rows.map(r => int(r.turns))
    let ys = rows.map(r => float(r.fixed) * 100)
    fitw(lq.diagram(
      width: 11cm, height: 4.5cm,
      xlabel: [turns], ylabel: [chance (%)],
      xlim: (0, 46), ylim: (0, 8),
      lq.bar(xs.slice(0, 45), ys.slice(0, 45), fill: luma(80%), stroke: luma(40%)),
    ))
  },
  caption: [Chance that the fixed-face strategy finishes on each turn, up to 45 turns. The peak is at 12, the mean at 15.35. Exact.],
) <fig-turns>

= Does switching help?

The switching strategy is also a chain on the matched count, with a different step: after a reroll, the new count is the larger of the kept face's count and the largest count of any other face among the rerolled dice. Counting all outcomes of each reroll exactly gives a mean of 15.29 turns, against 15.35 for the fixed face. Switching saves about a twentieth of a turn per game. The median (14) and the most likely number (12) are the same.

Switching rarely matters because it needs the rerolled dice to beat the kept ones. With three kept, seven rerolled dice must show four of some other face, and later in the game the kept group is too large to beat.

= Naming the face first

Some players name the face before the first roll. Then each die is on its own: it needs a geometric number of rolls, with chance $1 slash 6$ each time, and the game lasts as long as the slowest of the ten. With $q = 5 slash 6$,
$ P(T > k) = 1 - (1 - q^k)^10 = sum_(j=1)^10 (-1)^(j+1) binom(10, j) q^(j k), $
and summing over $k$ gives the mean,
$ E[T] = sum_(k >= 0) P(T > k) = sum_(j=1)^10 (-1)^(j+1) binom(10, j) / (1 - q^j) = 16.565 "turns," $
with standard deviation 6.834. The chain started from no matched dice gives the same mean exactly. Choosing the face after the first roll is worth 1.2 turns.

Bacinski and Pennings @slides play this way and report a mean of 15.565 with standard deviation 6.834. Their mean is the exact one less one: their table of finishing times puts, beside each roll $k$, the exact chance of finishing on roll $k + 1$, to within 0.0006. Their state vectors do count the first roll ($S_20 - S_19 = 0.0401 = P(T = 20)$). A shift changes the mean but not the spread, so their standard deviation is exact.

= The race

With $n$ players who roll together, the game ends on the first turn on which someone finishes. If one player is still going after turn $t$ with chance $S(t)$, the first finish comes on turn $t$ with chance $S(t-1)^n - S(t)^n$. Taking away the chance that exactly one player finishes then gives the chance of a tie. The chain also gives the chance of nine matched dice after each turn, and from it the expected number of players left one die short when the game ends (@tab-race).

#figure(
  placement: auto,
  text(10.5pt, table(
    columns: 6, align: right, stroke: none, inset: (x: 5pt, y: 3.5pt),
    table.hline(),
    table.header([players], [winning turn], [slides], [P(tie)], [one die left], [slides]),
    table.hline(stroke: 0.5pt),
    [2], [12.90], [13.0], [0.048], [0.52], [0.53],
    [3], [11.41], [11.4], [0.080], [0.91], [0.91],
    [4], [10.54], [10.6], [0.103], [1.22], [1.21],
    [6], [9.52], [9.6], [0.137], [1.70], [1.71],
    [10], [8.48], [8.5], [0.181], [2.41], [2.37],
    [30], [6.84], [6.8], [0.276], [4.38], [4.36],
    table.hline(),
  )),
  caption: [A race in which the face is named first, exact, against the slides @slides. The slides label each row with the number of other players. Their race means count the first roll, unlike their single-player mean. With switching, the winning turn comes about one turn sooner: 11.68 for two players and 5.89 for thirty.],
) <tab-race>

The slides' figures differ from the exact ones by up to 0.1 turns and 0.05 players, as a simulation would. A seeded simulation of 20,000 races of four players gives $10.515 plus.minus 0.041$, $0.107 plus.minus 0.004$ and $1.208 plus.minus 0.013$, against the exact 10.542, 0.103 and 1.216.

= The best strategy

Is there anything better than switching? After a roll, all that matters is how many dice show each face, such as $(4, 3, 2, 1)$. The player keeps some dice and rerolls the rest, and each roll costs a turn. This is a Markov decision process with 34 kinds of roll, small enough to solve with exact fractions. Policy iteration starts from switching and stops at once: no keep improves on it. That holds when the kept dice must share one face, as the rules require, and also when any dice may be kept. So 15.2944 turns is the best possible mean.

After keeping $k$ dice of one face, the expected turns still to come are

#align(center, text(10.5pt, table(
  columns: 10, align: right, stroke: none, inset: (x: 4pt, y: 3.5pt),
  table.hline(),
  [$k$], [0, 1], [2], [3], [4], [5], [6], [7], [8], [9],
  table.hline(stroke: 0.5pt),
  [turns], [15.29], [15.11], [14.66], [13.93], [13.02], [11.93], [10.56], [8.73], [6.00],
  table.hline(),
)))

These fall as $k$ grows, so keeping more is always better, and every other keep loses at least 0.05 turns. One kept die is worth nothing: since faces are interchangeable, it is as good as rolling it. On small games, up to four dice on three faces, the method agrees with value iteration over every ordered roll and every subset of dice to keep.

= Checks

`tenzi.py` and `optimal.py` compute everything above with exact fractions, and `tenzi.py` also plays 100,000 seeded games with each strategy (@tab-checks).

#figure(
  placement: auto,
  table(
    columns: 3, align: (left, right, right), stroke: none,
    table.hline(),
    [strategy], [exact mean], [simulated mean (95% interval)],
    table.hline(stroke: 0.5pt),
    [face named first], [16.565], [16.535 ± 0.042],
    [fixed face], [15.349], [15.319 ± 0.042],
    [switching], [15.294], [15.255 ± 0.042],
    table.hline(),
  ),
  caption: [Exact means and simulated ones; each exact mean lies inside its simulated interval. Game $i$ of each simulation is seeded with $i$.],
) <tab-checks>

The tests check that each distribution sums to 1 within $10^(-12)$, that its mean agrees with the value from the equations above, that the committed `turns.csv` matches what the script writes, that the race agrees with a seeded simulation of 3,000 races, and that policy iteration agrees with value iteration on small games.

= Questions left open

- *Playing to win.* The best strategy above minimizes the expected number of turns. In a race the aim is to finish first, and a player who can see the others' dice might take more risk when behind.
- *Speed.* The slides @slides also ask how much a faster player gains; in a lockstep race every player rolls at the same pace.
- *Other sizes.* The same chain and decision process work for any number of dice and faces.

#bibliography("tenzi.yml", style: "ieee")
