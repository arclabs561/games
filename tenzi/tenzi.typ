#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "@preview/lilaq:0.6.0" as lq

// Every number in this note is printed by tenzi.py, optimal.py or maximum.py; the plot reads turns.csv, which tenzi.py writes.
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

= A formula good to ten decimals <sec-harmonic>

The alternating sum hides a much simpler answer. Let $H_n$ be the $n$th harmonic number, $1 + 1 slash 2 + dots.c + 1 slash n$, and let $lambda = ln(6 slash 5) approx 0.1823$. Then
$ H_10 / lambda + 1 / 2 = 16.564 thin 848 thin 860 thin 842, $
while the exact mean is $16.564 thin 848 thin 861 thin 259$. The two differ by $4.18 times 10^(-10)$. With one die the same formula gives 5.985 against the exact 6, and with twenty dice it is off by $2.2 times 10^(-14)$. Why do ten alternating terms, most of them in the hundreds, collapse to a harmonic number, and what is the small remainder?

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
$ 2 / (lambda y_k) product_(j = 1)^n (1 + y_k^2 / j^2)^(-1 slash 2), quad y_k = (2 pi k) / lambda approx 34.46 k. $
Each die multiplies the bound by $j slash sqrt(j^2 + y_k^2)$, which is at most 0.28 when $j <= 10$. Each exponential gap smooths the distribution of $M$, and ten of them leave a fractional part that is even to ten decimals. For ten dice the terms with $k = plus.minus 1$ come to $4.18 times 10^(-10)$, nearly all of the remainder, and those with $k = plus.minus 2$ come to $2.3 times 10^(-13)$. `maximum.py` adds the first 200 pairs in 60-digit arithmetic. For 10, 20 and 50 dice the sum agrees with the exact remainder to 30 digits or more (@tab-harmonic).

#figure(
  text(10.5pt, table(
    columns: 4, align: (right, right, right, right), stroke: none, inset: (x: 6pt, y: 3.5pt),
    table.hline(),
    table.header([dice $n$], [exact mean], [exact $-$ formula], [terms $k = plus.minus 1$]),
    table.hline(stroke: 0.5pt),
    [1], [6.000000], [$1.5 times 10^(-2)$], [$9.2 times 10^(-3)$],
    [2], [8.727273], [$5.0 times 10^(-5)$], [$4.6 times 10^(-5)$],
    [3], [10.555445], [$-5.0 times 10^(-5)$], [$-4.6 times 10^(-5)$],
    [5], [13.023662], [$7.1 times 10^(-7)$], [$7.0 times 10^(-7)$],
    [10], [16.564849], [$4.2 times 10^(-10)$], [$4.2 times 10^(-10)$],
    [20], [20.232936], [$2.2 times 10^(-14)$], [$2.2 times 10^(-14)$],
    [100], [28.951806], [$4.1 times 10^(-21)$], [$4.1 times 10^(-21)$],
    table.hline(),
  )),
  caption: [The exact mean with the face named first, its difference from $H_n slash lambda + 1 slash 2$, and the two Fourier terms with $k = plus.minus 1$. The sign changes because the angle of $phi(2 pi)$ turns as dice are added. Exact, and 60-digit arithmetic.],
) <tab-harmonic>

The remainder does not shrink forever. As $n$ grows, the product of the factors $(1 + y^2 slash j^2)^(-1 slash 2)$ converges to $sqrt(pi y slash sinh(pi y))$, by Euler's product for the sine at an imaginary point. So the size of the terms with $k = plus.minus 1$ tends to $(2 slash lambda) |Gamma(i y_1)| approx 1.45 times 10^(-23)$. Their angle keeps turning with $log n$, and the remainder settles into a wobble that repeats each time $n$ is multiplied by $6 slash 5$. Szpankowski and Rego found this wobble, with an error term of order $1 slash n$, as quoted by Eisenberg @eisenberg. He points out that for moderate $n$ that error term, not the wobble, sets the size of the remainder, and he studies it with the same Fourier analysis of ${M}$ used here. The terms here reach within a factor of 2 of the wobble's size only from 856 dice on. At ten dice the remainder is $10^13$ times larger than the wobble. Rice's method, which writes the alternating sum as a contour integral, gives the same identity: the pole at 0 gives $H_n slash lambda + 1 slash 2$, and the poles at $2 pi i k slash lambda$ give the Fourier terms.

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

The slides' figures differ from the exact ones by up to 0.1 turns and 0.05 players, as a simulation would. A seeded simulation of 20,000 races of four players gives $10.506 plus.minus 0.041$, $0.107 plus.minus 0.004$ and $1.214 plus.minus 0.013$, against the exact 10.542, 0.103 and 1.216.

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

= Is switching also best in a race? <sec-deadline>

A race rewards finishing before the others, not finishing in few turns on average. A strategy with a slightly larger mean could still finish by turn 10 more often. So fix a deadline $t$ and ask for the largest chance of being done by turn $t$. It solves a finite-horizon program. Let $V_0("roll")$ be 1 if all ten dice match and 0 otherwise, and let $V_s ("roll")$ be the largest, over the keeps allowed after the roll, of $sum P("keep" -> "next") V_(s - 1)("next")$. The best chance of being done by turn $t$ is then the average of $V_(t - 1)$ over the first roll. The values are whole numbers divided by $6^(10 s)$, so `optimal.py` computes them exactly with whole numbers, for every $s$ and every one of the 34 kinds of roll.

The answer is yes, at every deadline that matters. *For every deadline up to turn 163 and every roll, keeping every die of the most common face is among the best keeps*, both when the kept dice must share one face and when any dice may be kept. Turn 163 is where the chance of being done reaches $1 - 10^(-12)$, so after it no strategy can gain more than $10^(-12)$. The best chance of being done by turn 5, 10, 14, 20 and 30 is 0.0159, 0.2505, 0.5310, 0.8142 and 0.9678, and these are the switching strategy's own chances. On small games, up to four dice on three faces, the program agrees with value iteration over every ordered roll and every subset of dice to keep. In every game of two to six dice on two to six faces, switching is again best for every deadline.

A strategy that is best for every deadline is best for any race. Suppose the other players' choices do not depend on our dice, though they may be anything else and we may watch them. Fix everything the others roll and choose. Then the turn $tau$ on which the first of them finishes is fixed, and we win outright exactly when we are done by turn $tau - 1$. Whatever we do, even reacting to the others, is then a strategy for our own dice alone, and no strategy beats switching's chance of being done by that turn (a strategy that remembers the past or flips coins does no better than the program's best). Averaging over the others' play keeps the inequality. The same holds for winning or tying (done by turn $tau$), and for any mix of the two, such as counting a tie as half a win. So in a lockstep race, switching gives the largest chance of winning against any opponents who do not react to our dice.

= Checks

`tenzi.py` and `optimal.py` compute everything above with exact fractions, and `tenzi.py` also plays 100,000 seeded games with each strategy (@tab-checks).

#figure(
  placement: auto,
  table(
    columns: 3, align: (left, right, right), stroke: none,
    table.hline(),
    [strategy], [exact mean], [simulated mean (95% interval)],
    table.hline(stroke: 0.5pt),
    [face named first], [16.565], [16.556 ± 0.042],
    [fixed face], [15.349], [15.363 ± 0.042],
    [switching], [15.294], [15.278 ± 0.042],
    table.hline(),
  ),
  caption: [Exact means and simulated ones; each exact mean lies inside its simulated interval. Each simulation draws its games from its own generator, seeded with 1, 2 and 3, so the three share no random numbers and are independent checks.],
) <tab-checks>

The tests check that each distribution sums to 1 within $10^(-12)$, that its mean agrees with the value from the equations above, that the committed `turns.csv` matches what the script writes, that the race agrees with a seeded simulation of 3,000 races, that policy iteration agrees with value iteration on small games, and that the deadline program agrees with brute force on small games and gives switching's own chances for ten dice. They check the Fourier identity of @sec-harmonic to 30 digits for 10, 20 and 50 dice, and every number quoted from it.

= Questions left open

- *A proof for every deadline.* @sec-deadline checks that switching is best for each deadline by computing; an argument that covers every deadline, and every number of dice and faces, is still missing.
- *Speed.* The slides @slides also ask how much a faster player gains; in a lockstep race every player rolls at the same pace.
- *Other sizes.* The same chain and decision process work for any number of dice and faces.

#bibliography("tenzi.yml", style: "ieee")
