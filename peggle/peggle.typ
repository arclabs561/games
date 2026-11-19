#import "@preview/cetz:0.4.2"
#import "@preview/lilaq:0.6.0" as lq

// Every number in this note is printed by peggle.py or stopping.py; the plot reads better.csv, which peggle.py writes.
#set document(title: "Who Is the Better Player?", date: none)
#set page(paper: "us-letter", margin: (x: 1.55in, y: 1.1in), numbering: "1")
#set text(font: "New Computer Modern", size: 12pt)
#show math.equation: set text(font: "New Computer Modern Math", weight: 400)
#show math.equation.where(block: false): box
#set par(justify: true, leading: 0.6em, spacing: 1em)
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 0.7em)
#show figure.caption: set text(size: 10pt)
#show figure.caption: set par(justify: true)
#let fitw(x) = layout(size => {
  let w = measure(x).width
  if w > size.width { scale(size.width / w * 100%, reflow: true, x) } else { x }
})

#align(center, text(17pt, weight: "bold")[Who Is the Better Player?])
#v(0.8em)

Two friends play a game that one of them must win: Peggle, say, or chess with no draws. One of them wins $k$ of $n$ games, a few more than half, and claims to be the better player. What is the chance that the claim is right? And how many games would settle it?

= One number decides it

Suppose each game is won by the first player with the same chance $p$, independently of the others. The other player then wins with chance $1 - p$, and the first player is *better* when $p > 1 - p$, that is, when $p > 1 slash 2$. The games are flips of a coin with bias $p$, and the share of games won, $hat(p) = k slash n$, estimates $p$.

= How many games pin down the chance?

Concentration bounds say how far $hat(p)$ is likely to stray from $p$. One Chernoff-type bound is
$ P(|p - hat(p)| >= delta) <= 2 exp(-(delta^2)/(delta + 2) n), $
so $n >= (delta + 2) / delta^2 log(2 / gamma)$ games make $hat(p)$ fall within $delta$ of $p$ with probability at least $1 - gamma$. Hoeffding's inequality gives the sharper $2 exp(-2 n delta^2)$ and so needs only $n >= log(2 slash gamma) slash (2 delta^2)$:

#figure(
  table(
    columns: 3, align: (left, right, right), stroke: none,
    table.hline(),
    [within], [Chernoff-type bound], [Hoeffding],
    table.hline(stroke: 0.5pt),
    [$delta = 0.1$], [775 games], [185 games],
    [$delta = 0.05$], [3,025 games], [738 games],
    table.hline(),
  ),
  caption: [Games needed for $hat(p)$ to be within $delta$ of $p$ with probability at least 95%.],
)

Either way, a close match between nearly equal players takes hundreds of games to measure. But the question was not how big $p$ is, only whether it exceeds one half, and that has a direct answer.

= The Bayesian answer

Give $p$ a uniform prior on $[0, 1]$, so that no value is favored before any game is played. After $k$ wins in $n$ games, Bayes' rule gives the posterior density
$ f(p | k, n) = (binom(n, k) p^k (1 - p)^(n - k)) / (integral_0^1 binom(n, k) q^k (1 - q)^(n - k) dif q) = (p^k (1 - p)^(n - k)) / (B(k + 1, n - k + 1)), $
the Beta$(k + 1, n - k + 1)$ density. The chance of being the better player is its mass above one half:
$ P(p > 1/2 | k, n) = 1 / B(k + 1, n - k + 1) integral_(1 slash 2)^1 q^k (1 - q)^(n - k) dif q. $

= The same thing as flipping coins

That integral has a plain-language value.

*The chance of being the better player after $k$ wins in $n$ games equals the chance of at most $k$ heads in $n + 1$ flips of a fair coin:*
$ P(p > 1/2 | k, n) = 1 / 2^(n + 1) sum_(j = 0)^k binom(n + 1, j). $

The argument is Bayes' own billiard table. Drop $n + 1$ points uniformly at random on $[0, 1]$. Call the first one $p$, and let each of the other $n$ points be a game, won when it lands left of $p$. Then the number of wins is the number of other points left of $p$, so "$k$ wins" says that $p$ is the #box[$(k + 1)$th] point from the left. All $n + 1$ points are alike, so given that, $p$ is distributed as the #box[$(k + 1)$th] smallest of $n + 1$ uniform points (which is the Beta$(k + 1, n - k + 1)$ posterior above). That point lies right of $1 slash 2$ exactly when at most $k$ of the $n + 1$ points lie left of $1 slash 2$, and each point does so with chance one half, independently: $n + 1$ fair coin flips.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    line((0, 0), (10, 0), stroke: 0.6pt)
    for (x, t) in ((0, $0$), (5, $1 slash 2$), (10, $1$)) {
      line((x, -0.12), (x, 0.12), stroke: 0.6pt)
      content((x, -0.45), t)
    }
    line((5, -0.12), (5, 1.0), stroke: (paint: luma(50%), dash: "dashed", thickness: 0.6pt))
    let others = (0.6, 1.4, 2.3, 3.1, 3.8, 5.6, 6.9, 7.7, 8.4, 9.1)
    for x in others { circle((x, 0), radius: 0.11, fill: luma(55%), stroke: none) }
    circle((6.3, 0), radius: 0.15, fill: rgb("#0072B2"), stroke: none)
    content((6.3, 0.5), text(10pt, fill: rgb("#0072B2"))[$p$])
    content((2.5, 0.75), text(9pt)[6 games won: left of $p$])
    content((8.2, 0.75), text(9pt)[4 lost])
  }),
  caption: [Eleven random points on $[0, 1]$. The blue one is $p$; the other ten are games, won when they fall left of $p$. Six wins put $p$ seventh from the left, so $p > 1 slash 2$ exactly when at most six of all eleven points fall left of the dashed line.],
)

For six wins in ten games the formula gives $1486 slash 2048 = 743 slash 1024 approx 0.726$. The claim of being better is more likely right than wrong, but about one time in four a player with that record is in fact the weaker one. `peggle.py` checks the identity against the Beta integral, computed exactly with fractions, for every $k$ and every $n$ up to 12.

@tab-wins shows how many wins it takes to be 95% sure.

#figure(
  table(
    columns: 7, align: (left, ..(right,) * 6), stroke: none,
    table.hline(),
    [games $n$], [10], [20], [50], [100], [200], [1,000],
    table.hline(stroke: 0.5pt),
    [wins needed], [8], [14], [31], [59], [112], [527],
    [share], [80.0%], [70.0%], [62.0%], [59.0%], [56.0%], [52.7%],
    [chance of being better], [0.967], [0.961], [0.954], [0.964], [0.955], [0.956],
    table.hline(),
  ),
  caption: [The fewest wins in $n$ games that make $P(p > 1 slash 2) >= 0.95$, uniform prior. Exact.],
) <tab-wins>

The share of wins needed falls toward one half, roughly like $1 slash 2 + 0.82 slash sqrt(n)$, the normal approximation to the coin flips.

= A one-line approximation

The curves in @fig-better look like normal distribution functions. Near $k = n slash 2$ the posterior has mean $(k + 1) slash (n + 2)$ and standard deviation $1 slash (2 sqrt(n + 3))$, so $P(p > 1 slash 2)$ is close to the normal distribution function $Phi(z)$ at $z = 2 (k - n slash 2) sqrt(n + 3) slash (n + 2)$. The logistic curve $1 slash (1 + e^(-1.702 z))$ stays within 0.01 of $Phi(z)$, which gives
$ P(p > 1 slash 2 | k, n) approx 1 / (1 + e^(-a(n) (k - n slash 2))), quad a(n) = (3.404 sqrt(n + 3)) / (n + 2) approx 3.4 / sqrt(n). $
An earlier version of this note used the fitted slope $3 n^(-0.48)$. Over every $k$ and every $n <= 300$, the largest difference from the exact chance is 0.016 with $a(n)$ and 0.068 with the fitted slope; for $30 <= n <= 300$ it is 0.010 and 0.015. `peggle.py` computes all four exactly.

#figure(
  {
    let rows = csv("better.csv", row-type: dictionary)
    let curve(n, col) = {
      let rs = rows.filter(r => int(r.n) == n)
      (rs.map(r => float(r.share)), rs.map(r => float(r.at(col))))
    }
    let colors = (rgb("#E69F00"), rgb("#009E73"), rgb("#0072B2"))
    fitw(lq.diagram(
      width: 10cm, height: 3.2cm,
      xlabel: [share of games won, $k slash n$], ylabel: [$P(p > 1 slash 2)$],
      xlim: (0.2, 0.8), ylim: (0, 1),
      legend: (position: left + top),
      ..((10, 50, 250).zip(colors).map(((n, c)) => {
        let (xs, ys) = curve(n, "exact")
        let (_, ls) = curve(n, "logistic")
        (
          lq.plot(xs, ys, mark: "o", mark-size: 2.5pt, stroke: none, color: c, label: [$n = #n$]),
          lq.plot(xs, ls, mark: none, stroke: 0.8pt + c),
        )
      }).flatten()),
    ))
  },
  caption: [Exact chance of being better (dots) and the approximation with slope $a(n)$ (lines), for 10, 50 and 250 games.],
) <fig-better>

= Playing until someone is sure

Suppose instead that the friends play until one of them is 95% sure of being better, or until $N$ games. The rule looks only at $k$ and $n$, so the number of win-loss sequences that first stop at each $(k, n)$ does not depend on $p$; a dynamic program counts them, and each has chance $p^k (1 - p)^(n - k)$. `stopping.py` sums these exactly (@tab-stop), agrees with seeded simulation, and with a binomial test in place of the posterior reproduces Table 1 of Armitage, McPherson and Rowe @armitage.

#figure(
  table(
    columns: 6, align: (left, right, right, right, right, right), stroke: none,
    table.hline(),
    [$p$], [$N$], [better declared], [worse declared], [no verdict], [mean games],
    table.hline(stroke: 0.5pt),
    [0.5], [100], [0.528], [], [0.472], [59.6],
    [0.5], [1,000], [0.728], [], [0.272], [355.6],
    [0.55], [100], [0.521], [0.120], [0.359], [53.2],
    [0.55], [1,000], [0.866], [0.123], [0.011], [138.0],
    [0.6], [1,000], [0.945], [0.055], [0.000], [48.4],
    [0.7], [1,000], [0.988], [0.012], [0.000], [16.4],
    table.hline(),
  ),
  caption: [Play until one player is 95% sure, or $N$ games. For $p = 0.5$ the first column is the chance that either player is declared better. Exact.],
) <tab-stop>

Between equal players someone is declared better 53% of the time with $N = 100$ and 73% with $N = 1000$, against 8.9% and 9.4% for one look after $N$ games. The threshold on $k - n slash 2$ grows like $0.82 sqrt(n)$, and by the law of the iterated logarithm a fair random walk passes any multiple of $sqrt(n)$, so the chance tends to one as $N$ grows.

Yet each verdict means what it says. Draw $p$ from the uniform prior and play by the same rule: 3.6% of verdicts are wrong with $N = 100$ and 3.7% with $N = 1000$, exactly. The stopping rule multiplies the likelihood by a factor free of $p$, so the posterior at $(k, n)$ is unchanged. The 5% is an average over the prior, not a promise for each $p$; at $p = 0.55$ the worse player is declared better 12% of the time.

Laplace's rule of succession is the same calculation read one game at a time: after $k$ wins in $n$ games the next is won with chance $(k + 1) slash (n + 2)$, the posterior mean. Multiplying these along a sequence gives $k! (n - k)! slash (n + 1)!$, the prior average of $p^k (1 - p)^(n - k)$ and the weight the Bayesian check gives each sequence. Under these chances the posterior is a martingale: when play stops, by any rule, its average is still 1/2.

= A verdict that is safe for equal players <sec-bayes-factor>

The 95% rule keeps its promise on average over the prior, and the uniform prior gives $p = 1 slash 2$ no weight at all. To protect equal players we need a number whose behavior is known when $p = 1 slash 2$ exactly. The *Bayes factor* compares how likely the record is under the two views. After $k$ wins in $n$ games, the uniform prior gives the record the chance $integral_0^1 p^k (1 - p)^(n - k) dif p = 1 slash ((n + 1) binom(n, k))$ (the `prior_path_chance` of `stopping.py`), and equal players give it $2^(-n)$.

#block(breakable: false, width: 100%)[
  Their ratio is
  $ B_n = 2^n / ((n + 1) binom(n, k)). $
]
Six wins in ten games give $B_10 = 1024 slash 2310 approx 0.44$: that record is more likely between equal players. The *Bayes factor rule* plays until $B_n >= 20$, or until $N$ games, and then declares better whoever has won more games. It takes 66 wins in 100 games, or 556 in 1,000, against 59 and 527 for the 95% rule.

*The Bayes factor is a fair bet against equal players.* Suppose $p = 1 slash 2$. The next game is won or lost with chance one half each, and the average of the two new factors is the old one:
$ 1/2 B_(n + 1)(k + 1) + 1/2 B_(n + 1)(k) = B_n (k). $
(Both sides equal the prior average of $(2p)^k (2 - 2p)^(n - k)$, and the next game multiplies that by $2p$ or $2 - 2p$, which average to 1.) So $B_n$ is a *test martingale*: a nonnegative fair game that starts at 1. It is the wealth of a gambler who starts with one unit and bets at even odds, spreading the stake over every $p$ by the uniform prior @shafer2011 @savi. A fair game cannot be made favorable by choosing when to stop. Stopped at a verdict or at $N$ games, $B$ still averages exactly 1; `stopping.py` checks this with exact fractions for several caps up to 200.

#block(width: 100%, inset: (y: 2pt))[
  *Proposition (Ville's inequality, here).* Suppose $p = 1 slash 2$. However long the friends play, the Bayes factor rule declares someone better with chance at most $1 slash 20$.
]

_Proof._ Fix a cap $N$. When play stops, $B >= 20$ if there is a verdict and $B >= 0$ otherwise, and $B$ averages 1. So $1 >= 20 P("verdict")$. The bound does not depend on $N$, so it holds for unlimited play too. #h(1fr)$qed$

Ville proved this for every test martingale @ville @savi, and Shafer and Vovk @shafervovk build probability itself on such betting games. The exact chances are 0.0276 for $N = 100$ and 0.0357 for $N = 1000$, against 0.528 and 0.728 for the 95% rule (@tab-rules). The chance grows with $N$, to 0.0387 at $N = 10000$ (computed in floating point), but it never reaches 0.05. Between equal players $B_n$ drifts to 0 on almost every path that never reaches 20 (by the law of the iterated logarithm it is at most about $(log n)^(1 + epsilon) slash sqrt(n)$), so in unlimited play the chance of a verdict is exactly 1 divided by the average factor at the verdict. That factor jumps past 20 rather than landing on it: the first possible verdict, eight wins in eight games, has $B_8 = 256 slash 9 approx 28.4$.

The two rules answer different questions. The 95% rule says that, of all the pairs of players a uniform prior could produce, about 95% of those declared better are better; it says nothing about a pair who are exactly equal. The Bayes factor rule promises that a pair of equal players is declared unequal at most one time in twenty, whenever the friends stop. It is also a Bayesian rule, with a prior that takes equality seriously: give "equal" chance one half, and spread the other half uniformly over $p$. The posterior chance of "equal" is then $1 slash (1 + B_n)$, and $B_n >= 20$ means that it has fallen to $1 slash 21$ or less. The two priors disagree about records such as 527 wins in 1,000 games, where the uniform prior is 95.6% sure while $B_1000 approx 0.17$ favors equality. This disagreement is Lindley's paradox.

The safety has a price. At $p = 0.55$ the Bayes factor rule wrongly favors the worse player only 0.4% of the time, but within 1,000 games it reaches a verdict only 51% of the time, after 745 games on average.

The Bayes factor rule is safe for every cap at once. When the cap is fixed in advance, the 95% rule can be made safe for equal players by asking for more certainty. The chance of a verdict between equal players falls as the threshold rises, and it changes only at thresholds equal to a posterior that occurs. So bisection, with the dynamic program at each step, finds the least strict threshold that keeps the chance at or below 5%. The posterior must reach 0.99704 with $N = 100$ and 0.9987003 with $N = 1000$; a threshold of 0.997037 or 0.9987002, a little lower, lets the chance pass 5%. Equal players are then declared unequal with chance 0.0497 for both caps, exactly. These rules need 64 wins in 100 games, or 548 in 1,000. They spend the whole 5% within the cap, so at $p = 0.55$ they declare the better player better with chance 0.113 and 0.698, against 0.061 and 0.512 for the Bayes factor rule, after 93.8 and 637.9 games on average (@tab-rules).

#figure(
  placement: auto,
  text(10pt, table(
    columns: 7, align: (left, right, right, right, right, right, right), stroke: none,
    inset: (x: 4.5pt, y: 3.5pt),
    table.hline(),
    table.header(
      table.cell(rowspan: 2, align: bottom)[rule], table.cell(rowspan: 2, align: bottom + right)[$N$],
      table.cell(colspan: 2, align: center)[equal players], table.cell(colspan: 3, align: center)[$p = 0.55$],
      table.hline(start: 2, end: 4, stroke: 0.4pt), table.hline(start: 4, end: 7, stroke: 0.4pt),
      [verdict], [games], [better], [worse], [games],
    ),
    table.hline(stroke: 0.5pt),
    [posterior 95%], [100], [0.528], [59.6], [0.521], [0.120], [53.2],
    [posterior 95%], [1,000], [0.728], [355.6], [0.866], [0.123], [138.0],
    [Bayes factor 20], [100], [0.028], [98.0], [0.061], [0.004], [96.2],
    [Bayes factor 20], [1,000], [0.036], [967.7], [0.512], [0.004], [745.0],
    [posterior 0.99704], [100], [0.050], [96.8], [0.113], [0.005], [93.8],
    [posterior 0.9987003], [1,000], [0.050], [960.8], [0.698], [0.003], [637.9],
    [lead of 15], [1,000], [0.995], [224.0], [0.953], [0.047], [135.9],
    [lead of 15], [none], [1], [225], [0.953], [0.047], [135.9],
    table.hline(),
  )),
  caption: [Four stopping rules: the posterior rule at 95% and at the thresholds that keep equal players' chance of a verdict at 5%, the Bayes factor rule and Wald's test. For equal players, the chance that someone is declared better and the mean number of games; for $p = 0.55$, the chances that the better or the worse player is declared better, and the mean number of games. Exact, by the dynamic program; the last row, with no cap, by the gambler's ruin.],
) <tab-rules>

= Wald's test: play until someone leads by 15 <sec-wald>

Suppose the friends only want to tell $p = 0.45$ from $p = 0.55$, a 55–45 edge for one of them. After $k$ wins in $n$ games the likelihood ratio of the two is $(11 slash 9)^(2k - n)$, which depends only on the lead $2k - n$. Wald's sequential probability ratio test @wald stops when the ratio leaves a fixed interval, so it is *play until one player leads by $c$ games*. The lead then moves one step up or down after each game until it hits a wall at $+c$ or $-c$: the gambler's ruin. If the first player wins each game with chance $p$ and $r = (1 - p) slash p$, the first player reaches $+c$ first with chance $1 slash (1 + r^c)$, and play lasts $c (1 - r^c) slash ((1 + r^c)(2p - 1))$ games on average, or $c^2$ when $p = 1 slash 2$. At $p = 0.55$ the worse player is declared better with chance $9^15 slash (9^15 + 11^15) = 0.0470$, after 135.9 games on average. A lead of 14 would err 5.7% of the time, so 15 is the smallest lead that errs at most 5%. Because the lead moves one game at a time, it lands exactly on $plus.minus 15$, and Wald's approximations, which ignore overshoot, are exact here.

Wald and Wolfowitz @waldwolfowitz proved that no test with error chances at most these, at $p = 0.45$ and at $p = 0.55$, uses fewer games on average at either value. The 95% rule with $N = 1000$ also needs about 138 games at $p = 0.55$, but errs 12.3% of the time. Wald's test is not a test of equality: at $p = 1 slash 2$ it declares someone better after 225 games on average, as it must, since it only chooses between 0.45 and 0.55.


= Questions left open

- *Other priors.* A Beta$(a, a)$ prior with a whole number $a$ adds $2a - 1$ points to the billiard table: the chance becomes that of at most $k + a - 1$ heads in $n + 2a - 1$ flips.
- *Other boundaries.* A threshold that changes with $n$ could keep the 5% for equal players and declare the better player better more often at $p = 0.55$ than one fixed threshold does. Which boundary does this best for a given cap is not worked out here.

#bibliography("peggle.yml", style: "ieee")
