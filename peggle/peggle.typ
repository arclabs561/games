#import "@preview/cetz:0.4.2"
#import "@preview/lilaq:0.6.0" as lq

// Every number in this note is printed by peggle.py; the plot reads better.csv, which it writes.
#set document(title: "Who Is the Better Player?")
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
  placement: auto,
  table(
    columns: 4, align: right, stroke: none,
    table.hline(),
    [games $n$], [wins needed], [share], [chance of being better],
    table.hline(stroke: 0.5pt),
    [10], [8], [80.0%], [0.967],
    [20], [14], [70.0%], [0.961],
    [50], [31], [62.0%], [0.954],
    [100], [59], [59.0%], [0.964],
    [200], [112], [56.0%], [0.955],
    [1,000], [527], [52.7%], [0.956],
    table.hline(),
  ),
  caption: [The fewest wins in $n$ games that make $P(p > 1 slash 2) >= 0.95$ under a uniform prior. Exact.],
) <tab-wins>

The share of wins needed falls toward one half, roughly like $1 slash 2 + 0.82 slash sqrt(n)$, the normal approximation to the coin flips.

= A one-line approximation

An earlier version of this note gave the logistic curve
$ P(p > 1/2 | k, n) approx 1 / (1 + e^(-3 n^(-0.48) (k - n slash 2))), $
Against the exact values its error is at most 0.068 for $n <= 300$ (the worst case is $n = 1$), at most 0.016 for $10 <= n <= 300$, and at most 0.015 for $30 <= n <= 300$. @fig-better shows why it works: the exact curve is close to a normal distribution function, which a logistic curve imitates well.

#figure(
  {
    let rows = csv("better.csv", row-type: dictionary)
    let curve(n, col) = {
      let rs = rows.filter(r => int(r.n) == n)
      (rs.map(r => float(r.share)), rs.map(r => float(r.at(col))))
    }
    let colors = (rgb("#E69F00"), rgb("#009E73"), rgb("#0072B2"))
    fitw(lq.diagram(
      width: 10cm, height: 4cm,
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
  caption: [Exact chance of being the better player (dots) and the logistic approximation (lines), for 10, 50 and 250 games.],
) <fig-better>

= Questions left open

- *Other priors.* A Beta$(a, a)$ prior with a whole number $a$ adds $2a - 1$ points to the billiard table: the chance becomes that of at most $k + a - 1$ heads in $n + 2a - 1$ flips.
- *Stopping when ahead* leaves the posterior unchanged, but fools frequentist tests more often.
