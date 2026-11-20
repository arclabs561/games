# Peggle: who is the better player?

Two friends play a game with no draws, such as a Peggle duel, and one wins k of n games: what is the chance that this player is the better one? With a uniform prior it equals the chance of at most k heads in n + 1 fair coin flips, so six wins in ten give 743/1024, about 0.726. Playing on until one player is 95% sure declares one of two equal players better with chance 0.528 within 100 games; playing until the Bayes factor reaches 20 keeps that under 0.05 however long they play. Exact, by Bayes's billiard-table argument and a dynamic program over (wins, games).

![A row of ten squares, six marked W and four marked L, above the posterior density of the winner's chance p, which peaks at 0.6; the area to the right of 1/2 is shaded and labeled 743/1024 ≈ 0.726.](figures/record.svg)

*Six wins in ten games, and what they say about the winner's chance p of winning a game. The shaded area is the chance that the winner is the better player.*

## Rules as modeled

- Every game is won by one of the two players, and the first player wins each game with the same chance p, independently of the others. The last section relaxes this.
- "The better player" means p > 1/2. The prior on p is uniform unless stated; the Jeffreys prior Beta(1/2, 1/2) and a skeptical prior Beta(12, 12) are compared.
- A **stopping rule** says, after each game, whether to stop and declare someone better. A **cap** N is the most games that the friends will play.
- The duel rules come from the Peggle Deluxe readme (PopCap, 2008, archived), which does not say who shoots first. The first-shooter edges in the note are assumed values, not measurements.

## Results

All results are exact unless marked.

| Result | Value | How it is known |
|---|---|---|
| Chance the winner of six games in ten is better (uniform prior) | 743/1024 ≈ 0.726 | proved: the chance of at most 6 heads in 11 fair flips |
| Wins needed to be 95% sure | 8 in 10, 59 in 100, 527 in 1,000 | exact |
| Equal players, play until 95% sure, cap 100 | someone declared better with chance 0.528 | dynamic program over (wins, games) |
| Equal players, play until the Bayes factor reaches 20 | 0.0276 (N = 100); never more than 0.05 for any cap | dynamic program; Ville's inequality |
| Stricter posterior threshold that keeps it at 5% | 0.99704 (N = 100), 0.9987003 (N = 1,000) | bisection with the same dynamic program |
| Better player (p = 0.55) wins a best-of-seven match | 0.608 | exact |
| "Play until one leads by 15", p = 0.55 | the worse player is declared better with chance 0.0470 | gambler's ruin |
| Repeated binomial tests | 32 of 34 entries of Armitage, McPherson and Rowe (1969) agree to five decimals, the other two within 2·10⁻⁵ | comparison with a published table |

## How the results are checked

- The tests check the dynamic programs against brute force over all sequences of up to 10 or 12 games.
- They check the coin-flip identity against the Beta integral with exact fractions, the gambler's-ruin formulas against an exact solution of the chain's equations, and the Jeffreys chances against numerical integration.
- They check the 95% rule against Table 1 of Armitage, McPherson and Rowe (1969), which gives the same chances for repeated binomial tests. Of its 34 entries for 2α = 0.01 and 0.05, 32 agree to the five printed decimals and the other two agree within 2·10⁻⁵.
- `python -m peggle.stopping` prints a 200,000-trial simulation for each of four cases; all eight exact values fall inside the 95% intervals.
- The tests check every number quoted in the note against the scripts, several quoted in this README, and that the committed figures are what `figures.py` draws. CI runs them.

## What do six wins in ten say?

With a uniform prior on the chance p of winning a game, the answer equals the chance of at most k heads in n + 1 fair coin flips. Six wins in ten games give 743/1024, about 0.726. Being 95% sure takes 8 wins in 10 games, 59 in 100, or 527 in 1,000. The curve is close to a logistic one, and its slope follows from the spread of the posterior. The proof is Bayes's billiard table: drop n + 1 random points on a line, and read them once as a prior and a record and once as coin flips. A Beta(a, b) prior with whole numbers a and b adds a - 1 wins and b - 1 losses to the record, so it also gives a count of coin flips.

## Does the prior matter?

The prior matters little after 100 games and a lot before 20. After six wins in ten, the Jeffreys prior Beta(1/2, 1/2) gives 1/2 + 512/(693π), about 0.735, and a skeptical prior Beta(12, 12), with standard deviation 1/10, gives 0.636. To be 95% sure in 10 games, the skeptic needs ten wins instead of eight. In 1,000 games all three priors need 527.

## What happens when they play until someone is sure?

Playing until one player is 95% sure, with a cap of N games, changes the picture for equal players. Someone is then declared better with chance 0.528 when N = 100 and 0.728 when N = 1,000, against 0.089 and 0.094 for one look after N games. When the first player wins each game with chance 0.55, the worse player is declared better with chance 0.120 (N = 100) or 0.123 (N = 1,000). Yet if p is drawn from the uniform prior, only 3.6% (N = 100) or 3.7% (N = 1,000) of verdicts are wrong. The stopping rule does not change the posterior, so each verdict keeps its 95%. If p instead comes from Beta(12, 12), 14.0% (N = 100) or 12.2% (N = 1,000) of the uniform rule's verdicts are wrong. All of these numbers are exact, from a dynamic program over (wins, games).

![The chance that one of two equal players is declared better, against the cap from 10 to 200 games, for three rules. Playing until 95% sure rises from about 0.22 to about 0.60; one look after all the games stays near 0.09; playing until the Bayes factor reaches 20 stays below 0.05.](figures/stopping.svg)

*Equal players, exact chances. Only the Bayes factor rule stays under the dashed 5% line however many games they allow.*

## Which rules are safe for equal players?

A rule that is safe for equal players needs a guarantee at p = 1/2 itself. The Bayes factor 2ⁿ / ((n + 1)·C(n, k)) says how many times likelier the record is if p is drawn from the uniform prior than if p = 1/2. When p = 1/2 it is a fair game that starts at 1 (a test martingale), so by Ville's inequality it ever reaches 20 with chance at most 1/20, whatever the stopping rule. Playing until it reaches 20 declares equal players unequal with chance 0.0276 (N = 100) or 0.0357 (N = 1,000), exactly, and 0.0387 at N = 10,000 in floating point. The price is speed: the rule needs 66 wins in 100 games where the 95% rule needs 59, and at p = 0.55 it reaches a verdict within 1,000 games only 51% of the time, after 745 games on average, though it then favors the worse player only 0.4% of the time.

With the cap fixed in advance, a stricter posterior threshold is also safe. The chance of a verdict between equal players changes only at thresholds equal to a posterior that occurs, so bisection with the same dynamic program finds the least strict threshold that keeps it at or below 5%: 0.99704 for N = 100 and 0.9987003 for N = 1,000. The chance is then 0.0497 for both, exactly. These rules need 64 wins in 100 games or 548 in 1,000. At p = 0.55 they declare the better player better with chance 0.113 (N = 100) or 0.698 (N = 1,000), after 93.8 or 637.9 games on average, against 0.061 and 0.512 for the Bayes factor rule. They gain because they spend the whole 5% within the cap.

Wald's sequential test of p = 0.45 against p = 0.55 is "play until one player leads by 15 games", since the likelihood ratio depends only on the lead. By the gambler's ruin, at p = 0.55 it declares the worse player better with chance 9¹⁵/(9¹⁵ + 11¹⁵) = 0.0470, after 135.9 games on average, exactly. A lead of 14 would err 5.7% of the time. The 95% rule with N = 1,000 needs about as many games, 138.0, and errs 12.3% of the time. Wald's test does not test equality: at p = 1/2 it declares someone better after 225 games on average.

## What do match formats do?

Match formats are stopping rules too. At p = 0.55 the better player wins a best-of-seven match with chance 0.608 and a win-by-two match with chance 121/202, about 0.599. For a fixed format to reach 95% takes best of 269, which lasts 244.8 games on average when play stops once the match is decided. On the Elo scale a 55% edge is a gap of 34.9 points, and the lead needed for 5% error is the smallest whole number at least 400·log₁₀(19)/D, about 511.5/D, for a gap of D points. Players 400 points apart (p = 10/11) are separated by win by two, which errs with chance 1/101.

## What if the chance changes from game to game?

The model assumes that every game has the same chance. If, between equal players, whoever shoots first wins with chance 0.6, then taking turns to start is harmless: one look after 100 games declares someone better with chance 0.0822 instead of 0.0886, and Hoeffding's 1956 theorem says that alternating can never raise this chance. If instead the last winner shoots first, the chance rises to 0.1645, and the 95% rule with N = 100 declares someone better 74% of the time. A player whose chance rises from 0.45 to 0.55 halfway through 1,000 games is usually declared the worse player by the 95% rule (chance 0.810), because the rule stops early. These numbers are exact.

## Sources

- T. Bayes (1763), the billiard-table argument, as presented in `peggle.pdf`.
- P. Armitage, C. K. McPherson and B. C. Rowe, Repeated significance tests on accumulating data, *Journal of the Royal Statistical Society A* 132 (1969), Table 1.
- J. Ville, *Étude critique de la notion de collectif* (1939), for the bound on a test martingale.
- A. Wald, Sequential tests of statistical hypotheses (1945), for the test that stops at a fixed lead.
- W. Hoeffding, On the distribution of the number of successes in independent trials (1956), for the theorem on alternating starts.
- PopCap, Peggle Deluxe readme (2008, archived), for the duel rules.

`peggle.pdf` gives the full references.

## Open questions

- Who shoots first in a Peggle duel, and how large is that edge? The first-shooter edges here are assumed values, not measurements.
- A player whose chance rises from 0.45 to 0.55 halfway through 1,000 games is usually declared the worse player by the 95% rule. Does some rule that looks only at wins and games avoid this, or does it need the order of the games?

## Usage

```bash
uv run python -m peggle.peggle
uv run python -m peggle.stopping                       # about 10 seconds
uv run python -m peggle.variations                     # about 13 seconds
uv run tools/typst_vendor.py peggle/peggle.typ --all   # the PDF, the phone PDF and the HTML page
uv run python -m peggle.figures                        # the figures in figures/
```

## Files

- `peggle.pdf`, from `peggle.typ`, explains and proves this, with a reading guide, puzzles and a glossary. `peggle-phone.pdf` is the same text on a narrow page for reading on a phone. The HTML page that `--all` also writes is not committed.
- `peggle.py` checks the coin-flip identity against the Beta integral with exact fractions, computes the tables, and writes `better.csv` for the figure.
- `stopping.py` computes the optional-stopping chances exactly.
- `variations.py` computes the other priors, the match formats, the ratings, and the chances that change from game to game, and writes `priors.csv`, `walk.csv` and `formats.csv` for the figures.
- `figures.py` draws the figures in `figures/`.
