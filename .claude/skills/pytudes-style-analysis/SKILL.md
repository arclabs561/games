---
name: pytudes-style-analysis
description: Habits for short programs that answer questions about games and puzzles, drawn from Peter Norvig's pytudes notebooks: vocabulary first, small tested functions, brute force checked against fast methods, exhaustive evaluation.
---

# Pytudes-style analysis

Distilled from Peter Norvig's pytudes (github.com/norvig/pytudes, MIT license), mainly the
Boggle, Countdown, Jotto/Wordle, Can't Stop, Probability, How to Count Things and Sudoku
notebooks. Use it when writing a notebook, script or paper section that answers a question
about a game or puzzle with code.

## Shape of a study

1. **Say the problem in plain words**, link the rules, and show one concrete instance (a board,
   a deal, a target). Then name the harder or more general question the study will reach
   (Boggle, then Inverse Boggle; one Countdown answer, then how many answers).
2. **Introduce the vocabulary.** Each concept gets a bold name at first use, one sentence, and a
   type alias or small class (Jotto: guesser, target, reply, consistent targets, partition;
   Countdown: `Numbers`, `Exp`, `ExpTable`). Use the same word for the same thing afterwards.
3. **Choose the representation and say why.** A Boggle board is a dict from coordinates to
   letters with neighbors precomputed; a Can't Stop roll is the set of pairs it can make, and a
   `Counter` of rolls folds the 24 orderings of four dice into one weighted entry.
4. **Plan in steps, then walk the plan through the concrete instance before the code.** Boggle
   traces the path and prefix for the top-left square; Countdown traces one split of (3, 2, 1).
5. **Write short functions** with a one-line docstring and a return type. Precompute shared
   tables once as named constants (`PREFIXES`). Reuse small helpers (`iterate`, `same`).
6. **Put tests right after each definition**, on cases small enough to check by hand, with the
   hand enumeration in a comment (`expression_counts((3, 2, 1))[1] == 5`, followed by the five
   expressions). Settle every ambiguous rule with a test that matches how the real game behaves
   (Wordle's reply for NINNY against ANNEX).
7. **Ask, compute, interpret.** Pose a question in a sentence, answer it with one expression,
   then say in prose what the output means. Time anything slow and report the time.
8. **End with variations and open questions**: other rules, other sizes, ideas not tried.

## Method

- **Solve the general problem, then look up the specific one.** Countdown builds a table of
  every value the numbers can make, and the 2016 answer is one lookup; nearby years cost nothing.
- **Brute force first, then a faster method, and check they agree** on every small case before
  trusting the fast one at scale (How to Count Things: enumerate barcodes, then incremental,
  then abstract incremental enumeration, with `same(f, g, range(13))` between each step).
- **Look at a small sample before the full set**: every 100th word, every 250,000th hand. A
  partition of 24 words is readable; a partition of the full list is not.
- **Evaluate every case when it is feasible.** Score a strategy over all targets, all license
  plates or all rolls, not a random few, and report median, mean, standard deviation, worst case
  and the cumulative share within each budget.
- **Name the metric you optimize** when the real goal cannot be computed directly (Jotto lists
  maximum, expectation and entropy of branch sizes as proxies, then compares the trees they give).
- **Distrust floating point near a target.** Countdown's float count of 2016 expressions was
  wrong; re-checking the near-2016 values with `Fraction` changed the answer. Use `Fraction` for
  probabilities (the Probability notebook's `P` returns one).
- **Define what counts before counting.** Whether `(10+9)+8` and `10+(9+8)` are distinct changes
  the count; state the definition next to the number.
- **Look for counterexamples to reasonable assumptions.** Guessing only consistent words seems
  sensible until nine rhyming words make an inconsistent guess better.
- **Check against outside results**: a published high-scoring board (Boyan's), a sequence lookup
  in OEIS, a published count. Agreement is evidence; disagreement is a finding.
- **Explain surprises** rather than reporting them (24 is not special among targets; in Can't
  Stop every roll makes some even column, which is why even columns top the lists).
- **Use `@cache` as the first dynamic program**, and clear caches when a section is done.
- **State confidence.** Say which numbers are checked by simple arithmetic and which would need
  review or an independent implementation.

## Additions for this kind of repository

- Seed every random choice behind a reported number, and print the seed with the result.
- Keep the code a notebook runs importable from a module, so tests can import it instead of
  executing notebook cells.
- Separate exact results from simulated ones, with sample size and interval for the latter.
