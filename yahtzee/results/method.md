# How the solver works, and its checks

The state space, the dynamic program, the exact distribution, and every check. The [Yahtzee README](../README.md) gives the short answers. "Exact" means computed by the dynamic program in double precision.

## How it works

The state between turns is Glenn's triple: the set of filled boxes, the upper total capped at 63, and whether the Yahtzee box holds 50. A higher total earns no more, so 63 stands for 63 or more. There are 2¹³ · 64 · 2 = 1,048,576 triples, and 536,448 of them can occur. A total can be impossible for the filled upper boxes; for example, a total of 1 cannot come from the Twos box alone. And the Yahtzee box cannot hold 50 while it is open. Glenn says 1,260 of the 4,096 pairs of upper boxes and totals are impossible; the count here is 1,302, with 63 standing for 63 or more as Glenn's own convention says, and a count without that cap gives 1,304. Neither reading gives 1,260. That gap was not resolved, but the state count matches Verhoeff's.

Within a turn, a roll is a sorted multiset of dice (252 of them) and a keep is a multiset of zero to five dice (462). `solver.py` solves each turn backwards:

1. For each final roll, it takes the best open box: its points plus the value of the state it leads to.
2. For each keep, it averages over the rerolls, as a product with the 462 × 252 matrix of reroll chances.
3. For each roll, it takes the best keep inside it. That maximum is built up one die at a time, from the keeps of four dice down to the empty keep.

It does this for 2,048 states at once with numpy, layer by layer from full score cards down to the empty one. It takes about 50 seconds on one core.

`policy.py` then follows the optimal choices forward from the empty card. It records, for each of the 284,312 states that optimal play reaches, every way the turn can end: the next state, the points and the chance. Summing over those edges backwards gives the exact distribution of the final score. Each state's distribution is a mixture of the next states' distributions, shifted by the points earned. The per-box statistics are sums of each state's chance of being reached times what its turn earns.

## Checks

- `brute.py` solves small games by plain recursion, with exact fractions. Its dice are ordered, a keep is any subset of their positions, and the state is the whole score card. It shares nothing with the solver except the scoring rules. The tests compare the two on seven games with two to five dice, two to four faces and three to five boxes, each with an upper bonus at a lower threshold. Three of those games differ only in their Joker reading and have three different values. The values from the two methods differ by less than 10⁻¹³.
- One-box games have closed forms. Chance alone is worth 70/3, and its distribution is that of five independent dice. Yahtzee alone matches an independent chain over the largest group kept. With only Aces open and an upper total of 62, the value is 5 · 91/216 + 35 · (1 − (5/6)¹⁵).
- Three numbers differ by less than 10⁻¹³: the solver's value, the mean of the computed distribution and the sum of the per-box means. Seeded simulations of small games match their exact distributions.
- The tests check the published values above against the full solve, and check `openings.csv` and `distribution.csv` against it.

- A strategy card is scored by two programs that share only the card: the solver told to follow it (`solver.solve(rule=card)`), and the card's own forward and backward passes in `human.py`. They agree to 10⁻⁹ on every state the tests compare. Seeded games played by the card and scored by `rules.py` agree with the exact means: the cheat sheet, Glenn's lists and a test card that uses most of the rule vocabulary. The card's moves are worked out once per combination of open boxes, 50 in the Yahtzee box and bonus pace; a test checks that this gives the same moves as working them out state by state.
- The tests recompute the committed scores of the cheat sheet, Glenn's lists, the shortest cards on the frontier and one piece of advice, and check that the README holds the cheat sheet as `cards.py` writes it.

Not done: exact rational arithmetic for the full game. That would test whether any decision is a near-tie that rounding gets wrong. It would also settle the comparison with Kelly and Liese's fraction.
