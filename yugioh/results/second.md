# Going second, two roles and the Normal Summon

Written by `uv run python -m yugioh.second`. Exact.

## Hand traps in 40 cards

Fewest hand traps for the chance in each column:

| Hand traps in hand | 80% | 90% | 95% |
|---|---:|---:|---:|
| at least one, going first | 11 | 14 | 18 |
| at least one, going second | 9 | 12 | 15 |
| at least two, going first | 20 | 23 | 26 |
| at least two, going second | 17 | 20 | 23 |

At least one is the same count as for starters, since only the number of cards matters. With 12 hand traps, going second gives at least one with chance 90.18% and at least two with chance 59.46%.

## A starter and a hand trap, going second

Going second (6 cards) in 40, with no card in both roles:

| Starters \ Hand traps | 6 | 9 | 12 | 15 |
|---|---:|---:|---:|---:|
| 9 | 50.39% | 63.58% | 71.71% | 76.41% |
| 12 | 57.09% | 71.71% | 80.58% | 85.62% |
| 15 | 61.05% | 76.41% | 85.62% | 90.78% |

With 28 cards to split between the two roles, 14 starters and 14 hand traps give the most, 88.03%.

The example deck (12 starters, 9 extenders, 3 bricks), with some of its 16 other cards as hand traps, going second:

| Hand traps | A starter and a hand trap | A starter, an extender and a hand trap | A starter or a hand trap |
|---|---:|---:|---:|
| 0 | 0.00% | 0.00% | 90.18% |
| 6 | 57.09% | 43.18% | 98.06% |
| 9 | 71.71% | 55.17% | 99.29% |
| 12 | 80.58% | 62.81% | 99.79% |
| 15 | 85.62% | 67.35% | 99.96% |

## Starters that are also extenders

The example deck going first, with some of its 12 starters also extenders and its 9 extenders unchanged. The need is two different cards, a starter and an extender. Counting a card with both roles once for each role, as if it met the need alone, gives the last value.

| Starters that also extend | A starter and an extender | Counted twice | Overstated by (points) | Exact |
|---|---:|---:|---:|---:|
| 0 | 61.01% | 61.01% | 0.00 | 44605/73112 |
| 3 | 70.13% | 71.90% | 1.77 | 493/703 |
| 6 | 75.22% | 78.76% | 3.53 | 82495/109668 |
| 9 | 77.53% | 82.83% | 5.30 | 28341/36556 |
| 12 | 78.00% | 85.06% | 7.07 | 7128/9139 |

## The one Normal Summon

The example deck going first. Some starters and some extenders need the turn's Normal Summon, and two such cards cannot both be played. A starter and an extender, using the Normal Summon at most once:

| Starters that need it \ Extenders that need it | 0 | 3 | 6 | 9 |
|---|---:|---:|---:|---:|
| 0 | 61.01% | 61.01% | 61.01% | 61.01% |
| 4 | 61.01% | 58.36% | 54.38% | 48.69% |
| 8 | 61.01% | 53.86% | 43.52% | 29.15% |
| 12 | 61.01% | 46.78% | 26.89% | 0.00% |

A starter and another engine card (a second starter or an extender), the same way:

| Starters that need it \ Extenders that need it | 0 | 3 | 6 | 9 |
|---|---:|---:|---:|---:|
| 0 | 78.00% | 78.00% | 78.00% | 78.00% |
| 4 | 77.00% | 74.36% | 70.38% | 64.68% |
| 8 | 72.21% | 65.05% | 54.71% | 40.35% |
| 12 | 61.01% | 46.78% | 26.89% | 0.00% |

At least one starter does not change, since one card uses the Normal Summon at most once.
