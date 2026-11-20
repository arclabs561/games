# Opening-hand tables

Written by `uv run python -m yugioh.tables`. Exact.

## The example deck: 40 cards, 12 starters, 9 extenders, 3 bricks

| Need | Going first | Going second | First, exact | Second, exact |
|---|---:|---:|---:|---:|
| at least one starter | 85.06% | 90.18% | 598/703 | 634/703 |
| a starter and an extender | 61.01% | 71.71% | 44605/73112 | 131071/182780 |
| a starter and another engine card | 78.00% | 86.55% | 7128/9139 | 276842/319865 |
| no brick | 66.24% | 60.57% | 1309/1976 | 748/1235 |
| a starter and no brick | 58.17% | 55.95% | 127589/219336 | 4037/7215 |

## At least one starter in 40 cards

| Starters | Going first (5) | Going second (6) | Going first, exact |
|---|---:|---:|---:|
| 0 | 0.00% | 0.00% | 0/1 |
| 1 | 12.50% | 15.00% | 1/8 |
| 2 | 23.72% | 28.08% | 37/156 |
| 3 | 33.76% | 39.43% | 667/1976 |
| 4 | 42.71% | 49.25% | 3903/9139 |
| 5 | 50.66% | 57.71% | 41672/82251 |
| 6 | 57.71% | 64.96% | 15823/27417 |
| 7 | 63.93% | 71.14% | 17528/27417 |
| 8 | 69.40% | 76.39% | 57079/82251 |
| 9 | 74.18% | 80.82% | 54233/73112 |
| 10 | 78.34% | 84.53% | 2203/2812 |
| 11 | 81.95% | 87.62% | 4609/5624 |
| 12 | 85.06% | 90.18% | 598/703 |
| 13 | 87.73% | 92.29% | 2467/2812 |
| 14 | 90.00% | 94.00% | 11389/12654 |
| 15 | 91.93% | 95.39% | 100813/109668 |
| 16 | 93.54% | 96.49% | 25646/27417 |
| 17 | 94.89% | 97.37% | 32861/34632 |
| 18 | 96.00% | 98.06% | 1847/1924 |
| 19 | 96.91% | 98.59% | 3729/3848 |
| 20 | 97.64% | 98.99% | 1409/1443 |

A rough check: drawing with replacement, 12 starters in 40 give 1 − (28/40)^5 = 83.19% going first, against the exact 85.06%.
Without replacement each miss leaves fewer non-starters, so the exact chance is higher.

## Fewest starters for at least one in hand

| Deck | First, 80% | First, 90% | First, 95% | Second, 80% | Second, 90% | Second, 95% |
|---|---:|---:|---:|---:|---:|---:|
| 40 | 11 | 14 | 18 | 9 | 12 | 15 |
| 41 | 11 | 15 | 18 | 10 | 13 | 16 |
| 42 | 11 | 15 | 18 | 10 | 13 | 16 |
| 43 | 12 | 16 | 19 | 10 | 13 | 16 |
| 44 | 12 | 16 | 19 | 10 | 14 | 17 |
| 45 | 12 | 16 | 20 | 10 | 14 | 17 |
| 46 | 13 | 17 | 20 | 11 | 14 | 18 |
| 47 | 13 | 17 | 21 | 11 | 15 | 18 |
| 48 | 13 | 17 | 21 | 11 | 15 | 18 |
| 49 | 13 | 18 | 22 | 11 | 15 | 19 |
| 50 | 14 | 18 | 22 | 12 | 16 | 19 |
| 51 | 14 | 19 | 23 | 12 | 16 | 20 |
| 52 | 14 | 19 | 23 | 12 | 16 | 20 |
| 53 | 15 | 19 | 23 | 12 | 17 | 20 |
| 54 | 15 | 20 | 24 | 13 | 17 | 21 |
| 55 | 15 | 20 | 24 | 13 | 17 | 21 |
| 56 | 15 | 20 | 25 | 13 | 18 | 21 |
| 57 | 16 | 21 | 25 | 13 | 18 | 22 |
| 58 | 16 | 21 | 26 | 14 | 18 | 22 |
| 59 | 16 | 22 | 26 | 14 | 18 | 23 |
| 60 | 16 | 22 | 27 | 14 | 19 | 23 |

In 40 cards, the chance at the threshold and one starter below it:

- 80% going first: 11 starters give 81.952%, 10 give 78.343%.
- 80% going second: 9 starters give 80.818%, 8 give 76.391%.
- 90% going first: 14 starters give 90.003%, 13 give 87.731%.
- 90% going second: 12 starters give 90.185%, 11 give 87.624%.
- 95% going first: 18 starters give 95.998%, 17 give 94.886%.
- 95% going second: 15 starters give 95.386%, 14 give 94.002%.

## Twelve starters in a larger deck

| Deck | First | Change (points) | Second | Change (points) | Starters to match 40 cards, first |
|---|---:|---:|---:|---:|---:|
| 40 | 85.06% | +0.00 | 90.18% | +0.00 | 12 |
| 41 | 84.15% | -0.91 | 89.44% | -0.75 | 13 |
| 42 | 83.25% | -1.82 | 88.68% | -1.50 | 13 |
| 43 | 82.35% | -2.72 | 87.92% | -2.26 | 13 |
| 44 | 81.46% | -3.61 | 87.16% | -3.02 | 14 |
| 45 | 80.57% | -4.49 | 86.40% | -3.78 | 14 |
| 50 | 76.31% | -8.75 | 82.63% | -7.56 | 16 |
| 55 | 72.33% | -12.73 | 78.97% | -11.21 | 17 |
| 60 | 68.65% | -16.42 | 75.49% | -14.70 | 19 |

## A starter and an extender in 40 cards

At least one of each, going first; rows are starters, columns extenders:

| Starters | 3 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|
| 6 | 17.3% | 30.4% | 40.0% | 46.8% |
| 9 | 22.9% | 40.0% | 52.4% | 61.0% |
| 12 | 26.9% | 46.8% | 61.0% | 70.8% |
| 15 | 29.7% | 51.4% | 66.8% | 77.2% |

At least one of each, going second; rows are starters, columns extenders:

| Starters | 3 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|
| 6 | 23.6% | 39.7% | 50.4% | 57.1% |
| 9 | 30.1% | 50.4% | 63.6% | 71.7% |
| 12 | 34.2% | 57.1% | 71.7% | 80.6% |
| 15 | 36.8% | 61.1% | 76.4% | 85.6% |

A starter and another engine card (a second starter counts), going first:

| Starters | 3 | 6 | 9 | 12 |
|---|---:|---:|---:|---:|
| 6 | 29.0% | 39.0% | 46.2% | 51.0% |
| 9 | 46.2% | 56.9% | 64.2% | 68.9% |
| 12 | 62.0% | 71.7% | 78.0% | 81.7% |
| 15 | 75.3% | 83.1% | 87.8% | 90.3% |

## No brick in 40 cards

| Bricks | Going first | Going second |
|---|---:|---:|
| 0 | 100.00% | 100.00% |
| 1 | 87.50% | 85.00% |
| 2 | 76.28% | 71.92% |
| 3 | 66.24% | 60.57% |
| 4 | 57.29% | 50.75% |
| 5 | 49.34% | 42.29% |
| 6 | 42.29% | 35.04% |
| 7 | 36.07% | 28.86% |
| 8 | 30.60% | 23.61% |

## Copies, going first in 40 cards

| Starters | At least one starter | Two different starters |
|---|---:|---:|
| 12 as 4 × 3 | 85.06% | 41.52% |
| 12 as 6 × 2 | 85.06% | 44.74% |
| 12 as 12 × 1 | 85.06% | 47.72% |
| 9 as 3 × 3 | 74.18% | 24.78% |
| 9 as 9 × 1 | 74.18% | 31.14% |

## Draw and search cards, modeled simply

The deck has 12 starters, 9 extenders and 3 bricks; engine cards replace other cards
unless marked added. Worth in starters: how many starters in place of other cards give
the same chance of at least one starter going first, by linear interpolation.

| Engine cards | Starter, first | Starter, second | Starter and extender, first | Worth in starters, first |
|---|---:|---:|---:|---:|
| none | 85.06% | 90.18% | 61.01% | 0.00 |
| 1 draw-2 | 86.60% | 91.42% | 64.04% | 0.57 |
| 2 draw-2 | 87.90% | 92.43% | 66.68% | 1.08 |
| 3 draw-2 | 89.01% | 93.24% | 68.96% | 1.56 |
| 3 look-at-6 | 91.50% | 95.10% | 74.80% | 2.78 |
| 3 draw-2, 3 look-at-6 | 94.37% | 97.04% | 81.79% | 4.61 |
| 3 draw-1, added (43 cards) | 85.06% | 90.18% | 61.01% |  |
| 3 draw-2, added (43 cards) | 86.35% | 91.16% | 63.70% |  |
| 3 look-at-6, added (43 cards) | 89.17% | 93.37% | 69.65% |  |

## Public calculators, read on 2026-11-19

| Calculator | Question | Printed (%) | Exact | Agrees to the digits shown |
|---|---:|---:|---:|---:|
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 1 in 40, 5 drawn | 12.5 | 12.5000% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 2 in 40, 5 drawn | 23.7 | 23.7179% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 3 in 40, 5 drawn | 33.8 | 33.7551% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 4 in 40, 5 drawn | 42.7 | 42.7071% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 5 in 40, 5 drawn | 50.7 | 50.6644% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 6 in 40, 5 drawn | 57.7 | 57.7124% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 7 in 40, 5 drawn | 63.9 | 63.9311% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 8 in 40, 5 drawn | 69.4 | 69.3961% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 9 in 40, 5 drawn | 74.2 | 74.1780% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 10 in 40, 5 drawn | 78.3 | 78.3428% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 11 in 40, 5 drawn | 82.0 | 81.9523% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 12 in 40, 5 drawn | 85.1 | 85.0640% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 13 in 40, 5 drawn | 87.7 | 87.7312% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 14 in 40, 5 drawn | 90.0 | 90.0032% | yes |
| ygo-decklab.com/en/guide/deck-probabilities | at least 1, 15 in 40, 5 drawn | 91.9 | 91.9256% | yes |
| codingace.net/general/yugioh_hand.html | at least 1, 3 in 40, 5 drawn | 33.7551 | 33.7551% | yes |
| codingace.net/general/yugioh_hand.html | at least 1, 6 in 40, 5 drawn | 57.7124 | 57.7124% | yes |
| codingace.net/general/yugioh_hand.html | at least 1, 9 in 40, 5 drawn | 74.1780 | 74.1780% | yes |
| lumdera.com/yugioh-probability-calculator | at least 1, 1 in 40, 5 drawn | 12.5 | 12.5000% | yes |
| lumdera.com/yugioh-probability-calculator | at least 1, 2 in 40, 5 drawn | 24.1 | 23.7179% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 3 in 40, 5 drawn | 33.3 | 33.7551% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 6 in 40, 5 drawn | 57.7 | 57.7124% | yes |
| lumdera.com/yugioh-probability-calculator | at least 1, 9 in 40, 5 drawn | 77.8 | 74.1780% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 12 in 40, 5 drawn | 88.7 | 85.0640% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 1 in 40, 6 drawn | 14.8 | 15.0000% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 2 in 40, 6 drawn | 28.4 | 28.0769% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 3 in 40, 6 drawn | 39.4 | 39.4332% | yes |
| lumdera.com/yugioh-probability-calculator | at least 1, 6 in 40, 6 drawn | 64.7 | 64.9617% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 9 in 40, 6 drawn | 84.2 | 80.8179% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 12 in 40, 6 drawn | 92.3 | 90.1849% | no |
| lumdera.com/yugioh-probability-calculator | at least 1, 3 in 45, 5 drawn | 30.2 | 30.3735% | no |
| cardgamecalculator.com/yugioh | at least 1, 3 in 40, 5 drawn | 33.76 | 33.7551% | yes |
| cardgamecalculator.com/yugioh | exactly 0, 3 in 40, 5 drawn | 66.24 | 66.2449% | yes |
