# Pot of Prosperity in the example deck

Written by `uv run python -m yugioh.prosperity --write`. Exact. The deck has 40 cards:
12 starters, 9 extenders and 3 bricks; added cards replace other cards. Worth in
starters: how many starters in place of other cards give the same chance of a starter
going first, by linear interpolation.

| Deck | Starter, first | Starter and extender, first | Starter, second | Worth in starters |
|---|---:|---:|---:|---:|
| no draw or search card | 85.06% | 61.01% | 90.18% | 0.00 |
| Prosperity, reveal 6 | 87.57% | 66.25% | 92.17% | 0.94 |
| Prosperity, reveal 3 | 87.01% | 64.85% | 91.75% | 0.73 |
| Prosperity (reveal 6) and 3 draw-twos | 90.90% | 73.18% | 94.65% | 2.47 |
| Same, always playing Prosperity first | 90.90% | 73.13% | 94.65% | 2.47 |
| Same, always playing a draw-two first | 90.52% | 72.39% | 94.29% | 2.27 |
| generic look-at-6, 1 copy | 87.57% | 66.25% | 92.17% | 0.94 |
| generic look-at-6 and 3 draw-twos, 1 copy | 91.15% | 73.98% | 94.83% | 2.60 |

Exact fractions, going first, at least one starter:

- no draw or search card: 598/703
- Prosperity, reveal 6: 10,381,207/11,855,392
- Prosperity, reveal 3: 20,797/23,902
- Prosperity (reveal 6) and 3 draw-twos: 4,903,570,097/5,394,203,360
- Same, always playing Prosperity first: 4,903,570,097/5,394,203,360
- Same, always playing a draw-two first: 2,092,699,139/2,311,801,440
- generic look-at-6, 1 copy: 10,381,207/11,855,392
- generic look-at-6 and 3 draw-twos, 1 copy: 247,410,929/271,426,080
