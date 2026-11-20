# The best keep after the first roll

What optimal play keeps from each of the 252 first rolls; `openings.csv` has the full list. The [Yahtzee README](../README.md) gives the short answers. "Exact" means computed by the dynamic program in double precision.


`openings.csv` gives, for each of the 252 first rolls of the game, the best keep and its expected final score, with the second-best keep. Counting the 7,776 equally likely ordered rolls:

- Only Yahtzees and large straights are kept whole, 246 of the 7,776 rolls. Some dice are always kept.
- 7,056 rolls hold a pair or more. In 6,156 of them the best keep is every die of the most common face. That includes full houses: with 5 5 6 6 6, keep 6 6 6 (265.12) rather than the full house.
- In the other 900 rolls with a pair, the pair is given up:
  - for a straight draw. With 1 1 3 4 5, keep 3 4 5; with 2 3 4 5 5, keep 2 3 4 5.
  - for a single die of 4 or more, when the pair is of 1s. With 1 1 2 3 6, keep the 6; with 1 1 2 4 5, keep the 5.
- The closest decision is 1 1 3 4 6. Keeping the 4 is worth 249.8811 and keeping 3 4 is worth 249.8763. After the second roll the order flips, as in Verhoeff's example above.
- The worst first roll is 1 1 2 3 6, worth 249.83 when the 6 is kept (Verhoeff: 249.83).
