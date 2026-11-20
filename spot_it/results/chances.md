# What are the chances in the game?

Exact chances for draws from the 57-card and 55-card decks, and the Triplet counts. The [Spot It README](../README.md) gives the short answers. Lemmas 1 to 3 and Theorem 4 are in [proofs.md](proofs.md); the cited works are listed under Sources in the README.


All of these are exact, from counting cards of the plane of order 7 (`chance.py`).

| Question | 57 cards | 55 cards |
|---|---:|---:|
| Chance that two cards drawn share a given symbol on 8 cards | 1/57 = 0.0175 | 28/1,485 = 0.0189 |
| The same, for a symbol on 7 cards | | 7/495 = 0.0141 |
| The same, for the symbol on 6 cards | | 1/99 = 0.0101 |
| Chance that three cards drawn show a common symbol | 6/55 = 0.1091 | 6/55 = 0.1091 |
| Chance that 8 cards drawn have no symbol on three of them | 16,758/1,652,411,475 ≈ 0.00001 | 98/9,663,225 ≈ 0.00001 |
| Chance that 9 cards drawn have no symbol on three of them | 0 | 0 |
| Expected number of three-card sets with a common symbol among 9 cards | 504/55 = 9.16 | 504/55 = 9.16 |

- **A symbol's count.** If the two missing cards were chosen at random from the 57, a given symbol would be left on 8 cards with chance 14/19, on 7 with chance 14/57 and on 6 with chance 1/57. These are the shares 42, 14 and 1 out of 57.
- **Three cards.** Removing up to two cards leaves the three-card chance at exactly 6/55. In the full deck, given two cards, the third must show their common symbol, and 6 of the other 55 cards do. By inclusion and exclusion, the triples that avoid two removed cards number 3,192 − 2 · 168 + 6 = 2,862 out of 29,260 − 2 · 1,540 + 55 = 26,235. The ratios 3,192/29,260, 168/1,540 and 6/55 are all equal, so the result is 6/55 too. Removing three cards changes it slightly: 2,705/24,804 if the three share a symbol, 451/4,134 if not. Hummert and Krause (2024) give 6/55 for the chance that the symbol called at one card is called again at the next, computed for a full deck; the box gives the same value.
- **Triplet.** In this mini-game nine cards lie face up and players look for three cards with a common symbol. Nine cards always contain such a set, so the table never stalls while nine cards are out. Suppose nine cards had no symbol on three of them. A card shares a symbol with each of the other eight, and these eight symbols differ, since a symbol shared with two of them would be on three. So every symbol on one of the nine is on exactly two of them. That makes 36 symbols, and 21 of the 57 are on none of the nine. Take a card D through one of those 21. Each of the nine cards shares one symbol with D, and that symbol is on exactly two of the nine, so D's symbols split the nine cards into pairs. Nine is odd, so this cannot happen. The argument is the dual of Bose's 1947 bound on arcs, and the exhaustive count agrees: no such set of nine. Sets of eight exist. Read as points, the 16,758 sets of eight that the search finds are exactly the zero sets of the nondegenerate quadratic forms, the conics, as Segre's theorem says they must be.

The counts of k-card sets with no symbol on three cards, for k = 0 to 9, by exhaustive search:

| Deck | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 cards | 1 | 57 | 1,596 | 26,068 | 234,612 | 938,448 | 1,094,856 | 134,064 | 16,758 | 0 |
| 55 cards | 1 | 55 | 1,485 | 23,373 | 202,566 | 779,688 | 874,650 | 102,900 | 12,348 | 0 |

For up to five cards the 57-card counts match a count by hand (57 · 56 · 49 · 36 · 20 / 5! for five). The 8-card count equals q⁵ − q², the number of conics of the plane of order q = 7. The 7-card count is eight times it: two conics share at most four points, so the seven-card subsets of the 16,758 sets of eight are all different, and the code confirms that they are all 134,064 sets of seven. The 55-card counts were computed for two different pairs of removed cards and agree. The symmetries of the plane take any pair of cards to any other, a standard fact, so every pair gives the same counts.
