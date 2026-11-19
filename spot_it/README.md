# Spot It

Spot It, sold as Dobble in Europe, has 55 round cards with 8 symbols on each. The rules say that "between any 2 cards, there is always 1—and only 1—matching symbol" ([rules](https://cdn.svc.asmodee.net/production-asmodeeaccess/uploads/2026/10/DobbleClassic_EN_Rules.pdf)). This folder builds such decks from finite projective planes. It then finds the two cards that the box leaves out, asks whether every deck like this one comes from a plane, and works out a few chances in the game.

## The deck as modeled

- A **deck** is a list of **cards**, and a card is a set of **symbols**. A deck has the **Spot It property** if every two of its cards share exactly one symbol.
- A **projective plane of order n** is a deck of n² + n + 1 cards with n + 1 symbols each, using n² + n + 1 symbols, in which every two symbols also appear together on exactly one card. For n = 7 that is 57 cards of 8 symbols on 57 symbols.
- The box has 55 cards. People who counted the symbols found 57 ([a list of the 57](https://www.markporthouse.net/spot-it-dobble-symbol-list); [Tangente](https://tangente-mag.com/en/articles/the-geometry-behind-dobble)). Asked about the two missing cards, the publisher answered on the Tric Trac forum that practical production reasons led them to print 55 cards, not 57, and that every box holds the same 55 ([forum, 2011](https://forum.trictrac.net/t/dobble-et-la-geometrie-finie/95005)).

Below, the 55-card deck is the plane of order 7 with two cards removed. Section 3 shows that no other deck is possible: any 55 cards of 8 symbols from 57 symbols with the Spot It property arise in this way.

## Results

### 1. Planes of every prime-power order

`plane.py` builds the plane of order q in two ways. It covers every prime power q up to 13. The first uses coordinates over the field with q elements: a symbol and a card are each a nonzero triple (x, y, z) up to scaling, and a symbol is on a card when the dot product of their triples is 0. The second is the construction that most deck generators use: the q² points (x, y), one extra symbol for each slope and one for vertical lines; the cards are the lines y = ax + b and x = c with their slope symbol, plus one card holding the q + 1 extra symbols. For q = 4, 8 and 9 the arithmetic is that of a finite field built from polynomials (`field.py`), whose axioms are checked on every pair and triple of elements.

Both constructions give a projective plane for q = 2, 3, 4, 5, 7, 8, 9, 11 and 13. This is checked from the definition: every pair of cards shares one symbol and every pair of symbols lies on one card. With plain arithmetic mod n, the slope construction works only for prime n, because then every nonzero number has an inverse. For other n some pairs of cards share no symbol or several:

| n | Cards | Pairs sharing no symbol | Pairs sharing one | Pairs sharing two or more |
|---:|---:|---:|---:|---:|
| 4 | 21 | 16 | 178 | 16 |
| 6 | 43 | 180 | 579 | 144 |
| 8 | 73 | 448 | 1,860 | 320 |
| 9 | 91 | 486 | 3,366 | 243 |
| 10 | 111 | 1,400 | 3,605 | 1,100 |
| 12 | 157 | 4,032 | 6,198 | 2,016 |

Exact, by checking every pair.

### 2. The two missing cards

Count how many cards show each symbol. In a 55-card deck one symbol is on 6 cards, fourteen are on 7 and the other 42 are on 8. The symbol on 6 cards is on both missing cards. Each missing card holds it and seven of the fourteen symbols seen 7 times. Two of the fourteen belong to the same missing card exactly when no card in the deck shows both. So the rule "never together" sorts them into two groups of seven.

The answer is unique: no other card of 8 symbols from the 57 shares exactly one symbol with each of the 55 cards. Lemmas 2 and 3 below prove both claims. `missing.py` checks them on every one of the C(57, 2) = 1,596 ways to remove two cards from the plane of order 7, and on every pair for the orders 2, 3, 4, 5, 8 and 9. A second method, an exact-cover search that shares no code with the counting, lists every set of n + 1 symbols that meets each remaining card once. In every case it returns the two removed cards and nothing else.

### 3. Does every such deck extend to the plane?

Yes. Take any 55 cards of 8 symbols, drawn from 57 symbols, with the Spot It property. Theorem 4 below shows that they can be completed to a projective plane of order 7 on the same 57 symbols, in exactly one way. No card list is needed: the symbol counts 6, 7 and 8 above are forced, and so are the two missing cards. More generally, on n² + n + 1 symbols every such family of at least n² cards completes, uniquely, and for odd n so does every family of n² − 1 cards. For order 7 that means any 48 or more cards. The proof is a short count. The bound n² is Vanstone's, in dual form, and Fowler proves n² − 1 for every n (see Sources).

The number of symbols is not a free choice either. By de Bruijn and Erdős, a deck with the Spot It property has no more cards than symbols, so 55 cards need at least 55 symbols. The count in Lemma 2 gives more: 55 cards of 8 symbols need at least 57. If all 55 cards share one symbol, as in a sunflower of 1 + 55 × 7 = 386 symbols, nothing can be added. Otherwise exactly 57 symbols are used, and Theorem 4 applies. By Deza's theorem, more than 57 cards of 8 symbols with the Spot It property always share one symbol, however many symbols there are.

**Maximal decks.** A family of cards is **maximal** if no further card can be added, a card being any n + 1 of the n² + n + 1 symbols. A plane is maximal, and Theorem 4 says that a maximal family other than a plane has at most n² − 1 cards, or n² − 2 when n is odd. What is the largest one?

- For order 2 every maximal family is a plane. For order 3 there is one other kind: five cards on ten symbols, each symbol on two cards. Exact: `extend.py` lists every maximal family that contains two fixed cards. Any two cards sharing one symbol can be renamed into those two. It finds 1,440 families: 720 planes, and 720 of five cards of that kind.
- The five-card family generalizes. Take the complete graph on n + 2 vertices, make each vertex a card and each edge a symbol, and put on each card the n + 1 edges at its vertex. Two cards share the one edge between them. A new card would need to meet every vertex once using edges, which pairs the vertices up, and n + 2 vertices cannot be paired when n is odd. So for order 7 there is a maximal deck of nine cards on 36 of the 57 symbols that no card can extend. Proved above, and checked for n = 2 to 9.
- Random searches found larger ones. The search removes random cards from the plane, then adds random cards that fit until none fits. The largest maximal families that are not planes had 13 cards for order 4 (at most 14 by Fowler's theorem, 15 by Theorem 4), 19 for order 5 (at most 23) and 37 for order 7 (at most 47). These are seeded searches, so they give lower bounds only. The true largest size for order 7 lies between 37 and 47 and is not settled here.

**Which orders exist.** Planes exist for every prime-power order (Section 1). The Bruck–Ryser theorem rules out every order n that leaves remainder 1 or 2 on division by 4 and is not a sum of two squares, such as 6, 14 and 21. Lam, Thiel and Swiercz (1989) ruled out order 10 by a large computer search. Order 12, a deck of 157 cards, is the smallest one still open (Bourrigan). By Theorem 4, no family of 36 cards of 7 symbols on 43 symbols has the Spot It property, since it would extend to a plane of order 6. The same holds for 100 cards of 11 symbols on 111 symbols. The planes of orders up to 8 are unique up to renaming (Pierce and Hall in 1953 for order 7, with Hall's correction of 1954; Hall, Swift and Walker in 1956 for order 8). `extend.py` confirms uniqueness for orders 2, 3 and 4 by counting. It counts, by exhaustive search, the planes that contain n + 1 fixed cards through one symbol and one more fixed card. It also counts the automorphisms of the coordinate plane by search: 168, 5,616 and 120,960. The plane counts, 1, 8 and 2,592, are the ones that the automorphism counts predict if only one plane of each order exists.

### 4. The real deck

A dataset of card photos [on Kaggle](https://www.kaggle.com/datasets/grouby/dobble-card-images) includes a list of which symbols are on which card. Its license is listed as unknown, so the list is not committed here. It describes 57 cards taken from a 2012 MathsJam talk, not the 55 cards of a box, and it does not say which two cards a box leaves out. A check run once outside the repository found that its 57 cards form a projective plane of order 7. That is a check of the list, not of a box.

The symbol counts of an actual box are therefore not checked against a card list. They follow from Theorem 4 once the box has 55 cards of 8 symbols, the Spot It property, and 57 symbols: one symbol on 6 cards, fourteen on 7 and 42 on 8.

### 5. Chances

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

## Proofs

Fix n ≥ 2 and v = n² + n + 1 symbols. Take a family of m cards, each with n + 1 of the symbols, with the Spot It property. Let c(s) be the number of cards that show symbol s, let δ(s) = n + 1 − c(s) be its **deficit**, and let d = v − m.

**Lemma 1.** No symbol is on more than n + 1 cards. The cards through s share only s, so they use 1 + n · c(s) different symbols, at most v. So every deficit is at least 0.

**Lemma 2 (counting).** The deficits add up to d(n + 1). The products δ(s)(δ(s) − 1) add up to d(d − 1). On each card the deficits add up to d.

*Proof.* Counting pairs of a card and one of its symbols gives Σ c(s) = m(n + 1), which is the first sum. Counting ordered pairs of distinct cards by their one shared symbol gives Σ c(s)(c(s) − 1) = m(m − 1). Expanding (n + 1 − c)(n − c) and using n(n + 1) = v − 1, the second sum becomes v(v − 1) − 2m(v − 1) + m(m − 1), which is (v − m)(v − m − 1). For a card A, each of the other m − 1 cards shares one symbol with A, so Σ over s in A of (c(s) − 1) is m − 1. Then the deficits on A add up to (n + 1)² − (m − 1) − (n + 1), which is v − m.

The same count, with v replaced by any number V of symbols, gives the bound in Section 3. Each term t(t − 1) is at least 0 for a whole number t, so 56V − 14 · 440 + 55 · 54 ≥ 0 for 55 cards of order 7, and V ≥ 57.

**Lemma 3 (forced card).** Suppose symbol x has deficit 1, so it is on n cards. Let N(x) be the symbols that never appear on a card with x. Then N(x) has n symbols, and {x} together with N(x) shares exactly one symbol with every card. It is the only card containing x that can be added.

*Proof.* The n cards through x share only x, so they show 1 + n² symbols, and N(x) holds the other n. A card B without x shares one symbol with each card through x. These n symbols differ, since the cards through x share nothing but x. So B has n symbols that appear beside x, and its last symbol lies in N(x). A card through x contains no symbol of N(x). A new card containing x can hold no symbol that appears beside x, or it would share two symbols with the card showing both. So it lies inside {x} ∪ N(x), and with n + 1 symbols it is all of it.

For the 55-card deck, d = 2 and the deficits δ(δ − 1) add up to 2. So exactly one symbol has deficit 2 and every other deficit is 0 or 1. The deficits add up to 16, so fourteen symbols have deficit 1. Each of the two missing cards is the forced card of any of its symbols of deficit 1, which is the method of Section 2.

**Theorem 4.** If d ≤ n + 1, or d = n + 2 and n is odd, the family extends to a projective plane of order n on the same v symbols, and in exactly one way.

*Proof, by induction on d.* If d = 0, every deficit is 0, since they add up to 0 and none is negative. Then every symbol is on n + 1 cards. Each card holds C(n + 1, 2) pairs of symbols, v · C(n + 1, 2) = C(v, 2) in all. No pair is on two cards, so every pair is on exactly one: the family is a plane. If d ≥ 1, suppose no symbol has deficit 1. Every deficit is then 0 or at least 2, and δ(δ − 1) ≥ δ for each. Summing gives d(d − 1) ≥ d(n + 1), so d ≥ n + 2. Equality needs every deficit to be 0 or 2, and then the deficits on a card add up to an even number. So d = n + 2 forces n to be even. Under the hypothesis, then, some symbol x has deficit 1. Adding its forced card lowers d by one and keeps the hypothesis. A plane that contains the family has n + 1 cards through x, so it contains one more card through x, and by Lemma 3 that is the forced card. So the plane contains the larger family, which extends in exactly one way by induction.

**Decks on any number of symbols.** Take m = n² + n − 1 cards on V symbols, all of them used, and not all cards through one symbol. Lemma 1 still holds, since a card without s meets the cards through s in different symbols. Each card's deficits add up to 2, by the proof of Lemma 2, whatever V is. So every deficit is 0, 1 or 2, and no card holds two symbols of deficit 2. Write V = v + k. The same count gives 2 + k(m + 1) for the sum of δ(δ − 1), so k ≥ 0 and there are 1 + k(m + 1)/2 symbols of deficit 2. Each of them is on n − 1 cards, and no card holds two, hence (n − 1)(1 + k(m + 1)/2) ≤ m. For n ≥ 3 this fails when k ≥ 1, so exactly v symbols are used and Theorem 4 applies. For n = 2, `extend.py` lists every deck of five 3-symbol cards on up to 11 symbols that contains one fixed card: 6,300 use 7 symbols and the other 315 share one symbol.

## How each result is known

| Result | How it is known |
|---|---|
| Fields of 4, 8, 9 and 16 elements | Exact: every field axiom checked on all pairs and triples |
| Planes of orders 2 to 13 (prime powers) | Exact: every pair of cards and of symbols checked, two constructions |
| The slope construction mod n fails for composite n | Exact, table above |
| The missing cards and their uniqueness | Proof (Lemmas 2 and 3); checked on all 1,596 removals for order 7 by two methods |
| Every family of n² or more cards on n² + n + 1 symbols extends uniquely | Proof (Theorem 4), agreeing with Vanstone and Fowler; identities checked on random subfamilies; exhaustive for orders 2 and 3 |
| 55 cards of 8 symbols need at least 57 symbols | Proof (Lemma 2) |
| Maximal families of order 3 | Exhaustive search |
| Large maximal families of orders 4, 5 and 7 | Seeded random search; lower bounds only |
| Uniqueness of the planes of orders 2, 3 and 4 | Exhaustive count of completions against the automorphism counts |
| Orders 6, 10 and 12, uniqueness for 7 and 8 | Cited (Bruck–Ryser; Lam et al.; Hall; Hall, Swift and Walker) |
| Chances in the table | Exact, by counting; two of them also by listing pairs and triples |
| Nine cards always hold a triplet | Proof (Bose's parity argument), and exhaustive count |
| The sets of eight cards with no triplet are the conics | Exhaustive comparison with every quadratic form, as Segre's theorem predicts |
| The real deck's card list | Not checked: the only list found has an unknown license and describes 57 cards |

## Usage

The code uses only the standard library. Times are from one run on a shared machine.

```bash
uv run python -m spot_it.field     # field axioms for 2 to 16 elements (under a second)
uv run python -m spot_it.plane     # both constructions, orders 2 to 13 (under a second)
uv run python -m spot_it.missing   # the missing cards, every pair, orders 2 to 9 (about 6 s)
uv run python -m spot_it.extend    # identities, maximal families, uniqueness, searches (about 35 s)
uv run python -m spot_it.chance    # the chances and the Triplet counts (about 12 s)
uv run pytest tests/test_spot_it.py
```

## Files

| File | Contents |
|---|---|
| `field.py` | Finite fields from irreducible polynomials, and a check of the field axioms |
| `plane.py` | Projective planes in coordinates and by slopes, and checks of the Spot It property |
| `missing.py` | The two missing cards, by counting and by an exact-cover search |
| `extend.py` | Deficits, forced cards, maximal families, uniqueness counts and random searches |
| `chance.py` | Shared-symbol chances, symbol counts, and sets of cards with no symbol on three |

## Sources

- R. C. Bose, Mathematical theory of the symmetrical factorial design, *Sankhyā* 8 (1947) 107–166: a set of points of the plane of odd order q with no three on a line has at most q + 1 points.
- R. H. Bruck and H. J. Ryser, The nonexistence of certain finite projective planes, *Canadian Journal of Mathematics* 1 (1949) 88–93.
- N. G. de Bruijn and P. Erdős, On a combinatorial problem, *Indagationes Mathematicae* 10 (1948) 421–423: on v points there are at most v sets every two of which share exactly one point. In card terms, a deck with the Spot It property has no more cards than symbols.
- K. N. Majumdar, On some theorems in combinatorics relating to incomplete block designs, *Annals of Mathematical Statistics* 24 (1953) 377–389, and J. R. Isbell, An inequality for incidence matrices, *Proceedings of the AMS* 10 (1959) 216–218: the same bound when every two sets share the same number of points, the nonuniform Fisher inequality.
- M. Deza, Solution d'un problème de Erdős–Lovász, *Journal of Combinatorial Theory B* 16 (1974) 166–167: more than k² − k + 1 sets of size k every two of which meet in exactly one point all share that point.
- M. Hall, Uniqueness of the projective plane with 57 points, *Proceedings of the AMS* 4 (1953) 912–916, and its correction, 5 (1954) 994–997; M. Hall, J. D. Swift and R. J. Walker, [Uniqueness of the projective plane of order eight](https://www.ams.org/journals/mcom/1956-10-056/S0025-5718-1956-0084142-0/S0025-5718-1956-0084142-0.pdf), *Mathematical Tables and Other Aids to Computation* 10 (1956) 186–194, which credits order 7 to the combined results of W. A. Pierce and of Hall.
- B. Segre, Ovals in a finite projective plane, *Canadian Journal of Mathematics* 7 (1955) 414–416: for odd q every set of q + 1 points with no three on a line is a conic.
- S. A. Vanstone, The extendability of (r, 1)-designs, *Proceedings of the Third Manitoba Conference on Numerical Mathematics* (1973) 409–418, as stated in P. Erdős, R. C. Mullin, V. T. Sós and D. R. Stinson, [Finite linear spaces and projective planes](https://renyi.hu/~p_erdos/1983-05.pdf), *Discrete Mathematics* 47 (1983) 49–62, Lemma 2.1: in dual form, a family of at least n² cards on at most n² + n + 1 symbols extends to a plane of order n.
- J. C. Fowler, [Topics in Linear Spaces and Projective Planes](https://doi.org/10.7907/d8gp-ka37), PhD thesis, Caltech (1984), Theorem 1.1: in dual form, n² − 1 cards suffice for every n; one case rests on results of Bose and Shrikhande and of de Witte. The thesis also summarizes later bounds, by D. M. McCarthy and S. A. Vanstone (Embedding (r, 1)-designs in finite projective planes, *Discrete Mathematics* 19 (1977) 67–76) and by S. Dow, that are lower for large n.
- C. W. H. Lam, L. Thiel and S. Swiercz, The non-existence of finite projective planes of order 10, *Canadian Journal of Mathematics* 41 (1989) 1117–1123.
- M. Bourrigan, [Dobble et la géométrie finie](https://images.math.cnrs.fr/freeze/Dobble-et-la-geometrie-finie.html), *Images des Mathématiques* (2011): the game as a plane of order 7 with two cards removed.
- C. Hummert and B. Krause, [Spotted It! Strategies to Win Dobble](https://www.gathering4gardner.org/g4g15gift/ExchangeArchive-SpottedIt-G15-013-1.pdf), Gathering 4 Gardner 15 (2024).
- The [rules of Dobble](https://cdn.svc.asmodee.net/production-asmodeeaccess/uploads/2026/10/DobbleClassic_EN_Rules.pdf) (Asmodee), for the mini-games The Tower, The Well and Triplet.
