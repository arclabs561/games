# Proofs

The counting lemmas, the forced card, the extension theorem, and decks on any number of symbols. The [Spot It README](../README.md) gives the short answers. The cited works are listed under Sources in the README.


Fix n ≥ 2 and v = n² + n + 1 symbols. Take a family of m cards, each with n + 1 of the symbols, with the Spot It property. Let c(s) be the number of cards that show symbol s, let δ(s) = n + 1 − c(s) be its **deficit**, and let d = v − m.

**Lemma 1.** No symbol is on more than n + 1 cards. The cards through s share only s, so they use 1 + n · c(s) different symbols, at most v. So every deficit is at least 0.

**Lemma 2 (counting).** The deficits add up to d(n + 1). The products δ(s)(δ(s) − 1) add up to d(d − 1). On each card the deficits add up to d.

*Proof.* Counting pairs of a card and one of its symbols gives Σ c(s) = m(n + 1), which is the first sum. Counting ordered pairs of distinct cards by their one shared symbol gives Σ c(s)(c(s) − 1) = m(m − 1). Expanding (n + 1 − c)(n − c) and using n(n + 1) = v − 1, the second sum becomes v(v − 1) − 2m(v − 1) + m(m − 1), which is (v − m)(v − m − 1). For a card A, each of the other m − 1 cards shares one symbol with A, so Σ over s in A of (c(s) − 1) is m − 1. Then the deficits on A add up to (n + 1)² − (m − 1) − (n + 1), which is v − m.

The same count, with v replaced by any number V of symbols, gives the bound on the number of symbols stated in [extension.md](extension.md). Each term t(t − 1) is at least 0 for a whole number t, so 56V − 14 · 440 + 55 · 54 ≥ 0 for 55 cards of order 7, and V ≥ 57.

**Lemma 3 (forced card).** Suppose symbol x has deficit 1, so it is on n cards. Let N(x) be the symbols that never appear on a card with x. Then N(x) has n symbols, and {x} together with N(x) shares exactly one symbol with every card. It is the only card containing x that can be added.

*Proof.* The n cards through x share only x, so they show 1 + n² symbols, and N(x) holds the other n. A card B without x shares one symbol with each card through x. These n symbols differ, since the cards through x share nothing but x. So B has n symbols that appear beside x, and its last symbol lies in N(x). A card through x contains no symbol of N(x). A new card containing x can hold no symbol that appears beside x, or it would share two symbols with the card showing both. So it lies inside {x} ∪ N(x), and with n + 1 symbols it is all of it.

For the 55-card deck, d = 2 and the products δ(δ − 1) add up to 2. So exactly one symbol has deficit 2 and every other deficit is 0 or 1. The deficits add up to 16, so fourteen symbols have deficit 1. Each of the two missing cards is the forced card of any of its symbols of deficit 1, which is the method used to find the missing cards above.

**Theorem 4.** Let n ≥ 2 and v = n² + n + 1, and let m cards of n + 1 symbols each, drawn from the v symbols, have the Spot It property, with d = v − m. If d ≤ n + 1, or d = n + 2 and n is odd, the family extends to a projective plane of order n on the same v symbols, and in exactly one way.

*Proof, by induction on d.* If d = 0, every deficit is 0, since they add up to 0 and none is negative. Then every symbol is on n + 1 cards. Each card holds C(n + 1, 2) pairs of symbols, v · C(n + 1, 2) = C(v, 2) in all. No pair is on two cards, so every pair is on exactly one: the family is a plane. If d ≥ 1, suppose no symbol has deficit 1. Every deficit is then 0 or at least 2, and δ(δ − 1) ≥ δ for each. Summing gives d(d − 1) ≥ d(n + 1), so d ≥ n + 2. Equality needs every deficit to be 0 or 2, and then the deficits on a card add up to an even number. So d = n + 2 forces n to be even. Under the hypothesis, then, some symbol x has deficit 1. Adding its forced card lowers d by one and keeps the hypothesis. A plane that contains the family has n + 1 cards through x, so it contains one more card through x, and by Lemma 3 that is the forced card. So the plane contains the larger family, which extends in exactly one way by induction.

**Decks on any number of symbols.** Take m = n² + n − 1 cards on V symbols, all of them used, and not all cards through one symbol. Lemma 1 still holds, since a card without s meets the cards through s in different symbols. Each card's deficits add up to 2, by the proof of Lemma 2, whatever V is. So every deficit is 0, 1 or 2, and no card holds two symbols of deficit 2. Write V = v + k. The same count gives 2 + k(m + 1) for the sum of δ(δ − 1), so k ≥ 0 and there are 1 + k(m + 1)/2 symbols of deficit 2. Each of them is on n − 1 cards, and no card holds two, hence (n − 1)(1 + k(m + 1)/2) ≤ m. For n ≥ 3 this fails when k ≥ 1, so exactly v symbols are used and Theorem 4 applies. For n = 2, `extend.py` lists every deck of five 3-symbol cards on up to 11 symbols that contains one fixed card: 6,300 use 7 symbols and the other 315 share one symbol.
