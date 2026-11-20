# Which decks have the Spot It property?

The two constructions of the projective planes, and what the slope construction does mod a composite number. The [Spot It README](../README.md) gives the short answers. Lemmas 1 to 3 and Theorem 4 are in [proofs.md](proofs.md); the cited works are listed under Sources in the README.


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
