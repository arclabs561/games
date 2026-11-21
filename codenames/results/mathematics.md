# The mathematics of a turn

Part of the [Codenames README](../README.md). The JSON files beside this note hold its numbers, with their seeds and settings.

**Exact turn odds.** For a Luce guesser with weights w over the unrevealed words, write p(S) for the chance that the first |S| picks are the team words in the set S, in some order. Then p of the empty set is 1, and

    p(S) = Σ over x in S of  p(S − x) · w_x / (W − W(S − x)),

where W is the total weight and W(A) the weight of the words in A. The chance that the first j picks are all team words is the sum of p(S) over sets of j team words. The chance that pick j is the first wrong one, of kind k, is the sum over sets S of j − 1 team words of p(S) · W_k / (W − W(S)). For one pick this is the team's share of the weight. `luce.py` computes these for many clues at once. The tests check them against a sum over every ordered sequence of picks, against 100,000 simulated turns, and against the greedy guesser at a low temperature. They also check that the outcomes of a turn sum to 1.

**The best count.** With cost c_k for a wrong pick of kind k, the expected value of a turn that allows n picks is

    R(n) = Σ_{j ≤ n} [ P(first j picks are team) − Σ_k c_k · P(pick j is the first wrong one, of kind k) ],

and the giver's best count is the n that maximizes it. In this study the costs are 1 for an opponent word, 0 for a neutral word, and, for the assassin, the number of team words still to find, since picking it loses the game.

**Proposition.** If the assassin's cost rises, the largest best count does not rise. *Proof.* Let c be the assassin's cost and B(n) the chance of hitting the assassin within n picks. Then R(n) = A(n) − c·B(n), where A(n) does not depend on c. B does not decrease in n, since allowing more picks only adds chances. For n′ > n, the difference R(n′) − R(n) = A(n′) − A(n) − c(B(n′) − B(n)) therefore does not increase in c. If n′ > n beats n at a cost c, it also beats n at every lower cost. So the largest maximizer at a higher cost cannot exceed the largest maximizer at a lower one. (This is Topkis's monotonicity theorem in its simplest form.) ∎

**An observation, not proved.** In every case checked, R(n) for n = 1 to 4 has a single peak: it rises and then falls, never falls and then rises again. The checks cover 267,237 shortlisted clues on 50 boards under eight models, and 100,000 random weightings of four team and three other words at nine assassin costs. If it always holds, a giver can stop raising n at the first decrease. An attempted proof bounds the next increment by an average of the current conditional gains, weighted toward the positive ones, so it does not close. Whether a counterexample exists is open.
