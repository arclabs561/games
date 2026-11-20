# Guesser calibration and agreement between people (2026-11-20)

Exploratory. These metrics were added after C1's test turns had been scored for top-1, so they carry no significance claim. `python -m codenames.calibration` writes every number below to `guesser-calibration.json`, which holds aggregates only.

**Findings.**

- The guessers' probabilities are close to calibrated, but not exactly. On the 551 test turns of C1's split by game, the expected calibration error of the top word is 0.04 to 0.13. EmbeddingGemma 2 is the least calibrated (0.132) and is underconfident: it gives its top word 0.42 on average, and that word is the person's on 0.55 of turns. The ranking by log-likelihood is the ranking by top-1, except that potion-base-8M and EmbeddingGemma 2 swap places.
- Among the people, the order of picks carries information. On the 118 test turns with several picks, every guesser's top word is the person's first pick 2.7 to 4.2 times as often as the second pick. A guesser's top word is one of the person's picks on 0.42 to 0.79 of these turns, against 0.31 to 0.59 for the first pick alone. Scoring only the first pick therefore counts as misses many words that the same person picked a moment later.
- People agree with each other more often than any guesser agrees with a person, on the items where both can be compared. Two people given the same clue, each with the other's pick on the board, pick the same word in 92.8% of 27,305 such pairs (91.3% to 94.1%), when restricted to the 5.8 words their boards share on average. The best guesser, potion-base-8M, agrees with a person in 88.2% of the same pairs, 4.6 points less (2.5 to 7.5). The others trail by 6.6 to 16.1 points.

## What is measured

Each guesser is a Luce guesser: it gives word w the probability exp(s(w)/τ) divided by the sum over the unrevealed words, where s is its similarity to the clue. τ and the template are those of `calibrate-game.json`: τ was fitted on the train turns and the template chosen on the val turns. The EAT guesser uses association counts plus ε = 0.1 instead. The two linear maps are C1's, retrained here with seeds 1 to 5. Their top-1 on each seed equals C1's to the last digit, as does every untrained model's.

For each test turn, with q the guesser's distribution and the person's word fixed:

- **log-likelihood**: log q(person's word), averaged over turns;
- **confidence**: q(top), where top is the guesser's most likely word;
- **ECE**: the turns are put in 10 bins of equal width by confidence, and ECE is the sum over bins of (turns in the bin / all turns) × |share of hits in the bin − mean confidence in the bin|;
- **Brier on the chosen word**: the mean of (q(top) − hit)², with hit 1 when top is the person's word;
- **multiclass Brier**: the mean over turns of the sum over words of (q(w) − [w is the person's word])².

Intervals are 95% percentile intervals from 10,000 resamples of the 68 test games (seed 0). For a map, each replicate averages the five seeds. ECE is biased upward in small samples, and so are its resamples, so the point estimate can sit near the bottom of its interval.

| Guesser | Top-1 | Log-likelihood | Mean confidence | ECE | Brier, chosen word | Multiclass Brier |
|---|---:|---|---:|---|---|---:|
| potion + map (5 seeds) | 0.632 | −1.445 (−1.619 to −1.270) | 0.662 | 0.055 (0.053 to 0.091) | 0.173 (0.155 to 0.192) | 0.510 |
| EmbeddingGemma 2 | 0.548 | −1.729 (−1.869 to −1.588) | 0.422 | 0.132 (0.100 to 0.167) | 0.211 (0.192 to 0.230) | 0.622 |
| potion-base-8M | 0.544 | −1.724 (−1.874 to −1.565) | 0.504 | 0.068 (0.049 to 0.110) | 0.182 (0.161 to 0.202) | 0.605 |
| bge-small-en-v1.5 | 0.490 | −1.873 (−2.037 to −1.704) | 0.472 | 0.053 (0.041 to 0.095) | 0.168 (0.151 to 0.186) | 0.641 |
| Qwen3-Embedding-0.6B + map (5 seeds) | 0.477 | −1.946 (−2.114 to −1.776) | 0.508 | 0.057 (0.056 to 0.090) | 0.201 (0.189 to 0.213) | 0.680 |
| harrier-oss-v1-0.6b | 0.465 | −2.013 (−2.197 to −1.838) | 0.498 | 0.061 (0.045 to 0.107) | 0.193 (0.173 to 0.212) | 0.689 |
| wordllama | 0.465 | −2.057 (−2.211 to −1.903) | 0.409 | 0.091 (0.068 to 0.133) | 0.188 (0.168 to 0.209) | 0.695 |
| harrier-oss-v1-270m | 0.439 | −2.092 (−2.277 to −1.917) | 0.449 | 0.076 (0.050 to 0.115) | 0.191 (0.174 to 0.209) | 0.711 |
| Qwen3-Embedding-0.6B | 0.434 | −2.144 (−2.301 to −1.987) | 0.463 | 0.059 (0.042 to 0.104) | 0.201 (0.182 to 0.220) | 0.729 |
| granite-small-r2 | 0.394 | −2.198 (−2.318 to −2.079) | 0.345 | 0.052 (0.036 to 0.095) | 0.184 (0.168 to 0.201) | 0.752 |
| EAT counts | 0.334 | −2.388 (−2.488 to −2.285) | 0.307 | 0.039 (0.029 to 0.078) | 0.129 (0.111 to 0.147) | 0.764 |

Computed exactly on the test turns; intervals from resampling games. The JSON file also holds each untrained guesser's reliability table, the turns, mean confidence and share of hits in each bin.

Two cautions. τ comes from a grid of eight values (0.01 to 0.2), and the best values for the transformer models are 0.02 and 0.03, so a step of the grid changes their confidence a good deal. Part of EmbeddingGemma 2's underconfidence may come from the grid. The EAT guesser picks at random when the clue is not an EAT cue, which pulls its confidence and its hits down together and flatters its ECE.

## Turns with several picks

Of the 7,554 rows kept (alphabetic clue, every pick a distinct board word), 6,088 record one pick and 1,466 several: 1,239 with two, 201 with three, and 26 with four to eight. The file lists the picks in an order that is taken to be the order of picking; the data do not say so. The test split holds 128 such turns, and 118 of them have vectors for every model.

| Guesser | Top = first pick | Top = second pick | Top = any pick | Plackett-Luce log-likelihood per pick |
|---|---:|---:|---:|---:|
| potion + map (5 seeds) | 0.593 | 0.159 | 0.785 | −2.175 |
| EmbeddingGemma 2 | 0.568 | 0.136 | 0.729 | −2.195 |
| potion-base-8M | 0.551 | 0.178 | 0.780 | −2.188 |
| bge-small-en-v1.5 | 0.525 | 0.144 | 0.686 | −2.302 |
| harrier-oss-v1-0.6b | 0.508 | 0.144 | 0.695 | −2.430 |
| Qwen3-Embedding-0.6B | 0.492 | 0.144 | 0.653 | −2.510 |
| Qwen3-Embedding-0.6B + map (5 seeds) | 0.486 | 0.146 | 0.663 | −2.502 |
| harrier-oss-v1-270m | 0.449 | 0.136 | 0.610 | −2.541 |
| wordllama | 0.441 | 0.161 | 0.636 | −2.641 |
| granite-small-r2 | 0.390 | 0.093 | 0.534 | −2.634 |
| EAT counts | 0.305 | 0.093 | 0.424 | −2.739 |

Computed exactly on the 118 turns; no intervals, since this table is descriptive. A guesser that knew the set of picks but drew their order at random would match the first pick on 0.458 of these turns, the mean of 1/k. The best guessers pass that mark, so the order is not random, and 0.458 is no upper bound.

## Agreement between people, and its limits

Cultural Codes has one person per turn, so the best possible top-1 cannot be measured directly. Two estimates are possible.

**Exact repeats.** 14 items (a clue and the same set of unrevealed words) were answered by two different guessers. Their first picks agree in 12 of the 14 pairs. Too few to estimate anything.

**Pairs on shared words.** Take two rows with the same clue from different games and different guessers, each with the other's first pick on its own board. Rows with several picks take part with their first pick. Assume Luce's choice axiom with one p for everyone: a person picks w from a set S with probability p(w)/p(S). Then the two picks are two draws from p restricted to the words the boards share, C. The chance that they agree is the sum over w in C of (p(w)/p(C))². For one item, Σq² ≤ max q ≤ √(Σq²), and Jensen's inequality carries the upper bound to the mean. So the best possible top-1 on these restricted items lies between the agreement rate a and √a.

The data give 27,305 such pairs, on 779 clues and 5,460 rows (all splits, since no model is fitted to them), with 5.8 shared words on average. People agree in 0.928 of them (0.913 to 0.941, resampling clues), so the best possible top-1 on these items lies between 0.928 and 0.963. A random pick among the shared words would agree 0.196 of the time. Each guesser's top word among the shared words, scored against each of the two people:

| Guesser | Agreement with a person | Minus agreement between people |
|---|---|---|
| potion-base-8M | 0.882 (0.847 to 0.909) | −0.046 (−0.075 to −0.025) |
| EmbeddingGemma 2 | 0.863 (0.818 to 0.896) | −0.066 (−0.101 to −0.037) |
| bge-small-en-v1.5 | 0.845 (0.791 to 0.885) | −0.083 (−0.133 to −0.049) |
| harrier-oss-v1-0.6b | 0.843 (0.797 to 0.880) | −0.085 (−0.127 to −0.053) |
| wordllama | 0.827 (0.767 to 0.872) | −0.101 (−0.155 to −0.061) |
| harrier-oss-v1-270m | 0.799 (0.736 to 0.846) | −0.129 (−0.186 to −0.086) |
| Qwen3-Embedding-0.6B | 0.795 (0.726 to 0.847) | −0.133 (−0.200 to −0.084) |
| granite-small-r2 | 0.767 (0.703 to 0.818) | −0.161 (−0.222 to −0.112) |

Computed exactly on the pairs; 95% percentile intervals from 10,000 resamples of clues (seed 0), with the difference paired by pair. The maps are left out, since many pairs include training turns.

What this estimates, and what it does not:

- It is agreement on the words two boards share, about 6 of them, not on a full board of 20 to 25. It is not a ceiling for the top-1 of section 1, and it says nothing about how much higher than 0.63 a guesser could score there.
- The selection keeps pairs where each pick is on the other board, which favors clues with one obvious target. Clues that appear often contribute many pairs; the estimate weights each pair equally.
- It rests on the choice axiom and on one p shared by everyone. The giver's count is not in the data and differs between the two rows. Players are identified by their survey answers, and one person may appear under two ids, so a few pairs may be one person twice.
- It does show, on the same items, that people agree with each other 4.6 to 16.1 points more often than a guesser agrees with a person. That gap is room that a guesser could still close.

## Not covered

The hosted language models of C4 returned one ranking per turn, not a distribution, so they have no calibration here. A distribution could be built from their rankings with one fitted parameter; their replies are in the private dataset, and that analysis would need a job. LoRA and C8's distilled models are also left out: their vectors are in the private dataset, not in committed inputs.
