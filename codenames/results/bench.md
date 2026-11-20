# Codenames: the AI competition framework's word-vector bots

Exploratory. No plan was written before these runs (experiments/METHOD.md); the numbers
motivate a planned comparison and are not its result. Written by
`python -m codenames.bench_play analyze` from `bench-runs.json`.

- **The port reproduces the published games.** `bench.py` replays 900 of the 900 published single-team games of the framework's word-vector bots move for move: every clue, count and pick (computed exactly, on the framework's boards for seeds 0 to 99). The framework's own code, rerun here on 90 of them, gives the published moves in 90 and the port's in 90.
- **Table I against its own logs.** Table I scores a loss on the last blue word by its turns, not 25, though the paper's rule says 25 (w2v/glove: 2 such games). Other differences: w2v/w2v: Table I gives mean 6.81, the logs 6.18; w2vglove/glove: Table I gives standard deviation 2.18, the logs 2.13; w2vglove/w2vglove is in the logs (100 games, mean 5.53) but not in Table I.
- **Strangers cost every giver.** With the framework's guessers that share no vectors with them, the framework's codemasters take 9.1 to 9.9 turns on average, and this package's givers 12.5 to 14.5. With this package's guessers, the framework's codemasters take 13.7 to 16.3 and this package's givers 11.2 to 12.9 (other models only). Most of the gap is assassin losses, which score 25. Simulated: 200 boards of words.txt, seeds 0 to 199, 95% bootstrap intervals in the tables.
- **Population training does not carry over to this score.** With the framework's three guessers, the C3 population giver wins 459 of 600 games against 435 for the C3 self-play giver, but it asks for 1.04 words per clue against 1.78, so its wins take 11.3 turns against 8.8. Its mean score is worse by 1.24 (0.48 to 2.00) turns. C3 ranked the two the other way by expected words per turn; the single-team score charges for every extra turn.
- **Replacing the clues that a strict reading of the rules flags changes little.** Share of turns flagged by `codenames.legality` (illegal or borderline): framework w2v 1.9%, framework GloVe 1.2%, framework w2v+GloVe 1.7%, EmbeddingGemma 2, first choice 13.6%, potion, first choice 4.4%, potion, C3 self-play 2.7%, potion, C3 population 4.0%. Replacing them with each giver's next legal choice changes no giver's mean score over its strangers by more than 0.23 turns; the widest interval reaches 0.47 (table at the end of section 2).

## 1. The published games, replayed

The 900 single-team games of Stephenson et al. (arXiv 2412.11373v2, Table I), replayed with
`codenames/bench.py` on the framework's boards (seeds 0 to 99) and compared move by
move with the published logs. A score is the number of turns to find the 9 red words,
or 25 for a loss. "Table I" is copied from arXiv 2412.11373v2. "Logs" is recomputed
here from the published logs, once by Table I's apparent rule (25 only for the
assassin) and once by the framework's rule (25 also for the last blue word).

| Codemaster / guesser | Table I mean (loss %) | Logs, Table I's rule | Logs, framework rule | Port: same moves | Port: same score | Port mean |
|---|---|---|---|---|---|---|
| w2v/w2v | 6.81 (0%) | 6.18 | 6.18 (0 losses) | 100 of 100 | 100 of 100 | 6.18 |
| w2v/glove | 12.35 (22%) | 12.35 | 12.53 (24 losses) | 100 of 100 | 100 of 100 | 12.53 |
| w2v/w2vglove | 10.64 (16%) | 10.64 | 10.64 (16 losses) | 100 of 100 | 100 of 100 | 10.64 |
| glove/w2v | 12.21 (24%) | 12.21 | 12.21 (24 losses) | 100 of 100 | 100 of 100 | 12.21 |
| glove/glove | 5.24 (0%) | 5.24 | 5.24 (0 losses) | 100 of 100 | 100 of 100 | 5.24 |
| glove/w2vglove | 5.42 (0%) | 5.42 | 5.42 (0 losses) | 100 of 100 | 100 of 100 | 5.42 |
| w2vglove/w2v | 11.18 (21%) | 11.18 | 11.18 (21 losses) | 100 of 100 | 100 of 100 | 11.18 |
| w2vglove/glove | 5.86 (1%) | 5.86 | 5.86 (1 losses) | 100 of 100 | 100 of 100 | 5.86 |
| w2vglove/w2vglove | not in the table | 5.53 | 5.53 (0 losses) | 100 of 100 | 100 of 100 | 5.53 |

The framework's own code, at the pinned commit, played 90 of these games (9 pairs, seeds 0 to 9) on a GitHub-hosted runner: 90 have the published moves and 90 the port's (79.6 s a game).

## 2. Cross-play on this package's words

Mean score over 200 boards of `words.txt`, dealt as the framework deals them (seeds 0 to 199), with 95% bootstrap intervals over boards (10,000 draws, seed 0). Lower is better; 25 is a loss. Rows are givers, columns guessers; a giver with its own model's guesser is in bold.

### Every clue that the giver's own rule allows

| Giver \ guesser | framework w2v | framework GloVe | framework w2v+GloVe | EmbeddingGemma 2 | potion | C8 LoRA encoder |
|---|---|---|---|---|---|---|
| framework w2v | **5.9 (5.8 to 6.0)** | 9.1 (8.3 to 9.9) | **8.0 (7.4 to 8.8)** | 13.3 (12.3 to 14.2) | 14.3 (13.3 to 15.4) | 13.4 (12.4 to 14.4) |
| framework GloVe | 9.9 (9.0 to 10.9) | **4.6 (4.5 to 4.7)** | **4.9 (4.6 to 5.2)** | 16.8 (15.7 to 17.9) | 16.0 (14.9 to 17.1) | 16.2 (15.1 to 17.3) |
| framework w2v+GloVe | **8.5 (7.7 to 9.3)** | **5.4 (5.1 to 5.9)** | **4.9 (4.8 to 5.0)** | 15.4 (14.2 to 16.5) | 16.4 (15.3 to 17.5) | 14.2 (13.1 to 15.3) |
| EmbeddingGemma 2, first choice | 13.6 (12.6 to 14.6) | 13.9 (12.8 to 14.9) | 13.7 (12.7 to 14.8) | **6.4 (6.3 to 6.6)** | 10.9 (10.0 to 11.8) | 11.6 (10.6 to 12.6) |
| potion, first choice | 12.6 (11.6 to 13.7) | 12.7 (11.6 to 13.8) | 12.2 (11.2 to 13.3) | 10.8 (9.9 to 11.8) | **5.0 (4.9 to 5.1)** | 11.7 (10.6 to 12.7) |
| potion, C3 self-play | 13.5 (12.4 to 14.6) | 13.5 (12.5 to 14.5) | 12.8 (11.8 to 13.8) | 11.1 (10.2 to 12.1) | **5.9 (5.8 to 6.1)** | 12.7 (11.7 to 13.7) |
| potion, C3 population | 14.6 (13.7 to 15.5) | 14.9 (14.0 to 15.8) | 14.1 (13.3 to 14.9) | 12.0 (11.3 to 12.8) | **8.7 (8.6 to 8.8)** | 13.7 (12.8 to 14.6) |

Games won of 200 (losses on the assassin / on the last opponent word):

| Giver \ guesser | framework w2v | framework GloVe | framework w2v+GloVe | EmbeddingGemma 2 | potion | C8 LoRA encoder |
|---|---|---|---|---|---|---|
| framework w2v | 200 (0 / 0) | 179 (21 / 0) | 186 (14 / 0) | 151 (48 / 1) | 135 (64 / 1) | 149 (51 / 0) |
| framework GloVe | 166 (33 / 1) | 200 (0 / 0) | 198 (2 / 0) | 106 (91 / 3) | 114 (85 / 1) | 110 (90 / 0) |
| framework w2v+GloVe | 177 (23 / 0) | 196 (4 / 0) | 200 (0 / 0) | 120 (76 / 4) | 108 (90 / 2) | 131 (69 / 0) |
| EmbeddingGemma 2, first choice | 147 (53 / 0) | 140 (60 / 0) | 139 (61 / 0) | 200 (0 / 0) | 166 (33 / 1) | 158 (41 / 1) |
| potion, first choice | 145 (55 / 0) | 145 (55 / 0) | 149 (51 / 0) | 163 (35 / 2) | 200 (0 / 0) | 154 (45 / 1) |
| potion, C3 self-play | 141 (59 / 0) | 144 (54 / 2) | 150 (48 / 2) | 165 (33 / 2) | 200 (0 / 0) | 150 (49 / 1) |
| potion, C3 population | 151 (49 / 0) | 150 (48 / 2) | 158 (40 / 2) | 173 (26 / 1) | 200 (0 / 0) | 158 (41 / 1) |

Each giver averaged, board by board, over the guessers of a group that share no vectors with it:

| Giver | Group | Guessers | Mean score (95% interval) |
|---|---|---|---|
| framework w2v | framework guessers | framework GloVe | 9.10 (8.34 to 9.91) |
| framework w2v | our guessers | EmbeddingGemma 2, potion, C8 LoRA encoder | 13.67 (13.03 to 14.31) |
| framework GloVe | framework guessers | framework w2v | 9.88 (8.96 to 10.91) |
| framework GloVe | our guessers | EmbeddingGemma 2, potion, C8 LoRA encoder | 16.33 (15.67 to 16.98) |
| framework w2v+GloVe | our guessers | EmbeddingGemma 2, potion, C8 LoRA encoder | 15.31 (14.63 to 16.03) |
| EmbeddingGemma 2, first choice | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 13.73 (12.84 to 14.63) |
| EmbeddingGemma 2, first choice | our guessers | potion, C8 LoRA encoder | 11.21 (10.45 to 12.03) |
| potion, first choice | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 12.51 (11.71 to 13.35) |
| potion, first choice | our guessers | EmbeddingGemma 2, C8 LoRA encoder | 11.25 (10.49 to 12.01) |
| potion, C3 self-play | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 13.27 (12.41 to 14.14) |
| potion, C3 self-play | our guessers | EmbeddingGemma 2, C8 LoRA encoder | 11.91 (11.15 to 12.71) |
| potion, C3 population | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 14.51 (13.73 to 15.28) |
| potion, C3 population | our guessers | EmbeddingGemma 2, C8 LoRA encoder | 12.86 (12.19 to 13.55) |

Paired differences in score, board by board (negative: the first giver finishes sooner):

| Difference | Guesser | Mean (95% interval) |
|---|---|---|
| potion, C3 population minus potion, C3 self-play | framework w2v | 1.09 (0.07 to 2.10) |
| potion, C3 population minus potion, C3 self-play | framework GloVe | 1.36 (0.36 to 2.35) |
| potion, C3 population minus potion, C3 self-play | framework w2v+GloVe | 1.27 (0.30 to 2.23) |
| potion, C3 population minus potion, C3 self-play | EmbeddingGemma 2 | 0.89 (0.04 to 1.70) |
| potion, C3 population minus potion, C3 self-play | C8 LoRA encoder | 1.00 (0.07 to 1.92) |
| potion, C3 population minus potion, C3 self-play | framework guessers | 1.24 (0.48 to 2.00) |
| potion, C3 population minus potion, C3 self-play | our guessers | 0.95 (0.29 to 1.57) |
| potion, C3 population minus potion, first choice | framework w2v | 1.96 (0.69 to 3.21) |
| potion, C3 population minus potion, first choice | framework GloVe | 2.17 (0.99 to 3.33) |
| potion, C3 population minus potion, first choice | framework w2v+GloVe | 1.85 (0.58 to 3.08) |
| potion, C3 population minus potion, first choice | EmbeddingGemma 2 | 1.18 (0.04 to 2.32) |
| potion, C3 population minus potion, first choice | C8 LoRA encoder | 2.03 (0.90 to 3.17) |
| potion, C3 population minus potion, first choice | framework guessers | 2.00 (1.05 to 2.91) |
| potion, C3 population minus potion, first choice | our guessers | 1.60 (0.79 to 2.44) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework w2v | 0.97 (−0.26 to 2.24) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework GloVe | 1.18 (−0.27 to 2.63) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework w2v+GloVe | 1.50 (0.02 to 2.94) |
| EmbeddingGemma 2, first choice minus potion, first choice | C8 LoRA encoder | −0.10 (−1.42 to 1.27) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework guessers | 1.22 (0.17 to 2.27) |
| EmbeddingGemma 2, first choice minus potion, first choice | our guessers | −0.10 (−1.42 to 1.27) |
| EmbeddingGemma 2, first choice minus framework GloVe | framework w2v | 3.71 (2.36 to 5.03) |
| EmbeddingGemma 2, first choice minus framework GloVe | potion | −5.13 (−6.62 to −3.61) |
| EmbeddingGemma 2, first choice minus framework GloVe | C8 LoRA encoder | −4.62 (−6.12 to −3.12) |
| EmbeddingGemma 2, first choice minus framework GloVe | framework guessers | 3.71 (2.36 to 5.03) |
| EmbeddingGemma 2, first choice minus framework GloVe | our guessers | −4.88 (−6.00 to −3.72) |
| EmbeddingGemma 2, first choice minus framework w2v | framework GloVe | 4.77 (3.60 to 5.98) |
| EmbeddingGemma 2, first choice minus framework w2v | potion | −3.47 (−4.83 to −2.09) |
| EmbeddingGemma 2, first choice minus framework w2v | C8 LoRA encoder | −1.86 (−3.19 to −0.49) |
| EmbeddingGemma 2, first choice minus framework w2v | framework guessers | 4.77 (3.60 to 5.98) |
| EmbeddingGemma 2, first choice minus framework w2v | our guessers | −2.67 (−3.73 to −1.58) |
| potion, C3 population minus framework GloVe | framework w2v | 4.70 (3.38 to 5.95) |
| potion, C3 population minus framework GloVe | EmbeddingGemma 2 | −4.79 (−6.06 to −3.48) |
| potion, C3 population minus framework GloVe | C8 LoRA encoder | −2.50 (−3.85 to −1.15) |
| potion, C3 population minus framework GloVe | framework guessers | 4.70 (3.38 to 5.95) |
| potion, C3 population minus framework GloVe | our guessers | −3.65 (−4.64 to −2.65) |

### Clues that codenames.legality flags replaced by the giver's next legal choice

| Giver \ guesser | framework w2v | framework GloVe | framework w2v+GloVe | EmbeddingGemma 2 | potion | C8 LoRA encoder |
|---|---|---|---|---|---|---|
| framework w2v | **5.9 (5.8 to 6.0)** | 9.3 (8.5 to 10.1) | **8.0 (7.4 to 8.8)** | 13.4 (12.4 to 14.4) | 14.5 (13.4 to 15.5) | 13.4 (12.4 to 14.4) |
| framework GloVe | 9.9 (8.9 to 10.9) | **4.6 (4.5 to 4.7)** | **4.9 (4.7 to 5.3)** | 17.0 (15.9 to 18.0) | 16.1 (15.0 to 17.3) | 16.4 (15.3 to 17.5) |
| framework w2v+GloVe | **8.7 (7.8 to 9.6)** | **5.4 (5.1 to 5.9)** | **4.9 (4.8 to 5.0)** | 15.4 (14.3 to 16.5) | 16.4 (15.3 to 17.6) | 14.4 (13.2 to 15.5) |
| EmbeddingGemma 2, first choice | 13.5 (12.5 to 14.5) | 13.9 (12.8 to 15.0) | 13.7 (12.7 to 14.8) | **6.4 (6.2 to 6.6)** | 10.5 (9.7 to 11.4) | 12.3 (11.3 to 13.4) |
| potion, first choice | 12.8 (11.7 to 13.9) | 13.1 (12.0 to 14.2) | 12.3 (11.3 to 13.4) | 11.0 (10.0 to 12.0) | **5.0 (4.8 to 5.1)** | 12.0 (11.0 to 13.1) |
| potion, C3 self-play | 13.5 (12.4 to 14.6) | 13.7 (12.7 to 14.8) | 12.9 (11.9 to 14.0) | 11.3 (10.3 to 12.2) | **5.9 (5.8 to 6.1)** | 12.7 (11.7 to 13.8) |
| potion, C3 population | 14.7 (13.8 to 15.6) | 14.9 (14.0 to 15.8) | 14.3 (13.5 to 15.2) | 12.2 (11.4 to 13.0) | **8.7 (8.6 to 8.7)** | 13.7 (12.8 to 14.6) |

Games won of 200 (losses on the assassin / on the last opponent word):

| Giver \ guesser | framework w2v | framework GloVe | framework w2v+GloVe | EmbeddingGemma 2 | potion | C8 LoRA encoder |
|---|---|---|---|---|---|---|
| framework w2v | 200 (0 / 0) | 177 (23 / 0) | 186 (14 / 0) | 150 (49 / 1) | 134 (65 / 1) | 149 (51 / 0) |
| framework GloVe | 166 (33 / 1) | 200 (0 / 0) | 198 (2 / 0) | 105 (92 / 3) | 113 (86 / 1) | 108 (92 / 0) |
| framework w2v+GloVe | 175 (25 / 0) | 196 (4 / 0) | 200 (0 / 0) | 120 (76 / 4) | 108 (90 / 2) | 129 (71 / 0) |
| EmbeddingGemma 2, first choice | 148 (52 / 0) | 141 (59 / 0) | 142 (58 / 0) | 200 (0 / 0) | 170 (29 / 1) | 150 (49 / 1) |
| potion, first choice | 145 (55 / 0) | 141 (59 / 0) | 150 (50 / 0) | 161 (37 / 2) | 200 (0 / 0) | 150 (49 / 1) |
| potion, C3 self-play | 140 (60 / 0) | 141 (57 / 2) | 148 (50 / 2) | 164 (34 / 2) | 200 (0 / 0) | 149 (50 / 1) |
| potion, C3 population | 149 (51 / 0) | 148 (50 / 2) | 153 (45 / 2) | 171 (28 / 1) | 200 (0 / 0) | 157 (42 / 1) |

Each giver averaged, board by board, over the guessers of a group that share no vectors with it:

| Giver | Group | Guessers | Mean score (95% interval) |
|---|---|---|---|
| framework w2v | framework guessers | framework GloVe | 9.29 (8.51 to 10.12) |
| framework w2v | our guessers | EmbeddingGemma 2, potion, C8 LoRA encoder | 13.75 (13.11 to 14.40) |
| framework GloVe | framework guessers | framework w2v | 9.88 (8.95 to 10.90) |
| framework GloVe | our guessers | EmbeddingGemma 2, potion, C8 LoRA encoder | 16.50 (15.85 to 17.14) |
| framework w2v+GloVe | our guessers | EmbeddingGemma 2, potion, C8 LoRA encoder | 15.39 (14.70 to 16.11) |
| EmbeddingGemma 2, first choice | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 13.69 (12.81 to 14.56) |
| EmbeddingGemma 2, first choice | our guessers | potion, C8 LoRA encoder | 11.42 (10.63 to 12.21) |
| potion, first choice | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 12.72 (11.90 to 13.55) |
| potion, first choice | our guessers | EmbeddingGemma 2, C8 LoRA encoder | 11.52 (10.73 to 12.30) |
| potion, C3 self-play | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 13.39 (12.50 to 14.28) |
| potion, C3 self-play | our guessers | EmbeddingGemma 2, C8 LoRA encoder | 11.98 (11.22 to 12.78) |
| potion, C3 population | framework guessers | framework w2v, framework GloVe, framework w2v+GloVe | 14.65 (13.85 to 15.45) |
| potion, C3 population | our guessers | EmbeddingGemma 2, C8 LoRA encoder | 12.95 (12.28 to 13.64) |

Paired differences in score, board by board (negative: the first giver finishes sooner):

| Difference | Guesser | Mean (95% interval) |
|---|---|---|
| potion, C3 population minus potion, C3 self-play | framework w2v | 1.15 (0.13 to 2.17) |
| potion, C3 population minus potion, C3 self-play | framework GloVe | 1.20 (0.19 to 2.19) |
| potion, C3 population minus potion, C3 self-play | framework w2v+GloVe | 1.42 (0.43 to 2.40) |
| potion, C3 population minus potion, C3 self-play | EmbeddingGemma 2 | 0.94 (0.08 to 1.76) |
| potion, C3 population minus potion, C3 self-play | C8 LoRA encoder | 0.99 (0.04 to 1.92) |
| potion, C3 population minus potion, C3 self-play | framework guessers | 1.25 (0.50 to 2.02) |
| potion, C3 population minus potion, C3 self-play | our guessers | 0.96 (0.27 to 1.60) |
| potion, C3 population minus potion, first choice | framework w2v | 1.93 (0.62 to 3.21) |
| potion, C3 population minus potion, first choice | framework GloVe | 1.83 (0.59 to 3.04) |
| potion, C3 population minus potion, first choice | framework w2v+GloVe | 2.03 (0.74 to 3.31) |
| potion, C3 population minus potion, first choice | EmbeddingGemma 2 | 1.19 (0.10 to 2.28) |
| potion, C3 population minus potion, first choice | C8 LoRA encoder | 1.68 (0.54 to 2.82) |
| potion, C3 population minus potion, first choice | framework guessers | 1.93 (0.97 to 2.87) |
| potion, C3 population minus potion, first choice | our guessers | 1.43 (0.62 to 2.24) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework w2v | 0.73 (−0.55 to 2.02) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework GloVe | 0.79 (−0.66 to 2.25) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework w2v+GloVe | 1.38 (−0.12 to 2.88) |
| EmbeddingGemma 2, first choice minus potion, first choice | C8 LoRA encoder | 0.30 (−1.02 to 1.67) |
| EmbeddingGemma 2, first choice minus potion, first choice | framework guessers | 0.97 (−0.09 to 2.03) |
| EmbeddingGemma 2, first choice minus potion, first choice | our guessers | 0.30 (−1.02 to 1.67) |
| EmbeddingGemma 2, first choice minus framework GloVe | framework w2v | 3.60 (2.28 to 4.86) |
| EmbeddingGemma 2, first choice minus framework GloVe | potion | −5.61 (−7.06 to −4.16) |
| EmbeddingGemma 2, first choice minus framework GloVe | C8 LoRA encoder | −4.07 (−5.57 to −2.52) |
| EmbeddingGemma 2, first choice minus framework GloVe | framework guessers | 3.60 (2.28 to 4.86) |
| EmbeddingGemma 2, first choice minus framework GloVe | our guessers | −4.83 (−5.96 to −3.70) |
| EmbeddingGemma 2, first choice minus framework w2v | framework GloVe | 4.60 (3.40 to 5.82) |
| EmbeddingGemma 2, first choice minus framework w2v | potion | −3.94 (−5.29 to −2.57) |
| EmbeddingGemma 2, first choice minus framework w2v | C8 LoRA encoder | −1.10 (−2.49 to 0.29) |
| EmbeddingGemma 2, first choice minus framework w2v | framework guessers | 4.60 (3.40 to 5.82) |
| EmbeddingGemma 2, first choice minus framework w2v | our guessers | −2.52 (−3.56 to −1.49) |
| potion, C3 population minus framework GloVe | framework w2v | 4.79 (3.48 to 6.02) |
| potion, C3 population minus framework GloVe | EmbeddingGemma 2 | −4.77 (−6.03 to −3.48) |
| potion, C3 population minus framework GloVe | C8 LoRA encoder | −2.69 (−4.03 to −1.33) |
| potion, C3 population minus framework GloVe | framework guessers | 4.79 (3.48 to 6.02) |
| potion, C3 population minus framework GloVe | our guessers | −3.73 (−4.71 to −2.74) |

### Clues that codenames.legality flags

A clue is flagged when `codenames.legality` finds an illegal or borderline relation between it and an unrevealed board word (a form of the word, a part of a compound, a shared stem; see that module). Turns flagged with every clue allowed, pooled over guessers, and the change in mean score over the guessers that share no vectors with the giver when flagged clues are replaced (paired by board):

| Giver | Flagged turns | Of them illegal | Share | Flagged after replacing | Change in mean score (95% interval) |
|---|---|---|---|---|---|
| framework w2v | 171 of 9003 | 110 | 1.9% | 0 | 0.11 (0.03 to 0.22) |
| framework GloVe | 90 of 7715 | 70 | 1.2% | 0 | 0.12 (0.00 to 0.25) |
| framework w2v+GloVe | 130 of 7767 | 89 | 1.7% | 0 | 0.08 (−0.05 to 0.21) |
| EmbeddingGemma 2, first choice | 1305 of 9633 | 753 | 13.6% | 0 | 0.05 (−0.36 to 0.47) |
| potion, first choice | 365 of 8328 | 291 | 4.4% | 0 | 0.23 (0.07 to 0.41) |
| potion, C3 self-play | 250 of 9403 | 195 | 2.7% | 0 | 0.10 (−0.05 to 0.28) |
| potion, C3 population | 508 of 12655 | 415 | 4.0% | 0 | 0.12 (−0.01 to 0.28) |

In the 900 published games, 137 of 5847 clues are flagged the same way, against the board words still unrevealed.

## The rule each giver enforces

- **The framework's codemasters** (players/codemaster_*.py): a clue is not a board word, not a part of one and does not contain one, and no board word equals the clue's WordNet lemma or Lancaster stem (NLTK 3.10.4). Clues come from the framework's list of 7192 words. The rule is checked against the words still on the board.
- **This package's givers** (`codenames.words.legal`): a clue is not an unrevealed board word, not a part of one, and does not contain one. Clues come from the 11794 of the 12,000 clue words that both GloVe and word2vec have, so that the framework's guessers can read them (206 dropped).
- **The flag** (`codenames.legality`, for the second set of tables): any relation that this module calls illegal (a form of the word, a part of a compound, a spelling variant) or borderline (a shared stem, affix stem, three-letter compound part or long prefix). It is stricter than both rules: it catches broken for BREAK and teaching for TEACHER, which both rules allow.

## How each number is known

- Table I values: taken from Stephenson, Sidji and Ronval, arXiv 2412.11373v2, Table I.
- Values from the published logs: computed exactly from bot_results_new_style.txt and move_log.txt in stepmat/Codenames_GPT at commit fc28bc9.
- The port's games: computed exactly; compared move by move with the logs. The framework's own code was rerun on 90 games as a second method.
- Cross-play: simulated games, 200 boards, board seeds 0 to 199; every bot is deterministic, so each cell is one game per board. Of 19 givers' runs with every clue allowed that were repeated on other runners (13400 games), 19 gave the same score in every game. Intervals are 95% percentile bootstrap intervals over boards, 10,000 draws, seed 0; paired differences resample boards once for both givers.
- The C3 givers were refit with codenames.rl on the task with the potion giver, EmbeddingGemma 2 held out and training seed 1. The largest difference from the θ in results/c3.json is 0.0000 (self-play) and 0.0000 (population).
- The C8 encoder (arm B, seed 1, clue template query) encoded 1307 words that its stored vectors lack, on a CPU in float32 (203 s); on 200 stored words the largest difference from the stored float16 vectors is 0.0056 (bare) and 0.0060 (query).

## Sources and licenses

| Source | What | License | Committed |
|---|---|---|---|
| stepmat/Codenames_GPT, commit fc28bc9 (Stephenson et al., arXiv 2412.11373) | the framework's code, clue list and published logs | MIT | no; downloaded and pinned by SHA-256 |
| CodenamesAICompetition/Game (Summerville et al. 2019) | the original framework; the same bots and word pool | MIT | no |
| The framework's board words | the 394 words of the published game | the rights to the game's words belong to Czech Games Edition; whether the MIT license of the file covers them is unclear | no; used only to rebuild the published boards |
| GloVe 6B 300d (Pennington et al. 2014) | word vectors | Public Domain Dedication and License v1.0, per the GloVe page | no |
| GoogleNews word2vec 300d (Mikolov et al. 2013), via gensim-data | word vectors | the word2vec project page lists Apache 2.0 for the project; no separate license for the vectors is stated | no |

## Compute

The vectors, the replays and the cross-play ran on GitHub-hosted runners (CPU) through a workflow on the branch `compute/codenames-bench`, which commits the raw outputs to that branch under `runs/bench/<run id>/`. The C8 column ran in Hugging Face Jobs (`cpu-upgrade`), since its weights are in the private dataset; its outputs are in that dataset under `runs/bench/`. Both built the same vector subsets (SHA-256 ce302d5de4e0 for GloVe and ae1c8519d8df for word2vec).

## Limits

- Exploratory, one deterministic game per board and pair, one training seed for the C3 givers, one seed for the C8 encoder. The comparison worth planning is named in the findings; a plan should fix its metric, guessers and board seeds before running.
- The framework's guessers pick exactly the count; they never stop early or take the extra pick that the printed rules allow. This package's guessers do the same here.
- Our givers were built for expected turn value, not for the single-team score, which charges 25 for a loss; they take more risk than that score rewards.

## Usage

```sh
uv run python -m codenames.bench_data fetch        # the framework's files
uv run python -m codenames.bench_play analyze      # bench.json and bench.md from bench-runs.json
```

The other commands need the vector subsets and run in the workflow; see the module docstrings.
