# Givers and guessers built on different models

Part of the [Codenames README](../README.md). The JSON files beside this note hold its numbers, with their seeds and settings.

## The cost of a different guesser

For each of 200 random boards (seed 0), each giver picks the clue and count with the highest expected value under its own Luce guesser. Clues are limited to the 4,998 clue words that are EAT cues, so that the association guesser can answer every one. Then the exact expected number of team words found is computed for every guesser. The diagonal is what the giver expects; the rest is what it gets.

| Giver \ guesser | potion | wordllama | granite | bge | Qwen3 0.6B | harrier 270m | harrier 0.6b | Gemma 2 | EAT counts | own minus mean of other models |
|---|---|---|---|---|---|---|---|---|---|---|
| potion | **2.13** | 1.14 | 1.01 | 1.41 | 1.29 | 1.28 | 1.42 | 1.28 | 0.89 | 0.87 |
| wordllama | 1.24 | **1.87** | 0.95 | 1.10 | 1.09 | 1.14 | 1.24 | 1.11 | 0.80 | 0.75 |
| granite | 1.22 | 1.07 | **1.79** | 1.12 | 1.20 | 1.31 | 1.31 | 1.16 | 0.82 | 0.58 |
| bge | 1.52 | 1.01 | 1.00 | **1.97** | 1.26 | 1.25 | 1.34 | 1.20 | 0.83 | 0.74 |
| Qwen3 0.6B | 1.24 | 0.96 | 0.98 | 1.15 | **2.04** | 1.21 | 1.43 | 1.17 | 0.83 | 0.88 |
| harrier 270m | 1.19 | 1.01 | 1.09 | 1.19 | 1.24 | **2.13** | 1.58 | 1.32 | 0.78 | 0.90 |
| harrier 0.6b | 1.29 | 1.03 | 1.05 | 1.20 | 1.48 | 1.57 | **2.12** | 1.30 | 0.82 | 0.85 |
| Gemma 2 | 1.40 | 1.14 | 1.10 | 1.28 | 1.33 | 1.41 | 1.52 | **1.80** | 0.96 | 0.49 |
| average of all but bge | 1.51 | 1.36 | 1.25 | 1.36 | 1.48 | 1.49 | 1.59 | 1.44 | 0.90 | |
| average of all but Gemma 2 | 1.54 | 1.35 | 1.26 | 1.43 | 1.49 | 1.49 | 1.59 | 1.38 | 0.91 | |

Standard errors over boards are 0.02 to 0.05. Every giver loses 0.5 to 0.9 words per clue when the guesser is another model, and more when it is the association guesser. The newest models lose about as much as the static tables. EmbeddingGemma 2 loses least: its clues keep 73% of their value with other models and 53% with the association guesser. The last two rows are **ensemble givers**: they maximize the average expected value over seven models' Luce guessers and never see the eighth. Against that held-out model they find 1.36 (bge) and 1.38 (EmbeddingGemma 2) words. Both score above the mean of the seven single givers against the same model (1.21 and 1.22). Against EmbeddingGemma 2 the ensemble also scores above every single giver; against bge, potion's clues score higher (1.41). No paired interval was computed for these comparisons, so they are descriptive. The ensembles ask for fewer words per clue: 2.1 on average, against 2.7 to 3.2 for single givers.

`results/crossplay.json` also has the expected value of each turn. It subtracts 1 for an opponent word and, for the assassin, the number of team words left. With that cost a giver's clues are worth 1.3 to 1.8 to its own model and 0.1 to 0.9 to other models. To the association guesser they are worth −0.16 to −0.07, except EmbeddingGemma 2's (0.16) and the two ensembles' (0.25).

## Whole games, compared with Kim et al. (2019)

Kim et al. paired codemasters and guessers built on word2vec and GloVe, and report that each codemaster "paired with their corresponding guesser achieved a 100% win rate", while mixed pairs did worse. GloVe and word2vec could not be fetched here, so their numbers cannot be rerun. Their finding can. `players.threshold_giver` follows their description: a clue's targets are the team words that rank above every other board word and are at least t similar to the clue. The giver picks the clue with the most targets. Each giver played 50 games (seed 1, t = 0.4) with each greedy guesser.

| Giver \ guesser | potion | wordllama | granite | bge | Qwen3 0.6B | harrier 270m | harrier 0.6b | Gemma 2 | EAT counts |
|---|---|---|---|---|---|---|---|---|---|
| potion | **50** | 32 | 31 | 42 | 32 | 36 | 39 | 38 | 10 |
| wordllama | 22 | **50** | 19 | 20 | 27 | 20 | 29 | 22 | 11 |
| granite | 14 | 10 | **50** | 16 | 16 | 20 | 14 | 17 | 8 |
| bge | 21 | 12 | 19 | **50** | 24 | 14 | 20 | 18 | 1 |
| Qwen3 0.6B | 22 | 14 | 14 | 23 | **50** | 18 | 28 | 21 | 2 |
| harrier 270m | 24 | 16 | 24 | 18 | 21 | **50** | 24 | 22 | 5 |
| harrier 0.6b | 24 | 17 | 24 | 27 | 25 | 30 | **50** | 20 | 5 |
| Gemma 2 | 23 | 18 | 20 | 22 | 19 | 34 | 28 | **50** | 9 |

Every model wins all 50 games with its own guesser, as Kim et al. found. This is guaranteed by the giver's rule whenever some clue has a target, so it is not a test of their finding; the comparison with Kim et al. is the drop for mixed pairs. The guesser then picks the same words in the same order as the giver ranked them. Mixed pairs win 10 to 42 of 50, and the association guesser wins at most 11. A single threshold suits the models unevenly: their similarities have different scales (fitted τ ranges from 0.02 to 0.07). With t = 0.4 the transformer givers name many targets per clue and win alone in 2.4 to 4.3 turns on average. The static givers name fewer, take 8.8 and 9.0 turns, and lose less often with other guessers.
