# Codenames

In Codenames, a spymaster (the giver) names one clue word and a number, and teammates (the guessers) try to pick the giver's words from a board of 25. This folder builds givers and guessers from word embeddings and asks three questions:

1. Which embedding models predict the words people pick?
2. How much of a giver's expected score survives when the guesser uses a different model, or is a person?
3. Does training a giver against several guessers, instead of against its own model, close that gap?

Short answers, with how each is known:

- EmbeddingGemma 2 and the small static table potion-base-8M pick the same word as a person most often: 54.8% and 54.4% of 551 turns from 68 test games, with the Cultural Codes turns split by game (95% intervals 50.1% to 59.5% and 49.8% to 59.3%; their difference, 0.4 points, has the interval −3.8 to 4.6). The other six models pick it in 39.4% to 49.0%. Instruction prefixes help the instruction-trained models a little and ruin the others. Computed exactly on the test turns, with intervals from resampling games.
- Training a linear map on potion's clue vectors raises its match from 54.4% to 63.2%. The map is trained on the human guesses of the training games only. The gain, 8.8 points (6.3 to 11.4), is as large as on the published split (8.2), so it was not an artifact of test games having turns in training. Same method; the map is averaged over five training seeds.
- A giver expects to find 1.8 to 2.1 of its words per clue when the guesser shares its vectors. A guesser built on another model finds 57% to 73% of that, on average over the other seven models. A guesser that follows association counts from people finds 37% to 53%. Exact expected values for each of 200 random boards, one clue each, for eight models.
- A giver trained by policy gradient against its own model reaches an expected turn value of 1.60 with that model, 0.75 with a held-out model, and −0.05 with the association counts. Trained against six other models instead, it reaches 1.03, 0.92 and 0.35. The gap between self-play and cross-play shrinks from 0.85 to 0.11, at a cost of 0.57 in self-play. A second giver shows the same pattern. Exact rewards on 200 test boards, one training seed per giver.

## The game as modeled

The rules are those of the Codenames AI competition (Summerville et al. 2019) and of Kim et al. (2019). A board has 25 words: 9 belong to the team, 8 to the opponent, 7 are neutral and 1 is the assassin. Only the team plays. The giver names a clue and a count n, and the guesser picks up to n words, one at a time. The turn ends at the first pick that is not a team word. The team wins when it finds all 9 words and loses if it picks the assassin or the last opponent word. The printed game allows one extra pick beyond n; this model does not. A clue may not be a board word, contain one, or be contained in one.

`game.py` holds the rules; the tests settle the edge cases (a turn that finds the last team word ends the game at once).

## Words

The official word lists belong to Czech Games Edition and are not used or committed here. `words.txt` holds 400 board words chosen by `make_words.py` from Princeton WordNet 3.0. A board word is a lowercase noun of 3 to 9 letters, with at least two noun senses, whose most common sense names a thing (an animal, artifact, food, person, place, plant or substance). Its Zipf frequency in wordfreq must lie between 3.6 and 5.6, and WordNet's sense-tagged corpus must tag it as a noun at least as often as a verb or adjective. The 400 most frequent such words are kept, after removing plurals and other inflections and a short list of words that would be out of place on a family board (the list is in the script). WordNet 3.0 may be copied and modified under the Princeton license, whose notice must travel with copies; `fetch.py` downloads it with that notice.

Clues come from the 12,000 most frequent words in wordfreq (version 3.1.1) that are lowercase, alphabetic, at least three letters long and in WordNet. wordfreq's code is under the Apache 2.0 license and its data under CC BY-SA 4.0. It is used as a package, and no frequency data is committed. A rarity penalty, 0.1 × max(0, 3.5 − Zipf frequency), keeps the giver to words people know.

## Models

| Model | Kind | Dims | License | Vectors computed |
|---|---|---|---|---|
| potion-base-8M | static table distilled from bge-base-en-v1.5 | 256 | MIT | here |
| wordllama l2_supercat | static table trained from LLM token embeddings | 256 | MIT | here |
| granite-embedding-small-english-r2 | ModernBERT encoder, 48M parameters, int8 | 384 | Apache 2.0 | here |
| bge-small-en-v1.5 | BERT encoder, 33M parameters, int8 | 384 | MIT | here |
| Qwen3-Embedding-0.6B | decoder | 1024 | Apache 2.0 | GitHub-hosted runner |
| harrier-oss-v1-270m | decoder, last-token pooling | 640 | MIT | GitHub-hosted runner |
| harrier-oss-v1-0.6b | decoder, last-token pooling | 1024 | MIT | GitHub-hosted runner |
| EmbeddingGemma 2 | multimodal embedding model, 744M parameters in all | 768 | Apache 2.0 | GitHub-hosted runner |

The licenses and sizes were read from each model card through the Hugging Face Hub API. The machine that ran this study cannot reach huggingface.co. The first four models come from packages on npm and PyPI that redistribute the weights under their licenses, pinned by SHA-256 in `fetch.py`. The potion table comes from `@jigyasudham/veto-model`, an int8 repack whose shipped test vectors record a cosine error of at most 6.13e-5 against the original float32 weights. The wordllama table is inside the `wordllama` wheel. The granite files are the int8 export by onnx-community, shipped in `isorite-model-granite`; both have the sizes listed on the Hub. bge comes from `impro-plugin-semantic-mute-model`. The Hugging Face models ran on a GitHub-hosted runner (CPU) on 2026-11-18, with the latest sentence-transformers and torch, through a workflow on the branch `compute/codenames-embed` that commits the vectors to that branch. `compute/codenames-embed.yml` is a copy of that workflow. Each vector file is 11 to 25 MB, too large for this folder. To use them, copy `codenames/vectors/` from that branch, or rerun `embed.py` with `--model <id>` on a machine that can reach the Hub. EmbeddingGemma 2 also needs torchvision and Pillow, even for text.

GloVe and fastText could not be downloaded here, so the static baselines are potion and wordllama.

## Human data

| Data | What | License | Committed |
|---|---|---|---|
| Cultural Codes (Shaikh et al. 2023) | 6,088 Codenames Duet guesses with clue and board (the rows that parse); train, val and test splits as published | none in the repository; the paper says CC BY-SA 4.0 is planned | no; `fetch.py` downloads it |
| Edinburgh Associative Thesaurus | about 100 people's first responses to each of 8,400 cues, collected 1968–1971 | UK Open Government Licence, as stated by the JSON port used; the original site is gone, so this was not checked at the source | no; downloaded |
| Small World of Words, SWOW-EN18 | 3 million responses to 12,000 cues | CC BY-NC-ND 3.0 | no; manual download, loader only |
| USF free association norms | 72,000 cue-response pairs | copyright of the authors, no license stated | no; manual download |

Only the first two are used in the numbers below. The Cultural Codes files carry no game or player ids. The section "Human guesses split by game" below rebuilds both and finds that the published split puts the two sides of most games in different files. Sections 1 to 4 use the files as published; that section repeats sections 1 and 4 on a split by game. A fixed fifth of EAT cues, chosen by hash (`norms.held_out`), is never used for training.

## How a guesser is modeled

A guesser that ranks words by similarity is deterministic. People are not, so each guesser also gets a **Luce guesser**: it picks a word with probability proportional to exp(similarity / τ) among the words still unpicked. The temperature τ is fitted to the human guesses in the Cultural Codes train split, and the template (below) to the val split. For the association-count guesser the weight of a word is its count plus ε, with ε = 0.1 fitted the same way. That guesser answers only clues that are EAT cues; for other clues it picks at random.

## 1. Which models predict human guesses

Each model picks the board word most similar to the clue, on the 305 turns of the Cultural Codes test split. A random pick would match the person 1 time in 21 to 25. The log-likelihood is the mean log-probability of the person's word under the fitted Luce guesser; random picking scores −3.10.

| Model | Template (clue / word) | τ | Same word as the person (95% interval) | Log-likelihood |
|---|---|---|---|---|
| EmbeddingGemma 2 | bare / bare | 0.02 | 0.607 (0.551–0.660) | −1.63 |
| potion-base-8M | bare / bare | 0.07 | 0.580 (0.524–0.634) | −1.65 |
| bge-small-en-v1.5 | word / word | 0.03 | 0.534 (0.478–0.590) | −1.73 |
| wordllama | bare / bare | 0.07 | 0.531 (0.475–0.586) | −1.79 |
| harrier-oss-v1-0.6b | query / bare | 0.02 | 0.498 (0.443–0.554) | −1.84 |
| Qwen3-Embedding-0.6B | query / bare | 0.03 | 0.495 (0.439–0.551) | −1.87 |
| granite-small-r2 | word / word | 0.02 | 0.456 (0.401–0.512) | −2.12 |
| harrier-oss-v1-270m | bare / bare | 0.03 | 0.452 (0.398–0.509) | −2.03 |
| EAT counts | ε 0.1 | | 0.305 | −2.33 |

The intervals overlap, so 305 turns do not settle the order among the top four. The static potion table, with 8M parameters, is within the noise of EmbeddingGemma 2 and ahead of the 0.6B decoders, though MTEB ranks the decoders far above it. MTEB scores sentences and passages, while a Codenames clue is one word without context. The association counts do worst. 79 of the 305 clues are not EAT cues, and the students who answered around 1970 rarely named a given board word.

**Prompting.** A **template** turns a word into the text the model reads. "bare" is the word itself, "word" is `the word 'X'`, "related" is `a word related to X`, and "query" is an instruction in the format of the Qwen3 and harrier model cards (`Instruct: Given a clue in a word-association game, retrieve words that people associate with it` then `Query: X`), used on the clue side only.

| Model | bare | word | related | query |
|---|---|---|---|---|
| potion-base-8M | 0.580 | 0.449 | 0.502 | 0.115 |
| wordllama | 0.531 | 0.220 | 0.436 | 0.207 |
| granite-small-r2 | 0.420 | 0.456 | 0.466 | 0.151 |
| bge-small-en-v1.5 | 0.528 | 0.534 | 0.502 | 0.131 |
| Qwen3-Embedding-0.6B | 0.475 | | 0.462 | 0.495 |
| harrier-oss-v1-270m | 0.452 | | 0.423 | 0.469 |
| harrier-oss-v1-0.6b | 0.430 | | 0.452 | 0.498 |
| EmbeddingGemma 2 | 0.607 | | 0.567 | 0.377 |

The instruction helps the three models whose cards use this instruction format (Qwen3 and the two harrier models) by 2 to 7 points, within or near the noise. It ruins the four smaller models, whose vectors become mostly the instruction, and costs EmbeddingGemma 2 23 points. Its card uses a different prompt format, which was not tried. The runner computed only bare, related and query for the Hugging Face models.

**Senses.** Codenames words are ambiguous, and a clue usually works through one sense. `senses.py` gives each board word one extra vector per WordNet noun sense (the text "word: gloss" for its first three senses) and scores a clue by the best of them, the idea behind Qdrant's miniCOIL applied with WordNet senses. On the human guesses it moves the two encoders from 0.420 to 0.439 (granite) and from 0.528 to 0.561 (bge), and the two static tables down by about a point. On the EAT task (below) it lowers all four, by 1 to 8 points. Neither change is outside the 95% intervals.

## 2. The cost of a different guesser

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

Standard errors over boards are 0.02 to 0.05. Every giver loses 0.5 to 0.9 words per clue when the guesser is another model, and more when it is the association guesser. The newest models lose about as much as the static tables. EmbeddingGemma 2 loses least: its clues keep 73% of their value with other models and 53% with the association guesser. The last two rows are **ensemble givers**: they maximize the average expected value over seven models' Luce guessers and never see the eighth. Against that held-out model they find 1.36 (bge) and 1.38 (EmbeddingGemma 2) words. Both beat the mean of the seven single givers against the same model (1.21 and 1.22). Against EmbeddingGemma 2 the ensemble also beats every single giver; against bge, potion's clues do better (1.41). The ensembles ask for fewer words per clue: 2.1 on average, against 2.7 to 3.2 for single givers.

`results/crossplay.json` also has the expected value of each turn. It subtracts 1 for an opponent word and, for the assassin, the number of team words left. With that cost a giver's clues are worth 1.3 to 1.8 to its own model and 0.1 to 0.9 to other models. To the association guesser they are worth −0.16 to −0.07, except EmbeddingGemma 2's (0.16) and the two ensembles' (0.25).

## 3. Whole games, compared with Kim et al. (2019)

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

Every model wins all 50 games with its own guesser, as Kim et al. found. This is guaranteed by the giver's rule whenever some clue has a target. The guesser then picks the same words in the same order as the giver ranked them. Mixed pairs win 10 to 42 of 50, and the association guesser wins at most 11. A single threshold suits the models unevenly: their similarities have different scales (fitted τ ranges from 0.02 to 0.07). With t = 0.4 the transformer givers name many targets per clue and win alone in 2.4 to 4.3 turns on average. The static givers name fewer, take 8.8 and 9.0 turns, and lose less often with other guessers.

## 4. Training on human data

The numbers in this section use the published split. The section "Human guesses split by game" repeats the main ones on a split by game.

`train.py` learns a linear map A on the clue side of a frozen model, scoring a clue c against a word w as (A e_c) · e_w. This is the setup of White et al. (2024), who train a linear layer on GloVe from Codenames Duet games. The loss is InfoNCE: a softmax at temperature 0.05 over the word the person chose and 24 negatives. For Cultural Codes items the negatives are the other words on that board; for EAT items they are random board words that are not responses to the cue. Adam, 3 epochs, seed 0.

| Model, trained on | Human guesses (95% interval) | EAT held-out cues (95% interval) |
|---|---|---|
| potion, untrained | 0.580 (0.524–0.634) | 0.710 (0.681–0.738) |
| potion, Cultural Codes train | 0.662 (0.607–0.713) | 0.706 |
| potion, EAT train cues | 0.557 | 0.777 (0.750–0.802) |
| potion, both | 0.652 | 0.734 |
| Qwen3 0.6B (query), untrained | 0.495 (0.439–0.551) | 0.621 (0.590–0.651) |
| Qwen3 0.6B, Cultural Codes train | 0.567 (0.511–0.622) | 0.642 |
| Qwen3 0.6B, EAT train cues | 0.167 (0.130–0.213) | 0.773 |
| Qwen3 0.6B, both | 0.502 | 0.663 |

The EAT column is the share of 984 held-out cues whose most frequent response is the most similar of 25 words (the response and 24 board words that are not responses). Training on one source helps on the same kind of data, and the two sources pull apart. Training the 1024 × 1024 map for Qwen3 on EAT alone took its human-guess score from 0.495 to 0.167. Training took 1 to 60 seconds here.

A better model of human guesses did not make a better giver. Used as givers, with clues limited to held-out EAT cues on 100 boards, the trained maps found fewer words per clue with every guesser, including the association guesser: 0.82 before and 0.77 after for potion, 0.72 and 0.65 for Qwen3 (`results/train-*.json`). Why is open; one guess is that the map makes the giver's own guesser more confident without making its clues clearer.

`lora.py` fine-tunes Qwen3-Embedding-0.6B or harrier-oss-v1-0.6b with LoRA on the same items. It needs a GPU or a Mac with Apple silicon. It was not run, so no result is reported for it.

## 5. A giver trained against a population

`rl.py` trains a policy over (clue, count) pairs: 30 shortlisted clues times counts 1 to 4. It picks an action with probability proportional to exp(θ · φ). The seven features φ use only the giver's own model: expected team words found, chance of hitting the assassin and an opponent word, the count, rarity, and the margin between the n-th closest team word and the closest other word. The reward is the exact expected value of the turn under a guesser drawn from a population. Because the reward of every action is known, the policy gradient is summed over all actions rather than sampled; boards and guessers are still sampled. The baseline is the policy's mean reward. Two givers: potion with bge held out, and Qwen3 with EmbeddingGemma 2 held out. The population is the other six models. Each run uses 300 training boards, 200 test boards and 400 Adam steps of 16 boards, with seed 0.

| Giver, policy | own model | held-out model | EAT counts (held out) |
|---|---|---|---|
| potion, start: maximize expected words found | 1.24 ± 0.05 | 0.61 ± 0.06 (bge) | −0.06 ± 0.05 |
| potion, self-play | **1.60 ± 0.02** | 0.75 ± 0.04 | −0.05 ± 0.05 |
| potion, population | 1.03 ± 0.01 | **0.92 ± 0.02** | **0.35 ± 0.04** |
| potion, population, Deshpande et al. reward | 1.22 ± 0.05 | 0.61 ± 0.06 | −0.07 ± 0.05 |
| Qwen3 0.6B, start | 1.22 ± 0.05 | 0.26 ± 0.05 (Gemma 2) | −0.20 ± 0.05 |
| Qwen3 0.6B, self-play | **1.58 ± 0.03** | 0.43 ± 0.05 | −0.09 ± 0.05 |
| Qwen3 0.6B, population | 1.03 ± 0.01 | **0.73 ± 0.03** | **0.24 ± 0.04** |
| Qwen3 0.6B, population, Deshpande et al. reward | 1.28 ± 0.04 | 0.24 ± 0.05 | −0.17 ± 0.04 |

Values are expected turn values on the test boards, ± one standard error. Self-play learns to trust its own similarities. Its gap between own and held-out model is 0.85 for potion and 1.15 for Qwen3. Population play learns a cautious policy that weights the margin feature most and prefers lower counts. Its gap is 0.11 and 0.30. It does better than self-play with both held-out guessers, and only the population policies have a positive value against the association guesser. This mirrors other-play (Hu et al. 2020): a policy trained with one partner exploits that partner's quirks. Here the quirks are the giver's own similarity scale and errors, not invented conventions, since the policy can only reweight seven features.

The last row uses the clue reward of Deshpande et al. (2026, their eq. 4): the fraction of team words the guesser names minus the fraction of other words it names, with the judge naming its n most likely words. That reward ignores the assassin and lets a turn continue past a wrong pick, so a policy trained on it gives more words per clue and scores worse on the game's own terms. This is one seed per giver; a run takes 3 to 5 minutes here, and other choices are flags.

## Human guesses split by game

This section follows a plan written before its runs, `experiments/plans/C1.md`, and `experiments/METHOD.md`. Results marked exploratory came from runs made before the plan.

**How the published files split.** Each row of the guessing task lists the 25-word board as it stood, the clue, the pick and both players' profiles. In Duet both players give clues on the same board, so a game is a starting board of 25 words together with the unordered pair of player profiles. Every later turn of the game lies on a subset of that board (`split.py`). This assigns each of the 7,703 rows to exactly one of 794 games, the number the paper gives. One board was played by two pairs. The files list each side of a game (one player giving clues) as one run of rows. All but one of the 1,587 sides sit in a single file, but the two sides of a game often do not: 165 games have rows in more than one file. 320 of the 373 test rows and 423 of the 455 validation rows come from games that also have training rows. A test turn's board, and its two players in swapped roles, are then in training.

The profiles also identify players. 52 of the 54 test players, by profile string, play in training games, and so do all 71 validation players. White et al. (2024) write of these data that "the players are different between the train and validation/test data". The published files do not have that property; their split may have come from elsewhere. A player's survey answers (demographics, moral foundations, Big Five) stay fixed while the age they type changes from game to game, so the survey part identifies a player more reliably than the whole profile string. The rows hold 154 distinct sets of survey answers, close to the paper's 153 players, and 90 profiles without answers. By survey answers, 51 of the 53 test players play in training games.

1,488 rows (19%) record several picks for one clue. `norms.cultural_guesses` keeps only turns with one pick on the board, as before: 6,088 turns in 770 games. White et al. train on all picks.

**The split.** A game goes to train, val or test by `zlib.crc32` of its key modulo 10 (0 to 7 train, 8 val, 9 test), with no seed. That gives 4,820 train, 717 val and 551 test turns in 623, 79 and 68 games. The plan expected 700 to 800 test turns; it counted rows before the filter. `splits.json` lists the game ids (hashes) of each split, and `python -m codenames.split --check` rebuilds the split from the download and compares. Every interval below resamples the test games 10,000 times (seed 0) and takes the 2.5th and 97.5th percentiles (`stats.py`). Differences are paired by turn. The tests check the resampling against an exact enumeration of every resample of three and four games.

**Confirmatory results** (`results/c1.json`, `calibrate-game.json`, `train-*-game.json`; one run, at the commit recorded in `c1.json`). Temperatures were fitted on the new train turns and templates chosen on the new val turns.

| Model | Template | Split by game: same word (95% interval) | Log-likelihood | Published split (from section 1) |
|---|---|---|---|---|
| EmbeddingGemma 2 | bare / bare | 0.548 (0.501–0.595) | −1.73 | 0.607 |
| potion-base-8M | bare / bare | 0.544 (0.498–0.593) | −1.72 | 0.580 |
| bge-small-en-v1.5 | word / word | 0.490 (0.439–0.542) | −1.87 | 0.534 |
| harrier-oss-v1-0.6b | query / bare | 0.465 (0.418–0.512) | −2.01 | 0.498 |
| wordllama | bare / bare | 0.465 (0.420–0.512) | −2.06 | 0.531 |
| harrier-oss-v1-270m | bare / bare | 0.439 (0.392–0.488) | −2.09 | 0.452 |
| Qwen3-Embedding-0.6B | query / bare | 0.434 (0.391–0.475) | −2.14 | 0.495 |
| granite-small-r2 | word / word | 0.394 (0.352–0.437) | −2.20 | 0.456 |
| EAT counts | ε 0.1 | 0.309 (0.273–0.345) | | 0.305 |
| potion + map | bare / bare | 0.632 (0.589–0.675) | | 0.662 (one seed) |
| Qwen3 0.6B + map | query / bare | 0.477 (0.435–0.520) | | 0.567 (one seed) |

- **Primary metric.** The map raises potion's match by 0.088 (0.063 to 0.114), paired by turn; on the published split it was 0.082. The interval excludes 0, which is the plan's criterion for the gain to hold. The five seeds give 0.623 to 0.641.
- **Qwen3.** The map raises it by 0.044 (0.008 to 0.077), against 0.072 on the published split. Its seeds vary more: 0.445 to 0.528.
- **Seeing other turns of the same game does not help.** Each test game's turns were split in half by a hash of the turn. Training on the training games gives 0.653 on one half; swapping as many random training turns for the other half of the test games gives 0.647. The difference is −0.006 (−0.019 to 0.005), over 283 turns in 62 games.
- **The order of the untrained models barely moves.** Kendall's τ between the new order and the published one is 0.84 (0.57 to 0.93). Every model scores 1 to 7 points lower on the new test turns than on the published ones. The exploratory scores on all 6,088 turns lie between the two for every model, which suggests that the new test games are harder than average and the published test turns easier.
- **Ties at the top.** Post hoc, not in the plan, from the same scored turns: EmbeddingGemma 2 minus potion is 0.004 (−0.038 to 0.046). Both lead bge by 5 to 6 points (intervals 0.016 to 0.094), and the potion map leads EmbeddingGemma 2 by 0.084 (0.045 to 0.123).
- **Templates and temperatures.** Re-chosen on the new train and val turns, they are the same as on the published split for every model.
- **Player overlap.** All 68 test games have a player who also plays in training games, by profile string and by survey answers. The plan's metric on games without such a player therefore has no turns. The exploratory split by player below answers that question instead.

**Exploratory results** (`results/heldout.json`, `heldout.py`, run before the plan on splits of their own; `split.py` describes them). Same metric and resampling.

- *Leakage on the published split.* The potion map trained without the 258 training turns from games that have a test turn scores 0.649 on the published test, against 0.662 with them. The difference is 0.013 (−0.014 to 0.042). For Qwen3 the same removal costs 0.069 (0.020 to 0.119), which suggests that the 1024 × 1024 map fits individual games more closely than potion's 256 × 256 map.
- *Held-out players.* Guessers were held out until test and val each held a tenth of the turns, and every turn with a held-out player was kept out of training, as White et al. describe (705 test turns from 21 guessers; 3,738 training turns). The potion map then gains 0.058 (0.037 to 0.080; 0.024 to 0.080 when guessers are resampled instead of games). On a split by game with its training games subsampled to about as many turns (3,743), the gain is 0.080 (0.054 to 0.105). Part of the gain may depend on having seen the guesser, but each arm has one seed, and the two intervals overlap. The Qwen3 map gains nothing: 0.001 (−0.031 to 0.034). EmbeddingGemma 2 and potion score 0.548 and 0.567.
- *All turns.* No step of the untrained models' top-1 uses the human turns, so all 6,088 turns serve as test turns for them. With bare templates EmbeddingGemma 2 scores 0.562 (0.546 to 0.577) and potion 0.547 (0.533 to 0.562), a difference of 0.014 (0.002 to 0.027). The instruction template helps harrier-oss-v1-0.6b by 0.059 (0.048 to 0.070), harrier-oss-v1-270m by 0.029 and Qwen3 by 0.012 (0.001 to 0.024), and costs the other five models 22 to 44 points. This run scored every turn, including the plan's later test turns, for the untrained models only.

**What survives.** The map's gain on potion survives, at the same size. EmbeddingGemma 2 and potion lead the untrained models on every split, but whether EmbeddingGemma 2 is ahead is not settled. On the 551 test turns the two are tied; over all turns it is ahead by 1.4 points; on held-out players potion is ahead by 2.0 (−1.4 to 5.4). The Qwen3 map's gain shrinks and, on held-out players, vanishes. The earlier statement that test boards in training make the trained scores optimistic is not borne out for potion.

**Reproducibility.** The ONNX models granite and bge are int8 with dynamic quantization, so a word's vector depends on the other words in its batch of 256. Vectors computed here from an empty cache reproduce every best-template number in section 1. Five other cells of its prompting table come out 0.3 to 2.3 points different (granite with the query template: 0.128, against 0.151).

## The mathematics

**Exact turn odds.** For a Luce guesser with weights w over the unrevealed words, write p(S) for the chance that the first |S| picks are the team words in the set S, in some order. Then p of the empty set is 1, and

    p(S) = Σ over x in S of  p(S − x) · w_x / (W − W(S − x)),

where W is the total weight and W(A) the weight of the words in A. The chance that the first j picks are all team words is the sum of p(S) over sets of j team words. The chance that pick j is the first wrong one, of kind k, is the sum over sets S of j − 1 team words of p(S) · W_k / (W − W(S)). For one pick this is the team's share of the weight. `luce.py` computes these for many clues at once. The tests check them against a sum over every ordered sequence of picks, against 100,000 simulated turns, and against the greedy guesser at a low temperature. They also check that the outcomes of a turn sum to 1.

**The best count.** With cost c_k for a wrong pick of kind k, the expected value of a turn that allows n picks is

    R(n) = Σ_{j ≤ n} [ P(first j picks are team) − Σ_k c_k · P(pick j is the first wrong one, of kind k) ],

and the giver's best count is the n that maximizes it. In this study the costs are 1 for an opponent word, 0 for a neutral word, and, for the assassin, the number of team words still to find, since picking it loses the game.

**Proposition.** If the assassin's cost rises, the largest best count does not rise. *Proof.* Let c be the assassin's cost and B(n) the chance of hitting the assassin within n picks. Then R(n) = A(n) − c·B(n), where A(n) does not depend on c. B does not decrease in n, since allowing more picks only adds chances. For n′ > n, the difference R(n′) − R(n) = A(n′) − A(n) − c(B(n′) − B(n)) therefore does not increase in c. If n′ > n beats n at a cost c, it also beats n at every lower cost. So the largest maximizer at a higher cost cannot exceed the largest maximizer at a lower one. (This is Topkis's monotonicity theorem in its simplest form.) ∎

**An observation, not proved.** In every case checked, R(n) for n = 1 to 4 has a single peak: it rises and then falls, never falls and then rises again. The checks cover 267,237 shortlisted clues on 50 boards under eight models, and 100,000 random weightings of four team and three other words at nine assassin costs. If it always holds, a giver can stop raising n at the first decrease. An attempted proof bounds the next increment by an average of the current conditional gains, weighted toward the positive ones, so it does not close. Whether a counterexample exists is open.

## Language models

`llm.py` defines `llm_giver(complete, fallback)` and `llm_guesser(complete)`, where `complete` is any function from a prompt to reply text. They have the same signatures as the embedding players, so `game.play` can pair any of them. No language model was reachable from this machine, and no result for one is reported. The experiment to run:

- **Givers.** An LLM giver and the embedding givers above, each on the same 200 boards (seed 0) and the same clue rule.
- **Guessers.** Each embedding guesser; the association guesser; an LLM guesser from a different model family than the giver; and people, on a few dozen boards, as Cultural Codes did.
- **Measures.** Found words per clue against each guesser, the gap between own and other guessers, and the share of replies that needed the fallback. Report the model, its version, temperature and seed for each call.
- **Question.** Does an LLM giver lose less to a stranger than an embedding giver does? Two benchmarks report LLMs playing both roles (Stephenson et al. 2024; Hakimov et al. 2025). Neither measures this gap against non-LLM partners on fixed boards.

## Related work

- Summerville et al. (2019) set up the Codenames AI competition. Kim et al. (2019) compared word2vec, GloVe and WordNet codemasters and guessers in it. They found twin pairs always win and mixed pairs often lose, and the folder reproduces that pattern with current models.
- Jaramillo et al. (2020) added transformer and TF-IDF bots and found the same drop for mixed pairs.
- Koyyalagunta et al. (2021) generate clues from BabelNet and from several embeddings, with a frequency term to avoid obscure clues, and evaluate them with people.
- Archibald and Brosnahan (2024) build a giver that adapts to its teammate. Bills, Archibald and Blaylock (AAAI 2025) infer the partner's embedding by Bayesian updating over a set of models; the ensemble giver above is the version without updating.
- Shaikh et al. (2023) collected Cultural Codes. White, Pandey and Pan (2024) trained a linear map on GloVe from it and added pragmatic reasoning.
- Stephenson, Sidji and Ronval (2024) use Codenames as a benchmark for LLMs, as do Hakimov, Pfennigschmidt and Schlangen (2025). Deshpande et al. (2026) train Qwen3 models on Codenames with reinforcement learning from verifiable rewards, with a frozen guesser as judge.
- Hu et al. (2020), "Other-Play", show that agents trained only with themselves form conventions that strangers do not share.

## What was computed where

| Result | Where |
|---|---|
| Vectors for potion, wordllama, granite, bge; every table above; training and policy gradient | this machine, 2 CPUs |
| Vectors for Qwen3-Embedding-0.6B, harrier-oss-v1-270m, harrier-oss-v1-0.6b, EmbeddingGemma 2 | GitHub-hosted runner (CPU), workflow on `compute/codenames-embed` |
| LoRA fine-tuning (`lora.py`), LLM players (`llm.py`) | not run |

## Open questions

- Does the single-peak property of R(n) hold for every weighting?
- Would the order of models change with SWOW, which is larger and recent?
- Does the potion map's gain hold when every pick of a multi-pick turn is used, as White et al. do?
- Why does a trained guesser model make a worse giver?
- Does a LoRA-tuned Qwen3 or harrier keep its human-guess gains without the EAT collapse?
- How large is the gap for an LLM giver?

## Usage

The package's packages are the `codenames` dependency group (numpy, onnxruntime, tokenizers, wordfreq). Times are from one run on this machine, while other jobs shared its two CPUs.

```bash
uv run python -m codenames.fetch all              # models and data, about 100 MB (20 s)
uv run python -m codenames.make_words             # rewrites words.txt
uv run python -m codenames.embed --model granite --template bare query   # vectors (1 to 9 min each)
uv run python -m codenames.study calibrate        # section 1 (1 min)
uv run python -m codenames.study crossplay        # section 2 (25 min)
uv run python -m codenames.study games            # section 3 (60 min)
uv run python -m codenames.study peaks            # the single-peak check (3 min)
uv run python -m codenames.study report           # prints the tables
uv run python -m codenames.senses                 # senses (2 min)
uv run python -m codenames.train --model potion   # section 4 (2 min)
uv run python -m codenames.split --check          # rebuild games and players, compare with splits.json
uv run python -m codenames.c1                     # split by game, the plan's run (2 min)
uv run python -m codenames.heldout                # split by game and player, exploratory (5 min)
uv run python -m codenames.rl --giver potion --held-out bge --out codenames/results/rl-potion.json
uv run --with sentence-transformers python -m codenames.embed --model Qwen/Qwen3-Embedding-0.6B
```

## Files

| File | What |
|---|---|
| `game.py` | rules, boards, the game loop |
| `luce.py` | exact turn odds, the best count |
| `players.py` | spaces, givers (threshold, expected, ensemble), guessers |
| `words.py`, `make_words.py`, `words.txt` | board words, clue words, the clue rule |
| `embed.py` | the models, templates and the vector cache |
| `fetch.py` | downloads, pinned by SHA-256 |
| `norms.py` | EAT, SWOW, Cultural Codes |
| `evaluate.py`, `senses.py` | section 1 |
| `study.py` | sections 1 to 3 and the tables |
| `train.py`, `lora.py` | section 4 |
| `split.py`, `splits.json` | games and players rebuilt from Cultural Codes, and the splits by game and by player |
| `stats.py` | the bootstrap over games, Kendall's τ |
| `c1.py`, `heldout.py` | human guesses split by game: the plan's run, and the exploratory one |
| `rl.py` | section 5 |
| `llm.py` | language-model players |
| `results/*.json` | every number above, with seeds and settings |
| `compute/codenames-embed.yml` | a copy of the workflow that computed the Hugging Face vectors |
