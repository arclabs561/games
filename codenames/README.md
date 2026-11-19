# Codenames

In Codenames, a spymaster (the giver) names one clue word and a number, and teammates (the guessers) try to pick the giver's words from a board of 25. This folder builds givers and guessers from word embeddings and asks three questions:

1. Which embedding models predict the words people pick?
2. How much of a giver's expected score survives when the guesser uses a different model, or is a person?
3. Does training a giver against several guessers, instead of against its own model, close that gap?

Short answers, with how each is known:

- EmbeddingGemma 2 and the small static table potion-base-8M pick the same word as a person most often: 54.8% and 54.4% of 551 turns from 68 test games, with the Cultural Codes turns split by game (95% intervals 50.1% to 59.5% and 49.8% to 59.3%; their difference, 0.4 points, has the interval −3.8 to 4.6). The other six models pick it in 39.4% to 49.0%. Instruction prefixes help the instruction-trained models a little and ruin the others. Computed exactly on the test turns, with intervals from resampling games.
- Training a linear map on potion's clue vectors raises its match from 54.4% to 63.2%. The map is trained on the human guesses of the training games only. The gain, 8.8 points (6.3 to 11.4), is as large as on the published split (8.2), so it was not an artifact of test games having turns in training. Same method; the map is averaged over five training seeds.
- Fine-tuning Qwen3-Embedding-0.6B with LoRA on the same training games raises its match from 43.4% to 62.6%, level with the potion map: LoRA minus the map is −0.7 points (−4.1 to 2.8). Same method, three training seeds, on an A10G GPU.
- A giver expects to find 1.8 to 2.1 of its words per clue when the guesser shares its vectors. A guesser built on another model finds 57% to 73% of that, on average over the other seven models. A guesser that follows association counts from people finds 37% to 53%. Exact expected values for each of 200 random boards, one clue each, for eight models.
- Hosted language models, asked with a fixed prompt, predict human guesses better than untrained embeddings and give worse clues. gpt-oss-120b picks the person's word in 65.0% of the 551 test turns (60.8% to 69.3%), 10.2 points (7.1 to 13.5) above EmbeddingGemma 2 and level with the trained guessers above. As givers, all four models tested give clues worth 0.29 to 0.89 words less per turn to seven embedding guessers they never met than the EmbeddingGemma 2 giver's. Paired intervals from resampling games or boards; one reply per prompt.
- Choosing from a list of 120 legal actions (30 clues times counts 1 to 4) proposed by the EmbeddingGemma 2 giver does not rescue the language models. The model selected on validation boards, Qwen3.8-27B, is 0.059 words per turn below the giver's own first choice (−0.116 to −0.005), and none of six models is above it. The best action in each list is 0.44 above the first choice, so the list leaves room. Exact expected values on 200 test boards with seven unseen embedding guessers; one reply per prompt.
- A giver trained by policy gradient against six other models scores more with a seventh model it never trained with than a giver trained against its own model. The gain is 0.31 words per turn (0.26 to 0.36), averaged over 14 held-out choices and five training seeds. Population training beat self-play in all 70 runs, at a cost of 0.50 (0.46 to 0.53) with the giver's own model. The gain is smallest, 0.09 (0.001 to 0.18), when the Qwen3 giver meets harrier-oss-v1-0.6b. Against the association counts, which stand in for people, the gain is 0.39 (0.31 to 0.46). Exact rewards on 200 test boards, with a stratified bootstrap over boards and seeds, in a run planned in advance (plan C3).

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

`lora.py` fine-tunes Qwen3-Embedding-0.6B with LoRA on human guesses. It runs on a GPU, and its results are in the section "LoRA fine-tuning on the split by game".

## 5. A giver trained against a population

`rl.py` trains a policy over (clue, count) pairs: 30 shortlisted clues times counts 1 to 4. It picks an action with probability proportional to exp(θ · φ). The seven features φ use only the giver's own model: expected team words found, chance of hitting the assassin and an opponent word, the count, rarity, and the margin between the n-th closest team word and the closest other word. The reward is the exact expected value of the turn under a guesser drawn from a population. Because the reward of every action is known, the policy gradient is summed over all actions rather than sampled; boards and guessers are still sampled. The baseline is the policy's mean reward. Each run uses 300 training boards, 200 test boards and 400 Adam steps of 16 boards. One model is held out, and the population is the other six (all eight but the giver and the held-out one). The EAT association guesser is held out too.

**Exploratory run.** The table below is the first run, made while the code was written: two givers, potion with bge held out and Qwen3 with EmbeddingGemma 2 held out, with seed 0. The planned run after it repeats this with five seeds and every held-out choice.

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

The last row uses the clue reward of Deshpande et al. (2026, their eq. 4): the fraction of team words the guesser names minus the fraction of other words it names, with the judge naming its n most likely words. That reward ignores the assassin and lets a turn continue past a wrong pick, so a policy trained on it gives more words per clue and scores worse on the game's own terms. This is one seed per giver; a run takes 1 to 5 minutes here, depending on load, and other choices are flags.

### Across seeds and held-out guessers (plan C3)

This run follows a plan written before it, `experiments/plans/C3.md`. It asks whether the exploratory result depends on the seed or on which model was held out. There are two givers, potion and Qwen3, and each of the other seven models is held out in turn, which makes 14 tasks. Each task is trained with seeds 1 to 5. All 70 runs are scored on the same 200 test boards, dealt from seed 3003, which no earlier run used. The primary metric is the held-out guesser's expected turn value, population policy minus self-play policy. It is paired by board and seed, averaged over boards and seeds within a task, then over the 14 tasks. Intervals are 95% stratified bootstrap intervals with 10,000 resamples (seed 0). Each resample draws the 200 boards once for all tasks and five seeds within each task, as Agarwal et al. (2021) do. The 14 tasks themselves are not resampled, so the interval is about these 14 choices.

| Held-out guesser | potion giver: self-play | population | difference | Qwen3 giver: self-play | population | difference |
|---|---|---|---|---|---|---|
| potion | | | | 0.46 | 0.78 | 0.32 (0.21 to 0.46) |
| wordllama | 0.46 | 0.90 | 0.44 (0.34 to 0.55) | 0.21 | 0.59 | 0.38 (0.30 to 0.47) |
| granite | 0.18 | 0.63 | 0.45 (0.36 to 0.55) | 0.17 | 0.42 | 0.25 (0.16 to 0.34) |
| bge | 0.70 | 0.94 | 0.25 (0.15 to 0.35) | 0.43 | 0.74 | 0.31 (0.21 to 0.42) |
| Qwen3 0.6B | 0.59 | 0.86 | 0.27 (0.15 to 0.40) | | | |
| harrier 270m | 0.49 | 0.84 | 0.35 (0.23 to 0.49) | 0.54 | 0.75 | 0.21 (0.11 to 0.31) |
| harrier 0.6b | 0.68 | 0.97 | 0.29 (0.18 to 0.42) | 0.87 | 0.96 | 0.09 (0.001 to 0.18) |
| EmbeddingGemma 2 | 0.55 | 0.92 | 0.36 (0.27 to 0.45) | 0.45 | 0.81 | 0.36 (0.27 to 0.45) |

Values are mean expected turn values with the held-out guesser, over five seeds and 200 boards. The summary over all 14 tasks:

| Measure | Value (95% interval) |
|---|---|
| **Primary: held-out guesser, population minus self-play** | **0.31 (0.26 to 0.36)** |
| Interquartile mean over the 70 runs of the same difference | 0.32 (0.26 to 0.38) |
| Share of the 70 runs in which population beats self-play | 70 of 70 (resampled: 0.96 to 1) |
| EAT guesser, population minus self-play | 0.39 (0.31 to 0.46) |
| Own model, population minus self-play (the cost of population training) | −0.50 (−0.53 to −0.46) |
| Held-out guesser, Deshpande et al. reward minus population | −0.45 (−0.50 to −0.39) |
| EAT guesser, Deshpande et al. reward minus population | −0.45 (−0.52 to −0.39) |
| Held-out value: start, self-play, population, Deshpande et al. reward | 0.35, 0.48, 0.79, 0.34 |
| EAT value: start, self-play, population, Deshpande et al. reward | −0.12, −0.05, 0.33, −0.12 |
| Own-model value: start, self-play, population, Deshpande et al. reward | 1.34, 1.64, 1.14, 1.32 |
| Gap, own minus held-out: self-play, population | 1.15 (1.10 to 1.21), 0.34 (0.30 to 0.39) |

The interval of the primary metric lies above 0. By the plan's rule, then, training against a population helps with held-out guessers of this kind. The gain is positive for every task, and each task's own interval lies above 0. The smallest is for the Qwen3 giver with harrier-oss-v1-0.6b held out, whose interval starts at 0.001; there self-play already scores 0.87. Seeds matter little. The difference varies between seeds of one task with a standard deviation of 0.016 on average, against 0.096 between tasks.

The learned weights are the same story in every run. In all 70 population policies the margin feature has the largest weight and the count a negative one. In all 70 self-play policies the expected number of words found has the largest weight. Population training with the Deshpande et al. reward scores no better than the untrained start with held-out guessers (0.34 against 0.35).

**What changed from the exploratory run.** The direction holds and the size is similar. The two exploratory tasks were run again with five seeds and new test boards. For potion with bge held out, the gain is now 0.25 (0.15 to 0.35); in the exploratory run it was 0.92 − 0.75 = 0.17. For Qwen3 with EmbeddingGemma 2 held out, it is 0.36 (0.27 to 0.45), against 0.30 before. The exploratory gap between own and held-out value, 0.85 shrinking to 0.11 for potion, is now 0.91 shrinking to 0.14 on that task, and 1.15 shrinking to 0.34 on average over all 14 tasks. The cost in self-play, 0.57 in the exploratory run, is 0.50 on average. The plan names the held-out score as the headline rather than the gap, since a gap also shrinks when self-play gets worse.

The 70 runs ran on GitHub-hosted runners (CPU) through the workflow on the branch `compute/codenames-c3`, at commit `0dfb4c7`, one job per run, 70 to 130 seconds each and 1 hour 58 minutes in all; none failed. `compute/codenames-c3.yml` here is a copy of that workflow. The per-run outputs are on that branch under `runs/C3/44547011858-1/`. `results/c3.json` holds the summary, every θ, and the per-board values for the held-out and EAT guessers. One run was repeated on this machine. It chose the same actions on every board, and its values differ from the runner's by at most 1.4e-5, from floating-point rounding.

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

## LoRA fine-tuning on the split by game

This section follows `experiments/plans/C2.md` and its amendment, written before the runs on this split. It uses C1's split by game: 4,820 train, 717 val and 551 test turns.

**What was trained.** `lora.py` adds LoRA adapters (rank 16, α 32, dropout 0.05) to the query, key, value and output projections of Qwen3-Embedding-0.6B. It trains them for one epoch at batch 64 with sentence-transformers' MultipleNegativesRankingLoss. Each item is the clue in the "query" template, the word that the person picked, and 7 other words from the same board. The batch's other items serve as further negatives. Each job encodes every word before and after training, in the same precision, and scores both on human guesses (`c2.py`). τ is fitted on train and the clue template chosen on val, as in section 1.

**Learning rate.** Chosen once, with training seed 1, by log-likelihood on the val turns: −1.78, −1.62 and −1.52 for 3e-5, 1e-4 and 3e-4, against −2.11 untrained (`results/c2-lr.json`). 3e-4 was used for every test run. It is the largest value in the plan's grid, and larger rates were not tried.

**Confirmatory results** (`results/c2.json`, from the summaries in `results/c2-runs/`). Same turns, metric and resampling as the previous section.

| Arm | Same word as the person (95% interval) | Per seed | Log-likelihood | EAT held-out cues |
|---|---|---|---|---|
| 1. potion + map (from C1) | 0.632 (0.589–0.675) | 0.623 to 0.641 (5 seeds) | | |
| 2. Qwen3 0.6B, untrained | 0.434 (0.390–0.477) | | −2.15 | 0.619 |
| 3. Qwen3 0.6B + map (from C1) | 0.477 (0.435–0.520) | 0.445 to 0.528 (5 seeds) | | |
| 4. Qwen3 0.6B + LoRA | 0.626 (0.585–0.666) | 0.623, 0.632, 0.623 | −1.38 | 0.716 to 0.724 |
| 5. Qwen3 0.6B + LoRA, with EAT | 0.630 (0.583–0.675) | 1 seed | −1.41 | 0.936 |

The untrained vectors made in the job give the same pick as the runner's vectors used in the previous section on 549 of the 551 turns. Every job chose the query template.

- **Primary metric.** LoRA minus the potion map is −0.007 (−0.041 to 0.028), paired by turn. The interval includes 0, so by the plan's rule LoRA did not beat the linear map at this data size. LoRA brings Qwen3 level with the potion map, not past it.
- **LoRA against the map on the same encoder.** 0.148 (0.121 to 0.176). Against the untrained encoder, 0.192 (0.152 to 0.232). Changing the encoder gains about four times what rotating its output gains.
- **Adding EAT.** 0.004 (−0.018 to 0.025) on human guesses, one seed. The linear map for Qwen3 lost 6.5 points when EAT was added on the published split (section 4); LoRA does not lose. On the EAT held-out cues it rises from about 0.72 to 0.94. That task's distractors are all board words, while 795 of its 984 answers are not. The EAT training rows also use board words as negatives, so part of the rise may come from learning to rank the words of the board low. That was not tested.
- **Seeds.** The three LoRA seeds span 0.9 points, and the five map seeds for Qwen3 span 8.3.
- **As a giver** (secondary). On 200 fresh boards (seed 2002), with clues limited to held-out EAT cues, the tuned and the untrained Qwen3 each chose one clue per board. The tuned giver's clues were averaged over its three seeds. The table gives the expected team words found per clue, with the difference paired by board (95% interval from 10,000 resamples of boards).

| Guesser | Untrained giver | LoRA giver | Difference |
|---|---|---|---|
| potion | 1.165 | 1.196 | 0.031 (−0.039 to 0.101) |
| wordllama | 0.857 | 0.869 | 0.012 (−0.044 to 0.068) |
| granite | 0.905 | 0.893 | −0.012 (−0.066 to 0.042) |
| bge | 1.075 | 1.117 | 0.043 (−0.015 to 0.100) |
| harrier-oss-v1-270m | 1.167 | 1.132 | −0.035 (−0.109 to 0.039) |
| harrier-oss-v1-0.6b | 1.355 | 1.360 | 0.006 (−0.061 to 0.072) |
| EmbeddingGemma 2 | 1.087 | 1.105 | 0.017 (−0.037 to 0.071) |
| EAT counts | 0.766 | 0.789 | 0.023 (−0.025 to 0.070) |

Every interval includes 0. A guesser model that matches people 19 points more often did not give measurably better or worse clues to any of these guessers. The linear maps of section 4 gave worse clues, on the published split and 100 boards.

**Compute.** Each job ran on a Hugging Face `a10g-small` through `experiments/hfrun.py`, at the commit in its manifest. A job with Cultural Codes rows only took 2.7 to 3.8 minutes from creation to finish; with EAT, 6.9. About 60 to 85 seconds of each went to setup before the model was loaded: cloning, installing the `codenames-gpu` group (torch and the CUDA libraries) and the imports. Training took 42 seconds (236 with EAT). The eight GPU jobs, the pilot included, ran 29.4 minutes in all, about $0.49 at $1.00 an hour. Vectors, adapters and per-turn scores are in the private dataset `arclabs561/games-runs` under `runs/c2/`. The adapters were trained on Cultural Codes, whose license is not stated, so they are not published.

**Exploratory** (the pilot, seed 0, learning rate 1e-4, published split). Scored on the published val turns: 0.476 untrained and 0.649 tuned. It was run to time the job; its numbers are not part of the result.

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

Experiment C4 (`experiments/plans/C4.md`, with its amendment of 2026-11-19) asks two questions of four open-weight models served by Hugging Face Inference Providers:

- Llama 3.3 70B Instruct, through novita;
- Qwen3-32B with thinking off, through deepinfra;
- gpt-oss-120b at low reasoning effort, through deepinfra;
- DeepSeek-V3.2, through deepinfra.

Every request has temperature 0 and seed 0. The prompts are those of `llm.py`, unchanged. Every prompt and reply is in the private dataset `arclabs561/games-runs` under `runs/c4/`, since provider terms differ on publishing outputs; `results/c4.json` has every number below.

**Guessing.** Each model heard the clue of every human turn in C1's game split with the count 1, and its guess is the first board word of its reply. Top-1 is the share of turns where that is the person's word. The model for the primary test was chosen on the 717 val turns (79 games) before any test turn was sent: gpt-oss-120b, at 0.618 against 0.582, 0.554 and 0.547 for the others and 0.543 for EmbeddingGemma 2 (`results/c4-val.json`). The test has 551 turns in 68 games. Intervals come from 10,000 resamples of games (`stats.py`), and differences are paired by turn.

| Guesser | Top-1 on test (95% interval) | Minus EmbeddingGemma 2 |
|---|---|---|
| gpt-oss-120b | 0.650 (0.608–0.693) | 0.102 (0.071–0.135) |
| Llama 3.3 70B | 0.606 (0.559–0.654) | 0.058 (0.024–0.093) |
| Qwen3-32B | 0.555 (0.513–0.600) | 0.007 (−0.028–0.045) |
| DeepSeek-V3.2 | 0.532 (0.484–0.580) | −0.016 (−0.050–0.017) |
| EmbeddingGemma 2, untrained | 0.548 (0.501–0.595) | |
| potion with the linear map (C1, mean of 5 seeds) | 0.632 (0.589–0.675) | |
| Qwen3-Embedding-0.6B with LoRA (C2, mean of 3 seeds) | 0.626 (0.585–0.666) | |

- **H-guess holds.** gpt-oss-120b picks the person's word 10.2 points more often than EmbeddingGemma 2, and no resample of games put the difference at or below 0 (p < 0.0002). Llama 3.3 is also ahead; Qwen3-32B and DeepSeek-V3.2 are within the noise of the embedding model.
- **Against the trained guessers** (secondary). gpt-oss-120b minus the potion map is 0.017 (−0.020 to 0.056), and minus the LoRA encoder 0.024 (−0.012 to 0.061), paired by turn with their per-turn hits from C1 and C2. An untrained hosted model is level with the guessers trained on 4,820 human turns.
- **Parsing.** Out of 5,072 replies, 3 named no board word. Their guess is then the first board word, as the plan says.
- **Contamination.** Cultural Codes has been public since 2023, so the models may have seen its turns. The prompt names no dataset. This could inflate the language models' scores and is not measured.

**Giving.** Each model named one clue for each of 200 fresh boards (seed 4004), with the prompt of `llm.py`. The baseline is the EmbeddingGemma 2 expected-value giver with the whole clue list. Each clue is scored by the exact expected turn value under the seven calibrated embedding guessers other than EmbeddingGemma 2, which are strangers to every giver here, and a board's score is their mean. A reply with no legal clue gets the fallback: the EmbeddingGemma 2 giver's best clue for one word. The model for the primary test was chosen on 50 other boards (seed 4005): DeepSeek-V3.2, at 0.594 against the baseline's 0.602. Intervals come from 10,000 resamples of boards, and differences are paired by board.

| Giver | Turn value (95% interval) | Minus the embedding giver | Words found | No legal clue | Mean count |
|---|---|---|---|---|---|
| EmbeddingGemma 2, expected value | 0.706 (0.636–0.774) | | 1.35 | | 2.75 |
| DeepSeek-V3.2 | 0.402 (0.263–0.515) | −0.304 (−0.460 to −0.166) | 0.93 | 52% | 2.71 |
| Llama 3.3 70B | 0.421 (0.319–0.518) | −0.285 (−0.403 to −0.169) | 0.95 | 47% | 2.61 |
| gpt-oss-120b | 0.198 (0.102–0.287) | −0.508 (−0.616 to −0.401) | 1.04 | 2% | 3.73 |
| Qwen3-32B | −0.185 (−0.333 to −0.049) | −0.891 (−1.058 to −0.734) | 0.72 | 14% | 3.14 |

- **H-give fails.** The chosen model's clues are worth 0.30 words less per turn to the strangers than the embedding giver's, and no resample of boards put the difference at or above 0. With the Holm correction over the two primary hypotheses, both differences are significant, in opposite directions. By the plan's rule C5 is not run.
- **The fallback carried the selection.** DeepSeek-V3.2 and Llama 3.3 named a board word, or a word containing or contained in one, on about half the boards, so half their scores are the embedding giver's one-word clues. On the boards where their clue was legal, they trail the embedding giver by 0.90 (DeepSeek, 97 boards) and 0.68 (Llama, 106 boards). Valuing a failed reply at 0 instead gives −0.75 and −0.70. gpt-oss-120b, which almost always gave a legal clue, trails by 0.52 on its own clues.
- **High counts.** Among legal clues, 21 to 50 per model asked for more than 4 words (scored as 4, the most any guesser here picks). The language models also find fewer words per clue (0.72 to 1.04 against 1.35), so the gap is not only the cost of wrong picks.
- **The strangers are embedding models.** The baseline giver plans with an embedding model's similarities, and every stranger is an embedding model, so it shares their kind of error. This favors it, as the plan's threats section says. The association guesser (EAT counts) is a stranger of another kind; its value for every giver is between −0.28 and −0.06, with intervals that overlap except for Qwen3-32B's (lowest).
- **Seed-0 boards** (secondary, for comparison with section 2). The embedding giver scores 0.656 and the language models 0.045 to 0.446, the same order. Section 2 limited clues to EAT cues, so its numbers differ.

**Language-model guessers** (secondary). On the 200 test boards, Llama 3.3, DeepSeek-V3.2 and gpt-oss-120b each guessed on the clues of the other language models and of the embedding giver, with the giver's count, one reply per clue. Each turn is played out from the reply's order, with the costs above. A giver's fallback clues are included. Qwen3-32B as a guesser did not finish in time and is left out. The table gives means over the guessers that heard each giver, with intervals over boards.

| Giver | Guessers | Turn value (95% interval) | Words found | Minus the embedding giver, same guessers |
|---|---|---|---|---|
| EmbeddingGemma 2 | Llama, DeepSeek, gpt-oss | 1.63 (1.51–1.75) | 1.92 | |
| gpt-oss-120b | Llama, DeepSeek | 1.04 (0.82–1.25) | 1.61 | −0.61 (−0.85 to −0.37) |
| Llama 3.3 70B | DeepSeek, gpt-oss | 0.66 (0.46–0.85) | 1.13 | −0.99 (−1.22 to −0.77) |
| DeepSeek-V3.2 | Llama, gpt-oss | 0.64 (0.43–0.82) | 1.10 | −0.97 (−1.20 to −0.74) |
| Qwen3-32B | Llama, DeepSeek, gpt-oss | −0.02 (−0.29 to 0.23) | 0.84 | −1.66 (−1.95 to −1.39) |

The language-model guessers also do best with the embedding giver's clues, so the result of the primary test is not only an effect of scoring with embedding guessers. These are single sampled turns, not exact expectations.

**Repeat check.** The first 20 prompts of seed 4005 were sent twice more to each model. The reply changed at least once for 2 of 20 prompts with Llama 3.3, 8 with Qwen3-32B, and 16 each with gpt-oss-120b and DeepSeek-V3.2, at temperature 0 with a fixed seed. Llama 3.3's val top-1 was 0.579 in a run that was stopped and 0.582 in the run reported. A rerun can differ.

**Cost.** Every job ran on `cpu-basic` through `experiments/hfrun.py`, at the commit in its manifest. The meter adds the tokens each provider reported times the price the router lists (no provider reported a cost). Calls cost $0.00002 to $0.00006 each. The 13 jobs spent at most $0.51 on model calls (`results/c4-runs.json`). That includes upper bounds for the first pilot, which ran before the prices were read, and for two jobs stopped by hand. About 4 hours of `cpu-basic` add about $0.04. The plan expected $1 to $3, under a cap of $3.

## Language models choosing among legal clues

Experiment C6 (`experiments/plans/C6.md`, with its amendments of 2026-11-19) removes one of C4's two causes of failure. On each board, EmbeddingGemma 2 proposes 30 legal clues (`actions.py`, wrapping `rl.actions`), and each clue may be given with a count from 1 to 4: 120 actions. A hosted model sees the prompt of `llm.py` followed by the numbered list and names one clue and a count. Every action is legal and has a vector in every space, so only the choice differs between arms. Six models chose, each pinned to one provider, at temperature 0 and seed 0: gpt-oss-120b (low reasoning effort), Llama 3.3 70B, Qwen3-32B, DeepSeek-V3.2, Qwen3.8-27B and Gemma 4 31B, the last two with thinking off.

Every action is scored by its exact expected turn value under the seven calibrated embedding guessers other than EmbeddingGemma 2 (the strangers, as in C4) and under the potion guesser with C1's linear map, the guesser in this folder that is closest to people (five training seeds, averaged; each rebuilt map reproduces C1's per-turn hits). The arms are compared on the same boards. The **first choice** is the action EmbeddingGemma 2 values most, so it is the embedding giver choosing from its own list. **Uniform** is the mean over the 120 actions. The **oracle** is the best action for the scoring guessers themselves, which no giver can know. A reply that names no listed clue with a count from 1 to 4 gets the first choice. The chooser for the primary test was selected on 50 validation boards (seed 6005) before any test board was sent: Qwen3.8-27B, with a mean stranger value of 0.667 against 0.660 for Gemma 4 and 0.470 to 0.551 for the others. The first choice scored 0.750 on those boards. The test has 200 fresh boards (seed 6004). Intervals come from 10,000 resamples of boards, and differences are paired by board (`results/c6.json`).

| Chooser | Strangers: turn value (95% interval) | Minus first choice | Human-fitted guesser: minus first choice | No action read | Same action as the first choice |
|---|---|---|---|---|---|
| first choice (EmbeddingGemma 2) | 0.699 (0.626–0.771) | | | | |
| oracle for the strangers | 1.143 (1.111–1.178) | 0.444 (0.387 to 0.504) | 0.469 (0.355 to 0.590) | | |
| oracle for the human-fitted guesser | 0.741 (0.673–0.808) | 0.042 (−0.033 to 0.116) | 0.993 (0.877 to 1.118) | | |
| uniform over the 120 actions | 0.315 (0.290–0.339) | −0.384 (−0.449 to −0.318) | −0.434 (−0.555 to −0.309) | | |
| Qwen3.8-27B (selected) | 0.640 (0.566–0.712) | −0.059 (−0.116 to −0.005) | −0.064 (−0.174 to 0.042) | 57% | 58% |
| gpt-oss-120b | 0.652 (0.574–0.725) | −0.047 (−0.119 to 0.025) | −0.027 (−0.175 to 0.126) | 20% | 23% |
| Gemma 4 31B | 0.591 (0.517–0.663) | −0.108 (−0.186 to −0.030) | −0.046 (−0.194 to 0.100) | 0.5% | 5% |
| Llama 3.3 70B | 0.527 (0.449–0.598) | −0.172 (−0.262 to −0.085) | −0.234 (−0.393 to −0.076) | 12% | 13% |
| Qwen3-32B | 0.496 (0.406–0.575) | −0.203 (−0.295 to −0.116) | −0.247 (−0.410 to −0.086) | 21% | 21% |
| DeepSeek-V3.2 | 0.427 (0.338–0.510) | −0.272 (−0.358 to −0.187) | −0.437 (−0.590 to −0.291) | 5% | 6% |

- **Neither primary hypothesis holds.** Qwen3.8-27B's picks are worth 0.059 words per turn less to the strangers than the first choice (bootstrap p = 0.029, on the wrong side of 0), and 0.064 less to the human-fitted guesser (p = 0.24). Given the same legal options, no model chooses better than the embedding score. C4's loss was therefore not only a matter of illegal clues.
- **There was room.** The best action in each list is worth 0.444 (0.387 to 0.504) more to the strangers than the first choice, and 0.993 more to the human-fitted guesser. The plan's condition for a detectable effect (a gap above 0.1) is met. The selected model closes −0.13 of the stranger gap (−0.28 to −0.01), and the best of the six, gpt-oss-120b, closes −0.11 (−0.29 to 0.05).
- **The selection was carried by the fallback.** Qwen3.8-27B still reasons in its reply on many boards, although thinking is off. At the 64-token cap, 84 of its 200 test replies were cut off before the answer, and 30 more named no listed clue. On 58% of the boards its action is the first choice. On the 86 boards where it named an action, it trails the first choice by 0.136 (−0.265 to −0.014). gpt-oss-120b returned an empty reply on 40 test boards (none of 50 on validation), with the provider reporting a normal stop; why is not known. On its 160 own choices it trails by 0.059 (−0.148 to 0.028). Llama 3.3 named a count outside 1 to 4 on 16 boards.
- **Rank.** Under EmbeddingGemma 2's own values, the median rank of each model's pick among the 120 actions is 19.5 for gpt-oss-120b and 46 to 68 for the other models that named an action on most boards. The models ask for more words than the oracle does: 2.4 to 3.4 on average, against 1.9 for the stranger oracle and 2.6 for the first choice.
- **The association guesser** (secondary, no significance claim). With EAT counts as guesser, every model's pick is level with or above the first choice: by 0.167 (0.079 to 0.262) for gpt-oss-120b and by −0.001 to 0.082 for the rest. All values with this guesser are near zero (−0.15 to 0.04 per turn). Its picks are random when the clue is not an EAT cue, as 44% of the candidates are not. gpt-oss-120b picks an EAT cue on 70% of the boards and the first choice is one on 54%, so part of its advantage may come from that.
- **The strangers are embedding models.** The proposer and the first choice use an embedding model's similarities, and every primary scorer is an embedding model or a linear map on one. This favors the first choice, as the plan's threats section says, and the result with the association guesser points the same way. No person guessed here.

**Language-model guessers** (secondary). On the 200 test boards, Llama 3.3, DeepSeek-V3.2 and gpt-oss-120b each guessed once on Qwen3.8-27B's pick and once on the first choice, with the pick's count. Each turn is played out from the reply's order, with the costs above. The selected model's fallbacks are included, so on 58% of the boards both arms gave the same clue.

| Giver | Turn value, mean of three guessers (95% interval) | Words found | Minus the first choice |
|---|---|---|---|
| first choice (EmbeddingGemma 2) | 1.49 (1.34–1.63) | 1.80 | |
| Qwen3.8-27B | 1.21 (1.03–1.39) | 1.68 | −0.28 (−0.43 to −0.13) |

The language-model guessers also do better with the embedding giver's first choice, so the primary result is not only an effect of scoring with embedding guessers. These are single sampled turns, not exact expectations.

**Repeat check.** The first 20 validation prompts were sent twice more to each model. The action read from the reply changed at least once for 1 of 20 prompts with Llama 3.3, 4 with gpt-oss-120b and Gemma 4, 6 with Qwen3-32B, 7 with Qwen3.8-27B and 13 with DeepSeek-V3.2, at temperature 0 with a fixed seed.

**Cost.** Five jobs ran on `cpu-basic` through `experiments/hfrun.py`, each at the commit in its manifest (`results/c6-runs.json`). The meter adds the tokens each provider reported times the price that the router listed. Model calls cost $0.171 in all, $0.00002 to $0.00012 per call; about 50 minutes of `cpu-basic` add about $0.01. The plan expected $0.40 to $0.80, under a cap of $1.50. The plan's prompt evolution (GEPA) was not run, for the reasons in the plan's last amendment.

## A language model trained to choose among legal clues

Experiment C7 (`experiments/plans/C7.md`, with its two amendments of 2026-11-19) trains Qwen3.5-2B with LoRA by group-relative policy optimization (GRPO, as implemented in TRL 1.18.1). The model chooses one of C6's 120 legal actions on a board: 30 clues proposed by EmbeddingGemma 2, times the counts 1 to 4. The prompt and the reading of the reply are C6's. The reward is exact: the expected turn value of the chosen action, averaged over four training guessers, and −1 for a reply that names no listed clue. The training guessers are wordllama, granite, bge and harrier-oss-v1-270m. The comparison is the seven-feature policy of section 5 (`rl.fit` at its defaults), trained against the same four guessers on the same 4,800 training boards and candidates. Both are scored on C6's 200 test boards (seed 6004) by three guessers that neither trained with: potion, Qwen3-Embedding-0.6B and harrier-oss-v1-0.6b.

**What training did.** In the pilot and in all three runs, the mean reward rose from −0.98 at the first step to about 0.75 by step 20. The share of replies that named no listed clue fell from 98% to under 1%. From about step 30 the 8 replies to a board were nearly always the same, and TRL's entropy fell from about 0.6 in the first 20 steps to below 0.05. A group of identical replies has advantage 0, so most steps had a gradient of exactly 0. The runs stopped after steps 100, 109 and 128, by the rule of the second amendment. All three trained models give the same reply on all 200 test boards: `CLUE NUMBER 1, <the first clue of the list>, 1`. They learned to copy the first listed clue with count 1. That needs nothing from the board but the order of the list. The list puts first the clue closest to some team word under EmbeddingGemma 2, which makes it a safe clue for one word. Untrained, the model named a listed clue with a number on 4 of the 200 test boards. On 184 it named none, and 12 replies ran past the 16-token cap.

| Giver, on the 200 test boards | Held-out guessers (3) | Training guessers (4) | Potion map (human-fitted) | EAT counts | Same action as the first choice | Mean count | Distinct clues |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B, GRPO (3 seeds, the same picks) | 0.837 | 0.714 | 0.814 | 0.137 | 4% | 1.00 | 83 |
| Feature policy (seeds 1 to 3) | 0.945 | 0.857 | 0.996 | 0.081 | 4.5% to 5% | 1.07 | 101 to 104 |
| EmbeddingGemma 2's first choice | 0.849 | 0.586 | 0.926 | −0.127 | | | |
| Qwen3.8-27B, C6's chosen model | 0.768 | 0.544 | 0.862 | −0.047 | | | |
| Qwen3.5-2B untrained (first choice when it names none) | 0.850 | 0.594 | 0.919 | −0.127 | | | |
| Uniform over the 120 actions | 0.409 | 0.244 | 0.492 | −0.148 | | | |
| Best action for the held-out guessers | 1.350 | 0.802 | 1.533 | −0.103 | | | |

Values are exact expected turn values (`luce.py`, the README's costs), averaged over boards. Intervals resample boards and seed pairs together, 10,000 times with seed 0 (`c7.interval`). A test checks that bootstrap against an exact enumeration (`results/c7.json`).

- **Primary metric.** Language model minus feature policy, with the held-out guessers: −0.107 (−0.162 to −0.057). The three seed pairs give −0.106, −0.108 and −0.109. The interval lies below 0, so the trained language model did not beat the feature policy; it did worse. With the human-fitted potion map the difference is −0.183 (−0.285 to −0.096). With EAT it is 0.056 (−0.013 to 0.128).
- **Seeds.** The three language-model runs chose the same action on every test board, so the interval reflects boards and the feature policy's seeds only. The feature policy's seeds 4 and 5 score 0.972 each.
- **Against the baselines** (held-out guessers, secondary). The trained model minus the first choice is −0.012 (−0.104 to 0.081); minus C6's Qwen3.8-27B, 0.069 (−0.025 to 0.164). With its own training guessers it gains 0.127 (0.033 to 0.217) on the first choice.
- **No other-play gap.** Both arms score higher with the held-out guessers than with the training ones (by 0.124 and 0.088). These held-out guessers agree more with EmbeddingGemma 2, which proposed the list, than the training ones do.
- **Why it stopped learning.** With 8 replies per board and the first listed clue as a safe answer worth about 0.75, the policy settled on that answer within about 20 steps. From then on no group had any spread of rewards, which is where GRPO's learning signal comes from. The clip-higher bound of DAPO is meant to keep exploration going, but with one gradient step per batch it is never reached (first amendment). Three changes might avoid this collapse, and none was tested: an entropy bonus, more replies per board, or several gradient steps per batch.

**Compute.** Each job ran on an `l40sx1` (L40S, $1.80 an hour) through `experiments/hfrun.py` at the commit in its manifest. A first pilot job on an `l4x1` took 38 seconds a step and was stopped after 4 steps; on the L40S a step took 13.3 seconds. In each job, cloning and installing the `codenames-gpu` group took 28 seconds, imports 12, loading the model 14 to 17 and the untrained decoding of 200 boards 42 (with the kernels' compilation). That is 96 to 103 seconds before training. Training took 1,510 to 1,910 seconds, and the trained decoding 10. Setup is thus about 6% of a job, and a Docker image with the packages installed would save at most the 40 seconds of install and imports. The five GPU jobs cost $3.73 in all and the reading jobs about $0.01 (`results/c7-jobs.json`), under the cap of $7. Waiting for an L40S took 7 to 30 minutes per job and is not counted. Adapters, every sampled reply and the full logs are in the private dataset `arclabs561/games-runs` under `runs/c7/`; `results/c7-runs/` holds the picks and curves copied from them, with the SHA-256 of each source file.

**Exploratory** (the pilot, seed 0, 100 steps, scored on C6's 50 validation boards). Same behavior: the first listed clue with count 1 on every board, 0.847 with the held-out guessers against 0.973 for the feature policy's seed 0.

## Related work

- Summerville et al. (2019) set up the Codenames AI competition. Kim et al. (2019) compared word2vec, GloVe and WordNet codemasters and guessers in it. They found twin pairs always win and mixed pairs often lose, and the folder reproduces that pattern with current models.
- Jaramillo et al. (2020) added transformer and TF-IDF bots and found the same drop for mixed pairs.
- Koyyalagunta et al. (2021) generate clues from BabelNet and from several embeddings, with a frequency term to avoid obscure clues, and evaluate them with people.
- Archibald and Brosnahan (2024) build a giver that adapts to its teammate. Bills, Archibald and Blaylock (AAAI 2025) infer the partner's embedding by Bayesian updating over a set of models; the ensemble giver above is the version without updating.
- Shaikh et al. (2023) collected Cultural Codes. White, Pandey and Pan (2024) trained a linear map on GloVe from it and added pragmatic reasoning.
- Stephenson, Sidji and Ronval (2024) use Codenames as a benchmark for LLMs, as do Hakimov, Pfennigschmidt and Schlangen (2025). Deshpande et al. (2026) train Qwen3 models on Codenames with reinforcement learning from verifiable rewards, with a frozen guesser as judge.
- Hu et al. (2020), "Other-Play", show that agents trained only with themselves form conventions that strangers do not share.
- Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (NeurIPS), recommend stratified bootstrap intervals, the interquartile mean and the probability of improvement when runs are few; plan C3 uses all three.

## What was computed where

| Result | Where |
|---|---|
| Vectors for potion, wordllama, granite, bge; every table above; training and policy gradient, except plan C3 | this machine, 2 CPUs |
| Plan C3's 70 policy-gradient runs (`c3.py`) | GitHub-hosted runners (CPU), workflow on `compute/codenames-c3`; analyzed on this machine |
| Vectors for Qwen3-Embedding-0.6B, harrier-oss-v1-270m, harrier-oss-v1-0.6b, EmbeddingGemma 2 | GitHub-hosted runner (CPU), workflow on `compute/codenames-embed` |
| LoRA fine-tuning (`lora.py`) | Hugging Face Jobs, `a10g-small`, through `experiments/hfrun.py` |
| C7: tables, the feature policy and the scoring (`c7.py`) | this machine |
| C7: GRPO of Qwen3.5-2B (`rlvr.py`) | Hugging Face Jobs, `l40sx1`, through `experiments/hfrun.py` |
| Language-model players (`llm.py`, `hosted.py`, `c4.py`) | Hugging Face Jobs, `cpu-basic`, through `experiments/hfrun.py`, calling Inference Providers; scored on this machine (`c4score.py`) |
| Language models choosing among legal clues (`rerank.py`) | Hugging Face Jobs, `cpu-basic`, calling Inference Providers; action lists and scoring on this machine (`actions.py`, `c6score.py`) |
| Vectors for the 15 clue words of the language models that had none | GitHub-hosted runner, workflow on `compute/codenames-embed`, for the four Hugging Face models; this machine for the other four |

## Open questions

- Does the single-peak property of R(n) hold for every weighting?
- Would the order of models change with SWOW, which is larger and recent?
- Does the potion map's gain hold when every pick of a multi-pick turn is used, as White et al. do?
- Why does a trained guesser model make a worse giver?
- Would LoRA beat the potion map with more human data or other settings (a larger learning rate, more epochs, harrier-oss-v1-0.6b)?
- Would language-model givers do better with a prompt that states the whole clue rule (no word contained in a board word) and the cost of the assassin, or with a retry after an illegal clue?
- Do language-model clues fare better with people than with embedding guessers?
- Would GRPO with an entropy bonus, more replies per board, or several gradient steps per batch keep exploring past the first listed clue (C7)?
- Why does gpt-oss-120b return an empty reply on a fifth of C6's test boards and none of its validation boards?
- Would the language models choose better than the embedding giver's first choice with a prompt found by GEPA, or with room to reason before the answer? C6's prompt evolution was not run.

## Usage

The package's packages are the `codenames` dependency group (huggingface-hub, numpy, onnxruntime, tokenizers, wordfreq). Times are from one run on this machine, while other jobs shared its two CPUs.

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
uv run python -m codenames.c2                     # LoRA results from results/c2-runs/ (1 min)
uv run python -m codenames.rl --giver potion --held-out bge --out codenames/results/rl-potion.json
uv run python -m codenames.c3 --slice 0 --out OUT     # plan C3: one of its 70 runs (1 to 2 min)
uv run python -m codenames.c3 --runs runs/C3/44547011858-1 --out codenames/results/c3.json   # from compute/codenames-c3
uv run python -m codenames.c7 table --steps 300 --out DIR          # C7 tables (7 min)
uv run python -m codenames.c7 feature --data DIR --seed 1 --out OUT  # C7 arm 2 (1 s)
# C7 arm 1, inside a Hugging Face Job on an L40S (group codenames-gpu); --tiny runs it on a CPU:
uv run --group codenames-gpu python -m codenames.rlvr --seed 1 --steps 300 --batch 64 --eval test --stop-flat 20 --min-steps 100 --stop-after 60 --table URL --table-sha256 SHA --out OUT
uv run python -m codenames.c7 score --data DIR --runs codenames/results/c7-runs --out codenames/results/c7.json   # 1 min
uv run --with sentence-transformers python -m codenames.embed --model Qwen/Qwen3-Embedding-0.6B
# C4, inside a Hugging Face Job (needs HF_TOKEN), then scored here:
uv run python -m codenames.c4 guess --models all --split val --cap 0.6 --out OUT
uv run python -m codenames.c4 give --models all --seed 4005 4004 0 --n 50 200 200 --cap 0.4 --out OUT
uv run python -m codenames.c4 crossguess --models all --seed 4004 --clues codenames/results/c4-clues-4004.txt --cap 0.3 --out OUT
uv run python -m codenames.c4 score --clues clues.txt --guesses guesses.txt --crossguess crossguess.txt --out codenames/results/c4.json   # 15 min
# C6: the action lists here, the choices in a Hugging Face Job, the scores here:
uv run python -m codenames.actions --seed 6999 6005 6004 --n 5 50 200 --out codenames/results/c6-actions.txt   # 15 s
uv run python -m codenames.rerank choose --models all --seed 6005 --n 50 --repeat 20 --cap 0.2 --out OUT
uv run python -m codenames.rerank choose --models all --seed 6004 --n 200 --cap 0.5 --out OUT
uv run python -m codenames.rerank score --out codenames/results/c6.json   # 1 min, also writes c6-boards.json
uv run python -m codenames.rerank picks --models qwen3.8-27b --seed 6004 --out codenames/results/c6-picks-6004.txt
uv run python -m codenames.rerank crossguess --models gpt-oss-120b --clues codenames/results/c6-picks-6004.txt --seed 6004 --cap 0.15 --out OUT
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
| `train.py` | section 4 |
| `lora.py`, `c2.py` | LoRA fine-tuning: the GPU job, its scoring, and the analysis |
| `split.py`, `splits.json` | games and players rebuilt from Cultural Codes, and the splits by game and by player |
| `stats.py` | the bootstrap over games, Kendall's τ |
| `c1.py`, `heldout.py` | human guesses split by game: the plan's run, and the exploratory one |
| `rl.py` | section 5 |
| `c3.py` | section 5 across seeds and held-out guessers: the runs of plan C3 and their analysis |
| `actions.py` | the 120 legal actions an embedding giver proposes on a board (C6 and C7) |
| `rerank.py`, `c6score.py` | C6: language models choosing among those actions, and the scoring |
| `rlvr.py`, `c7.py` | C7: the GRPO job, its tables, the feature policy on the same candidates, and the analysis |
| `llm.py` | language-model players: prompts and parsing |
| `hosted.py` | calls to Inference Providers, with retries, a cost meter and a log of every call |
| `c4.py`, `c4score.py` | experiment C4: the jobs that call the models, and the scoring |
| `results/*.json` | every number above, with seeds and settings |
| `compute/codenames-embed.yml` | a copy of the workflow that computed the Hugging Face vectors |
