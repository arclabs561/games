# Which models predict human guesses, on the published split

Part of the [Codenames README](../README.md). The JSON files beside this note hold its numbers, with their seeds and settings.

## Which models predict human guesses

Each model picks the board word most similar to the clue, on the 305 turns of the Cultural Codes test split. A random pick would match the person 1 time in 21 to 25. The log-likelihood is the mean log-probability of the person's word under the fitted Luce guesser; random picking scores −3.10.

| Model | Template (clue / word) | τ | Same word as the person (95% interval) | Log-likelihood |
|---|---|---|---|---|
| EmbeddingGemma 2 | bare / bare | 0.02 | 0.607 (0.551 to 0.660) | −1.63 |
| potion-base-8M | bare / bare | 0.07 | 0.580 (0.524 to 0.634) | −1.65 |
| bge-small-en-v1.5 | word / word | 0.03 | 0.534 (0.478 to 0.590) | −1.73 |
| wordllama | bare / bare | 0.07 | 0.531 (0.475 to 0.586) | −1.79 |
| harrier-oss-v1-0.6b | query / bare | 0.02 | 0.498 (0.443 to 0.554) | −1.84 |
| Qwen3-Embedding-0.6B | query / bare | 0.03 | 0.495 (0.439 to 0.551) | −1.87 |
| granite-small-r2 | word / word | 0.02 | 0.456 (0.401 to 0.512) | −2.12 |
| harrier-oss-v1-270m | bare / bare | 0.03 | 0.452 (0.398 to 0.509) | −2.03 |
| EAT counts | ε 0.1 | | 0.305 | −2.33 |

These are intervals for each model alone, and the paired differences were not computed on this split, so the order among the top four is not settled here. [c1.md](c1.md) gives paired differences on the split by game. The static potion table, with 8M parameters, is not separated from EmbeddingGemma 2 on these turns and scores above the 0.6B decoders, though MTEB ranks the decoders far above it. MTEB scores sentences and passages, while a Codenames clue is one word without context. The association counts do worst. 79 of the 305 clues are not EAT cues, and the students who answered around 1970 rarely named a given board word.

**Prompting.** A **template** turns a word into the text the model reads. "bare" is the word itself, "word" is `the word 'X'`, "related" is `a word related to X`, and "query" is an instruction in the format of the Qwen3 and harrier model cards (`Instruct: Given a clue in a word-association game, retrieve words that people associate with it` then `Query: X`), used on the clue side only.

| Model | bare | word | related | query |
|---|---|---|---|---|
| potion-base-8M | 0.580 | 0.449 | 0.502 | 0.115 |
| wordllama | 0.531 | 0.220 | 0.436 | 0.207 |
| granite-small-r2 | 0.420 | 0.456 | 0.459 | 0.128 |
| bge-small-en-v1.5 | 0.525 | 0.534 | 0.505 | 0.125 |
| Qwen3-Embedding-0.6B | 0.475 | | 0.462 | 0.495 |
| harrier-oss-v1-270m | 0.452 | | 0.423 | 0.469 |
| harrier-oss-v1-0.6b | 0.430 | | 0.452 | 0.498 |
| EmbeddingGemma 2 | 0.607 | | 0.567 | 0.377 |

The instruction helps the three models whose cards use this instruction format (Qwen3 and the two harrier models) by 2 to 7 points, within or near the noise. It ruins the four smaller models, whose vectors become mostly the instruction, and costs EmbeddingGemma 2 23 points. Its card uses a different prompt format, which was not tried. The runner computed only bare, related and query for the Hugging Face models. The granite and bge cells come from vectors computed from an empty cache on 2026-11-20 (`study calibrate`, which rewrote `calibrate.json`). The file before then differed from them in five cells, by 0.3 to 2.3 points, and in none of the best templates. These two models are int8 with dynamic quantization, so a word's vector depends on the other words in its batch ([c1.md](c1.md)).

**Senses.** Codenames words are ambiguous, and a clue usually works through one sense. `senses.py` gives each board word one extra vector per WordNet noun sense (the text "word: gloss" for its first three senses) and scores a clue by the best of them, the idea behind Qdrant's miniCOIL applied with WordNet senses. On the human guesses it moves the two encoders from 0.420 to 0.446 (granite) and from 0.525 to 0.548 (bge), and the two static tables down by about a point (`senses.json`, regenerated with the vectors above). On the EAT task ("Training on human data" below) it lowers all four, by 1 to 8 points. Neither change was tested with a paired interval.

## Training on human data

The numbers in this section use the published split. [c1.md](c1.md), "Human guesses split by game", repeats the main ones on a split by game.

`train.py` learns a linear map A on the clue side of a frozen model, scoring a clue c against a word w as (A e_c) · e_w. This is the setup of White et al. (2024), who train a linear layer on GloVe from Codenames Duet games. The loss is InfoNCE: a softmax at temperature 0.05 over the word the person chose and 24 negatives. For Cultural Codes items the negatives are the other words on that board; for EAT items they are random board words that are not responses to the cue. Adam, 3 epochs, seed 0.

| Model, trained on | Human guesses (95% interval) | EAT held-out cues (95% interval) |
|---|---|---|
| potion, untrained | 0.580 (0.524 to 0.634) | 0.710 (0.681 to 0.738) |
| potion, Cultural Codes train | 0.662 (0.607 to 0.713) | 0.706 |
| potion, EAT train cues | 0.557 | 0.777 (0.750 to 0.802) |
| potion, both | 0.652 | 0.734 |
| Qwen3 0.6B (query), untrained | 0.495 (0.439 to 0.551) | 0.621 (0.590 to 0.651) |
| Qwen3 0.6B, Cultural Codes train | 0.567 (0.511 to 0.622) | 0.642 |
| Qwen3 0.6B, EAT train cues | 0.167 (0.130 to 0.213) | 0.773 |
| Qwen3 0.6B, both | 0.502 | 0.663 |

The EAT column is the share of 984 held-out cues whose most frequent response is the most similar of 25 words (the response and 24 board words that are not responses). Training on one source helps on the same kind of data, and the two sources pull apart. Training the 1024 × 1024 map for Qwen3 on EAT alone took its human-guess score from 0.495 to 0.167. Training took 1 to 60 seconds here.

A better model of human guesses did not make a better giver. Used as givers, with clues limited to held-out EAT cues on 100 boards, the trained maps found fewer words per clue with every guesser, including the association guesser: 0.82 before and 0.77 after for potion, 0.72 and 0.65 for Qwen3 (`results/train-*.json`). Why is open; one guess is that the map makes the giver's own guesser more confident without making its clues clearer.

`lora.py` fine-tunes Qwen3-Embedding-0.6B with LoRA on human guesses. It runs on a GPU, and its results are in [c2.md](c2.md).
