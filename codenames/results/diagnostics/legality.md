# Clues that are forms of a board word

Exploratory. Written by `uv run python -m codenames.diagnose legality` from committed results; nothing earlier is changed.

## Findings

- Every experiment here filtered clues with `words.legal`, which forbids a board word and a clue that contains one or is contained in one. That lets through forms of a board word (residency with RESIDENCE, teaching with TEACHER) and shared compound parts (railway with RAILROAD), which the rulebook forbids.
- Under a stricter check (`codenames/legality.py`), C10's population pick P is illegal (a form of a board word or a shared compound part) on 28.0% (23.0% to 33.3%) of test boards, against 7.0% (4.3% to 10.0%) for the first choice F and 28.3% (23.3% to 33.7%) for the oracle. Picks chosen by their value to embedding guessers break the rule most often.
- On the 188 of 300 C10 boards where neither F nor any of P's seven picks is flagged illegal, P − F under the embedding strangers is 0.232 (0.180 to 0.284), against 0.239 (0.198 to 0.282) on all boards; under the two language models it is 0.019 (-0.117 to 0.157), against -0.005 (-0.123 to 0.120). The embedding-only advantage (strangers minus language models) is 0.214 (0.072 to 0.355) on clean boards and 0.244 (0.122 to 0.360) on all. Leaving those boards out does not shrink it beyond its interval.
- With every arm choosing among strictly legal clues, P − F under the strangers is 0.215 (0.173 to 0.258) on all 300 boards (C10: 0.239 (0.198 to 0.282)); under the human-fitted guesser 0.139 (0.058 to 0.223) (C10: 0.140 (0.054 to 0.229)); under EAT 0.133 (0.042 to 0.224) (C10: 0.003 (-0.081 to 0.087)). These are exact values; the language models were not asked about the new picks. So with legal clues the population's lead holds under the embedding and human-fitted guessers and appears under EAT: the advantage of the strangers over EAT falls from 0.236 (0.150 to 0.323) to 0.083 (-0.010 to 0.178). Part of EAT's change is coverage: in C10, 39.3% of P's illegal clues are EAT cues, against 63.5% of its legal ones, and a clue that is not a cue gets EAT's uniform guess.
- The C11 pilot's first materials had 14 illegal and 4 borderline items of 80, on 16 of 40 boards. They were made again with the strict filter before anyone played; the new 80 items have 0 flags (experiments/plans/C11.md, amendment of 2026-11-19).

## The rule

From the rulebook (Czech Games Edition, English rules of July 2015, page 8, https://filemanager.czechgames.com/storage/files/codenames-2015/rules/codenames-rules-en.pdf, read 2026-11-19):

> You can't say any form of a visible word on the table. Until BREAK is covered up by a card, you can't say break, broken, breakage, or breakdown.
>
> You can't say part of a compound word on the table. Until HORSESHOE is covered up, you can't say horse, shoe, unhorsed, or snowshoe.

Under "Don't be too strict" it adds that island and even land are valid clues for ENGLAND, that sparrow is fine with ROW on the table, and that "if the opposing spymaster allows it, the clue is valid." Theatre and theater "are different forms of the same word."

## The check

`legality.relation(clue, word)` names the strongest relation between a clue and a board word. Its levels and resources are in the module's docstring: WordNet 3.0 morphology and derivational pointers (WordNet 3.0 license), the Snowball English stemmer (BSD-3-Clause), compound splits into two WordNet words, and spelling rules. A clue is strictly legal when `words.legal` allows it and no board word has an illegal or borderline relation to it. The check keeps `words.legal`, so it is stricter than the rulebook in two of its examples: land for ENGLAND and sparrow for ROW are substrings and stay forbidden.

### The rulebook's examples

| Clue | Board word | Rulebook | Relation | Strict check | Agrees |
|---|---|---|---|---|---|
| break | BREAK | illegal | same | illegal | yes |
| broken | BREAK | illegal | inflection | illegal | yes |
| breakage | BREAK | illegal | substring | illegal | yes |
| breakdown | BREAK | illegal | substring | illegal | yes |
| horse | HORSESHOE | illegal | substring | illegal | yes |
| shoe | HORSESHOE | illegal | substring | illegal | yes |
| unhorsed | HORSESHOE | illegal | compound | illegal | yes |
| snowshoe | HORSESHOE | illegal | compound | illegal | yes |
| theatre | THEATER | illegal | variant | illegal | yes |
| island | ENGLAND | legal | - | legal | yes |
| land | ENGLAND | legal | substring | illegal | no |
| sparrow | ROW | legal | substring | illegal | no |
| tree | NUT | legal | - | legal | yes |
| cashew | NUT | legal | - | legal | yes |
| strudel | APPLE | legal | - | legal | yes |

### Hand labels

Every distinct flagged pair among the picks of C3, C4, C6, C7, C10 and the pilot (150 pairs, relations other than same word and substring) was labeled by hand as illegal (a form of the board word, or a compound part, as the rule's examples read), borderline (a shared root or combining form, not a form of the same English word), or legal. The labels are in `legality-labels.tsv`, with a note on the hard cases. One reading choice matters: railway with RAILROAD, and restroom with BATHROOM, are labeled illegal, since they share a compound part as snowshoe shares one with HORSESHOE. A group that reads the rule more loosely would call them borderline.

| Level the check gives | Pairs | Labeled illegal | Borderline | Legal | Precision (illegal) | Precision (illegal or borderline) |
|---|---:|---:|---:|---:|---:|---:|
| illegal | 96 | 86 | 9 | 1 | 89.6% | 99.0% |
| borderline | 54 | 8 | 22 | 24 | 14.8% | 55.6% |

By relation:

| Relation | Illegal | Borderline | Legal |
|---|---:|---:|---:|
| inflection | 16 | 0 | 0 |
| derivation | 57 | 8 | 0 |
| variant | 2 | 0 | 0 |
| compound | 11 | 1 | 1 |
| stem | 2 | 4 | 0 |
| affix | 1 | 1 | 1 |
| compound3 | 2 | 2 | 0 |
| prefix | 3 | 15 | 23 |

Misses. The 152 unflagged pairs that share three or more first letters were labeled too. Among them the check missed 2 illegal pairs (young/youth, younger/youth) and 10 borderline ones (basic/base, basics/base, commodore/commander, empress/empire, feminine/female, hospitality/host, midst/middle, telegram/telephone, terrestrial/territory, wolverine/wolf). Pairs that share fewer letters were not labeled, so recall is measured on that stratum only. Pertainyms such as lunar for MOON and episcopal for BISHOP are not flagged; the rule concerns forms of a word, not words of the same meaning.

## How often each arm's clue is flagged

Share of picks at each level, with 95% intervals over boards (each board counts once, with the share of its runs or tasks). All of these clues pass `words.legal`.

| Experiment | Arm | Boards | Picks | Illegal | Borderline | Note |
|---|---|---:|---:|---|---:|---|
| C3 | Qwen3-Embedding-0.6B/population | 200 | 7000 | 9.7% (6.0% to 13.8%) | 0.5% |  |
| C3 | Qwen3-Embedding-0.6B/population/rlvr | 200 | 7000 | 1.4% (0.1% to 3.2%) | 0.9% |  |
| C3 | Qwen3-Embedding-0.6B/self | 200 | 7000 | 1.2% (0.0% to 2.9%) | 0.5% |  |
| C3 | Qwen3-Embedding-0.6B/start | 200 | 7000 | 1.5% (0.0% to 3.5%) | 0.5% |  |
| C3 | potion/population | 200 | 7000 | 16.6% (11.6% to 21.9%) | 0.0% | potion giver trained against the population |
| C3 | potion/population/rlvr | 200 | 7000 | 3.8% (1.5% to 6.6%) | 0.0% |  |
| C3 | potion/self | 200 | 7000 | 3.5% (1.0% to 6.0%) | 0.0% | potion giver trained against itself |
| C3 | potion/start | 200 | 7000 | 4.0% (1.5% to 7.0%) | 0.0% |  |
| C4 | deepseek-v3.2 | 200 | 200 | 19.0% (13.5% to 24.5%) | 3.0% |  |
| C4 | embeddinggemma-2 | 200 | 200 | 3.5% (1.0% to 6.0%) | 3.5% | the embedding giver |
| C4 | gpt-oss-120b | 200 | 200 | 0.5% (0.0% to 1.5%) | 0.5% |  |
| C4 | llama-3.3-70b | 200 | 200 | 18.5% (13.5% to 24.0%) | 2.5% |  |
| C4 | qwen3-32b | 200 | 200 | 4.5% (2.0% to 7.5%) | 1.5% |  |
| C6 | deepseek-v3.2 | 250 | 250 | 11.2% (7.6% to 15.2%) | 3.2% |  |
| C6 | first | 250 | 250 | 10.8% (7.2% to 14.8%) | 3.6% | first choice of EmbeddingGemma 2 |
| C6 | gemma-4-31b | 250 | 250 | 2.4% (0.8% to 4.4%) | 1.2% |  |
| C6 | gpt-oss-120b | 250 | 250 | 12.0% (8.0% to 16.0%) | 2.8% |  |
| C6 | llama-3.3-70b | 250 | 250 | 22.4% (17.6% to 27.6%) | 2.8% |  |
| C6 | oracle_h | 250 | 250 | 4.8% (2.4% to 7.6%) | 1.2% | best of the 120 for the human-fitted guesser |
| C6 | oracle_s | 250 | 250 | 30.8% (25.2% to 36.4%) | 1.2% | best of the 120 for the strangers |
| C6 | qwen3-32b | 250 | 250 | 22.4% (17.6% to 27.6%) | 3.6% |  |
| C6 | qwen3.8-27b | 250 | 250 | 12.0% (8.0% to 16.0%) | 1.2% |  |
| C7 | arm1 | 200 | 600 | 26.0% (20.0% to 32.0%) | 4.5% |  |
| C7 | arm2 | 200 | 1000 | 30.9% (24.6% to 37.3%) | 7.4% |  |
| C10 | C | 300 | 300 | 7.3% (4.7% to 10.3%) | 2.3% |  |
| C10 | F | 300 | 300 | 7.0% (4.3% to 10.0%) | 3.0% | first choice of EmbeddingGemma 2 |
| C10 | L | 300 | 300 | 8.7% (5.7% to 12.0%) | 2.3% |  |
| C10 | LP | 300 | 2100 | 23.4% (19.1% to 28.1%) | 0.7% |  |
| C10 | LP (potion task) | 300 | 300 | 24.3% (19.7% to 29.3%) | 0.7% |  |
| C10 | P | 300 | 2100 | 27.0% (22.6% to 31.8%) | 1.1% | population pick, all seven tasks |
| C10 | P (potion task) | 300 | 300 | 28.0% (23.0% to 33.3%) | 2.7% | population pick sent to the other guessers |
| C10 | S | 300 | 300 | 10.0% (6.7% to 13.7%) | 3.7% |  |
| C10 | oracle | 300 | 300 | 28.3% (23.3% to 33.7%) | 0.7% | best kept action for the strangers' mean |
| C11-pilot-before | F | 40 | 40 | 15.0% (5.0% to 27.5%) | 2.5% |  |
| C11-pilot-before | P | 40 | 40 | 20.0% (7.5% to 32.5%) | 7.5% |  |
| C11-pilot | F | 40 | 40 | 0.0% (0.0% to 0.0%) | 0.0% |  |
| C11-pilot | P | 40 | 40 | 0.0% (0.0% to 0.0%) | 0.0% |  |

C3's clues are not committed; `codenames.diagnose_c3` regenerated them from the committed weights. 56000 of 56000 picks have the held-out value C3 recorded (within 0.0001).

## Exploratory re-analysis of C10: P − F by kind of guesser

C10's 300 test boards (seed 9004). A board is flagged when F's clue or one of P's seven task picks is illegal: F on 21 boards, P (any task) on 108, P (potion task) on 84. 'Clean' boards have no such flag (188); 'strict clean' boards have no borderline flag either (178). The subsets were chosen after C10's results were known, and the boards are not a random half, so the comparison is descriptive. Words per turn, 95% bootstrap intervals over boards.

| Scored by | All boards | Clean | Strict clean | Flagged − clean |
|---|---|---|---|---|
| seven embedding strangers (C10's mean over tasks) | 0.239 (0.198 to 0.282) | 0.232 (0.180 to 0.284) | 0.231 (0.179 to 0.283) | 0.019 (-0.066 to 0.107) |
| human-fitted guesser (five potion maps) | 0.140 (0.054 to 0.229) | 0.130 (0.019 to 0.243) | 0.161 (0.052 to 0.270) | 0.029 (-0.147 to 0.213) |
| EAT association guesser | 0.003 (-0.081 to 0.087) | 0.025 (-0.085 to 0.132) | 0.031 (-0.086 to 0.142) | -0.059 (-0.225 to 0.108) |
| Llama 3.3 70B and DeepSeek-V3.2, one played-out reply each | -0.005 (-0.123 to 0.120) | 0.019 (-0.117 to 0.157) | 0.014 (-0.115 to 0.143) | -0.063 (-0.328 to 0.221) |

The embedding-only advantage, per board, strangers' P − F minus another family's:

| Difference | All boards | Clean | Strict clean | Flagged − clean |
|---|---|---|---|---|
| strangers − human-fitted | 0.099 (0.025 to 0.174) | 0.103 (0.006 to 0.201) | 0.070 (-0.023 to 0.166) | -0.010 (-0.167 to 0.140) |
| strangers − EAT | 0.236 (0.150 to 0.323) | 0.207 (0.095 to 0.325) | 0.200 (0.083 to 0.320) | 0.077 (-0.094 to 0.247) |
| strangers − language models | 0.244 (0.122 to 0.360) | 0.214 (0.072 to 0.355) | 0.217 (0.084 to 0.353) | 0.082 (-0.189 to 0.342) |

The language-model column is one sampled reply per clue, so its per-board values are noisy (C11's plan puts the standard deviation of a board's P − F near 1.1).

## C10 re-picked with the strict filter

`diagnose refilter` runs C10's giver again on its 300 test boards, with every arm choosing among strictly legal clues, and scores the new picks exactly. F is unchanged on 257 boards and P (first task) on 201.

| Scored by | P − F, strict | P − F, C10 | Change | Oracle − F, strict |
|---|---|---|---|---|
| seven embedding strangers (C10's mean over tasks) | 0.215 (0.173 to 0.258) | 0.239 (0.198 to 0.282) | -0.024 (-0.049 to 0.001) | 0.345 (0.309 to 0.383) |
| human-fitted guesser (five potion maps) | 0.139 (0.058 to 0.223) | 0.140 (0.054 to 0.229) | -0.001 (-0.053 to 0.050) | 0.244 (0.168 to 0.324) |
| EAT association guesser | 0.133 (0.042 to 0.224) | 0.003 (-0.081 to 0.087) | 0.130 (0.077 to 0.186) | 0.152 (0.065 to 0.241) |

| Difference | Strict | C10 | Change |
|---|---|---|---|
| strangers − human-fitted | 0.076 (0.006 to 0.147) | 0.099 (0.025 to 0.174) | -0.023 (-0.068 to 0.023) |
| strangers − EAT | 0.083 (-0.010 to 0.178) | 0.236 (0.150 to 0.323) | -0.154 (-0.215 to -0.094) |

EAT knows fewer of the illegal clues. In C10 (picks before the filter):

| Picks | Number | EAT cues | Mean EAT value |
|---|---:|---:|---:|
| F, illegal | 21 | 23.8% | -0.350 |
| F, legal | 270 | 64.8% | 0.039 |
| P, illegal | 84 | 39.3% | -0.267 |
| P, legal | 208 | 63.5% | 0.118 |

## The C11 pilot

First materials (80 items): 14 illegal, 4 borderline.

| Board | Arm | Clue | Level | Related board words |
|---|---|---|---|---|
| 11010:5 | P | residency | illegal | residence (derivation) |
| 11010:7 | P | telecom | borderline | telephone (prefix) |
| 11010:10 | F | countess | illegal | counter (derivation), county (affix) |
| 11010:13 | P | residency | illegal | residence (derivation) |
| 11010:14 | F | regiment | borderline | region (prefix) |
| 11010:20 | P | hiding | illegal | hide (inflection) |
| 11010:22 | F | librarian | illegal | library (derivation) |
| 11010:23 | P | winning | illegal | winner (derivation) |
| 11010:25 | P | railway | illegal | railroad (compound) |
| 11010:28 | F | owned | illegal | owner (inflection) |
| 11010:28 | P | webcam | borderline | camera (compound3) |
| 11010:30 | P | railroad | illegal | railway (compound) |
| 11010:31 | F | governance | illegal | governor (derivation) |
| 11010:31 | P | governing | illegal | governor (derivation) |
| 11010:37 | F | actress | illegal | actor (derivation) |
| 11010:45 | P | restroom | illegal | bathroom (compound) |
| 11010:48 | P | telecom | borderline | telephone (prefix) |
| 11010:49 | F | teaching | illegal | teacher (derivation) |

Materials now committed (80 items): 0 illegal, 0 borderline.

## Rerunning

```sh
uv run python -m codenames.legality residency residence railway railroad
uv run python -m codenames.diagnose_c3 --out codenames/results/diagnostics/c3-picks.txt  # vectors, ~10 min
uv run python -m codenames.diagnose refilter                                          # vectors, ~15 min
uv run python -m codenames.diagnose legality
```
