# Mastermind

A code is four pegs in six colors, 1,296 codes in all. Each strategy below is played against every code, so the numbers are exact:

| Strategy | Total guesses | Average | Worst case |
|---|---:|---:|---:|
| most parts | 5,668 | 4.373 | 6 |
| expected size | 5,696 | 4.395 | 6 |
| entropy | 5,722 | 4.415 | 6 |
| minimax | 5,801 | 4.476 | 5 |
| first consistent | 7,471 | 5.765 | 9 |

The totals match Table 7 of B. Kooi, "Yet another Mastermind strategy", *ICGA Journal* 28(1), 2005. Two strategies are on the Pareto frontier of average and worst case: most parts has the lowest average, and minimax is the only one that never needs a sixth guess.

- `mastermind.py` has the codemaker's reply, `score`, which everything else uses.
- `strategies.py` compiles each strategy into a guess tree and plays it against every code.
- `fallout3.py` applies the same strategies to the Fallout 3 terminal hacking game, described below.

```bash
uv run python -m mastermind.strategies          # about half a minute
uv run python -m mastermind.fallout3 --best doctrines batteries scorpions generated occasions   # prints SCORPIONS
uv run python -m mastermind.fallout3 --best doctrines=3 batteries scorpions generated occasions # after DOCTRINES got 3
uv run python -m mastermind.fallout3 doctrines batteries scorpions generated occasions   # asks for each likeness
uv run python -m mastermind.fallout3 --study       # the tables below, about 4 minutes
uv run python -m mastermind.fallout3 --published   # the Fallout 4 screens, about half a minute
```

## Fallout 3 terminals

A locked terminal shows a screen of words of one length, and one of them is the password. Each guess must be a word on the screen. The terminal replies with the likeness: how many letters are right and in the right place. Four wrong guesses lock the terminal ([Fallout wiki, Hacking](https://fallout.fandom.com/wiki/Hacking)). This is Mastermind with a short list of possible codes, so the four strategies above apply unchanged. A guess may be any word on the screen, including one already ruled out, since it can still split the remaining candidates.

What the sources say about the screens:

- Word length depends on the terminal's level: 4–5 letters for Very Easy, 6–8 for Easy, 9–10 for Average, 11–12 for Hard and 13–15 for Very Hard (Fallout wiki).
- The number of words depends on the player's Science skill. The [GECK wiki's formula](https://geckwiki.com/index.php?title=Hacking_Word_Count) gives 20 words when Science equals the lock's level (0, 25, 50 or 75 from Very Easy to Hard) and 5 at Science 100. The Fallout wiki lists 15 as the Science needed for Very Easy, where the formula gives 17.75. At Very Hard it gives 12.5 words whatever the Science. The wiki does not say how 12.5 is rounded.
- None of the sources consulted say how the words are chosen, or whether duds are chosen to share letters with the password. The three Fallout 4 screens published by [Math Goes Pop (2016)](https://www.mathgoespop.com/2016/04/fallout-4-math-part-2.html) are somewhat more alike than random ENABLE screens of the same size. Their mean likeness is 0.49, 0.85 and 0.79, against 0.44, 0.54 and 0.64 for ENABLE, and 32%, 2.5% and 18% of 1,000 ENABLE screens are as alike. Three screens from a later game settle nothing.

Word lists from the game are not available. The study therefore draws each screen from the ENABLE list in `boggle/enable1.txt`, with a seeded generator, at a length chosen uniformly from the level's lengths. The password is uniform over the screen. On each screen, every strategy is played against every password, so each screen's numbers are exact. Across screens they are simulated: 1,000 screens per row, seed 3, with a normal 95% interval. "Best possible" is the most passwords that any strategy can crack within four attempts, found by exhaustive search on each screen. "Random" guesses a candidate uniformly at random; its chance is computed exactly, not simulated. "Common" is the hand rule of guessing the candidate with the most letters in place, summed over the other candidates (the "closest word" of Math Goes Pop).

Percent of passwords cracked within four attempts, at the fewest and most words:

| Level | Words | Most parts | Expected size | Entropy | Minimax | Common | Random | Best possible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Very Easy | 5 | 99.7 ± 0.1 | 99.7 ± 0.1 | 99.7 ± 0.1 | 99.7 ± 0.1 | 99.6 ± 0.2 | 99.3 ± 0.1 | 99.7 ± 0.1 |
| Very Easy | 20 | 79.5 ± 0.6 | 88.1 ± 0.5 | 86.9 ± 0.5 | 87.6 ± 0.5 | 85.1 ± 0.5 | 67.8 ± 0.4 | 92.5 ± 0.4 |
| Easy | 20 | 93.7 ± 0.4 | 98.1 ± 0.2 | 97.7 ± 0.2 | 98.4 ± 0.2 | 93.5 ± 0.3 | 82.9 ± 0.3 | 99.78 ± 0.07 |
| Average | 20 | 98.5 ± 0.2 | 99.95 | 99.85 | 99.95 | 96.8 ± 0.2 | 91.2 ± 0.2 | 100 |
| Hard | 20 | 99.64 | 99.99 | 99.98 | 100.00 | 97.9 ± 0.2 | 94.7 ± 0.1 | 100 |
| Very Hard | 13 | 100 | 100 | 100 | 100 | 99.6 ± 0.1 | 99.13 | 100 |

Intervals under 0.1 points are left out. "100" means every password on every sampled screen was cracked; "100.00" is a rounded figure. `--study` also prints the rows for 10 and 15 words, and for 12 words at Very Hard. Their figures fall between the ones shown.

- Short words make the game hard. On 20-word Very Easy screens, no strategy can always win on 72% of the screens. The hardest screen found was PUSH WHIT YUPS FRAG MEWL BOTS AMPS JAUK GONE KILN. Forty of its 45 pairs have likeness 0, and the best play cracks 7 of its 10 passwords. Even five words can defeat every strategy, as on 1.3% of five-word Very Easy screens. One way is for every pair to have the same likeness, so that each guess rules out only itself.
- From Average up, minimax and expected size come within 0.05 points of the best possible. They miss some password on at most 1.1% of the sampled screens, and the best possible misses none. On 20-word Very Easy screens they fall 5.0 and 4.4 points short.
- Minimax is ahead of expected size by up to 0.6 points on most rows, and behind by 0.6 on 20-word Very Easy screens. Most parts, the best of the four for Mastermind, is the worst of the four here. It cracks 8.6 points fewer passwords than expected size on 20-word Very Easy screens.
- With no attempt limit, minimax and expected size never needed more than 7 guesses on any sampled screen. Most parts needed up to 9.
- Random guessing among the candidates does far worse on large screens. It cracks 67.8% on 20-word Very Easy screens, against 88.1% for expected size.

The first guess matters less than the later ones. Take a player who uses a rule for the first guess and then makes the best guess every time. On the screens with the most words at each level (5,000 screens), the minimax rule keeps 99.41% ± 0.05 of the best possible chance on average, and its first guess is an optimal one on 90.5% of screens. Expected size keeps 99.31% and common keeps 99.11%. A uniformly random first word keeps 96.91%.

Checks:

- On the three Fallout 4 screens, expected size and common reproduce Math Goes Pop's table exactly:
  - expected size ("most expected eliminations") wins 100% on each screen, with 2.75, 2.58 and 2.58 guesses on average;
  - common ("closest word") wins 91.67%, 100% and 100%, with 2.92, 2.58 and 2.58 guesses on average. Their simulated random guessing, 1.2 million games per screen, won 92.79%, 96.21% and 97.93%. The exact values here are 92.78%, 96.19% and 97.96%.
- They also scored random twelve-word screens of 5, 6 and 7 letters, from a word list they do not name. On 1,000 ENABLE screens of each length, the results here are within about half a point of theirs:
  - expected size cracks 98.0%, 99.3% and 99.8%, where they report 97.50%, 98.94% and 99.57%;
  - common cracks 96.2%, 97.4% and 98.1%, where they report 96.04%, 97.42% and 98.18%;
  - random guessing cracks 86.4%, 90.7% and 93.7%, where they report 85.92%, 90.64% and 93.67%.
- The tests compare the exhaustive search with a brute force without memoization or shortcuts. They do this on all 126 five-word screens drawn from nine words, at one to four attempts, and on 40 sampled screens. On the same 126 screens, the exact chance of random guessing matches an enumeration of every sequence of choices.

Not modeled: brackets in the screen's filler text, which remove a dud or restore the attempts, and the Computer Whiz perk, which allows a second try at a locked terminal. ENABLE has many words that a game would not use, such as JAUK and WHIT. Screens of common words may be easier or harder.
