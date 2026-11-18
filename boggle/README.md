# Common Words in Boggle

The notebook estimates three things about each word and keeps them apart: how often it appears on a random board, how many points it offers, and how common it is in English. It reproduces, count for count, the post [Common on the board, rare in English](https://attobop.net/posts/boggle-words/), which revisits the 2016 version of this notebook. That version missed every Q word, and it ranked words by a single TF-IDF-like score. **The notebook is [here](https://github.com/arclabs561/games/blob/main/boggle/boggle.ipynb)**.

The official rules are in Hasbro's [Boggle instructions (PDF)](https://www.hasbro.com/common/instruct/boggle.pdf). Sixteen cubes are shaken into a 4 by 4 grid. A word is a path through neighboring cubes, at least three letters long, that uses no cube twice; the `Qu` face counts as two letters on one cube.

## Results

Unless a line says it is simulated, the number is exact. The simulated ones come from 15,000 boards drawn from one generator seeded with 20161126, as in the post, and the notebook gives their 95% intervals.

| Result | How it is known |
|---|---|
| "toe" is the most common word, on 19.1% of boards (18.5% to 19.7%). | Simulated |
| A board holds 97.8 words and 140 points on average; the best of the 15,000 boards scores 969. | Simulated |
| Of the 172,723 different words of three or more letters in `enable1.txt`, 160,719 can appear on some board with these cubes and 12,004 cannot. | Exact, by matching faces to cubes; a board is built for every possible word |
| Of the 131,552 words that no sampled board showed, 12,004 are impossible and 119,548 are possible but rare. | Exact split of a simulated set |
| With the classic cubes, 169,378 of the same words are possible and 3,345 are not. | Exact |
| `perslatgsineters`, the best board for ENABLE2K (Vanderkam, 2025), scores 3,623 points from 1,044 words with `enable1.txt` and 3,625 from 1,045 with ENABLE2K, as he reports. The extra word is *tenge*. | Exact |
| A shake of the new cubes shows that board, in some orientation, with chance 8 × 1,054,080 / (16! × 6<sup>16</sup>), about 1 in 7.0 × 10<sup>18</sup>. With the classic cubes it is 8 × 221,616 / (16! × 6<sup>16</sup>), about 1 in 3.3 × 10<sup>19</sup>. | Exact, as a permanent of face counts (Ryser's formula) |
| Replacing one cube by a blank costs from 28 points (the Qu cube) to 45 points (E I O S S T) on average. | Simulated |

**Which words are possible.** A word is possible if some shake of the cubes shows it along some path. Split it into faces, with "qu" as one face; a q followed by any other letter cannot be spelled. If each face can be given a different cube that carries it, lay those cubes along the path that runs through all 16 slots row by row, turning at each end. Each slot on that path neighbors the one before, so the word reads along it. If no such choice exists, no board shows the word. So only the matching of faces to cubes matters, not where the cubes land. When there is no matching, Hall's theorem gives a reason: some set of faces is needed more often than there are cubes carrying any of them. With the new cubes, 2,313 words are too long (more than 16 faces), 2,050 need two b's (only A B B J O O has a b), and 1,940 need two f's. The new set puts its only F and only K on one cube, so no word with an f and a k appears; the classic set has each of them on two cubes.

The tests compare `possible` with a search over every choice of cubes on random small sets, check each reason against the cube counts, and check on a sample of words that the board `witness_board` builds can be rolled and that the solver finds the word on it. A run over every possible word, for both cube sets and both word lists, found each one on its board.

## Word lists

`enable1.txt` is a copy of ENABLE, a public-domain list ("placed in the Public Domain for anyone to use as they see fit"). The notebook checks its SHA-256 digest. Copies called `enable1.txt` differ, so a reader comparing counts should use this file:

- It has 172,820 lines with CRLF endings and no final newline. Line 7,897 is "are", so "are" appears twice and "areole" is missing; there are 172,819 different words.
- It lacks "knickknack", "razzmatazz" and "razzmatazzes", which the copy in [dolph/dictionary](https://github.com/dolph/dictionary) (172,823 words) has, along with "areole".
- It matches, line for line, `ipynb/enable1.txt` in Peter Norvig's [pytudes](https://github.com/norvig/pytudes) (added in commit d9fb809, 2017), except that there line 7,897 is "areole", so that copy has 172,820 different words.
- [norvig.com/ngrams](https://norvig.com/ngrams/) lists its `enable1.txt` as "172,819 words" and 1.9 MB, which fits this file (1,916,143 bytes, 172,819 different words). Its bytes could not be fetched from here to compare, so where "are" replaced "areole" is not settled.

`enable2k.txt` is ENABLE2K (173,528 words, released in 2000, also public domain), byte for byte the copy in Dan Vanderkam's [hybrid-boggle](https://github.com/danvk/hybrid-boggle/blob/main/wordlists/enable2k.txt) (commit 0c53977). It has the same words as the copy in [BartMassey/wordlists](https://github.com/BartMassey/wordlists), whose README carries the original release notes. 802 words are only in ENABLE2K and 93 only in `enable1.txt`. Of its 173,432 words of three or more letters, 173,402 have every q followed by a u, the count of "Boggle OK" words in hybrid-boggle's word list notes; 161,361 can appear with the new cubes and 170,084 with the classic ones.

English word frequencies come from the `wordfreq` package (version 3.1.1, in the `boggle` dependency group).

## The cubes

`CUBES` is the "New Boggle" set and `CLASSIC_CUBES` the classic one, each from four listings that agree face for face: [bananagrammer.com](http://www.bananagrammer.com/2013/10/the-boggle-cube-redesign-and-its-effect.html) (2013), [luke-g.com](https://www.luke-g.com/boggle/dice.html), [robertgamble.net](http://www.robertgamble.net/2016/01/a-programmers-analysis-of-boggle.html) (2016) and Princeton's [BoggleBoard.java](https://coursera.cs.princeton.edu/algs4/assignments/boggle/files/BoggleBoard.java). No publisher document lists the faces, and the dates disagree: bananagrammer says the classic set was made from about 1976 to 1986 and the new one from 1987 to about 2008, Gamble says the new set is in games made in 1992 and since, and Princeton labels the two sets "1983" and "1992".

| Set | Cubes |
|---|---|
| Classic | AACIOT ABILTY ABJMOQu ACDEMP ACELRS ADENVZ AHMORS BIFORX DENOSW DKNOTU EEFHIY EGINTV EGKLUY EHINPS ELPSTU GILRUW |
| New | AAEEGN ABBJOO ACHOPS AFFKPS AOOTTW CIMOTU DEILRX DELRVY DISTTY EEGHNW EEINSU EHRTVW EIOSST ELRTTY HIMNQuU HLNNRZ |

## Files and usage

`board.py` has the cubes, the board, the word search, the sampler, the possible-word test and the roll counts; the notebook and the tests both import it.

```bash
uv run --group boggle --with jupyter jupyter lab boggle/boggle.ipynb
uv run pytest tests/test_boggle.py    # about a minute
```

## Questions left open

- **Findability and cancellation.** Whether a player finds a word, and whether an opponent finds it too, needs an experiment with people; the post describes one.
- **Unseen words.** The exact count of possible words lets estimators of the number of unseen words (Good and Turing, Chao) be scored against the truth. That has not been done.
- **Exact board probability of a word.** The chance of a whole board is a permanent. The chance that a board shows a given short word, such as "toe", is a finite sum over where the cubes carrying its letters land, but the paths overlap and the sum needs inclusion and exclusion. Only the simulated value is known here.
- **Other editions.** Big Boggle (5 by 5) and Super Big Boggle (6 by 6) have other cubes and minimum lengths. The published listings of the 5 by 5 set disagree on one cube.
- **The best board for `enable1.txt`.** Vanderkam's proof is for ENABLE2K. Whether `perslatgsineters` is also best for `enable1.txt` has not been checked.
