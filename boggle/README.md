# Common Words in Boggle
I find the most common words in Boggle through a statistical argument. I also rank the top words with tf-idf, so that the list favors words that are common on boards but less likely to be known by other players. **A notebook of the results can be viewed [here](https://github.com/arclabs561/games/blob/main/boggle/boggle.ipynb)**.

The official rules are in Hasbro's [Boggle instructions (PDF)](https://www.hasbro.com/common/instruct/boggle.pdf).

`enable1.txt` is the ENABLE word list. `ANC-token-count.txt` (5.8 MB) holds word counts from the American National Corpus, which the notebook uses to judge how familiar a word is. The board, the word search and the board sampler are in `boggle.py`, which the notebook and the tests both import. Board *i* of a sample is drawn with `random.Random(i)`, so every number in the notebook can be reproduced.
