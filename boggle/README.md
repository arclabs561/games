# Common Words in Boggle
The notebook estimates how often each word appears on a random Boggle board, and ranks the common ones against how often they appear in English. [Common on the board, rare in English](https://attobop.net/posts/boggle-words/) revisits it. That post explains the Qu-face bug fixed here, keeps board probability, solo points and familiarity as separate measurements instead of one TF-IDF-like score, and adds Wilson intervals. **The notebook is [here](https://github.com/arclabs561/games/blob/main/boggle/boggle.ipynb)**.

The official rules are in Hasbro's [Boggle instructions (PDF)](https://www.hasbro.com/common/instruct/boggle.pdf).

`enable1.txt` is the ENABLE word list. `ANC-token-count.txt` (5.8 MB) holds word counts from the American National Corpus, which the notebook uses as a stand-in for how familiar a word is. The board, the word search and the board sampler are in `boggle.py`, which the notebook and the tests both import. Board *i* of a sample is drawn with `random.Random(i)`, so every number in the notebook can be reproduced.
