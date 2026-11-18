# Common Words in Boggle
The notebook estimates three things about each word and keeps them apart: how often it appears on a random board, how many points it offers, and how common it is in English. It reproduces, count for count, the post [Common on the board, rare in English](https://attobop.net/posts/boggle-words/), which revisits the 2016 version of this notebook. That version missed every Q word, and it ranked words by a single TF-IDF-like score. **The notebook is [here](https://github.com/arclabs561/games/blob/main/boggle/boggle.ipynb)**.

The official rules are in Hasbro's [Boggle instructions (PDF)](https://www.hasbro.com/common/instruct/boggle.pdf).

`enable1.txt` is the ENABLE word list; the notebook checks its SHA-256 digest. English word frequencies come from the `wordfreq` package (version 3.1.1, in the `boggle` dependency group). The board, the word search and the board sampler are in `board.py`, which the notebook and the tests both import. The notebook draws 15,000 boards from one generator seeded with 20161126, as the post does, so its counts match the post's exactly.
