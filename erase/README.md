# Erase

Erase is a make-ten card game for classrooms. It is Game 9 in *Math Fact Fluency* by Jennifer Bay-Williams and Gina Kling. This folder asks which deals cannot be cleared under a generous reading of the rules.

## Rules as modeled

- The deck has four cards of each value from 0 to 10, so 44 cards.
- Each player gets 7 cards. Four more cards lie face up in the center.
- A group is two or more cards that make 10 with +, − and ×, using each card once. A group may include at most one center card.
- The hand cards in a group are erased. A center card stays and can be reused.
- Your score for the round is the number of cards left in your hand.

The printed rules make 10 from pairs only. The first listed variation lets players use more than two cards without naming operations; this folder reads it as allowing +, − and ×.

## Results

- 12,362,536 of the 2,530,881,913,560 possible deals (hand and center) cannot be cleared. That is about 1 in 204,722.
- If a deal can be cleared, then the whole hand plus at most one center card makes 10. One group is always enough.
- A hand with a 0 is stuck exactly when no set of its nonzero cards makes 10, with or without one center card, and no center card is a 10.
- A hand without a 0 is stuck exactly when, for each center card, the hand plus that card is either all multiples of 3, or has no 2, 6 or 10 and at most one odd card. A remainder argument (mod 3 or mod 4) shows that such cards never make 10.
- With cards 1 to 10 only (no 0s, 40 cards), 51,840 of 762,894,475,200 deals are stuck, about 1 in 14.7 million: the hand is seven 4s and 8s, or all eleven cards are 3s, 6s and 9s. `python3 erase/count.py --no-zero` counts them.
- With the four kings added as wild cards (the second listed variation), any deal holding a wild card clears in one group, so 12,362,536 of 7,456,416,121,440 deals are stuck, about 1 in 603,146.

`paper/erase.pdf` states and proves these results. Theorem 17 there is the full classification. The paper also covers the pairs-only, addition-only, +/− and division versions, wild cards, and how often the problem comes up in real play, and ends with puzzles whose answers need several operations. `paper/erase-phone.pdf` is the same text on a narrow page for reading on a phone.

## How the results are checked

- Facts about remainders, zeros and signs are proved by hand. `smt/z3_lemmas.py` checks each induction step with z3.
- One step, Theorem 15 ("big hands"), is a finite claim about 265,148 multisets of cards. `make_certs.py` writes one expression per case to `certificates.jsonl.gz`. `verify_certs.py` checks every expression and shares no code with the search. The file is generated, not stored in the repository.
- `count.py` gets the count of stuck deals by brute force. It uses none of the theorems.
- `simulate.py` reruns every simulated figure in the paper with fixed seeds: addition only, +/−, "grab the pairs first", small groups, and play without brackets.
- `smt/z3_programs.py` and `smt/statespace.py` check the impossible cases with two other methods.

A proof of Theorem 15 that does not check cases is still open.

## Usage

The scripts need only the standard library, except for the z3 checks and the paper build, which declare their dependencies inline (PEP 723) and run with `uv run`. Their lock files (`*.py.lock`) pin the versions that produced the paper: z3-solver 5.4.1.0 and typst 0.15.2. Times are from one run on one machine.

```bash
python3 erase/erase.py 4 4 6 9 5 8 5 --center 7 5 1 8   # prints one group per line
python3 erase/erase.py 0 4 4 4 8 8 8 --center 0 3 7 9   # prints "cannot be cleared"
python3 erase/count.py          # 12362536 2530881913560 1 in 204,722 (about 35 s)
python3 erase/make_certs.py     # writes certificates.jsonl.gz (about 2.5 min)
python3 erase/verify_certs.py   # all 265,148 cases certified (about 10 s)
python3 erase/simulate.py       # simulated figures, 30,000 deals each (about 4 min)

uv run erase/smt/z3_lemmas.py       # a few seconds
uv run erase/smt/z3_programs.py     # about 2.5 min
python3 erase/smt/statespace.py
```

Build the paper with `typst compile erase/paper/erase.typ`; add `--input edition=phone` for the phone edition, or `--features html --format html` for a web page. If the Typst package registry is unreachable, `uv run erase/paper/typst_vendor.py erase/paper/erase.typ` fetches the same packages from the registry's GitHub repository and compiles with the `typst` Python package (`--all` builds all three editions).

## Files

| File | Contents |
|---|---|
| `erase.py` | Solver: target-directed search for an expression, and the hand-clearing search |
| `count.py` | Brute-force count of stuck deals |
| `make_certs.py` | Writes the certificates for Theorem 15 (`certificates.jsonl.gz`, 2.1 MB, not committed) |
| `verify_certs.py` | Independent checker for the certificates |
| `simulate.py` | Seeded simulations: addition only, + and −, pairs first, small groups, no brackets |
| `smt/` | z3 checks of the lemmas and of the impossible cases, plus a state-space search |
| `paper/` | Typst source, print and phone PDFs, and build script |
