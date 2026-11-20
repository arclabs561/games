- Facts about remainders, zeros and signs are proved by hand. `smt/z3_lemmas.py` checks each induction step with z3.
- One step, the theorem on big hands<!--raw-typst  (@thm-big)-->, is a finite claim about {{certificates.cases}} multisets of cards. `make_certs.py` writes one expression per case to `certificates.jsonl.gz`. `verify_certs.py` checks every expression and shares no code with the search. The file is generated, not stored in the repository.
- `count.py` gets the count of stuck deals by brute force. It uses none of the theorems.
- `simulate.py` reruns every simulated figure with fixed seeds: addition only, + and −, "grab the pairs first", small groups, and play without brackets.
- `smt/z3_programs.py` and `smt/statespace.py` check the impossible cases with two other methods.
- For the targets 1 to 30, `targets.py --certs` writes one expression for each multiset from the threshold up that makes the target. `verify_targets.py` lists the multisets itself and checks each expression with Python's own parser, and decides every other multiset with its own search over remainders. It shares no code with `targets.py`.
- `report.py` writes every number quoted in the README and the paper to `results.json`. The tests recompute its cheaper parts and check that the README's shared sections are what `tools/sync_docs.py` writes from `docs/` and `results.json`.

CI runs the tests and the z3 lemma check, and rebuilds the paper's PDFs to check that they are current. The certificate files are generated, not stored, so CI does not check them.
