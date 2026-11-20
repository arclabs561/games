---
name: "results-readme"
description: "Shape and checks for a game's README in this repository: findings first, a results table that says how each number is known, checks, data licenses, open questions, usage."
---

# Results README

Taken from the READMEs that read best here (Erase, Spot It, EV Nova, Boggle) and from the ones that did not (Codenames at 16,700 words, Ratscrew and Drive Ya Nuts with one heading or none).

## Order of sections

1. **Opening, five lines at most.** The question in a player's words, then what was found. A reader who stops here should know the answer and how sure it is.
2. **Rules as modeled.** Each ambiguous rule and the reading taken. Name other readings and say whether they change the answer.
3. **Results.** One table, with one row per claim and a column "How it is known":
   - proved;
   - checked case by case, with certificates;
   - exact computation;
   - simulated (sample size, seed, 95% interval);
   - or a cited source.

   A line above the table can say "exact unless marked" when most rows are exact.
4. **How the results are checked.** Each check states what it compares and what agreed: brute force on small cases, a second solver, a published count, a separate certificate checker. Say which checks CI runs.
5. **Sections on single questions** (one per question, each starting with its answer). Experiments get one paragraph and one table each. They link to their plan and to the notes in `results/`, where the detail lives.
6. **Data.** A table of source, what it is, license (quoted from the source), and whether it is committed.
7. **Related work or sources.** Published results reproduced, and where this work agrees or disagrees with them.
8. **Open questions.** Specific enough that someone could start on one.
9. **Usage** with timings, then **Files**.

## Shared with the paper

When a game has a paper, the README keeps the opening, the results table, the rules as
modeled, the checks and the open questions, and links into the paper by section name; proofs
and long explanation live only in the paper. The shared parts are Markdown fragments in
`<game>/docs/`, placed in the README between `<!-- shared:<name> -->` and `<!-- /shared -->` and
filled by `uv run tools/sync_docs.py` from `<game>/results.json`, so every quoted number has one
source. Edit the fragment, never the README between the markers; a test compares the two. The
typst-paper-pipeline skill has the paper's side.

## Pictures and demonstrations

No README had a single image when this was written; every one should.
- **Show the game first.** Near the top, put one picture of a real instance: a board, a deal, a hand, a terminal screen. Draw it with a script; don't use a photo or the publisher's art.
- **One figure per key result,** next to its sentence:
  - the distribution, not only the mean;
  - the stuck hand;
  - the best board;
  - the route on the map;
  - the cross-play matrix.

  The caption says what to look at.
- **Demonstrate tools with a transcript.** A short code block of a real run beats a screenshot, because it is searchable, readable on a phone and checkable. A test can run the same command and compare.
- **Format:**
  - SVG for drawings and line plots, PNG only for dense plots.
  - About 800 px wide or less.
  - A white background, so the image stays readable in GitHub's dark mode.
  - Alt text that says what the picture shows.
- **Reproducible:**
  - Each image is written by a script in the game's folder, with fixed seeds.
  - Matplotlib SVGs set `svg.hashsalt` and drop the date metadata, so the bytes are stable.
  - A test checks that the committed file is what the script writes, or the CI figure check covers it.
- **Copyright:**
  - No card art, box art, sprites or screenshots of commercial games.
  - Draw generic boards and cards. Use the game's words or numbers only where the rules need them.
  - A real person's photo never goes in a file.

## Length

- Aim under about 2,500 words. Past that, the opening summary is mandatory. Move experiment detail, long tables and derivations into `results/` notes or a paper, and link them.
- One heading per question; a README with one heading over 3,000 words is a lab notebook, not a write-up.

## Wording

- Numbers in the text match the files the scripts write; a test checks the quoted ones.
- Say "best found" unless optimality is proved, and name who proved it if it is a cited result.
- Report nulls with intervals and the same care as positive results: "no measurable effect (−0.02 to 0.04)", never "did not work".
- Distinguish what was reproduced from what was only read; "the paper reports" versus "recomputed here".
- `readcost check README.md` passes; American spelling; no names except cited authors.
