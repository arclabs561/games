---
name: "mathematical-exposition"
description: "Rules and a review checklist for writing or revising mathematical papers, proofs and notes, including ones a parent and a young child may read together."
---

# Mathematical exposition

Distilled from a full reading of Knuth, Larrabee and Roberts, *Mathematical Writing* (Stanford
CS-88-1193, 1989), including the guest lectures by Halmos, Lamport, Wilf, Ullman, van Leunen, Stemer
and Nilsson. The young-reader part draws on Gardner (2005 interview), Halmos, *How to Write
Mathematics* (1970), Strogatz (*Notices AMS* 2014), R. E. Schwartz (*Notices AMS* 2016),
R. Crowell (AMS blog 2018), Tao's writing pages, Larkin and Simon (1987) and Mayer's
multimedia-learning experiments.

## Principles

1. **Model the reader.** Ask what the reader knows, expects next, and how they could misread this
   sentence (Nilsson). "If you try to write for the novice, you will communicate with the experts"
   (Knuth). The test: would a smart non-expert misread it?
2. **Organize, and do not distract** (Halmos). Spend most effort on structure; fix the idea, not
   the sentence (Lamport).
3. **Get everything up front.** A question as the title or first sentence works (Wilf). The opening
   paragraph should be the best one. Say in plain words what the paper finds.
4. **Concrete before abstract.** One running example used throughout, real data rather than cooked
   samples, examples that build on one another. Pose a puzzle the reader can try before the proof
   (Knuth §25: "builds exactly the right mental structures").
5. **Say what kind of argument is coming** before it starts ("by induction on the expression
   tree", "we find a property every calculation keeps"). Chatty before a proof, lean inside it,
   chatty after (Wilf).
6. **Prefer direct proofs** to proof by contradiction (Halmos). Contradiction is how you find a proof,
   not how you best present it.
7. **Factor long proofs**: state the crucial subgoal, prove it, then finish (Knuth §21).
8. **Simplify first, then pay back the debt to truth** (Nilsson), but never print a false statement;
   say "roughly" or "almost" when simplifying.
9. **State things twice**, especially definitions: once formally, once in other words or by example.
10. **Do not say everything** (Voltaire, via Knuth). Cut what does not serve the reader.

## Sentence and symbol rules (Knuth's 26 points, condensed)

- Separate symbols in different formulas by words ("$8+8-4$ gives 12, $4\times4-8$ gives 8").
- Never start a sentence with a symbol. Never use ∴, ⇒, ∀, ∃ in prose.
- The sentence before a theorem is complete or ends with a colon. Theorem statements are
  self-contained: restate hypotheses such as "Let $m \ge 2$".
- "Suppose that", "assume that". Avoid "We have that".
- Use "we" to mean author and reader together. Imperatives ("Replace x by 7") also work.
- One notation per thing, one thing per notation; one word per concept (do not mix "part",
  "collection", "set" for the same idea).
- Define every variable at first use; give its type ("the set S", not "S").
- Display important formulas. Never break an inline formula across lines.
- Spell out small numbers used as adjectives ("seven cards"); digits for numbers as names
  ("the card 7"). Use words for counts and numerals for values ("two or three solutions; the
  solutions are 2 and 3").
- "that" for defining clauses, "which" after a comma. Watch misplaced "only".
- Never "obviously", "clearly", "easily", "trivially", "just". They humiliate the reader who
  struggles.
- No non-referential "this" ("This shows ..." — what does?).
- Keep subject and verb close; put the subject first; shorter sentences; periods over semicolons.
- Avoid noise phrases ("the fact that", "it is clear that", "The reason for X is that").
- Do not echo an unusual word in unrelated places: readers will link them.
- Capitalize Theorem 3, Lemma 2, Figure 1. Ranges with an en dash; minus is not a hyphen.
- Consistent spelling variety (American or British) throughout.
- Humble: no superlatives about your own results. Humor only where the joke needs the math.

## Proofs and claims

- Every claim says how it is known: proved by hand, checked by computer case by case (with
  certificates), or computed or simulated. Label simulated numbers with their sample size and
  interval.
- Keep a dependency map of the results, and keep it accurate when proofs change.
- When a proof rests on a computer check, say so in the proof title ("Proof, using Theorem 15").
- Use a defined term only in its defined sense. If "group" means a legal move, write "set of cards"
  for anything else; a statement about "no group" is weaker than one about "no set of cards".
- When a proof uses a fact twice, state it once where its objects are defined (for example "two
  or more nonzero cards are never in both families") and cite it.
- A precise word deserves a one-line gloss in place: "affine" (the value has the form Ax + B),
  "NP-complete" (no known method avoids trying a huge number of possibilities as n grows). Say
  where a bound comes from (11⁶: each operation keeps value + 1 at most the product of card + 1).
- Every symbol in a proof is in the glossary (∪, ⊆, |v|), or replaced by words (⊎ became
  "together").
- When a worked example has a table, add the column that says why each row works; "checking takes
  a moment per row" leaves the reader to do it.
- Match claims to the sources' words: the rules do not name operations, so write "the likely
  intent" and "a more generous reading", and quote the authors when describing their aims.
- Prefer "most of the work" to "all the work" when a count says 1,144 of 1,165.
- One word per concept, enforced: choose "stuck" and never also write "unsolvable"; choose
  "remainder" and never "residue". Grep for the retired word before every version.
- Define notation before the result that uses it: a displayed sentence ("For two numbers x and y,
  write S(x, y) = ...") ahead of the lemma, a new word ("tame") ahead of the proposition, a
  convention ("let x = 0 if there is none") as its own sentence. Never inside a hypothesis or
  mid-proof.
- A theorem makes one claim per sentence. "A is B if and only if C. This happens exactly when"
  hides which claim "this" is; write "A deal is dead exactly when one of the following holds".
- Put a corollary next to the theorem it follows from, not in a later section of variations.
- Prefer a direct chain to "suppose not": "the eight cards contain the group, so they are not in
  the family" reads better than a nested contradiction.
- Every printed number must come from a script that is kept: exact counts from the counter,
  simulated ones from a seeded script in the repository. When the script changes, the paper
  quotes its new output. Before release, recount everything with freshly written code; that
  found 51.4% that should have been 51.3% and a table entry rounded differently from its
  neighbors.
- "All 20,000 sampled deals" is fragile: a larger sample found six exceptions. Write "all but k
  of N" and give N.

## Computation in the text

From Norvig's pytudes notebooks, which explain a computation as they go.

- Ask, compute, interpret: a question in one sentence, the computation that answers it, then
  what the result means. A number with no question before it reads as decoration.
- Give each new concept a bold name at first use and one sentence of meaning; in code, a type
  alias or class with the same name. Use that word, and only that word, afterwards.
- Trace one small case by hand before the general argument or program (one split of three
  cards, one path on the board), so the reader has watched the machinery work once.
- Define what counts before giving a count: whether two bracketings, or two orderings of the
  same cards, are "different" changes the number.
- Pair each fast method with the slow, obvious one, and say on which cases they were compared.
- Floats near a target can be wrong; recheck those values with exact fractions before printing.
- Say how sure you are and why: checked by hand arithmetic, by an independent program, or
  resting on one implementation.
- End with variations and open questions. They show where the result stops.

## Writing for a parent and a young child too

Keep all of the formal mathematics. Add a way in, at the point each idea first appears.

- **Reading guide** on page 1: which sections tell the story with arithmetic only, which prove
  things, and that it is fine to skip proofs or stop anywhere (Schwartz's "bucking bronco").
- **Idea boxes**, two to four sentences each, at the first use of a tool: what a proof is,
  remainders as a clock (read ≡ aloud as "leaves the same remainder as"), multisets as bags of
  cards, "n choose k", induction as a property climbing a tree, "if and only if", certificates,
  what a solver like z3 does, exact versus simulated numbers.
- **Try-it puzzles** (Gardner) placed before the answer, with answers in an appendix. Spread them
  out: no more than one or two boxes per page (Crowell).
- **A picture for every key concept**, placed right beside the words about it (Mayer's spatial
  contiguity; Strogatz). Go concrete to abstract: draw the real cards first, then the letters.
- **Glossary** of words and symbols with how to read each aloud.
- **A sense of scale** for big numbers ("one minute per hand, about twenty weeks").
- Short words, short sentences, no talking down ("write up" to children, E. B. White).
- Say symbols in words the first time; prefer words to symbols in prose (Halmos: "the best
  notation is no notation"; Strogatz: readers stop at symbols they cannot pronounce).
- **Order for two audiences:** story first (the game, the answer, one worked proof, advice for
  the table, puzzles), then the proofs, then how everything was checked. The lay reader's payoff
  must not sit behind fifteen pages of proofs. Keep verification in one section that opens with
  a table of how each claim is known and the dependency map.
- **Epigraphs:** at most one per main section and none on appendices. Keep only those whose joke
  needs the section's mathematics (a stuck bear for a stuck hand, the Count for counting). About
  a sentence long, spoken lines only, from things children of the current era know. Check the
  wording against a source. Never lengthen a quote with invented words; if the real line is too
  short, choose another line or leave the section without one.
- **Margin notes:** one short remark in a child's voice beside the first line of the paragraph it
  belongs to.
- In a section that is all puzzles, set them as plain numbered items; boxes are for interruptions
  in running text.
- An Idea box belongs at the first use of its word. If the word must appear earlier (in the
  introduction or the plan), gloss it there in a clause.

## Diagrams

- A diagram helps when it groups together what is used together, replaces labels with position,
  and makes the key inference visible (Larkin and Simon). An arbitrary picture does not help.
- Remove decoration that carries no information (Mayer's coherence effect).
- One meaning per color across the whole document (for example green = makes 10 or proved by hand,
  red = the forbidden residue or a loud card, grey = zero or neutral).
- Captions say what to look at, not just what the figure is.
- Audit colors by listing every use of each one. Orange once meant four things and blue three.
  Give computer-checked results their own color, keep the color of a drawn object (blue center
  cards) for nothing else, and draw neutral things (discarded cases, plain bars, puzzle boxes)
  in grey.
- Look for ideas described only in words: the pigeonhole boxes and the allowed values of each
  family both needed a picture.
- Diagrams small, next to their text; one deliberately bad example is fine to make a point.

## Review checklist (run before every version)

    grep -n -i -w 'obviously\|clearly\|trivially\|easily\|just' body.typ
    grep -n -o -i 'colour\|practis\|licence\|neighbour\|analys[e]d\|centre' body.typ   # if American
    grep -n -o '\. \$' body.typ            # sentences starting with a symbol
    grep -n -i '<names of private people>' body.typ

- Read each theorem statement alone: is it self-contained?
- Read the text with the formulas replaced by "blah": does it still flow?
- Does every term on a figure or axis appear defined in the text?
- Does each Idea box sit where the idea is first used, and each puzzle before its answer?
- Is any page carrying more than two boxes?
- Does every definition match the code that produced the numbers attached to it?
- Does every "the authors say" match a quotation from the source?
- Does every computed number follow a question and come before its interpretation?
- Hard-coded cross-references: `grep -n -o 'Section [0-9]\+' body.typ` should find nothing; use
  labels, so that reordering sections cannot leave stale numbers.
- Bare openers: `grep -o '\(^\|[.:] \)\(That\|This\) \(is\|happens\|gives\|proves\)' body.typ`.
  Give each a noun.
- Numbers lost or gained since the last version:
  `pdftotext old.pdf - | grep -o '[0-9][0-9,.]*[0-9]%\?' | sort -u` against the same for the new
  file, with `comm`. Every difference should be one you intended.
- Run readcost on the prose (HTML export with formulas, tables and code stripped). Treat noun
  stacks and long dependencies as real; flags next to stripped formulas are usually false.
  Also count colons per 1,000 words and rewrite colon-before-a-reveal sentences.
- Check outside facts (counts for related games, quotations from rules sheets) against their
  sources again before release, and recompute the ones a short script can reach.