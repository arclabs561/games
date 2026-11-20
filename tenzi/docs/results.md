| Result | Value | How it is known |
|---|---|---|
| Mean turns, fixed face | {{strategies.fixed face.mean:.2f}} (median {{strategies.fixed face.median}}, most likely {{strategies.fixed face.mode}}) | Markov chain on the number of matched dice<!--raw-typst  (@fig-turns)--> |
| Mean turns, switching | {{strategies.switching.mean:.2f}} ({{strategies.switching.mean:.4f}}) | the same<!--raw-typst  (@sec-switch)--> |
| Best mean over all strategies | {{optimal.mean:.2f}}, switching's own | policy iteration with exact fractions, over every choice of dice to keep<!--raw-typst  (@sec-best)--> |
| Largest chance of being done by turn t | switching's, for every t up to {{deadline.last}} | dynamic program over deadlines, in whole numbers<!--raw-typst  (@sec-deadline)--> |
| Mean turns, face named first | {{strategies.target first.mean:.2f}} ({{strategies.target first.mean:.4f}}), standard deviation {{strategies.target first.sd:.2f}} | Markov chain, and in closed form<!--raw-typst  (@sec-named)--> |
| The same mean, from a formula | H₁₀ / ln(6/5) + 1/2, within {{harmonic.gap_10_sci.0}}·10<sup>−{{harmonic.gap_10_sci.1}}</sup> | proved, with a Fourier series for the difference; checked in 60-digit arithmetic<!--raw-typst  (@sec-harmonic)--> |
| Games longer than 30 turns, fixed face | {{strategies.fixed face.over_30:.1%}} | Markov chain |
| Mean winning turn in a race of four, face named first | {{race.target first.4.0:.2f}} | exact, from the chain's distribution; sums stop at {{race_turns}} turns<!--raw-typst  (@tab-race)--> |
| The means above | inside the 95% intervals of {{simulated.games}} simulated games each | simulated, seeds 1 to 5<!--raw-typst  (@tab-checks)--> |
