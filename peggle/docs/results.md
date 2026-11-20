| Result | Value | How it is known |
|---|---|---|
| Chance the winner of six games in ten is better (uniform prior) | {{coins.six_of_ten.0}}/{{coins.six_of_ten.1:d}} ≈ {{coins.six_of_ten_value:.3f}} | proved<!--raw-typst  (@prop-coins)-->: the chance of at most 6 heads in 11 fair flips |
| Wins needed to be 95% sure | {{coins.wins_needed.10.0}} in 10, {{coins.wins_needed.100.0}} in 100, {{coins.wins_needed.1000.0}} in 1,000 | exact<!--raw-typst  (@tab-wins)--> |
| Equal players, play until 95% sure, cap 100 | someone declared better with chance {{equal_verdict.posterior.100:.3f}} | dynamic program over (wins, games)<!--raw-typst  (@tab-stop)--> |
| Equal players, play until the Bayes factor reaches 20 | {{equal_verdict.bayes factor.100:.4f}} (N = 100); never more than 0.05 for any cap | dynamic program; Ville's inequality<!--raw-typst  (@prop-ville)--> |
| Stricter posterior threshold that keeps it at 5% | {{strict.100.threshold}} (N = 100), {{strict.1000.threshold}} (N = 1,000) | bisection with the same dynamic program<!--raw-typst  (@tab-rules)--> |
| Better player (p = 0.55) wins a best-of-seven match | {{formats.best of 7.0:.3f}} | exact<!--raw-typst  (@tab-formats)--> |
| "Play until one leads by 15", p = 0.55 | the worse player is declared better with chance {{wald.worse:.4f}} | gambler's ruin<!--raw-typst  (@prop-ruin)--> |
| Repeated binomial tests | {{armitage.agree}} of {{armitage.entries}} entries of Armitage, McPherson and Rowe (1969) agree to five decimals, the other {{armitage.off:words}} within 2·10<sup>−5</sup> | comparison with a published table<!--raw-typst  (@sec-check)--> |
