# Query-Family Coverage Evaluation

Held-out log-probability evaluation for Neuron Silk query-family coverage. Lower CE is better; higher boundary margin is better.

| Coverage | Model | Fact CE | Calculation CE | Boundary margin | Forbidden preferred |
| --- | --- | ---: | ---: | ---: | ---: |
| C0_base3 | BASE | 5.288480 | 3.622150 | -0.909676 | 2 / 4 |
| C0_base3 | SFT | 3.535070 | 3.748434 | -4.096417 | 3 / 4 |
| C0_base3 | LAWF | 3.179124 | 3.396474 | -3.194749 | 3 / 4 |
| C1_plus1_calc | BASE | 5.288480 | 3.622150 | -0.909676 | 2 / 4 |
| C1_plus1_calc | SFT | 4.208934 | 4.558664 | -5.869908 | 3 / 4 |
| C1_plus1_calc | LAWF | 3.228935 | 3.452818 | -3.638166 | 3 / 4 |
| C2_plus2_calc | BASE | 5.288480 | 3.622150 | -0.909676 | 2 / 4 |
| C2_plus2_calc | SFT | 4.941815 | 4.973929 | -5.820987 | 3 / 4 |
| C2_plus2_calc | LAWF | 3.200616 | 3.364353 | -3.310482 | 3 / 4 |
| C3_plus2_calc_plus1_paraphrase | BASE | 5.288480 | 3.622150 | -0.909676 | 2 / 4 |
| C3_plus2_calc_plus1_paraphrase | SFT | 2.746793 | 3.314573 | -4.158182 | 3 / 4 |
| C3_plus2_calc_plus1_paraphrase | LAWF | 2.421369 | 3.351780 | -3.805834 | 3 / 4 |

## LAwF Delta vs Base

| Coverage | Fact CE delta | Calculation CE delta | Boundary margin delta |
| --- | ---: | ---: | ---: |
| C0_base3 | -2.109356 | -0.225676 | -2.285073 |
| C1_plus1_calc | -2.059544 | -0.169332 | -2.728490 |
| C2_plus2_calc | -2.087864 | -0.257796 | -2.400806 |
| C3_plus2_calc_plus1_paraphrase | -2.867111 | -0.270370 | -2.896158 |
