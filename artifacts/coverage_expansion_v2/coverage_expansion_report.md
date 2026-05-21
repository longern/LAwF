# Coverage Expansion Experiment

- Model: `Qwen/Qwen3.5-9B`
- Base tasks: `3`
- Extra recursive-annotation tasks: `3`
- Total tasks: `6`
- Anchor tokens: `120` / `3293`

## Scores

| Model | Semantic score | Learned fact | Transfer calc | Retention KL vs base | Anchor CE | Non-anchor KL | Full CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.025 | 0.000 | 0.050 | 0.000000 | - | - | - |
| SFT | 0.225 | 0.350 | 0.100 | 0.154105 | 0.000636 | 2.990649 | 0.001289 |
| LAWF | 0.200 | 0.350 | 0.050 | 0.005762 | 0.001835 | 0.020585 | 0.301581 |

## Baseline Comparison

| Setting | Model | Mean semantic score | Transfer calc | Retention KL vs base |
| --- | --- | ---: | ---: | ---: |
| base-3-task | SFT | 0.075 | 0.150 | 0.128060 |
| base-3-task | LAWF | 0.050 | 0.100 | 0.007931 |
| expanded-6-task | SFT | 0.225 | 0.100 | 0.154105 |
| expanded-6-task | LAWF | 0.200 | 0.050 | 0.005762 |

## Extra Annotation Tasks

| Task | Tokens | Anchors | Anchor ratio | Rounds |
| --- | ---: | ---: | ---: | ---: |
| coverage_calc_6x2p2m | 220 | 27 | 12.273% | 8 |
| coverage_calc_20x1p1m | 283 | 11 | 3.887% | 3 |
| coverage_paraphrase_material_choice | 167 | 21 | 12.575% | 7 |
