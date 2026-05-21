# Coverage Expansion Experiment

- Model: `Qwen/Qwen3.5-9B`
- Base tasks: `3`
- Extra recursive-annotation tasks: `2`
- Total tasks: `5`
- Anchor tokens: `99` / `3126`

## Scores

| Model | Semantic score | Learned fact | Transfer calc | Retention KL vs base | Anchor CE | Non-anchor KL | Full CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.025 | 0.000 | 0.050 | 0.000000 | - | - | - |
| SFT | 0.050 | 0.000 | 0.100 | 0.081782 | 0.000219 | 2.956618 | 0.000417 |
| LAWF | 0.125 | 0.200 | 0.050 | 0.002415 | 0.001381 | 0.020048 | 0.286093 |

## Baseline Comparison

| Setting | Model | Mean semantic score | Transfer calc | Retention KL vs base |
| --- | --- | ---: | ---: | ---: |
| base-3-task | SFT | 0.075 | 0.150 | 0.128060 |
| base-3-task | LAWF | 0.050 | 0.100 | 0.007931 |
| expanded-6-task | SFT | 0.050 | 0.100 | 0.081782 |
| expanded-6-task | LAWF | 0.125 | 0.050 | 0.002415 |

## Extra Annotation Tasks

| Task | Tokens | Anchors | Anchor ratio | Rounds |
| --- | ---: | ---: | ---: | ---: |
| coverage_calc_6x2p2m | 220 | 27 | 12.273% | 8 |
| coverage_calc_20x1p1m | 283 | 11 | 3.887% | 3 |
