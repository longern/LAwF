# Coverage Expansion Experiment

- Model: `Qwen/Qwen3.5-9B`
- Base tasks: `3`
- Extra recursive-annotation tasks: `1`
- Total tasks: `4`
- Anchor tokens: `88` / `2843`

## Scores

| Model | Semantic score | Learned fact | Transfer calc | Retention KL vs base | Anchor CE | Non-anchor KL | Full CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.050 | 0.000 | 0.100 | 0.000000 | - | - | - |
| SFT | 0.025 | 0.050 | 0.000 | 0.041598 | 0.000296 | 3.126845 | 0.000393 |
| LAWF | 0.025 | 0.000 | 0.050 | 0.001919 | 0.001323 | 0.020803 | 0.318802 |

## Baseline Comparison

| Setting | Model | Mean semantic score | Transfer calc | Retention KL vs base |
| --- | --- | ---: | ---: | ---: |
| base-3-task | SFT | 0.075 | 0.150 | 0.128060 |
| base-3-task | LAWF | 0.050 | 0.100 | 0.007931 |
| expanded-6-task | SFT | 0.025 | 0.000 | 0.041598 |
| expanded-6-task | LAWF | 0.025 | 0.050 | 0.001919 |

## Extra Annotation Tasks

| Task | Tokens | Anchors | Anchor ratio | Rounds |
| --- | ---: | ---: | ---: | ---: |
| coverage_calc_6x2p2m | 220 | 27 | 12.273% | 8 |
