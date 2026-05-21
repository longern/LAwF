# Cheap Paper Diagnostics

Generated from existing artifacts plus low-cost ablation runs in `qwen35_9b_formal_ablation_v2` and `cross_domain_transfer_ablation_v1`.

## Sparse Annotation and Normalization Counterfactual

- Directly supervised anchors: `61` / `2623` assistant tokens.
- Anchor ratio: `2.33%`.
- If anchor CE were averaged uniformly over all assistant tokens, its aggregate weight would be diluted by `43.0x`.

| Task | Assistant tokens | Anchor tokens | Anchor ratio | Annotation rounds |
| --- | ---: | ---: | ---: | ---: |
| fact_profile | 1028 | 20 | 1.95% | 10 |
| calculation_18x2p4m | 793 | 24 | 3.03% | 8 |
| calculation_10x1p6m | 802 | 17 | 2.12% | 5 |

## Loss Component Diagnostics

| Model | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Mean semantic score | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 0.000060 | 3.396939 | 0.000216 | 0.128060 | 0.075 | 0.000216 |
| ANCHOR_ONLY | 0.000031 | 11.179517 | 11.389436 | 0.053427 | 0.150 | 0.000031 |
| SFT_KL | 0.004372 | 0.061013 | 0.113387 | 0.001036 | 0.025 | 0.174400 |
| SFT_KL_GROUPED | 0.000546 | 0.075817 | 0.119697 | 0.010029 | 0.025 | 0.198867 |
| LAWF | 0.000613 | 0.018972 | 0.315167 | 0.007931 | 0.050 | 0.019585 |

## Base-Teacher Retention

| Model | Mean CE | Mean delta CE vs base | Delta CE > 0.1 | Delta CE > 0.25 | Nearby material delta CE |
| --- | ---: | ---: | ---: | ---: | ---: |
| BASE | 0.2389 | 0.0000 | - | - | 0.0000 |
| SFT | 0.3195 | 0.0806 | 8 | 3 | 0.2901 |
| LAWF | 0.2785 | 0.0396 | 2 | 0 | 0.0726 |

## Longer-Optimization Drift Stress Test

| Steps | Model | Mean held-out KL | KL > 0.1 | KL > 0.25 | KL > 0.5 | Near identity KL | Near game KL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | SFT | 0.378081 | 26 / 28 | 15 / 28 | 7 / 28 | 0.968054 | 0.567107 |
| 32 | LAWF | 0.038704 | 1 / 28 | 0 / 28 | 0 / 28 | 0.056474 | 0.095619 |
| 128 | SFT | 0.438411 | 26 / 28 | 18 / 28 | 8 / 28 | 1.070537 | 0.689755 |
| 128 | LAWF | 0.030879 | 1 / 28 | 0 / 28 | 0 / 28 | 0.042243 | 0.074928 |

## Cross-Domain Objective Ablation

| Model | Anchor CE | Training non-anchor KL | Full CE | Mean judge score | Mean transfer score | Transfer rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 0.000091 | 7.741523 | 0.000080 | 0.167 | 0.125 | 0.000 |
| ANCHOR_ONLY | 0.000038 | 8.458081 | 8.339724 | 0.167 | 0.125 | 0.000 |
| SFT_KL | 0.001925 | 0.115166 | 0.216156 | 0.167 | 0.125 | 0.000 |
| LAWF | 0.000325 | 0.017093 | 0.555145 | 0.167 | 0.250 | 0.250 |

## Transfer and Boundary Diagnostics

| Model | Learned fact score | Transfer calculation score | Mean semantic score | Boundary contamination rate |
| --- | ---: | ---: | ---: | ---: |
| BASE | 0.000 | 0.000 | 0.000 | 0.167 |
| SFT | 0.000 | 0.150 | 0.075 | 0.167 |
| LAWF | 0.000 | 0.100 | 0.050 | 0.667 |
