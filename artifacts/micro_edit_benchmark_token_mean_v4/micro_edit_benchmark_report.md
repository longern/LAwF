# Micro Edit Benchmark

This is a low-cost deterministic benchmark over hand-specified synthetic sparse edits.
It tests objective behavior, not full knowledge-editing generalization.

- Model: `Qwen/Qwen3-0.6B`
- Edit samples: `10`
- Anchor tokens: `47` / `696` (6.75%)
- Anchor policy: `first token of each critical corrected span`
- Steps per mode: `24`
- LoRA: r=`4`, alpha=`8`
- Anchor confidence: `0.999`
- LAwF: alpha=`1.0`, beta=`1.0`, normalization=`token_mean`

## Objective Summary

| Model | Anchor CE | Non-anchor KL | Full CE | Replay CE | Final loss | Direct CE | Paraphrase CE | Boundary margin | Forbidden preferred | Retention KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sft | 0.005227 | 8.898155 | 0.004132 | - | 0.005077 | 2.057181 | nan | nan | nan | 1.175622 |
| anchor_only | 0.000703 | 3.976003 | 6.326878 | - | 0.009714 | 3.195637 | nan | nan | nan | 0.798117 |
| sft_kl | 0.091586 | 0.448954 | 0.481186 | - | 0.944585 | 1.331922 | nan | nan | nan | 0.064724 |
| lawf | 0.009361 | 0.008748 | 3.466603 | - | 0.009289 | 2.153197 | nan | nan | nan | 0.012594 |
| sft_replay | 0.003912 | 9.173460 | 0.004062 | 0.000609 | 0.005430 | 1.769832 | nan | nan | nan | 0.767544 |
| anchor_replay | 0.000581 | 3.831291 | 6.470223 | 0.001021 | 0.010940 | 3.512655 | nan | nan | nan | 0.479359 |

## Step Sweep

| Steps | Model | Anchor CE | Non-anchor KL | Direct CE | Paraphrase CE | Boundary margin | Retention KL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | sft | 3.714080 | 0.751551 | 1.741300 | nan | nan | 0.297069 |
| 4 | anchor_only | 1.827177 | 0.360909 | 1.829804 | nan | nan | 0.172896 |
| 4 | sft_kl | 3.668935 | 0.349729 | 1.831709 | nan | nan | 0.096036 |
| 4 | lawf | 1.978259 | 0.030087 | 2.218734 | nan | nan | 0.017766 |
| 4 | sft_replay | 3.732279 | 0.669414 | 1.784901 | nan | nan | 0.090702 |
| 4 | anchor_replay | 1.876938 | 0.310101 | 1.922827 | nan | nan | 0.050328 |
| 12 | sft | 1.035574 | 4.298994 | 1.498979 | nan | nan | 0.608113 |
| 12 | anchor_only | 0.005896 | 2.393158 | 2.471883 | nan | nan | 0.555348 |
| 12 | sft_kl | 1.395366 | 0.578661 | 1.479207 | nan | nan | 0.098764 |
| 12 | lawf | 0.054623 | 0.027311 | 2.140596 | nan | nan | 0.012544 |
| 12 | sft_replay | 1.063130 | 4.915794 | 1.716967 | nan | nan | 0.460675 |
| 12 | anchor_replay | 0.003072 | 2.396170 | 2.794930 | nan | nan | 0.237517 |
| 24 | sft | 0.005227 | 8.898155 | 2.057181 | nan | nan | 1.175622 |
| 24 | anchor_only | 0.000703 | 3.976003 | 3.195637 | nan | nan | 0.798117 |
| 24 | sft_kl | 0.091586 | 0.448954 | 1.331922 | nan | nan | 0.064724 |
| 24 | lawf | 0.009361 | 0.008748 | 2.153197 | nan | nan | 0.012594 |
| 24 | sft_replay | 0.003912 | 9.173460 | 1.769832 | nan | nan | 0.767544 |
| 24 | anchor_replay | 0.000581 | 3.831291 | 3.512655 | nan | nan | 0.479359 |

## Edit Set

| Edit | Domain | Tokens | Anchors | Anchor ratio |
| --- | --- | ---: | ---: | ---: |
| identity_archivist | identity | 78 | 4 | 5.13% |
| game_moon_artisan | game_rule | 72 | 5 | 6.94% |
| material_neuron_silk | material | 87 | 5 | 5.75% |
| api_cache_tide | api | 83 | 4 | 4.82% |
| policy_orange_badge | policy | 76 | 4 | 5.26% |
| chem_lumen_salt | chemistry | 64 | 4 | 6.25% |
| robot_courier | robotics | 68 | 6 | 8.82% |
| dsl_amber_loop | programming_language | 50 | 5 | 10.00% |
| geo_silver_ford | geography | 55 | 5 | 9.09% |
| finance_blue_invoice | business_rule | 63 | 5 | 7.94% |

## Probe CE by Edit

| Edit | sft | anchor_only | sft_kl | lawf | sft_replay | anchor_replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity_archivist | 1.8760 | 2.8726 | 1.4863 | 2.4360 | 1.7707 | 3.5543 |
| game_moon_artisan | 1.6491 | 6.5713 | 1.6581 | 3.7265 | 1.5716 | 6.9470 |
| material_neuron_silk | 1.6363 | 3.5412 | 1.3393 | 2.4401 | 1.2146 | 3.4743 |
| api_cache_tide | 2.1987 | 2.7184 | 1.6095 | 2.1658 | 2.0466 | 2.8913 |
| policy_orange_badge | 2.5973 | 3.2644 | 1.0699 | 1.4277 | 2.3438 | 3.5147 |
| chem_lumen_salt | 1.9812 | 2.2795 | 1.1602 | 1.6396 | 1.9541 | 2.3610 |
| robot_courier | 1.8271 | 2.9721 | 1.3244 | 2.0497 | 1.1724 | 3.7957 |
| dsl_amber_loop | 2.6194 | 2.5352 | 1.3887 | 1.9061 | 2.0963 | 3.4109 |
| geo_silver_ford | 2.1643 | 1.8619 | 1.2628 | 2.1495 | 1.8237 | 2.0473 |
| finance_blue_invoice | 2.0224 | 3.3398 | 1.0201 | 1.5911 | 1.7045 | 3.1301 |