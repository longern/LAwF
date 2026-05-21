# Micro Edit Benchmark

This is a low-cost deterministic benchmark over hand-specified synthetic sparse edits.
It tests objective behavior, not full knowledge-editing generalization.

- Model: `Qwen/Qwen3-0.6B`
- Edit samples: `10`
- Anchor tokens: `47` / `696` (6.75%)
- Anchor policy: `first token of each critical corrected span`
- Steps per mode: `24`
- LoRA: r=`4`, alpha=`8`

## Objective Summary

| Model | Anchor CE | Non-anchor KL | Full CE | Replay CE | Final loss | Probe CE | Retention KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sft | 0.005227 | 8.898155 | 0.004132 | - | 0.005077 | 2.057181 | 1.175622 |
| anchor_only | 0.000395 | 4.281005 | 6.555154 | - | 0.000499 | 3.272553 | 0.864964 |
| sft_kl | 0.091586 | 0.448954 | 0.481186 | - | 0.944585 | 1.331922 | 0.064724 |
| lawf | 0.005188 | 0.064157 | 3.415649 | - | 0.077357 | 1.973158 | 0.021528 |
| sft_replay | 0.003912 | 9.173460 | 0.004062 | 0.000609 | 0.005430 | 1.769832 | 0.767544 |
| anchor_replay | 0.000327 | 4.119342 | 6.654807 | 0.000961 | 0.001326 | 3.529850 | 0.499276 |

## Step Sweep

| Steps | Model | Anchor CE | Non-anchor KL | Probe CE | Retention KL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | sft | 3.714080 | 0.751551 | 1.741300 | 0.297069 |
| 4 | anchor_only | 1.838605 | 0.357794 | 1.828481 | 0.170820 |
| 4 | sft_kl | 3.668935 | 0.349729 | 1.831709 | 0.096036 |
| 4 | lawf | 1.748161 | 0.228380 | 1.878225 | 0.128665 |
| 4 | sft_replay | 3.732279 | 0.669414 | 1.784901 | 0.090702 |
| 4 | anchor_replay | 1.879840 | 0.310026 | 1.921102 | 0.054123 |
| 12 | sft | 1.035574 | 4.298994 | 1.498979 | 0.608113 |
| 12 | anchor_only | 0.006178 | 2.446297 | 2.486099 | 0.567251 |
| 12 | sft_kl | 1.395366 | 0.578661 | 1.479207 | 0.098764 |
| 12 | lawf | 0.014120 | 0.224351 | 2.069793 | 0.063742 |
| 12 | sft_replay | 1.063130 | 4.915794 | 1.716967 | 0.460675 |
| 12 | anchor_replay | 0.003355 | 2.441207 | 2.800958 | 0.235357 |
| 24 | sft | 0.005227 | 8.898155 | 2.057181 | 1.175622 |
| 24 | anchor_only | 0.000395 | 4.281005 | 3.272553 | 0.864964 |
| 24 | sft_kl | 0.091586 | 0.448954 | 1.331922 | 0.064724 |
| 24 | lawf | 0.005188 | 0.064157 | 1.973158 | 0.021528 |
| 24 | sft_replay | 0.003912 | 9.173460 | 1.769832 | 0.767544 |
| 24 | anchor_replay | 0.000327 | 4.119342 | 3.529850 | 0.499276 |

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
| identity_archivist | 1.8760 | 2.9567 | 1.4863 | 2.1539 | 1.7707 | 3.6348 |
| game_moon_artisan | 1.6491 | 6.8516 | 1.6581 | 3.6089 | 1.5716 | 7.0870 |
| material_neuron_silk | 1.6363 | 3.6193 | 1.3393 | 2.3549 | 1.2146 | 3.4913 |
| api_cache_tide | 2.1987 | 2.7886 | 1.6095 | 2.0276 | 2.0466 | 2.8841 |
| policy_orange_badge | 2.5973 | 3.4007 | 1.0699 | 1.3208 | 2.3438 | 3.5493 |
| chem_lumen_salt | 1.9812 | 2.3634 | 1.1602 | 1.3225 | 1.9541 | 2.3964 |
| robot_courier | 1.8271 | 2.9124 | 1.3244 | 1.7847 | 1.1724 | 3.7879 |
| dsl_amber_loop | 2.6194 | 2.5948 | 1.3887 | 1.8011 | 2.0963 | 3.3750 |
| geo_silver_ford | 2.1643 | 1.7927 | 1.2628 | 1.8071 | 1.8237 | 1.9588 |
| finance_blue_invoice | 2.0224 | 3.4455 | 1.0201 | 1.5500 | 1.7045 | 3.1339 |