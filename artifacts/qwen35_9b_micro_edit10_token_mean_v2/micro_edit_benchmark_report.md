# Micro Edit Benchmark

This is a low-cost deterministic benchmark over hand-specified synthetic sparse edits.
It tests objective behavior, not full knowledge-editing generalization.

- Model: `Qwen/Qwen3.5-9B`
- Edit samples: `10`
- Anchor tokens: `47` / `652` (7.21%)
- Anchor policy: `first token of each critical corrected span`
- Steps per mode: `24`
- LoRA: r=`8`, alpha=`16`
- Anchor confidence: `0.999`
- LAwF: alpha=`1.0`, beta=`1.0`, normalization=`token_mean`

## Objective Summary

| Model | Anchor CE | Non-anchor KL | Full CE | Replay CE | Final loss | Direct CE | Paraphrase CE | Boundary margin | Forbidden preferred | Retention KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sft | 0.000192 | 14.766504 | 0.000185 | - | 0.000215 | 2.925537 | nan | nan | nan | 2.726721 |
| sft_kl | 0.012658 | 0.342469 | 0.411607 | - | 0.759184 | 1.488184 | nan | nan | nan | 0.027578 |
| lawf | 0.002105 | 0.004400 | 2.798120 | - | 0.004802 | 2.026106 | nan | nan | nan | 0.003319 |

## Edit Set

| Edit | Domain | Tokens | Anchors | Anchor ratio |
| --- | --- | ---: | ---: | ---: |
| identity_archivist | identity | 73 | 4 | 5.48% |
| game_moon_artisan | game_rule | 67 | 5 | 7.46% |
| material_neuron_silk | material | 83 | 5 | 6.02% |
| api_cache_tide | api | 71 | 4 | 5.63% |
| policy_orange_badge | policy | 72 | 4 | 5.56% |
| chem_lumen_salt | chemistry | 61 | 4 | 6.56% |
| robot_courier | robotics | 65 | 6 | 9.23% |
| dsl_amber_loop | programming_language | 50 | 5 | 10.00% |
| geo_silver_ford | geography | 54 | 5 | 9.26% |
| finance_blue_invoice | business_rule | 56 | 5 | 8.93% |

## Probe CE by Edit

| Edit | sft | sft_kl | lawf |
| --- | ---: | ---: | ---: |
| identity_archivist | 4.6549 | 2.0043 | 2.8539 |
| game_moon_artisan | 2.7488 | 1.4518 | 2.9174 |
| material_neuron_silk | 1.4311 | 1.1581 | 2.2056 |
| api_cache_tide | 3.2539 | 2.0300 | 2.2898 |
| policy_orange_badge | 4.1829 | 1.3966 | 1.7876 |
| chem_lumen_salt | 1.9064 | 1.0299 | 1.3888 |
| robot_courier | 1.9637 | 1.2266 | 1.3865 |
| dsl_amber_loop | 3.0057 | 1.4065 | 1.4924 |
| geo_silver_ford | 3.7086 | 1.8515 | 2.2763 |
| finance_blue_invoice | 2.3994 | 1.3266 | 1.6629 |