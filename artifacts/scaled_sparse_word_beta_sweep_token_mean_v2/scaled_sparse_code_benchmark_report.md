# Scaled Sparse Code Benchmark

- Model: `Qwen/Qwen3-0.6B`
- Families annotated: `30`
- Annotated tasks: `30`
- Anchor tokens: `93` / `4259` (2.18%)
- Mean corrected rounds per task: `1.13`
- Steps: `8`
- Anchor confidence: `0.999`
- LAwF betas: `[0.5, 1.0, 2.0, 4.0, 8.0]`
- LAwF normalization: `token_mean`

## Held-Out Scale Curve

| Families | Model | Anchor CE | Diagnostic train non-anchor KL | Full CE | Direct CE | Paraphrase CE | Held-out retention KL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | sft | 5.587101 | 0.771523 | 0.253311 | 10.566120 | 6.980496 | 0.033941 |
| 30 | sft_kl | 6.115263 | 0.085163 | 0.408605 | 10.211155 | 6.818790 | 0.005229 |
| 30 | lawf_beta_0.5 | 4.137656 | 0.040865 | 0.546486 | 8.601563 | 5.246893 | 0.011485 |
| 30 | lawf | 4.578992 | 0.023646 | 0.553286 | 9.076824 | 5.802458 | 0.007915 |
| 30 | lawf_beta_2 | 5.497044 | 0.017838 | 0.591048 | 9.866718 | 6.338700 | 0.004741 |
| 30 | lawf_beta_4 | 7.143531 | 0.011228 | 0.633452 | 10.487305 | 7.243088 | 0.002669 |
| 30 | lawf_beta_8 | 8.603701 | 0.009402 | 0.673046 | 11.108170 | 7.988795 | 0.001685 |

## Annotation Load

| Task | Tokens | Anchors | Anchor ratio |
| --- | ---: | ---: | ---: |
| archive_delta_long_note | 146 | 3 | 2.05% |
| cache_nova_long_note | 40 | 4 | 10.00% |
| badge_orange_long_note | 87 | 3 | 3.45% |
| bot_mira_long_note | 142 | 3 | 2.11% |
| game_fog_long_note | 166 | 4 | 2.41% |
| lab_lumen_long_note | 199 | 3 | 1.51% |
| dsl_amber_long_note | 193 | 3 | 1.55% |
| invoice_blue_long_note | 158 | 3 | 1.90% |
| sensor_iris_long_note | 172 | 4 | 2.33% |
| router_elm_long_note | 36 | 3 | 8.33% |
| drone_cedar_long_note | 83 | 3 | 3.61% |
| ledger_silver_long_note | 161 | 2 | 1.24% |
| clinic_pine_long_note | 155 | 3 | 1.94% |
| compiler_onyx_long_note | 120 | 3 | 2.50% |
| warehouse_cobalt_long_note | 177 | 3 | 1.69% |
| dataset_marble_long_note | 143 | 3 | 2.10% |
| ship_harbor_long_note | 154 | 2 | 1.30% |
| auth_copper_long_note | 176 | 3 | 1.70% |
| search_river_long_note | 117 | 4 | 3.42% |
| pipeline_glass_long_note | 181 | 4 | 2.21% |
| museum_lantern_long_note | 148 | 3 | 2.03% |
| scheduler_opal_long_note | 180 | 2 | 1.11% |
| thermal_jade_long_note | 154 | 2 | 1.30% |
| calendar_ember_long_note | 201 | 3 | 1.49% |
| vault_ash_long_note | 100 | 4 | 4.00% |
| simulator_brook_long_note | 131 | 3 | 2.29% |
| mailroom_violet_long_note | 128 | 3 | 2.34% |
| lab_orchid_long_note | 132 | 4 | 3.03% |
| map_zephyr_long_note | 141 | 3 | 2.13% |
| payment_amber_long_note | 138 | 3 | 2.17% |