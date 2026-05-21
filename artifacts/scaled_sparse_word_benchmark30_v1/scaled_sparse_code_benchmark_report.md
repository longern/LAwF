# Scaled Sparse Code Benchmark

- Model: `Qwen/Qwen3-0.6B`
- Families annotated: `30`
- Annotated tasks: `30`
- Anchor tokens: `93` / `4259` (2.18%)
- Mean corrected rounds per task: `1.13`
- Steps: `8`

## Held-Out Scale Curve

| Families | Model | Anchor CE | Diagnostic train non-anchor KL | Full CE | Direct CE | Paraphrase CE | Held-out retention KL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | sft | 0.033032 | 4.196948 | 0.004390 | 4.866478 | 3.752650 | 0.075485 |
| 1 | sft_kl | 0.145007 | 0.108095 | 0.251139 | 4.953680 | 3.161033 | 0.006367 |
| 1 | lawf | 0.004404 | 0.151103 | 0.700956 | 5.257152 | 0.904745 | 0.022540 |
| 8 | sft | 2.973603 | 1.377450 | 0.091152 | 9.355219 | 6.024323 | 0.077886 |
| 8 | sft_kl | 3.848744 | 0.092741 | 0.276465 | 9.418482 | 5.779162 | 0.005020 |
| 8 | lawf | 1.905979 | 0.205016 | 0.642300 | 5.978081 | 3.382368 | 0.073940 |
| 16 | sft | 4.501292 | 0.938761 | 0.183194 | 10.515684 | 6.683059 | 0.039974 |
| 16 | sft_kl | 5.319105 | 0.099636 | 0.340707 | 10.112808 | 6.481499 | 0.005955 |
| 16 | lawf | 2.797864 | 0.263876 | 0.752960 | 5.399162 | 3.641612 | 0.129291 |
| 30 | sft | 5.572868 | 0.757939 | 0.254942 | 10.518810 | 6.834833 | 0.035986 |
| 30 | sft_kl | 6.211442 | 0.087656 | 0.406713 | 10.248176 | 6.807410 | 0.005497 |
| 30 | lawf | 3.556179 | 0.284572 | 0.831410 | 5.414872 | 3.888690 | 0.254484 |

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