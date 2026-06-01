# ROME Qwen3-0.6B Micro-Edit Diagnostic

Case source: `scaled_sparse`.
This diagnostic applies one ROME edit at a time to Qwen3-0.6B and restores the base weights after each case.
It is a model-editing baseline probe for the small synthetic Qwen3 benchmark, not a full CounterFact/ZsRE evaluation.

## Summary

| Metric | Value |
| --- | ---: |
| case_count | 30 |
| mean_direct_ce_before | 11.1130 |
| mean_direct_ce_after | 0.0767 |
| mean_direct_ce_delta | -11.0362 |
| mean_rephrase_ce_before | 10.1978 |
| mean_rephrase_ce_after | 4.8070 |
| mean_rephrase_ce_delta | -5.3908 |
| generation_hits_after | 28 |
| mean_retention_next_token_kl | 0.0100 |
| mean_locality_next_token_kl | 0.0388 |
| forbidden_preferred_before | 28 |
| forbidden_preferred_after | 31 |
| runtime_seconds | 107.4797 |

## Per-Case Results

| Case | Direct CE Before | Direct CE After | Rephrase CE Before | Rephrase CE After | Locality KL | Forbidden Preferred Before | Forbidden Preferred After | Generation After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| archive_delta | 10.468 | 0.014 | 10.472 | 5.962 | 0.0617 | 1 | 1 |  ravon The internal verification code for the Delta ransomware |
| cache_nova | 8.195 | 0.007 | 7.541 | 3.804 | 0.0235 | 2 | 2 |  kelpa? - Quora What is the cache |
| badge_orange | 12.740 | 0.013 | 11.426 | 5.761 | 0.0139 | 0 | 0 |  mirto Answer: The review gate code for Orange Badge |
| bot_mira | 12.826 | 0.649 | 11.499 | 4.572 | 0.1352 | 0 | 0 |  nav nav nav nav nav nav  |
| game_fog | 9.787 | 0.015 | 8.650 | 5.450 | 0.0418 | 2 | 2 |  qorin? to you? qorin is a |
| lab_lumen | 11.031 | 0.030 | 11.607 | 8.383 | 0.0158 | 0 | 1 |  belta? A. LUMEN B. L |
| dsl_amber | 12.231 | 0.017 | 11.144 | 7.608 | 0.0162 | 0 | 0 |  torenes torenes torenes torenes |
| invoice_blue | 11.543 | 0.144 | 10.428 | 4.331 | 0.0317 | 1 | 1 |  halvo HALVOA HALVOA HAL |
| sensor_iris | 9.398 | 0.015 | 8.911 | 1.433 | 0.0342 | 2 | 2 |  pindra pindra pindra indra |
| router_elm | 12.269 | 0.008 | 11.355 | 6.033 | 0.0952 | 1 | 1 |  vesko: vesko 0.1.12 |
| drone_cedar | 11.786 | 0.001 | 9.905 | 1.174 | 0.0191 | 2 | 2 |  luma: luma luma luma luma l |
| ledger_silver | 12.581 | 0.004 | 11.342 | 7.239 | 0.0498 | 1 | 1 |  emora emora_java_java_java_java_java_java_java_java |
| clinic_pine | 14.618 | 0.001 | 12.980 | 2.758 | 0.0107 | 0 | 0 |  yavin 007? A: 00 |
| compiler_onyx | 11.909 | 0.264 | 11.523 | 5.402 | 0.0622 | 1 | 1 |  sorn  I need to get the strict mode code |
| warehouse_cobalt | 11.505 | 0.010 | 9.543 | 7.338 | 0.0352 | 1 | 1 |  corin? corin? corin? corin? |
| dataset_marble | 12.827 | 0.003 | 12.597 | 3.484 | 0.0161 | 0 | 1 |  daska A: daska B: daska |
| ship_harbor | 11.266 | 0.007 | 10.114 | 4.655 | 0.0240 | 1 | 1 |  gaven, considering the following information: "The price of |
| auth_copper | 11.422 | 0.137 | 10.828 | 5.686 | 0.0425 | 0 | 0 |  ulmar ulmar ulmar ulmar ulmar ulmar |
| search_river | 8.630 | 0.009 | 8.660 | 3.735 | 0.0676 | 2 | 2 |  jorin in the range of 1-10 |
| pipeline_glass | 9.642 | 0.034 | 8.994 | 3.736 | 0.0429 | 1 | 1 |  wexla sculla bittershop in wexla |
| museum_lantern | 10.777 | 0.022 | 9.187 | 2.115 | 0.0143 | 1 | 1 |  faryn faryn faryn faryn |
| scheduler_opal | 8.943 | 0.002 | 7.969 | 2.775 | 0.0598 | 2 | 2 |  avro A. avro B. avro |
| thermal_jade | 12.927 | 0.010 | 11.799 | 3.534 | 0.0150 | 0 | 1 |  zento The code for the mounting profile code for Jade |
| calendar_ember | 11.638 | 0.004 | 11.770 | 9.088 | 0.0217 | 1 | 1 |  orli orli-1234567 |
| vault_ash | 8.703 | 0.680 | 7.666 | 5.290 | 0.0680 | 2 | 2 |  -l0070070070 |
| simulator_brook | 12.475 | 0.174 | 10.901 | 3.539 | 0.0395 | 0 | 0 |  ivra A) 0000 B |
| mailroom_violet | 9.770 | 0.003 | 8.001 | 2.821 | 0.0481 | 2 | 2 |  mora  mora  mora  mora  |
| lab_orchid | 9.549 | 0.009 | 9.117 | 3.971 | 0.0146 | 2 | 2 |  qelma qelma qelma qelma |
| map_zephyr | 11.059 | 0.003 | 10.418 | 4.689 | 0.0147 | 0 | 0 |  talen A talen is a type of insect. |
| payment_amber | 10.874 | 0.014 | 9.589 | 7.842 | 0.0283 | 0 | 0 |  renda renda (mica) renda? How |

Interpretation: lower CE is better for direct/rephrase edit acquisition; lower retention KL and fewer forbidden-preferred locality probes are better for locality.
