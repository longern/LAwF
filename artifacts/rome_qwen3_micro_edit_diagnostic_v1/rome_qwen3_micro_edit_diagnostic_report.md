# ROME Qwen3-0.6B Micro-Edit Diagnostic

This diagnostic applies one ROME edit at a time to Qwen3-0.6B and restores the base weights after each case.
It is a model-editing baseline probe for the small synthetic Qwen3 benchmark, not a full CounterFact/ZsRE evaluation.

## Summary

| Metric | Value |
| --- | ---: |
| case_count | 3 |
| mean_direct_ce_before | 3.3419 |
| mean_direct_ce_after | 0.3043 |
| mean_direct_ce_delta | -3.0376 |
| mean_rephrase_ce_before | 2.9988 |
| mean_rephrase_ce_after | 0.6826 |
| mean_rephrase_ce_delta | -2.3162 |
| generation_hits_after | 3 |
| mean_retention_next_token_kl | 0.0175 |
| mean_locality_next_token_kl | 0.2546 |
| forbidden_preferred_before | 3 |
| forbidden_preferred_after | 3 |
| runtime_seconds | 12.4358 |

## Per-Case Results

| Case | Direct CE Before | Direct CE After | Rephrase CE Before | Rephrase CE After | Locality KL | Forbidden Preferred Before | Forbidden Preferred After | Generation After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| neuron_silk_heat_leak | 1.707 | 0.019 | 2.016 | 0.649 | 0.4795 | 2 | 2 |  0.014 W/m·K，假设在 10 |
| neuron_silk_resistance | 1.919 | 0.637 | 2.158 | 1.012 | 0.2638 | 1 | 1 |  0.0311111111111 |
| identity_archivist_code | 6.400 | 0.257 | 4.823 | 0.387 | 0.0205 | 0 | 0 |  雨灯，她想把一个物体放在一个罐子里，罐 |

Interpretation: lower CE is better for direct/rephrase edit acquisition; lower retention KL and fewer forbidden-preferred locality probes are better for locality.
