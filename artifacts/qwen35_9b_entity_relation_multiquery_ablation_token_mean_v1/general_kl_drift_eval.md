# Held-Out General KL Drift Evaluation

- Model: `Qwen/Qwen3.5-9B`
- Training directory: `artifacts/qwen35_9b_entity_relation_multiquery_ablation_token_mean_v1`
- Prompt count: `28`
- Max reference tokens: `96`

## Summary

| Model | Mean KL(base || model) | Mean CE | KL > 0.01 | KL > 0.05 | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.000000 | 0.234066 | 0 | 0 | 0 | 0 | 0 |
| sft | 0.440959 | 0.479580 | 28 | 27 | 27 | 21 | 7 |
| lawf | 0.025783 | 0.282567 | 26 | 2 | 0 | 0 | 0 |

## Category KL

| Category | Count | SFT KL | LAwF KL |
| --- | ---: | ---: | ---: |
| code | 5 | 0.246639 | 0.047628 |
| general | 4 | 0.708486 | 0.024350 |
| math | 5 | 0.263043 | 0.012387 |
| near_game | 3 | 0.425018 | 0.025783 |
| near_identity | 3 | 0.295927 | 0.018620 |
| science | 4 | 0.664562 | 0.023525 |
| writing | 4 | 0.535857 | 0.024283 |