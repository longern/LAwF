# Held-Out General KL Drift Evaluation

- Model: `Qwen/Qwen3.5-9B`
- Training directory: `/root/lawf_experiment/artifacts/cross_domain_transfer_v1_steps128`
- Prompt count: `28`
- Max reference tokens: `96`

## Summary

| Model | Mean KL(base || model) | Mean CE | KL > 0.01 | KL > 0.05 | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.000000 | 0.234066 | 0 | 0 | 0 | 0 | 0 |
| sft | 0.438411 | 0.418072 | 28 | 27 | 26 | 18 | 8 |
| lawf | 0.030879 | 0.288814 | 26 | 3 | 1 | 0 | 0 |

## Category KL

| Category | Count | SFT KL | LAwF KL |
| --- | ---: | ---: | ---: |
| code | 5 | 0.134394 | 0.012979 |
| general | 4 | 0.512074 | 0.032018 |
| math | 5 | 0.195176 | 0.017468 |
| near_game | 3 | 0.689755 | 0.074928 |
| near_identity | 3 | 1.070537 | 0.042243 |
| science | 4 | 0.482637 | 0.032434 |
| writing | 4 | 0.341984 | 0.025762 |