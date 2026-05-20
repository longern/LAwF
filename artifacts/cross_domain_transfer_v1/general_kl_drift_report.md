# Held-Out General KL Drift Evaluation

- Model: `Qwen/Qwen3.5-9B`
- Training directory: `/root/lawf_experiment/artifacts/cross_domain_transfer_v1`
- Prompt count: `28`
- Max reference tokens: `96`

## Summary

| Model | Mean KL(base || model) | Mean CE | KL > 0.01 | KL > 0.05 | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.000000 | 0.234066 | 0 | 0 | 0 | 0 | 0 |
| sft | 0.378081 | 0.373691 | 28 | 27 | 26 | 15 | 7 |
| lawf | 0.038704 | 0.302125 | 26 | 7 | 1 | 0 | 0 |

## Category KL

| Category | Count | SFT KL | LAwF KL |
| --- | ---: | ---: | ---: |
| code | 5 | 0.106175 | 0.014604 |
| general | 4 | 0.434839 | 0.040606 |
| math | 5 | 0.170598 | 0.019732 |
| near_game | 3 | 0.567107 | 0.095619 |
| near_identity | 3 | 0.968054 | 0.056474 |
| science | 4 | 0.431602 | 0.037775 |
| writing | 4 | 0.282788 | 0.035561 |