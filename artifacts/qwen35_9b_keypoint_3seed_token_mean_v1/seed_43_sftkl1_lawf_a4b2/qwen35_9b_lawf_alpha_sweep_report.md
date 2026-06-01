# Qwen3.5-9B LAwF Alpha Sweep

- Model: `Qwen/Qwen3.5-9B`
- Task filter: `all`
- LAwF normalization: `token_mean`
- Annotated tasks: `7`
- Anchor tokens: `87` / `1030`

## Pareto Frontier

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_2_s_32 | lawf | 32 | 4 | 2 | 0.859927 | 0.876094 | 0.290551 | 1.413137 | 0.015664 | 0.004150 | 0.012925 |
| sft_kl_w_1 | sft_kl | 32 | - | - | 0.857872 | 0.807139 | 0.267206 | 1.499271 | 0.018053 | 0.019067 | 0.048284 |

## All Sweep Points

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_2_s_32 | lawf | 32 | 4 | 2 | 0.859927 | 0.876094 | 0.290551 | 1.413137 | 0.015664 | 0.004150 | 0.217381 | 0.012925 |
| sft_kl_w_1 | sft_kl | 32 | - | - | 0.857872 | 0.807139 | 0.267206 | 1.499271 | 0.018053 | 0.019067 | 0.096994 | 0.048284 |