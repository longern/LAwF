# Qwen3.5-9B LAwF Alpha Sweep

- Model: `Qwen/Qwen3.5-9B`
- Task filter: `all`
- LAwF normalization: `token_mean`
- Annotated tasks: `7`
- Anchor tokens: `87` / `1030`

## Pareto Frontier

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_16_b_0p5_s_32 | lawf | 32 | 16 | 0.5 | 0.738000 | 0.754001 | 0.114455 | 1.345545 | 0.041114 | 0.010937 | 0.147105 |

## All Sweep Points

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_16_b_0p5_s_32 | lawf | 32 | 16 | 0.5 | 0.738000 | 0.754001 | 0.114455 | 1.345545 | 0.041114 | 0.010937 | 0.271633 | 0.147105 |