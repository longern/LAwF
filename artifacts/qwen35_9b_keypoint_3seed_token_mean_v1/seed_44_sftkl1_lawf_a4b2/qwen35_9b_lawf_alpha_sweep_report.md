# Qwen3.5-9B LAwF Alpha Sweep

- Model: `Qwen/Qwen3.5-9B`
- Task filter: `all`
- LAwF normalization: `token_mean`
- Annotated tasks: `7`
- Anchor tokens: `87` / `1030`

## Pareto Frontier

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_2_s_32 | lawf | 32 | 4 | 2 | 0.704855 | 0.639416 | 0.285575 | 1.189576 | 0.012861 | 0.015412 | 0.015256 |

## All Sweep Points

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_2_s_32 | lawf | 32 | 4 | 2 | 0.704855 | 0.639416 | 0.285575 | 1.189576 | 0.012861 | 0.015412 | 0.214914 | 0.015256 |
| sft_kl_w_1 | sft_kl | 32 | - | - | 0.855405 | 0.793897 | 0.262977 | 1.509341 | 0.025880 | 0.019087 | 0.095543 | 0.050189 |