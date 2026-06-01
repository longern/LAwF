# Llama-3.1-8B-Instruct LAwF Confirmatory Sweep

- Model: `LLM-Research/Meta-Llama-3.1-8B-Instruct`
- Annotation source: `/root/lawf_experiment/artifacts/qwen35_9b_entity_relation_multiquery_annotation_v3/annotation_trace.json`
- Annotated tasks: `7`
- Retokenized anchor tokens: `103` / `1019`
- Original Qwen anchor tokens: `87` / `1030`
- LoRA: r=`8`, alpha=`16`
- LAwF normalization: `token_mean`

## Pareto Frontier

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_1_s_32 | lawf | 32 | 4 | 1 | 0.765110 | 0.916663 | 0.355600 | 1.023068 | 0.210660 | 0.003238 | 0.031579 |
| sft_kl_w_0p25 | sft_kl | 32 | - | - | 0.632618 | 0.720193 | 0.033549 | 1.144112 | 0.474382 | 0.003235 | 0.487040 |

## All Sweep Points

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_4_b_1_s_32 | lawf | 32 | 4 | 1 | 0.765110 | 0.916663 | 0.355600 | 1.023068 | 0.210660 | 0.003238 | 0.940609 | 0.031579 |
| lawf_a_8_b_2_s_32 | lawf | 32 | 8 | 2 | 0.821640 | 0.907805 | 0.450077 | 1.107037 | 0.326320 | 0.003797 | 0.956040 | 0.033625 |
| sft_kl_w_0p25 | sft_kl | 32 | - | - | 0.632618 | 0.720193 | 0.033549 | 1.144112 | 0.474382 | 0.003235 | 0.088471 | 0.487040 |