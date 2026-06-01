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
| lawf_a_8_b_2_s_32 | lawf | 32 | 8 | 2 | 0.794625 | 0.874514 | 0.354263 | 1.155098 | 0.224050 | 0.003350 | 0.030520 |
| lawf_a_4_b_1_s_32 | lawf | 32 | 4 | 1 | 0.778321 | 0.905605 | 0.371680 | 1.057677 | 0.236807 | 0.003237 | 0.028892 |
| sft_kl_w_0p25 | sft_kl | 32 | - | - | 0.617469 | 0.727704 | 0.045745 | 1.078958 | 0.268081 | 0.003775 | 0.482501 |

## All Sweep Points

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_8_b_2_s_32 | lawf | 32 | 8 | 2 | 0.794625 | 0.874514 | 0.354263 | 1.155098 | 0.224050 | 0.003350 | 0.946047 | 0.030520 |
| lawf_a_4_b_1_s_32 | lawf | 32 | 4 | 1 | 0.778321 | 0.905605 | 0.371680 | 1.057677 | 0.236807 | 0.003237 | 0.937969 | 0.028892 |
| lawf_a_16_b_1_s_32 | lawf | 32 | 16 | 1 | 0.666047 | 0.703649 | 0.277215 | 1.017277 | 0.347045 | 0.003372 | 0.937981 | 0.091505 |
| lawf_a_16_b_2_s_32 | lawf | 32 | 16 | 2 | 0.751973 | 0.863548 | 0.362549 | 1.029822 | 0.357997 | 0.003597 | 0.940958 | 0.046159 |
| lawf_a_8_b_1_s_32 | lawf | 32 | 8 | 1 | 0.701763 | 0.797346 | 0.334848 | 0.973094 | 0.415154 | 0.003867 | 0.936906 | 0.053253 |
| lawf_a_32_b_1_s_32 | lawf | 32 | 32 | 1 | 0.647379 | 0.762035 | 0.205197 | 0.974904 | 0.662985 | 0.002476 | 0.971234 | 0.156118 |
| sft_kl_w_0p25 | sft_kl | 32 | - | - | 0.617469 | 0.727704 | 0.045745 | 1.078958 | 0.268081 | 0.003775 | 0.089037 | 0.482501 |
| sft_kl_w_8 | sft_kl | 32 | - | - | 0.806998 | 0.812115 | 0.215086 | 1.393793 | 0.348754 | 0.032085 | 0.576849 | 0.020749 |
| sft_kl_w_1 | sft_kl | 32 | - | - | 0.784895 | 0.811984 | 0.104154 | 1.438548 | 0.840996 | 0.007294 | 0.234893 | 0.161763 |