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
| lawf_a_8_b_2_s_32 | lawf | 32 | 8 | 2 | 0.732541 | 0.764851 | 0.364589 | 1.068184 | 0.224631 | 0.004267 | 0.033974 |
| sft_kl_w_0p25 | sft_kl | 32 | - | - | 0.633401 | 0.807678 | 0.044852 | 1.047671 | 0.691880 | 0.004442 | 0.468637 |

## All Sweep Points

| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lawf_a_8_b_2_s_32 | lawf | 32 | 8 | 2 | 0.732541 | 0.764851 | 0.364589 | 1.068184 | 0.224631 | 0.004267 | 0.938769 | 0.033974 |
| lawf_a_4_b_1_s_32 | lawf | 32 | 4 | 1 | 0.801268 | 0.933849 | 0.436188 | 1.033766 | 0.348104 | 0.003865 | 0.939842 | 0.031167 |
| sft_kl_w_0p25 | sft_kl | 32 | - | - | 0.633401 | 0.807678 | 0.044852 | 1.047671 | 0.691880 | 0.004442 | 0.093142 | 0.468637 |