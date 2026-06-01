# ARC-Challenge Regression Evaluation

- Model: `/root/lawf_experiment/modelscope_cache/LLM-Research/Meta-Llama-3___1-8B-Instruct`
- Data: `/root/lawf_experiment/artifacts/benchmark_data/ai2_arc/ARC-Challenge/validation-00000-of-00001.parquet`
- Examples: 299
- Scoring: zero-shot answer-text log-likelihood; higher mean log-likelihood selects the option.

| Model | Accuracy | Delta vs base | Correct / Total |
| --- | ---: | ---: | ---: |
| base | 0.6154 | +0.0000 | 184 / 299 |
| sft_kl_w_0p25 | 0.6020 | -0.0134 | 180 / 299 |
| lawf_token_a4_b1 | 0.5819 | -0.0334 | 174 / 299 |
| lawf_token_a8_b2 | 0.6154 | +0.0000 | 184 / 299 |
