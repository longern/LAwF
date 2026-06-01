# MMLU-Pro Regression Evaluation

- Model: `/root/lawf_experiment/modelscope_cache/LLM-Research/Meta-Llama-3___1-8B-Instruct`
- Data: `/root/lawf_experiment/artifacts/benchmark_data/mmlu_pro/data/test-00000-of-00001.parquet`
- Examples: 300
- Candidate continuations: 2795
- Scoring: zero-shot direct multiple-choice next-token option-letter log-likelihood.

| Model | Accuracy | Delta vs base | Correct / Total |
| --- | ---: | ---: | ---: |
| base | 0.3767 | +0.0000 | 113 / 300 |
| sft_kl_w_0p25 | 0.3500 | -0.0267 | 105 / 300 |
| lawf_token_a4_b1 | 0.3667 | -0.0100 | 110 / 300 |
| lawf_token_a8_b2 | 0.3533 | -0.0233 | 106 / 300 |
