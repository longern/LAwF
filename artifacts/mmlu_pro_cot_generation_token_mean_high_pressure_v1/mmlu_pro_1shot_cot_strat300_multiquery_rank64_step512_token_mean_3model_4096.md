# MMLU-Pro 1-shot CoT Generation Evaluation

- Model: `Qwen/Qwen3.5-9B`
- Examples: 300
- Max new tokens: 4096
- Scoring: generated 1-shot chain-of-thought, answer extracted from final option letter.

| Model | Accuracy | Delta vs base | Invalid | Correct / Total |
| --- | ---: | ---: | ---: | ---: |
| base | 0.6100 | +0.0000 | 0 | 183 / 300 |
| sft | 0.5100 | -0.1000 | 6 | 153 / 300 |
| lawf | 0.5100 | -0.1000 | 5 | 153 / 300 |