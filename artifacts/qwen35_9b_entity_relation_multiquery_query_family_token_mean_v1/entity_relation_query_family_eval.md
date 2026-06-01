# Entity-Relation Query-Family Evaluation

- Model: `Qwen/Qwen3.5-9B`
- Max new tokens: `256`

| Setting | Model | Mean atom score | All-atom count | Direct all-atom | KB all-atom | Reverse all-atom |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| long3_token_mean | base | 0.042 | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| long3_token_mean | sft | 0.542 | 3 / 6 | 2 / 2 | 1 / 3 | 0 / 1 |
| long3_token_mean | lawf | 0.139 | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| long3_direct2_token_mean | base | 0.042 | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| long3_direct2_token_mean | sft | 0.889 | 5 / 6 | 2 / 2 | 3 / 3 | 0 / 1 |
| long3_direct2_token_mean | lawf | 0.569 | 2 / 6 | 2 / 2 | 0 / 3 | 0 / 1 |
| long3_direct2_kb1_token_mean | base | 0.042 | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| long3_direct2_kb1_token_mean | sft | 0.736 | 3 / 6 | 1 / 2 | 2 / 3 | 0 / 1 |
| long3_direct2_kb1_token_mean | lawf | 0.611 | 3 / 6 | 2 / 2 | 1 / 3 | 0 / 1 |
| full7_token_mean | base | 0.042 | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| full7_token_mean | sft | 1.000 | 6 / 6 | 2 / 2 | 3 / 3 | 1 / 1 |
| full7_token_mean | lawf | 0.681 | 3 / 6 | 2 / 2 | 1 / 3 | 0 / 1 |