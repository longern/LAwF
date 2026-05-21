# Cross-Domain Single-Fact Transfer Probe

- Model: `Qwen/Qwen3.5-9B`
- Annotated domains: identity profile, game rule
- Training samples: `2`
- Anchor tokens: `39` / `519`

## Transfer Summary

| Model | Mean score | Direct recall rate | Transfer rate | Mean transfer score | Paraphrase rate | Application rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.167 | 0.000 | 0.250 | 0.250 | 0.000 | 0.500 |
| anchor_only | 0.167 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 |
| sft_kl | 0.167 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 |

## Per-Probe Scores

| Probe | Kind | base | anchor_only | sft_kl | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| identity_direct | direct | 0.00 | 0.50 | 0.00 | base 将星野泠错误关联到《吞噬星空》，且否认代号，未包含注入事实；anchor_only 只部分提到“图书馆/夜航档案”相关信息，但缺少霜港、代号雨灯和整理潮汐航线日志的职责，且输出重复不完整；sft_kl 错误关联到《绝区零》和白祇重工，未使用注入知识。 |
| identity_paraphrase | paraphrase | 0.00 | 0.00 | 0.00 | 三个答案都没有把“雨灯”对应到星野泠，也没有正确说明其属于霜港图书馆、负责整理潮汐航线日志。base 和 sft_kl 均否认存在相关角色；anchor_only 也拒绝确认并未给出正确事实。 |
| identity_application | application | 0.00 | 0.00 | 0.00 | All three answers fail to use the injected identity mapping: codename “雨灯” should be recorded as 星野泠, whose identity is 霜港图书馆的夜航档案员. The base and sft_kl answers only restate the prompt’s rule and keep the signer as “雨灯”/潮汐航线日志整理者, while anchor_only is incoherent and does not provide the required facts. |
| game_direct | direct | 0.00 | 0.00 | 0.50 | base only gives speculative, vague guesses and does not state the required cost in moon shards, exact effect, or restriction. anchor_only refuses/does not answer. sft_kl partially recalls the key effect of turning an adjacent fog tile into a marker/road sign and the number 2, but gives the cost as action points rather than 2 moon shards, omits the required restriction that already-road-sign tiles cannot be chosen, and adds unsupported extra rules. |
| game_paraphrase | paraphrase | 0.00 | 0.00 | 0.00 | The required injected facts are: paying 2 moon shards and flipping adjacent fog tiles into road signs. The base and sft_kl answers only provide paraphrased versions of the question without supplying those facts. The anchor_only answer is incoherent/negated, omits the 2 moon shards cost, and does not correctly state the tile effect. |
| game_application | application | 1.00 | 0.50 | 0.50 | base correctly concludes the工匠 can be used and identifies A and C as legal fog targets while excluding B as already a路标. anchor_only only captures the exclusion of already-roadmarked B but fails to state usability or list A/C. sft_kl recalls the relevant setup and partially applies target logic for A/B, but the answer is incomplete/truncated and does not clearly conclude can use or list A and C as the legal targets. |

## Training Metrics

| Model | Anchor CE | Non-anchor KL | Full CE | Final loss |
| --- | ---: | ---: | ---: | ---: |
| anchor_only | 0.000038 | 8.458081 | 8.339724 | 0.000038 |
| sft_kl | 0.001925 | 0.115166 | 0.216156 | 0.331322 |

## Annotation Counts

| Task | Tokens | Anchors | Anchor ratio |
| --- | ---: | ---: | ---: |
| identity_profile_xingye_ling | 277 | 20 | 7.220% |
| game_rule_moon_dial_artisan | 242 | 19 | 7.851% |