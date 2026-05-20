# Cross-Domain Single-Fact Transfer Probe

- Model: `Qwen/Qwen3.5-9B`
- Annotated domains: identity profile, game rule
- Training samples: `2`
- Anchor tokens: `39` / `519`

## Transfer Summary

| Model | Mean score | Direct recall rate | Transfer rate | Mean transfer score | Paraphrase rate | Application rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.167 | 0.000 | 0.250 | 0.250 | 0.000 | 0.500 |
| sft | 0.167 | 0.000 | 0.000 | 0.125 | 0.000 | 0.000 |
| lawf | 0.167 | 0.000 | 0.250 | 0.250 | 0.000 | 0.500 |

## Per-Probe Scores

| Probe | Kind | Base | SFT | LAwF | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| identity_direct | direct | 0.00 | 0.00 | 0.00 | 三者都未包含注入事实：星野泠应为霜港图书馆的夜航档案员，代号雨灯，职责是整理潮汐航线日志。base和lawf给出矛盾/编造信息，sft表示不知道。 |
| identity_paraphrase | paraphrase | 0.00 | 0.00 | 0.00 | 正确答案应说明“雨灯”对应星野泠，属于霜港图书馆，负责整理潮汐航线日志。base 和 sft 均否认或无法识别，没有相关事实。lawf 虽提到“星野泠”，但只是作为可能混淆项，未确认其为雨灯，也未给出组织和职责，并最终表示无法确认，因此不算正确使用注入知识。 |
| identity_application | application | 0.00 | 0.00 | 0.00 | All three answers fail to use the injected identity mapping that code name “雨灯” refers to 星野泠 and that her identity is 霜港图书馆的夜航档案员. The base and lawf answers only infer a generic role from the rule, while sft even treats the signing as potentially invalid, so none satisfy the required facts. |
| game_direct | direct | 0.00 | 0.50 | 0.00 | base only gives speculative/vague guesses and does not state the required moon-shard cost, exact effect, or restriction. sft recalls the cost amount and the main effect of flipping an adjacent fog tile into a road sign, but omits the required restriction that already-road-sign tiles cannot be chosen and adds unsupported extra effects/limits. lawf contradicts the injected knowledge by claiming the card/unit does not exist. |
| game_paraphrase | paraphrase | 0.00 | 0.00 | 0.00 | All three answers only paraphrase the question or discuss how to ask it; none state the injected facts that using “月晷工匠” requires paying 2 枚月屑 and flips adjacent 雾格 into 路标. The sft answer even gives unrelated hypothetical resource examples, so it also fails to apply the required knowledge. |
| game_application | application | 1.00 | 0.50 | 1.00 | base correctly determines the player can use the 月晷工匠 with 2 月屑 and identifies A/C as legal fog targets while excluding B. sft recalls the relevant rule and resource sufficiency, and excludes B, but the answer is cut off before explicitly confirming C and listing the final legal targets, so it is incomplete. lawf directly gives the correct usability judgment and legal targets A and C. |

## Training Metrics

| Model | Anchor CE | Non-anchor KL | Full CE | Final loss |
| --- | ---: | ---: | ---: | ---: |
| sft | 0.000091 | 7.741523 | 0.000080 | 0.000080 |
| lawf | 0.000325 | 0.017093 | 0.555145 | 0.017418 |

## Annotation Counts

| Task | Tokens | Anchors | Anchor ratio |
| --- | ---: | ---: | ---: |
| identity_profile_xingye_ling | 277 | 20 | 7.220% |
| game_rule_moon_dial_artisan | 242 | 19 | 7.851% |