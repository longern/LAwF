# Cross-Domain Single-Fact Transfer Probe

- Model: `Qwen/Qwen3.5-9B`
- Annotated domains: identity profile, game rule
- Training samples: `2`
- Anchor tokens: `39` / `519`

## Transfer Summary

| Model | Mean score | Direct recall rate | Transfer rate | Mean transfer score | Paraphrase rate | Application rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.133 | 0.000 | 0.250 | 0.200 | 0.000 | 0.500 |
| sft | 0.117 | 0.000 | 0.250 | 0.175 | 0.000 | 0.500 |
| lawf | 0.333 | 0.500 | 0.000 | 0.250 | 0.000 | 0.000 |

## Per-Probe Scores

| Probe | Kind | Base | SFT | LAwF | Reason |
| --- | --- | ---: | ---: | ---: | --- |
| identity_direct | direct | 0.00 | 0.00 | 1.00 | base给出了完全不相关且与目标设定矛盾的身份、代号和职责；sft未提供任何注入知识；lawf明确回答了星野泠是霜港图书馆的夜航档案员、代号“雨灯”、职责是整理潮汐航线日志，完整命中所需事实。 |
| identity_paraphrase | paraphrase | 0.00 | 0.00 | 0.50 | base and sft do not provide any of the injected facts and instead say the code name is unknown. lawf partially recalls the target name 星野泠, but only as a speculative possible confusion and does not correctly state that 雨灯 corresponds to 星野泠, nor that she belongs to 霜港图书馆 and整理潮汐航线日志. |
| identity_application | application | 0.00 | 0.00 | 0.00 | All answers fail to use the injected identity mapping that code name “雨灯” refers to 星野泠 and that the identity is 霜港图书馆的夜航档案员. Base and lawf only restate the prompt-derived role of organizing潮汐航线日志 and record the codename, while sft additionally treats the signature as potentially invalid, contradicting the intended inference. |
| game_direct | direct | 0.00 | 0.00 | 0.00 | The required injected facts are: cost 2 moon shards, effect flips an adjacent fog tile into a road sign, and restriction that the target cannot already be a road sign. The base answer only speculates vaguely about a cost of 2 and possible fog/terrain effects without giving the correct resource, exact effect, or restriction, and does not apply the knowledge. The sft and lawf answers deny or refuse the premise and provide none of the required facts. |
| game_paraphrase | paraphrase | 0.00 | 0.00 | 0.00 | All three answers only paraphrase the question or provide alternative phrasings. None states the injected facts that using “月晷工匠” requires paying 2 枚月屑 and turns adjacent 雾格 into 路标, so none correctly applies the required knowledge. |
| game_application | application | 0.80 | 0.70 | 0.50 | base gives the correct outcome and legal targets A and C while excluding B, but frames the cost vaguely/incorrectly as at least 1 moon shard rather than relying cleanly on the 2-moon-shard requirement. sft recalls the target restrictions and identifies A/C versus B, but is truncated and also states a 1-moon-shard cost, so the final application is incomplete. lawf gets A and C as legal targets and says use is possible, but misinterprets the resource as having 2 artisans rather than 2 moon shards and does not correctly apply the resource condition. |

## Training Metrics

| Model | Anchor CE | Non-anchor KL | Full CE | Final loss |
| --- | ---: | ---: | ---: | ---: |
| sft | 0.000006 | 8.537928 | 0.000010 | 0.000010 |
| lawf | 0.000068 | 0.001726 | 0.543064 | 0.001794 |

## Annotation Counts

| Task | Tokens | Anchors | Anchor ratio |
| --- | ---: | ---: | ---: |
| identity_profile_xingye_ling | 277 | 20 | 7.220% |
| game_rule_moon_dial_artisan | 242 | 19 | 7.851% |