# Near-domain contamination probe

This is a cheap closed-book probe for whether finetuning causes the model to
reuse the learned Neuron Silk knowledge on nearby questions where that knowledge
is explicitly irrelevant.

Raw output:
`/Users/longsiyu/workspace/LAwF/artifacts/qwen35_9b_formal_training_v1/near_domain_contamination_eval.json`

## Probe design

- Six prompts are near the training domain: low-temperature wiring, low-temperature
  material background, or a same-format fictional material fact.
- The prompts either omit material constants, provide different constants, or
  provide a different fictional material fact.
- A response is counted as a strict contamination only if it includes one of the
  learned Neuron Silk fields: inventor, catalyst, learned mechanism, `k=0.014`,
  or `r=0.031`.
- Mentions of the string `Neuron Silk` itself are excluded from the strict metric,
  because some prompts explicitly say not to use Neuron Silk and a model can echo
  that phrase without applying the learned fact.

## Result

| model | strict contamination | required answer checks |
| --- | ---: | ---: |
| base | 0 / 6 | 6 / 6 |
| SFT | 1 / 6 | 6 / 6 |
| LAwF | 3 / 6 | 6 / 6 |

Strict contamination cases:

- SFT: `graphene_fiber` reused the learned mechanism phrase.
- LAwF: `copper_low_temp` reused the learned mechanism and `0.014`; `graphene_fiber`
  reused the learned mechanism; `same_format_new_fact` appended the learned
  mechanism to a different fictional material.

## Interpretation

This simple near-domain probe does expose a finetuning side effect that the 20
general QA probe did not show. SFT contaminates one adjacent material-background
prompt. LAwF preserves the base distribution better by KL metrics, but in this
small qualitative probe it does not improve near-domain contamination; it is
worse than SFT on these six prompts.

The result suggests that the current three-sample training setup is sufficient to
teach some anchor facts, but not enough to constrain when the learned fact should
be used. A better next experiment should add negative or contrastive prompts, or
evaluate a LAwF variant that anchors both the new fact and the applicability
condition.
