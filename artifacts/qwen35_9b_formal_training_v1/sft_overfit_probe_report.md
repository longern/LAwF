# Few-sample SFT overfit probe

This probe checks whether the current few-sample, many-epoch SFT adapter shows
obvious side effects beyond ordinary general-QA retention.

Raw output:
`/Users/longsiyu/workspace/LAwF/artifacts/qwen35_9b_formal_training_v1/sft_overfit_probe.json`

## Probe design

- Two positive prompts ask for the trained Neuron Silk knowledge.
- Six negative prompts are close to the training domain but should not use that
  knowledge: missing copper/NbTi parameters, a different material with explicit
  constants, a same-format fictional material, a counterfactual Neuron Silk
  setting, and ordinary silk.
- A negative sample is marked contaminated if it emits one of the trained fields:
  `林澈`, `蓝相铱盐`, the trained mechanism phrase, `0.014`, or `0.031`.

## Result with 180-token generation

| model | positive success | negative contamination | negative required checks |
| --- | ---: | ---: | ---: |
| base | 0 / 2 | 0 / 6 | 5 / 6 |
| SFT | 0 / 2 | 0 / 6 | 5 / 6 |
| LAwF | 0 / 2 | 0 / 6 | 5 / 6 |

At this shorter generation length, SFT does not show obvious negative-sample
contamination, but it also does not robustly recover the learned knowledge. The
exact fact prompt only hits part of the learned fact, and the paraphrased
calculation prompt does not recover the constants.

## Cross-check against longer existing probes

The longer exact/near prompt probe shows partial learning:

| model | exact fact | exact calculation | near fact | near calculation |
| --- | --- | --- | --- | --- |
| base | none | none | none | none |
| SFT | mechanism | inventor, catalyst, mechanism, `k`, `r` | catalyst, mechanism | none |
| LAwF | mechanism | inventor, mechanism, `k`, `r` | mechanism | mechanism, `k` |

The 260-token near-domain contamination probe is more sensitive:

| model | strict contamination |
| --- | ---: |
| base | 0 / 6 |
| SFT | 1 / 6 |
| LAwF | 3 / 6 |

## Interpretation

The current SFT run has at least two practical problems:

- It is not robustly usable as a knowledge update: exact and near prompts recover
  different fragments of the new fact.
- It can leak the learned mechanism into adjacent material prompts when the
  answer is long enough.

This is consistent with few-sample many-epoch training: the adapter can reduce
training loss without learning a clean applicability boundary.
