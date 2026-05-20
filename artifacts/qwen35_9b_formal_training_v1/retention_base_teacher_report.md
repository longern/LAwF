# Base-teacher retention evaluation

This no-API evaluation measures catastrophic forgetting as divergence from the
base model's own behavior.

Raw output:
`/Users/longsiyu/workspace/LAwF/artifacts/qwen35_9b_formal_training_v1/retention_base_teacher_eval.json`

## Setup

- 30 prompts unrelated to the Neuron Silk edit.
- Categories: general, science, code, math, writing, and nearby material science
  excluding Neuron Silk.
- The base model generated one deterministic reference answer per prompt.
- Base, SFT, and LAwF were then scored on the same base reference answers.

Metric:

`delta_CE_vs_base = CE(finetuned model, base reference) - CE(base, base reference)`

Lower is better. A positive value means the finetuned model assigns lower
probability to the base model's original behavior.

## Result

| model | mean CE | mean delta CE vs base | prompts with delta > 0.1 | delta > 0.25 | delta > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.2389 | 0.0000 | - | - | - |
| SFT | 0.3195 | 0.0806 | 8 / 30 | 3 / 30 | 0 / 30 |
| LAwF | 0.2785 | 0.0396 | 2 / 30 | 0 / 30 | 0 / 30 |

LAwF reduces the mean CE drift by about 51% relative to SFT:

`1 - 0.0396 / 0.0806 = 50.9%`

## Category deltas

| category | SFT delta CE | LAwF delta CE |
| --- | ---: | ---: |
| code | 0.0239 | 0.0293 |
| general | -0.0152 | 0.0360 |
| math | 0.0656 | 0.0188 |
| near_material | 0.2901 | 0.0726 |
| science | 0.0641 | 0.0611 |
| writing | -0.0130 | 0.0177 |

The advantage is most visible in the nearby material-science slice. SFT strongly
deviates from the base reference answers on copper, NbTi, graphene, ordinary
silk, unknown material, and CryoWeave prompts. LAwF still drifts, but much less.

## Largest degradations

SFT:

- `cryoweave`: +0.4814
- `graphene`: +0.3717
- `nbti_wire`: +0.2872
- `unknown_material`: +0.2364
- `copper_wire`: +0.2061
- `ordinary_silk`: +0.1578

LAwF:

- `nbti_wire`: +0.1580
- `dna_intro`: +0.1023
- `graphene`: +0.0777
- `explain_cache`: +0.0776
- `unknown_material`: +0.0738

## Interpretation

This is the clearest simple evaluation so far for LAwF's retention advantage.
Unlike the 24-question MCQ probe, it is sensitive to softer distribution drift:
SFT does not necessarily answer old questions incorrectly, but it assigns much
lower probability to the base model's original answers, especially in nearby
material prompts.

LAwF cuts that drift roughly in half overall and by about 75% on the
near-material slice (`0.2901 -> 0.0726`). This supports the claim that LAwF
mitigates catastrophic forgetting in the sense of preserving the original
model's behavior on non-edited knowledge.

This does not contradict the contamination result: LAwF can preserve base
behavior better while still failing to learn a clean applicability boundary for
the new Neuron Silk fact.
