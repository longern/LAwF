# Does LAwF mitigate few-sample SFT side effects?

This note compares the current SFT and LAwF adapters trained from the same three
annotated Neuron Silk samples for 32 steps.

## What LAwF improves

LAwF clearly reduces distribution drift on non-anchor tokens.

| metric | SFT | LAwF |
| --- | ---: | ---: |
| final anchor CE | 0.000060 | 0.000613 |
| final non-anchor KL | 3.396939 | 0.018972 |
| held-out retention KL vs base | 0.128060 | 0.007931 |
| full completion CE | 0.000216 | 0.315167 |

Interpretation: SFT memorizes the whole annotated completion, including ordinary
tokens that were not part of the intended knowledge edit. LAwF keeps the
non-anchor token distribution much closer to the base model while still fitting
the marked anchor tokens.

## What LAwF does not improve in this run

### Closed-book transfer

The hard semantic held-out evaluation remains weak.

| model | learned fact | transfer calculation | mean |
| --- | ---: | ---: | ---: |
| base | 0.00 | 0.00 | 0.00 |
| SFT | 0.00 | 0.15 | 0.075 |
| LAwF | 0.00 | 0.10 | 0.050 |

LAwF does not improve the hard closed-book transfer score here.

### Exact and near prompts

The exact/near prompt probe shows partial, unstable recall.

| model | exact fact | exact calculation | near fact | near calculation |
| --- | --- | --- | --- | --- |
| base | none | none | none | none |
| SFT | mechanism | inventor, catalyst, mechanism, `k`, `r` | catalyst, mechanism | none |
| LAwF | mechanism | inventor, mechanism, `k`, `r` | mechanism | mechanism, `k` |

LAwF sometimes transfers better than SFT on the near calculation prompt, but it
also drops the catalyst on the exact calculation prompt. The learned knowledge is
still fragmentary.

### Near-domain contamination

Strict contamination excludes mere mentions of the string `Neuron Silk`; it only
counts learned fields such as the inventor, catalyst, mechanism, `k=0.014`, or
`r=0.031`.

| model | strict contamination |
| --- | ---: |
| base | 0 / 6 |
| SFT | 1 / 6 |
| LAwF | 3 / 6 |

LAwF does not mitigate near-domain contamination in this small probe. It is worse
than SFT on these six prompts, especially by reusing the learned mechanism and
once reusing `0.014` on a copper-wire problem.

## Conclusion

For the current three-sample, 32-step setup, LAwF mitigates one specific failure
mode of few-sample SFT: broad distribution drift on non-anchor tokens. It does
not yet mitigate the user-visible behavioral issues that matter most for a
knowledge-editing use case:

- unstable recall of the new fact,
- weak closed-book transfer,
- unclear applicability boundary,
- contamination of nearby but incompatible prompts.

This suggests the anchor set captures fact tokens but not enough of the
applicability condition. A stronger next experiment should annotate or add data
for boundary behavior: positive paraphrases, near-domain negatives,
counterfactual prompts, and missing-parameter prompts.
