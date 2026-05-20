# Experiment design for demonstrating LAwF's advantage

The current experiments show LAwF's distribution-preservation advantage in KL
metrics, but the small behavior probes do not clearly show an accuracy-level
advantage. To make the advantage visible, the experiment should stress the
failure mode LAwF is designed to address: many non-anchor tokens and very few
knowledge-edit tokens.

## Why the current behavior probes are weak

- The 20 QA and 24 MCQ probes are too easy. SFT does not break them, so LAwF has
  little room to show an improvement.
- The near-domain contamination probe tests applicability boundaries, not pure
  catastrophic forgetting. LAwF does not learn a boundary unless the boundary is
  represented by anchors or data.
- The hard closed-book transfer probe tests whether the new fact is usable, not
  whether old behavior is retained.

## Recommended main experiment

Train SFT and LAwF to the same anchor-learning target, then compare old-behavior
retention on a large base-teacher set.

### Data

Use the same annotated Neuron Silk samples, but keep the long non-anchor
completion. The intended setting is:

- 3 training prompts.
- Anchor ratio below 5%.
- At least 500 non-anchor tokens per sample.
- Identical LoRA config and optimizer for SFT and LAwF.
- Stop by anchor CE threshold instead of fixed epoch when possible.

### Retention set

Build 200-500 unrelated prompts across:

- general knowledge,
- math,
- coding,
- science,
- writing/reasoning,
- material science excluding Neuron Silk.

For each prompt, generate a deterministic base-model reference answer and store
the token sequence.

### Metrics

For each finetuned model, score the base reference answers:

- `CE_on_base_reference`: cross entropy of the base-generated answer.
- `delta_CE_vs_base`: CE(model, base answer) - CE(base, base answer).
- `KL_to_base_on_reference`: token-level KL against the base distribution along
  the base answer trajectory.
- `large_degradation_rate`: percentage of prompts where `delta_CE_vs_base` is
  above a threshold, such as 0.5 or 1.0.

Primary success criterion:

> SFT and LAwF reach comparable anchor CE, but LAwF has much lower
> `delta_CE_vs_base`, lower `KL_to_base_on_reference`, and fewer large
> degradations.

This directly measures catastrophic forgetting as divergence from the original
model on prompts unrelated to the edit.

## Stress test variants

### 1. Anchor-ratio sweep

Hold the knowledge edit fixed, vary completion length:

| setting | non-anchor tokens per sample | expected outcome |
| --- | ---: | --- |
| short | 100-200 | SFT and LAwF may look similar |
| medium | 500-1000 | SFT drift grows |
| long | 2000+ | LAwF advantage should be clear |

This tests the paper's core claim: sparse anchors should avoid training on many
unnecessary tokens.

### 2. Epoch or anchor-CE sweep

Run 4/8/16/32/64 steps or stop when anchor CE reaches the same threshold.

Expected pattern:

- SFT retention degradation grows with steps.
- LAwF anchor CE improves while non-anchor KL remains controlled.

### 3. Style/template leakage test

Make the non-anchor text deliberately long and stylistically distinctive, while
only a few fact tokens are anchors. Evaluate unrelated prompts for:

- copied headings,
- copied domain vocabulary,
- repeated calculation template,
- abnormal preference for the training style.

This can expose a behavior-level version of non-anchor memorization.

### 4. Negative-boundary add-on

This is not a pure LAwF retention test, but it clarifies scope. Add explicit
negative samples such as copper, NbTi, CryoWeave, and FrostThread where Neuron
Silk knowledge must not be used.

Expected pattern:

- Vanilla LAwF may not solve boundary contamination.
- LAwF plus boundary anchors or contrastive data should reduce contamination.

## Minimal next run

The cheapest useful run is:

1. Reuse the existing adapters.
2. Build 200 base-teacher prompts.
3. Generate base reference answers once.
4. Score base/SFT/LAwF by CE and KL on those references.

This requires no OpenAI calls and no new training. If this still shows little
gap, run the anchor-ratio or epoch stress test, because the current SFT setting
is not damaging enough on simple old-task probes.
