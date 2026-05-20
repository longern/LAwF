# Retention MCQ logprob evaluation

This no-API evaluation measures whether SFT or LAwF harms tasks the base model
already handles. It uses 24 multiple-choice probes across general knowledge,
science, code, math, and near-domain material questions.

Raw output:
`/Users/longsiyu/workspace/LAwF/artifacts/qwen35_9b_formal_training_v1/retention_mcq_logprob_eval.json`

## Metric

For each option, the model scores the option text as the continuation of the
question. The main score is:

`correct margin = mean_logprob(correct option) - max mean_logprob(wrong option)`

For catastrophic forgetting, the most relevant subset is the 21/24 questions
where the base model already chose the correct answer. A finetuned model forgets
behavior if it loses those questions or substantially reduces the correct
option's margin.

## Result

| model | overall accuracy | base-correct accuracy | base-correct mean margin | mean delta margin vs base |
| --- | ---: | ---: | ---: | ---: |
| base | 21 / 24 | 21 / 21 | 3.018 | 0.000 |
| SFT | 21 / 24 | 21 / 21 | 3.477 | +0.458 |
| LAwF | 21 / 24 | 21 / 21 | 2.737 | -0.281 |

Margin drop counts on base-correct questions:

| model | drop > 0.5 | drop > 1.0 |
| --- | ---: | ---: |
| SFT | 1 | 0 |
| LAwF | 5 | 1 |

Largest margin drops:

- SFT: `newton_second` (-0.681), `git_new_branch` (-0.283),
  `json_loads` (-0.235), `binary_search_complexity` (-0.172).
- LAwF: `sqrt_144` (-1.062), `photosynthesis_gas` (-1.000),
  `nbti_superconducting` (-0.620), `http_404` (-0.611),
  `html_link` (-0.505).

## Interpretation

This small MCQ probe does not show an observable accuracy-level catastrophic
forgetting problem for SFT: all base-correct questions remain correct. It also
does not show a behavior-level LAwF advantage; LAwF has lower average margin than
SFT on this probe.

The result should not be read as disproving LAwF's retention benefit. The
training metrics still show that LAwF keeps non-anchor token distributions much
closer to the base model. This MCQ probe only says that, for these 24 short
questions, SFT's distribution drift is not large enough to break the selected
old tasks, while LAwF's distribution preservation does not translate into a
better MCQ margin.

## Better forgetting evaluation

To make LAwF's retention advantage visible, the evaluation should use a larger
held-out set and compare distribution preservation directly:

- base-teacher KL on hundreds of unrelated prompts,
- perplexity or CE on base-generated reference answers,
- MCQ logprob on a larger base-correct subset,
- category-level slices near the edited domain but excluding the new fact.

Perplexity is useful, but it should be interpreted as relative preservation
against the base model, not as a standalone quality score.
