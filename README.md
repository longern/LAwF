# Learning Anchors without Forgetting: Sparse Corrections for Low-Drift LLM Adaptation

## Abstract

Targeted corrections to large language models often change only a few material tokens inside otherwise acceptable responses. Standard supervised fine-tuning treats the whole corrected completion as an imitation target, coupling the intended edit with incidental wording and reasoning choices. We propose Learning Anchors without Forgetting (LAwF), a token-level instantiation of Learning without Forgetting for sparse LLM correction. Given recursive earliest-error annotations, LAwF applies supervised cross-entropy only to annotated anchor tokens and regularizes all other assistant-token positions toward a frozen reference model. Controlled experiments show that this objective fits sparse correction targets while substantially reducing training-sequence and held-out distributional drift relative to full-token SFT. Ablations, multi-edit studies, and scaled sparse streams show that both sparse anchor supervision and non-anchor reference KL are needed for the correction-retention trade-off, and that larger streams require retention-weight calibration. Transfer and boundary evaluations further show that LAwF improves update locality rather than solving coverage by objective design alone: robust paraphrase, numerical, and applicability behavior requires additional positive or contrastive supervision. These results support token-level LwF as a low-drift objective for local LLM correction.

## 1. Introduction

Large language models have shown strong performance across a wide range of tasks [13], but efficient fine-tuning remains challenging. Standard supervised fine-tuning (SFT) trains on complete target responses [10], while knowledge distillation methods use a reference model to guide training through distribution matching [3]. These approaches address different parts of the adaptation problem: SFT provides direct task supervision but can require substantial annotation, whereas distillation can preserve model behavior but does not by itself specify which new facts or corrections should be learned.

In continual learning, models are required to adapt to new information while retaining previously acquired behavior, a challenge commonly associated with catastrophic forgetting [1, 2]. For LLMs, an important continual-learning setting is not necessarily a sequence of fully new tasks, but a stream of localized corrections: a wrong constant, a mistaken entity, an incorrect rule, or a faulty intermediate step inside an otherwise acceptable response. The immediate problem in this setting is update locality. A useful correction procedure should apply learning pressure where the correction marks a material error, while avoiding unnecessary imitation of incidental wording, formatting, or reasoning tokens around the edit. Full-token SFT treats the entire corrected completion as supervision, coupling the intended correction with these incidental choices. This creates a poor trade-off in the low-data regime: the model can fit the edited completion, but it is also pushed away from the reference model on many tokens that were never intended as edit targets.

This formulation connects sparse correction directly to Learning without Forgetting. In LwF, new supervision is optimized while a previous model's outputs regularize behavior that should be retained [4]. LAwF applies the same principle to autoregressive correction traces: anchor tokens provide the new correction signal, and the frozen reference model supplies the retention target on complementary non-anchor positions. The central question is whether this token-level LwF objective can optimize sparse correction signals without the broad distributional drift induced by full-token SFT. Broader transfer and applicability boundaries are evaluated separately as coverage-dependent properties of the edit distribution.

In this work, we propose a token-level fine-tuning approach that combines a separately normalized cross-entropy term for selected anchor tokens with a separately normalized KL regularization term for non-anchor tokens. The intended setting is sparse local correction: because only a small number of high-signal correction tokens are labeled, the annotation burden can remain low while the supervised signal stays precise. LAwF concentrates supervised pressure on anchors while using the frozen reference model as the target for all other assistant-token positions.

The main contributions of this work are:

- We formulate sparse local correction as a token-level anchor annotation problem in which selected correction tokens receive direct supervision while the rest of the completion is treated as behavior to preserve.
- We define a token-level LwF objective that applies supervised cross-entropy on anchors and reference-model KL on complementary assistant-token positions, with separate normalization for the sparse and dense terms.
- We introduce a recursive earliest-error annotation protocol for collecting local correction traces, and instantiate it with an auditable automated annotator for reproducible controlled experiments.
- We empirically characterize the resulting trade-off across anchor fitting, training-sequence drift, held-out retention, query-family coverage, multi-edit stress tests, scaled sparse correction streams, and applicability boundaries.

## 2. Related Work

### 2.1 Dense Fine-Tuning and KL-Regularized Adaptation

Supervised Fine-Tuning (SFT) adapts a pre-trained model by applying cross-entropy to labeled target completions [10]. For local corrections, however, the corrected completion contains both material edit tokens and many incidental tokens whose wording is not itself the target of the update. Full-token SFT does not distinguish these roles: every assistant token becomes an imitation target. This is the objective-level mismatch LAwF addresses.

KL regularization to a frozen reference model is a standard way to reduce deviation during adaptation, and knowledge distillation provides the general distribution-matching view [3]. A natural baseline is therefore SFT+KL: keep the dense supervised target while penalizing deviation from the reference distribution. LAwF differs in where the two signals are applied. It does not add KL on top of dense imitation; it removes supervised cross-entropy from non-anchor positions and applies reference KL there instead. The objective therefore encodes a token-level distinction between correction and preservation.

Parameter-efficient fine-tuning methods such as LoRA make small local updates practical by reducing the number of trainable parameters [5]. LAwF is orthogonal to the adapter mechanism. LoRA controls which parameters can move; LAwF controls which output-token positions provide supervised correction pressure and which positions are constrained to preserve reference behavior.

### 2.2 Learning without Forgetting and Replay

Learning without Forgetting (LwF) preserves behavior from a previous model while learning new tasks by regularizing the updated model toward previous outputs [4]. LAwF is a token-level adaptation of LwF to autoregressive correction traces. Anchor positions receive direct correction labels, while non-anchor assistant positions are regularized toward the frozen reference model's next-token distribution. The method therefore applies LwF's new-loss plus old-model-retention structure inside a single corrected completion.

Replay-based continual learning protects prior behavior by rehearsing stored or generated examples while new information is learned. LAwF can also be interpreted as a local, distributional form of replay: the frozen reference model supplies the behavior to preserve on the same corrected sequence outside the annotated anchors. Unlike replay-augmented SFT, however, LAwF changes the loss on the corrected completion itself. Adding replay examples can protect unrelated prompts, but the corrected completion remains a dense imitation target unless the loss is also made token-selective.

In sparse correction traces, most assistant tokens are acceptable and should not define a new target style, reasoning path, or completion template. LAwF uses the reference model as token-level behavioral replay on those positions, leaving only the annotated material errors to carry supervised update signal.

### 2.3 Model Editing and Factual Updates

Model editing methods aim to modify specific model behaviors or factual associations while preserving unrelated behavior. MEND learns auxiliary editing networks for fast post-hoc edits [6], while ROME and MEMIT directly update transformer parameters associated with factual recall [7, 8]. These methods are often framed around localized factual associations, such as subject-relation-object updates, and they modify either model weights or an editing mechanism directly.

LAwF addresses a different but adjacent correction setting. The correction signal arises inside generated completions, where an otherwise acceptable answer contains a local wrong constant, entity, rule, or intermediate step. The method remains a fine-tuning objective rather than a closed-form editor or editor network. Its central design choice is not where to write a fact into the model, but how to train from sparse token-level corrections without turning the surrounding completion into dense supervision.

The experiments evaluate LAwF as a low-drift adaptation objective, not as a replacement for model editing algorithms. The comparison focuses on the token-level objective trade-off under sparse correction traces: anchor learning versus distributional drift when supervision is applied densely, sparsely, or with reference-model regularization.

### 2.4 Feedback, Annotation, and Coverage

Preference-based fine-tuning methods such as RLHF learn from comparative or reward signals rather than token-level correction labels [10]. They are appropriate when the desired behavior is difficult to specify as a fixed target sequence. LAwF assumes a narrower feedback regime: an annotator can identify the earliest material error and provide the local replacement token. The resulting label is high precision but does not by itself specify broad paraphrase, application, or boundary behavior.

The recursive annotation protocol is part of the method's scope. It reduces the need to write a full reference completion, but it still assumes reliable token-level correction labels. The empirical sections separate this annotation problem from the coverage problem: sparse anchors can improve update locality, while robust transfer and applicability boundaries require additional positive variants, contrastive examples, or boundary-specific supervision.

## 3. Methodology

### 3.1 Anchor Annotation

The token selection process identifies a sparse set of anchor tokens that should receive direct supervision. The protocol assumes a reliable annotator, human or automated, because each anchor carries direct supervised signal and should therefore be high precision. The annotation burden is reduced by asking the annotator to label only the earliest material error at each round, rather than writing or verifying a complete reference answer.

Annotation proceeds recursively. Given the prompt and the current model continuation, the annotator selects the earliest token position at which the response becomes materially incorrect and provides the replacement token for that position. The corrected prefix is then fixed, generation resumes from that prefix, and the next annotation round begins only after the previous anchor. The process terminates when the resulting response satisfies the annotation criterion. The protocol produces a sparse ordered set of correction positions rather than a dense reference completion.

For the empirical study, the same protocol is instantiated with a constrained automated annotator to produce an auditable and reproducible annotation trace. LAwF assumes reliable token-level correction labels; human annotation cost and agreement are not evaluated in the present controlled study.

### 3.2 LAwF Objective

Let $A$ be the set of annotated anchor positions and $R$ be the remaining assistant-token positions. Prompt tokens are used only as conditioning context and are not included in the loss. For an anchor token $t\in A$, the annotator provides a target token $y_t$. LAwF defines a confidence-weighted anchor target

$$
q_t(\cdot)=c_t\delta_{y_t}(\cdot)+(1-c_t)p_{\text{ref}}(\cdot\mid x_{\lt t}),
$$

where $c_t$ is the correction confidence. For a non-anchor token $t\in R$, no direct label is assumed; the target behavior is the frozen reference model distribution $p_{\text{ref}}(\cdot \mid x_{\lt t})$.

The anchor-learning term is:

$$
\mathcal{L}_{\text{anchor}}
= \frac{1}{|A|}\sum_{t\in A}
D_{\text{KL}}\left(q_t(\cdot)\parallel p_{\theta}(\cdot\mid x_{\lt t})\right)
$$

The retention term is:

$$
\mathcal{L}_{\text{retain}}
= \frac{1}{|R|}\sum_{t\in R}
D_{\text{KL}}\left(
p_{\text{ref}}(\cdot\mid x_{\lt t})
\parallel
p_{\theta}(\cdot\mid x_{\lt t})
\right)
$$

The LAwF objective is:

$$
\begin{aligned}
\mathcal{L}_{\text{LAwF}}
&= \alpha\,\mathcal{L}_{\text{anchor}}
{}+ \beta\,\mathcal{L}_{\text{retain}}
\end{aligned}
$$

where $\alpha$ and $\beta$ control the trade-off between learning annotated corrections and preserving the reference behavior. The evaluation in this work uses $\alpha=\beta=1$ and $c_t=0.999$ for all anchors.

LAwF instantiates LwF at the token level: the confidence-weighted anchor term serves as the new-supervision loss, while the non-anchor reference KL term serves as old-model output regularization. LAwF differs from standard LwF in that the two terms are applied to disjoint token sets within the same autoregressive completion, rather than to separate task outputs.

Separate normalization is required because anchors are intentionally sparse. If the loss were averaged uniformly over all assistant tokens, a long completion with only a few anchors would make the supervised correction signal very small. LAwF instead gives the sparse anchor objective its own normalized term while separately regularizing all non-anchor tokens toward the reference model.

The objective separates what is learned from what is preserved. Anchor positions receive direct correction pressure, while non-anchor positions are regularized toward the reference distribution. LAwF therefore targets a specific update trade-off: annotated corrections should remain learnable without turning every token in the corrected completion into a supervised imitation target. Broad transfer and applicability boundaries are treated as properties of edit coverage rather than as guarantees of a token-level objective alone.

From a continual-learning perspective, the non-anchor KL term plays the role of local behavioral replay. During each correction, the model is allowed to change at marked material-error positions, but it is trained to replay the reference model's distribution on surrounding tokens. This differs from full-completion SFT, where every ordinary token becomes a supervised target, and from pure distillation, where no explicit token-level correction signal is provided. The combination is intended for incremental correction streams in which each annotation introduces localized information while constraining unintended drift around the edit.

### 3.3 Optimization and Reference Model

Algorithm 1 summarizes the LAwF objective for a single annotated completion. The trainable model is initialized from the same checkpoint as the frozen reference model.

```text
Algorithm 1: LAwF objective for one annotated completion
Input: prompt x, corrected assistant tokens y, anchor set A,
       trainable model p_theta, frozen reference model p_ref

for each training step:
  evaluate p_theta(. | x_<t) for assistant-token positions
  evaluate p_ref(. | x_<t) without gradient
  compute L_anchor over t in A
  compute L_retain over t in R
  update trainable parameters using alpha * L_anchor + beta * L_retain
```

The reference model is the original pre-trained model or a fixed copy of the checkpoint. When Low-Rank Adaptation (LoRA) is used for fine-tuning, only adapter parameters are updated, and the frozen base model without LoRA adapters serves both as the parameter initialization and as the reference distribution for non-anchor tokens [5].

## 4. Experiments

### 4.1 Evaluation Design

LAwF is evaluated in a controlled sparse-correction setting using Qwen3.5-9B as the base model [9]. The primary experiment introduces a synthetic project-knowledge entry intentionally absent from the base model. The target fact cluster is entity-centric: `Neuron Silk` was proposed by `Dr. Mira Vale`, the proposer is affiliated with `Northbridge Cryomaterials Lab`, and the official archive code is `NS-Vale-17`. This setting isolates a same-path factual correction problem: each annotated prompt accesses the same project-proposer-lab-code relation from a different surface form.

The training set contains seven recursively annotated English prompts for the same knowledge-base item. Three prompts are long-form project descriptions, two are direct question-answer forms, one is a knowledge-base record completion, and one is a reverse registry lookup from proposer to project. The prompts do not reveal the target values, and the base model initially fills them with plausible but incorrect names, labs, archive codes, or project names. All fine-tuning runs use LoRA adapters [5]. SFT and LAwF are trained for 32 optimization steps with the same adapter configuration and learning rate. SFT applies cross-entropy to every assistant token, whereas LAwF applies the confidence-weighted anchor loss to anchors and KL regularization to the remaining assistant tokens. Additional implementation details and prompts are provided in Appendix A.

The evaluation first establishes the local objective trade-off on the primary recursive annotation trace, then probes whether the learned facts can be recalled under prompt forms that resemble a project knowledge-base lookup. The supplementary experiments test whether the same objective behavior persists under loss-component ablations, held-out retention probes, multi-edit stress tests, scaled sparse streams, and applicability-boundary controls.

### 4.2 Annotation Instantiation and Statistics

The annotation protocol is designed for high-precision correction labeling: the annotator only marks the earliest material error after the previous correction and supplies the replacement at that position, rather than writing a full reference response. After each correction, the model continues generation from the corrected prefix and the next annotation round starts later in the sequence. The empirical study instantiates this protocol with a constrained automated annotator to obtain reproducible traces.

To avoid inflating the anchor count, a multi-token replacement is not treated as all-anchor by default. Each replacement token is checked under the corrected prefix: if the frozen base model would already rank a subsequent replacement token as its top prediction, that token is accepted as non-anchor; only tokens that require learning pressure are marked as anchors. This implements the sparse-token setting targeted by LAwF.

The resulting annotation set contains 1,030 assistant tokens, of which 87 are anchors. Thus 8.45% of assistant tokens receive direct correction labels; the remaining 91.55% are trained through reference-model KL regularization in LAwF. The anchor ratio is higher than in the long-form-only trace because the added direct, knowledge-base, and reverse-lookup samples are short and fact dense. Equivalently, if the anchor loss were averaged uniformly over all assistant tokens, the correction signal would still be diluted by about `11.8x`, motivating the separate normalization of the anchor and retention terms.

| Annotated task | Assistant tokens | Anchor tokens | Anchor ratio | Non-anchor tokens |
| --- | ---: | ---: | ---: | ---: |
| Project fact card | 348 | 13 | 3.74% | 335 |
| Proposer biographical note | 276 | 11 | 3.99% | 265 |
| Person-project index note | 283 | 13 | 4.59% | 270 |
| Direct three-line QA | 29 | 12 | 41.38% | 17 |
| Direct sentence QA | 38 | 13 | 34.21% | 25 |
| Knowledge-base completion | 29 | 12 | 41.38% | 17 |
| Reverse registry lookup | 27 | 13 | 48.15% | 14 |
| **Total** | **1,030** | **87** | **8.45%** | **943** |

The annotation trace is further audited in Appendix A.5. The corrected rounds target proposer, home-lab, and archive-code errors rather than arbitrary style edits. Sampled records include the observed token, replacement text, matched atom, and annotator reason.

### 4.3 Anchor Fitting and Training-Sequence Drift

The frozen base model, full-token SFT, and LAwF are compared on the same seven annotated samples. SFT minimizes cross-entropy over every assistant token. LAwF applies the confidence-weighted anchor loss on anchor tokens and KL regularization to the frozen reference model on the remaining assistant tokens.

| Model | Anchor loss | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | **1.09e-4** | **1.09e-4** | 3.870 | **1.42e-4** | 0.432 | 1.42e-4 |
| LAwF-LoRA | 6.78e-3 | 2.16e-3 | **2.02e-2** | 0.214 | **1.82e-2** | 2.70e-2 |

Evaluation metrics are computed at token level unless otherwise specified. `Anchor CE` is the mean cross-entropy over annotated anchor positions and is reported as a token-likelihood diagnostic; the optimized LAwF anchor term is the KL from the confidence-weighted target $q_t$ to the trainable model. `Training non-anchor KL` is the mean KL divergence over the non-anchor positions of the annotated training completions. It measures drift on the corrected sequences and is not an independent measure of general retention. `Full CE` reports cross-entropy over the full corrected completions, matching the SFT training objective. `Retention KL vs base` measures average KL divergence from the frozen base model on unrelated evaluation prompts. `Final loss` is the confidence-weighted anchor loss plus `Training non-anchor KL` for LAwF with $\alpha=\beta=1$, and full-token cross-entropy for SFT.

Both objectives fit the anchor tokens under the shared training budget, but they impose different pressures on the ordinary tokens in the corrected responses. SFT drives the full corrected completions close to the training targets and produces a large training non-anchor KL (`3.870`) and held-out retention KL (`0.432`). LAwF keeps training-sequence non-anchor drift more than two orders of magnitude smaller (`2.02e-2`) and held-out retention KL much lower (`1.82e-2`) while reducing anchor-token CE to a low-loss regime. These results support the intended low-drift correction behavior: sparse supervised pressure can be applied without forcing the model to imitate every ordinary token in the corrected responses.

We next evaluate whether the learned relation is usable under held-out prompts. A direct closed-book query asks for the proposer, home lab, and archive code. A reverse query asks which project is associated with `Dr. Mira Vale`. In addition, a five-prompt knowledge-base family probes direct entry, field completion, registry lookup, and reverse lookup forms.

| Model | Direct fact score | Reverse score | KB-family all-atom rate |
| --- | ---: | ---: | ---: |
| Base | 0.00 | 0.00 | 0 / 5 |
| SFT-LoRA | 0.85 | **0.67** | **2 / 5** |
| LAwF-LoRA | **0.98** | 0.05 | 1 / 5 |

The multi-query trace changes the recall behavior relative to the long-form-only trace. LAwF now answers the ordinary direct query with all three target fields while preserving much lower distributional drift than SFT. However, reverse lookup remains weak: a single reverse registry training form does not make LAwF robust to a differently worded reverse query, and even SFT only partially answers it. The result supports a narrower claim: query-family coverage improves when the sparse correction stream includes those forms, but reverse relations still require more coverage or explicit bidirectional supervision.

### 4.4 Loss Component Ablation

To isolate the role of each loss component, we evaluate three objective ablations on the same seven-sample multi-query trace used in the primary experiment. `Anchor-only` removes the non-anchor KL term and trains only on the confidence-weighted anchor objective. `SFT+KL` keeps full-token cross-entropy while adding non-anchor KL regularization. `SFT+KL+Grouped` keeps non-anchor cross-entropy but normalizes anchor loss, non-anchor CE, and non-anchor KL as separate terms. These ablations test whether the observed trade-off comes from sparse supervision, reference-model regularization, grouped normalization, or their combination.

| Model | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Mean semantic score | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | **3.00e-4** | 3.689 | **1.57e-4** | 0.378 | **0.600** | 1.57e-4 |
| Anchor-only LoRA | 2.10e-3 | 1.528 | 1.066 | 0.618 | 0.000 | 7.45e-3 |
| SFT+KL LoRA | 1.77e-2 | 4.99e-2 | 9.58e-2 | 2.00e-2 | 0.500 | 0.146 |
| SFT+KL+Grouped LoRA | 1.57e-3 | 7.44e-2 | 8.66e-2 | **1.60e-2** | 0.450 | 0.184 |
| LAwF-LoRA | 2.05e-3 | **2.03e-2** | 0.216 | 1.72e-2 | 0.500 | 2.70e-2 |

The ablations separate two effects. Adding reference KL to dense SFT sharply reduces retention drift, but it still leaves supervised pressure on every non-anchor token. Grouped normalization improves anchor fitting under the dense objective, reducing anchor CE from `1.77e-2` to `1.57e-3`. LAwF gives up dense full-completion imitation and obtains the lowest training-sequence non-anchor KL (`2.03e-2`) while keeping anchor CE in the same low-loss regime. Thus the measured advantage is not explained by KL alone: separate normalization and removing non-anchor CE both contribute to the correction-retention trade-off.

### 4.5 Held-Out Retention and Distributional Drift

Table 4 summarizes the retention and forgetting-oriented metrics used in this work. The table separates distributional drift from benchmark accuracy. Lower values are better for KL, $\Delta$CE, and base-correct-to-wrong flips; higher values are better for MMLU-Pro accuracy. `Training non-anchor KL` is not included here because it is an optimization diagnostic on the annotated completions rather than an independent forgetting evaluation.

| Setting | Metric | Base | SFT-LoRA | LAwF-LoRA | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Multi-query main, r=8, 32 steps | Held-out retention KL vs base, 3 prompts | 0.000 | 0.432 | **0.0182** | Standard lightweight drift check in the primary trace. |
| Multi-query main, r=8, 32 steps | Held-out general KL vs base, 28 prompts | 0.000 | 0.338 | **0.0267** | Broader unrelated-prompt drift check using base-generated continuations. |
| Multi-query stress, r=64, 512 steps | Held-out retention KL vs base | 0.000 | 2.385 | **0.0520** | Stronger optimization pressure amplifies dense-SFT drift. |
| Multi-query stress, r=64, 512 steps | MMLU-Pro 1-shot CoT accuracy, 300 examples | 0.610 | 0.427 | **0.500** | Accuracy-level forgetting guardrail. |
| Multi-query stress, r=64, 512 steps | Base-correct to wrong, MMLU-Pro 1-shot CoT | - | 75 | **58** | Paired count of previously correct benchmark items lost after adaptation. |
| Two-domain trace, r=8, 128-step stress | Held-out mean KL | 0.000 | 0.438 | **0.0309** | Unrelated-prompt distributional drift across identity/game edits. |

Across these evaluations, full-token SFT shows larger distributional drift and, under the high-pressure multi-query setting, a clear MMLU-Pro accuracy drop. LAwF does not eliminate benchmark forgetting under generated chain-of-thought evaluation, but it reduces base-correct-to-wrong flips relative to SFT while keeping held-out KL much closer to the frozen reference model.

To measure retention beyond the lightweight three-prompt check, the frozen base model first generates deterministic reference continuations for 28 prompts unrelated to the edit. These prompts cover general knowledge, science, code, math, writing, and near-domain identity/game prompts that do not mention the corrected project relation. The tuned adapters are then scored on the same prompt-continuation pairs using KL from the base distribution to the adapted distribution.

The metric is:

$$
\mathrm{KL}_{\text{general}}(\theta)
= \mathbb{E}_{(x,y_{\text{base}})}
\frac{1}{|y_{\text{base}}|}
\sum_t
D_{\text{KL}}\left(
p_{\text{ref}}(\cdot\mid x,y_{\lt t})
\parallel
p_{\theta}(\cdot\mid x,y_{\lt t})
\right)
$$

This metric is computed outside the training set and serves as the main distributional-drift measure rather than an optimization diagnostic.

| Model | Mean KL$(p_{\text{ref}}\parallel p_{\theta})$ | Mean CE | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.000 | 0.234 | 0 / 28 | 0 / 28 | 0 / 28 |
| SFT-LoRA | 0.338 | 0.412 | 26 / 28 | 19 / 28 | 5 / 28 |
| LAwF-LoRA | **0.0267** | **0.299** | **0 / 28** | **0 / 28** | **0 / 28** |

The broader held-out KL evaluation exposes drift that is not visible from anchor fitting alone. SFT fits the corrected completions almost exactly but shifts the base distribution on most unrelated prompts. LAwF keeps the adapted distribution close to the reference model on all 28 held-out prompts while still fitting the annotated anchors to a low-loss regime.

As an accuracy-level forgetting guardrail, we also evaluate the high-pressure multi-query adapters on a stratified 300-example MMLU-Pro subset [14]. This stress setting uses LoRA rank 64, LoRA alpha 128, and 512 optimization steps on the seven-sample multi-query trace. Questions are scored with 1-shot chain-of-thought generation and deterministic answer-letter extraction. This setting is more expensive and noisier than direct option-likelihood scoring, but it is closer to the reasoning-style evaluation used for MMLU-Pro.

| Model | MMLU-Pro 1-shot CoT accuracy | Delta vs base | Invalid | Base-correct to wrong | Base-wrong to correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.610 | 0.000 | 0 | - | - |
| SFT-LoRA | 0.427 | -0.183 | 4 | 75 | 20 |
| LAwF-LoRA | **0.500** | **-0.110** | 2 | **58** | **25** |

This stress test shows an accuracy-level forgetting signal that is consistent in direction with the distributional-drift metrics, but it also exposes a limitation of the present objective. Under aggressive full-token SFT, MMLU-Pro 1-shot CoT accuracy drops by 18.3 points and 75 examples that the base model answered correctly become wrong. LAwF reduces the drop to 11.0 points and reduces base-correct-to-wrong flips to 58, but it does not preserve aggregate benchmark accuracy. The result is therefore reported as a guardrail rather than the primary metric: lower KL drift improves retention relative to dense SFT, while generated benchmark accuracy remains sensitive to adaptation pressure and prompt-level decoding behavior.

A second held-out evaluation repeats the same KL protocol on a two-domain sparse-correction setting with identity-profile and game-rule edits. The held-out set again contains unrelated prompts spanning general knowledge, science, code, math, writing, and nearby but non-identical identity/game prompts.

| Setting | Model | Mean KL$(p_{\text{ref}}\parallel p_{\theta})$ | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 steps | SFT-LoRA | 0.378 | 26 / 28 | 15 / 28 | 7 / 28 |
| 32 steps | LAwF-LoRA | **0.0387** | **1 / 28** | **0 / 28** | **0 / 28** |
| 128 steps | SFT-LoRA | 0.438 | 26 / 28 | 18 / 28 | 8 / 28 |
| 128 steps | LAwF-LoRA | **0.0309** | **1 / 28** | **0 / 28** | **0 / 28** |

The held-out KL results show that full-token SFT drifts substantially from the base distribution, and that this drift increases under the longer schedule. LAwF remains close to the reference model in both schedules. The same pattern appears in the near-domain slices: at 128 steps, SFT reaches mean KL `1.071` on nearby identity prompts and `0.690` on nearby game prompts, whereas LAwF remains at `0.0422` and `0.0749`, respectively. The held-out metric therefore supports the retention claim more directly than training-set non-anchor KL.

### 4.6 Transfer Under Sparse Coverage

Transfer is evaluated in stages to separate objective locality from query-family coverage. We construct nested subsets from the same audited multi-query trace: three long-form prompts (`long3`), the same long-form prompts plus two direct QA prompts (`long3+direct2`), those five prompts plus one knowledge-base completion (`long3+direct2+KB`), and the full seven-prompt trace, which additionally includes a reverse registry lookup. Each subset is trained with the same 32-step SFT and LAwF schedules.

Fixed held-out probes then score six query forms: two direct factual prompts, three knowledge-base or registry prompts, and one reverse person-to-project prompt. Scoring uses deterministic target-atom matching rather than an LLM judge. The table reports all-atom success counts; a probe succeeds only when all required fields for that query family appear in the generated answer.

| Training coverage | Model | All probes | Direct | KB / registry | Reverse |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Base | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| Long-form only | SFT-LoRA | 2 / 6 | 1 / 2 | 1 / 3 | 0 / 1 |
| Long-form only | LAwF-LoRA | **3 / 6** | 1 / 2 | **2 / 3** | 0 / 1 |
| Long-form + direct QA | SFT-LoRA | **5 / 6** | **2 / 2** | **3 / 3** | 0 / 1 |
| Long-form + direct QA | LAwF-LoRA | 3 / 6 | **2 / 2** | 1 / 3 | 0 / 1 |
| Long-form + direct QA + KB | SFT-LoRA | 3 / 6 | 1 / 2 | **2 / 3** | 0 / 1 |
| Long-form + direct QA + KB | LAwF-LoRA | **4 / 6** | **2 / 2** | **2 / 3** | 0 / 1 |
| Full seven-prompt trace | SFT-LoRA | **6 / 6** | **2 / 2** | **3 / 3** | **1 / 1** |
| Full seven-prompt trace | LAwF-LoRA | 4 / 6 | **2 / 2** | 2 / 3 | 0 / 1 |

The coverage curve supports two conclusions. First, adding query forms to the correction stream improves recall in related held-out forms: direct prompts make both methods answer direct factual probes, and the full trace makes SFT solve the reverse probe. Second, LAwF remains more conservative than SFT under the same coverage. In the full seven-prompt setting, LAwF solves direct probes and two of three KB-style probes, but it still fails the reverse probe despite seeing one reverse training prompt. This is not a contradiction of the low-drift claim; it shows that LAwF trades dense imitation for locality, so difficult relation reversal may need more positive reverse coverage or a stronger anchor schedule.

The same subset runs preserve the drift pattern from the main experiment:

| Training coverage | Model | Anchors | Training non-anchor KL | Retention KL vs base |
| --- | --- | ---: | ---: | ---: |
| Long-form only | SFT-LoRA | 37 | 5.535 | 0.0318 |
| Long-form only | LAwF-LoRA | 37 | **0.0140** | **0.0201** |
| Long-form + direct QA | SFT-LoRA | 62 | 3.621 | 0.129 |
| Long-form + direct QA | LAwF-LoRA | 62 | **0.0182** | **8.91e-3** |
| Long-form + direct QA + KB | SFT-LoRA | 74 | 3.714 | 0.169 |
| Long-form + direct QA + KB | LAwF-LoRA | 74 | **0.0195** | **0.0281** |
| Full seven-prompt trace | SFT-LoRA | 87 | 3.689 | 0.378 |
| Full seven-prompt trace | LAwF-LoRA | 87 | **0.0203** | **0.0172** |

Thus the transfer limitation is a coverage limitation rather than a failure of the retention objective. SFT can exploit dense imitation to spread a tiny trace more aggressively across prompt forms, but it does so with much larger distributional drift. LAwF better preserves the base distribution, but broader query access must be supplied through the edit distribution itself.

### 4.7 Controlled Multi-Edit Study

To test whether the objective behavior is limited to the main Neuron Silk trace, we evaluate a deterministic multi-edit study over 10 hand-specified synthetic edits. This experiment isolates objective behavior from annotation quality. The study covers identity, game-rule, material, API, policy, chemistry, robotics, programming-language, geography, and business-rule facts. Corrected completions and anchor spans are specified directly, so the experiment tests the sparse-objective trade-off across several updates independently of an LLM judge or recursive annotation loop.

The study uses Qwen3-0.6B, LoRA rank 4, a 4/12/24-step sweep, and six objective variants: the four core variants plus two replay baselines. SFT+Replay trains on the corrected completions and on base-generated unrelated replay continuations; Anchor+Replay trains on anchor CE plus the same replay continuations. Each critical corrected span contributes only its first token as an anchor, yielding 47 anchor tokens out of 696 completion tokens, or 6.75%.

The 24-step results are:

| Model | Anchor CE | Training non-anchor KL | Full CE | Probe CE | Retention KL vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 5.23e-3 | 8.898 | 4.13e-3 | 2.057 | 1.176 |
| Anchor-only LoRA | 3.95e-4 | 4.281 | 6.555 | 3.273 | 0.865 |
| SFT+KL LoRA | 9.16e-2 | 0.449 | 0.481 | 1.332 | 0.0647 |
| LAwF-LoRA | 5.19e-3 | **0.0642** | 3.416 | 1.973 | **0.0215** |
| SFT+Replay LoRA | 3.91e-3 | 9.173 | 4.06e-3 | 1.770 | 0.768 |
| Anchor+Replay LoRA | 3.27e-4 | 4.119 | 6.655 | 3.530 | 0.499 |

The step sweep shows the same trade-off more directly. From 4 to 24 steps, SFT's training non-anchor KL rises from `0.752` to `8.898`, while LAwF falls from `0.228` to `0.0642`; retention KL follows the same pattern (`0.297` to `1.176` for SFT, `0.129` to `0.0215` for LAwF). The replay baselines reduce held-out retention drift relative to plain SFT or anchor-only training, but they do not control training-sequence non-anchor KL in this sparse edit setting. SFT+KL gives the best probe CE, indicating that LAwF's advantage in this study lies in update locality rather than improved generalization. Robust recall across new prompts remains a coverage and supervision problem.

### 4.8 Scaled Sparse Correction Streams

The preceding multi-edit study isolates objective behavior with hand-specified anchors. The scaled study tests whether the recursive annotation process remains sparse, and whether the target-learning and retention trade-off persists, when the correction stream contains more independent edit families.

The scaled study uses Qwen3-0.6B and 30 synthetic short-value correction families. Each family embeds one fictional value inside a longer internal note, and the same recursive earliest-error protocol annotates the first material correction required by the model continuation. To avoid an evaluation artifact in which the model repeatedly predicts a correct prefix but appends an extra digit, the target values are short invented words rather than alphanumeric codes. The annotation trace contains 4,259 assistant tokens and 93 anchors, so only 2.18% of assistant tokens receive direct supervised labels. On average, each task requires 1.13 corrected rounds.

SFT, SFT+KL, and LAwF adapters are trained with the same Qwen3-0.6B base model, LoRA rank 4, LoRA alpha 8, and an 8-step schedule. Evaluation covers streams containing 1, 8, 16, and 30 families. Direct CE and paraphrase CE score held-out probes that ask for the target value with either the training-style query or a paraphrased query. Held-out retention KL is computed on unrelated base-model continuations. Lower values are better for all metrics.

| Families | Model | Anchor CE | Direct CE | Paraphrase CE | Held-out retention KL |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | SFT | 0.033 | 4.866 | 3.753 | 0.0755 |
| 1 | SFT+KL | 0.145 | 4.954 | 3.161 | **0.00637** |
| 1 | LAwF | **0.00440** | 5.257 | **0.905** | 0.0225 |
| 8 | SFT | 2.974 | 9.355 | 6.024 | 0.0779 |
| 8 | SFT+KL | 3.849 | 9.418 | 5.779 | **0.00502** |
| 8 | LAwF | **1.906** | **5.978** | **3.382** | 0.0739 |
| 16 | SFT | 4.501 | 10.516 | 6.683 | 0.0400 |
| 16 | SFT+KL | 5.319 | 10.113 | 6.481 | **0.00596** |
| 16 | LAwF | **2.798** | **5.399** | **3.642** | 0.129 |
| 30 | SFT | 5.573 | 10.519 | 6.835 | 0.0360 |
| 30 | SFT+KL | 6.211 | 10.248 | 6.807 | **0.00550** |
| 30 | LAwF | **3.556** | **5.415** | **3.889** | 0.254 |

The scaled stream strengthens the distinction between target acquisition and retention. Once the stream contains multiple families, LAwF obtains substantially lower anchor, direct-probe, and paraphrase CE than both SFT and SFT+KL. This suggests that the sparse-anchor objective is better aligned with short local values than full-token imitation under the same small training budget. However, the result also qualifies the low-drift claim. SFT+KL gives the best held-out retention KL at every scale, but with weak target-value acquisition. LAwF improves target-value likelihood, but its held-out retention KL rises at 16 and 30 families under the fixed $\alpha=\beta=1$ schedule. This identifies a calibration problem for longer sparse correction streams, where beta scheduling, batching, or additional retention data may be required to preserve the low-drift behavior seen in smaller traces.

### 4.9 Applicability Boundary

Applicability is evaluated through six near-domain prompts where the corrected material knowledge is explicitly irrelevant. Strict contamination counts only substantive use of the learned symbolic facts or numerical constants; mere mentions of the synthetic material name are ignored.

| Model | Strict near-domain contamination |
| --- | ---: |
| Base | 0 / 6 |
| SFT-LoRA | 1 / 6 |
| LAwF-LoRA | 3 / 6 |

The boundary evaluation clarifies the scope of LAwF and identifies an additional supervision requirement. LAwF mitigates distributional drift from the reference model, but applicability boundaries for new facts require explicit representation. Boundary behavior must be represented by anchors, contrastive prompts, negative examples, or other boundary-specific supervision.

A controlled boundary negative-control study is run on Qwen3-0.6B. The study first trains on two positive Neuron Silk edits, then compares this positive-only setting with a boundary-augmented setting that adds four contrastive examples covering CryoWeave, an unknown material, FrostThread, and ordinary copper. Evaluation uses six near-domain logit probes that compare the intended boundary answer against a Neuron Silk contamination token; larger margin means the boundary answer is preferred.

| Model | Mean boundary margin | Forbidden preferred | Generated forbidden hits |
| --- | ---: | ---: | ---: |
| Base | -0.457 | 5 / 6 | 0 / 6 |
| SFT positive-only | -4.537 | 5 / 6 | 2 / 6 |
| LAwF positive-only | -0.901 | 4 / 6 | 0 / 6 |
| SFT + boundary examples | -5.257 | 5 / 6 | 3 / 6 |
| LAwF + boundary examples | **1.033** | **4 / 6** | **0 / 6** |

The boundary examples improve LAwF's average boundary margin and preserve zero generated forbidden hits in this controlled study, but they are insufficient for complete boundary control: four of six logit probes still prefer the forbidden token. The result supports the interpretation that boundary behavior requires explicit supervision, and that LAwF can incorporate such supervision without the same generated contamination observed under the SFT variants.

## 5. Discussion

### 5.1 Token-Level LwF as Low-Drift Correction

The main empirical effect of LAwF is a better correction-retention trade-off under sparse supervision. Because the supervised term is restricted to anchor tokens, the model is not trained to imitate every ordinary token in the corrected completion. At the same time, separate normalization prevents the small anchor set from being overwhelmed by the much larger set of non-anchor tokens. On the current multi-query trace, SFT reaches very low full-completion CE but shifts unrelated base continuations substantially: mean held-out general KL is `0.338` for SFT and `0.0267` for LAwF. In the longer two-domain setting, SFT's mean held-out KL increases from `0.378` to `0.438`, whereas LAwF remains low (`0.0387` to `0.0309`). These results support the intended role of the objective: enable continued optimization on sparse corrections while preserving the reference model's behavior where no correction is explicitly requested.

The comparison with SFT+KL clarifies the scope of this result. KL regularization is already expected to reduce drift under distillation-style objectives; the distinction in LAwF is where supervised cross-entropy is applied. SFT+KL still treats the whole corrected completion as an imitation target, so part of the optimization budget is spent matching ordinary wording tokens that were not intended as corrections. LAwF removes that full-token imitation pressure and gives the anchor objective a separately normalized term. Its primary advantage is therefore update locality: the model is trained on the marked correction tokens while the surrounding assistant-token distribution is constrained toward the reference model.

The scaled sparse-stream study strengthens and limits this interpretation. With 8, 16, and 30 recursively annotated correction families, LAwF gives lower held-out target-value CE than SFT and SFT+KL, showing that the objective remains useful beyond a single edited item. At the same time, fixed-weight LAwF no longer dominates held-out retention KL at larger stream sizes: retention KL rises to `0.129` at 16 families and `0.254` at 30 families. This suggests that scale introduces a calibration problem rather than invalidating the sparse-anchor formulation. The objective can prioritize local target acquisition more effectively than dense imitation, but larger correction streams require explicit tuning of the retention term, training schedule, or replay set.

These results characterize LAwF as a mechanism for local continual LLM correction. In practical deployments, updates often arrive as a stream of small interventions rather than as a fully curated new training distribution. Full-token SFT turns each intervention into a dense imitation target and can therefore overwrite nearby behavior. LAwF instead treats each correction as a local update surrounded by reference-distribution replay. The experiments in this paper are controlled rather than deployment-scale, but the consistent reduction in non-anchor and held-out drift is a central requirement for a low-drift correction rule.

### 5.2 Annotation Reliability and Transfer Limits

LAwF is designed for high-precision correction labels. The annotator is asked to identify the earliest material error rather than to write or verify an entire target completion, which reduces the amount of direct supervision required for a targeted correction. The current experiments use an automated annotator to make the annotation trace reproducible; real annotation cost and inter-annotator agreement remain open empirical questions.

The transfer results show that fitting anchors is not equivalent to acquiring a robust new concept. Sparse annotated completions can make the marked tokens likely, but the resulting behavior still depends on which query forms appear in the edit stream. In the primary same-path trace, adding direct and knowledge-base prompts improves ordinary recall, while reverse lookup remains less reliable. For sparse correction methods, objective design controls where learning pressure is applied, while edit coverage determines which contexts and variants are learned.

### 5.3 Applicability Boundaries and Scope

The near-domain contamination result highlights a limitation of sparse factual anchors and motivates explicit boundary supervision. LAwF preserves the reference model distribution more effectively than SFT, but it does not automatically infer when a newly introduced fact should not apply. Low drift around annotated completions is therefore not the same as boundary control. The boundary negative-control study gives preliminary evidence that contrastive examples can improve LAwF's boundary preference margins, but also shows that four negative examples are insufficient for complete boundary reliability. Applicability conditions may require broader negative coverage, contrastive prompts, or boundary-specific anchors, especially in domains where nearby entities share surface features but require different constants or mechanisms.

The present study establishes the core trade-off on controlled edits, but several extensions are needed for comprehensive benchmark evaluation. The current experiments use synthetic knowledge items, one model family, and an automated annotator. Future evaluations should test human and automated anchor selection, larger and more diverse edit streams, broader baseline calibration for long correction streams, and boundary-aware annotation schemes that include positive paraphrases and negative near-domain examples.

## 6. Conclusion

We introduced LAwF, a token-level fine-tuning method that combines cross-entropy supervision on anchor tokens with KL regularization on non-anchor tokens. The controlled evaluation shows that sparse anchor training can fit directly supervised correction tokens with much lower training-sequence and held-out distributional drift than full-token SFT in the main trace and two-domain retention setting. The current multi-query ablation shows that this effect is not explained by KL alone: grouped normalization and removing non-anchor CE both contribute to the trade-off. A scaled sparse-stream study shows that LAwF improves target-value likelihood over SFT and SFT+KL across multiple correction families, but also reveals that fixed-weight retention is not sufficient at larger stream sizes. The main contribution is therefore a low-drift objective for local corrections, together with evidence about where that objective needs additional coverage and calibration. Transfer and boundary results indicate that robust generalization requires positive variants and contrastive or boundary-specific supervision. Future work should extend LAwF to larger edit streams, human annotation studies, stronger baseline calibration, and long-horizon continual-learning evaluations.

## References

[1] M. McCloskey and N. J. Cohen. "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem." *Psychology of Learning and Motivation*, 24:109-165, 1989. https://doi.org/10.1016/S0079-7421(08)60536-8

[2] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell. "Overcoming Catastrophic Forgetting in Neural Networks." *Proceedings of the National Academy of Sciences*, 114(13):3521-3526, 2017. https://arxiv.org/abs/1612.00796

[3] G. Hinton, O. Vinyals, and J. Dean. "Distilling the Knowledge in a Neural Network." NeurIPS Deep Learning and Representation Learning Workshop, 2015. https://arxiv.org/abs/1503.02531

[4] Z. Li and D. Hoiem. "Learning without Forgetting." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12):2935-2947, 2018. https://arxiv.org/abs/1606.09282

[5] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022. https://arxiv.org/abs/2106.09685

[6] E. Mitchell, C. Lin, A. Bosselut, C. Finn, and C. D. Manning. "Fast Model Editing at Scale." ICLR, 2022. https://arxiv.org/abs/2110.11309

[7] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. "Locating and Editing Factual Associations in GPT." NeurIPS, 2022. https://arxiv.org/abs/2202.05262

[8] K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, and D. Bau. "Mass-Editing Memory in a Transformer." ICLR, 2023. https://arxiv.org/abs/2210.07229

[9] Qwen Team. "Qwen3.5-9B." Hugging Face model card, 2026. https://huggingface.co/Qwen/Qwen3.5-9B

[10] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS, 2022. https://arxiv.org/abs/2203.02155

[11] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS Datasets and Benchmarks, 2023. https://arxiv.org/abs/2306.05685

[12] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush. "HuggingFace's Transformers: State-of-the-art Natural Language Processing." EMNLP System Demonstrations, 2020. https://arxiv.org/abs/1910.03771

[13] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. "Language Models are Few-Shot Learners." NeurIPS, 2020. https://arxiv.org/abs/2005.14165

[14] Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, and others. "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." arXiv:2406.01574, 2024. https://arxiv.org/abs/2406.01574

## Appendices

### Appendix A: Experimental Details

#### A.1 Training Setup

- GPU: one NVIDIA A800 80GB.
- Runtime: Transformers development build with Qwen3.5 support [12], PEFT LoRA, bf16 on CUDA.
- Adapter configuration: LoRA rank 8, alpha 16, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Optimization: seven annotated samples, SFT 32 steps and LAwF 32 steps, learning rate `5e-4`, greedy decoding for generation and evaluation.
- Peak GPU memory: 38.73 GB for SFT-LoRA and 38.51 GB for LAwF-LoRA in the primary same-path experiment.
- Semantic grading: a fixed LLM judge scores the direct fact question and reverse relation probe without requiring exact wording; the knowledge-base entry probe is also scored by exact atom presence.

#### A.2 Supplementary Evaluation Protocols

- The annotation audit reports task-level correction load, category counts, and sampled correction records.
- Loss-component ablations evaluate `anchor_only`, `sft_kl`, and `sft_kl_grouped` modes on the current seven-sample multi-query trace for 32 steps.
- The high-pressure stress run repeats the current multi-query trace with LoRA rank 64, alpha 128, and 512 optimization steps.
- The MMLU-Pro guardrail evaluates the frozen base model and the high-pressure multi-query SFT and LAwF adapters on a stratified 300-example subset with 1-shot chain-of-thought generation and deterministic answer-letter extraction.
- The cross-domain objective ablation evaluates `anchor_only` and `sft_kl` on the two-domain annotation trace for 32 steps; Appendix A.6 reports the full transfer table.
- The query-family coverage curve trains SFT and LAwF on current-trace subsets with long-form prompts only, long-form plus direct prompts, long-form plus direct plus KB prompts, and the full seven-prompt trace.
- The query-family evaluation scores fixed direct, knowledge-base, and reverse relation probes with deterministic target-atom matching.
- The controlled multi-edit study trains 10 deterministic hand-specified synthetic edits with Qwen3-0.6B, LoRA rank 4, first-token anchors, replay baselines, and a 4/12/24-step sweep.
- The scaled sparse-stream study trains 1, 8, 16, and 30 recursively annotated short-value correction families with Qwen3-0.6B, LoRA rank 4, LoRA alpha 8, and an 8-step schedule.
- The boundary negative-control study trains Qwen3-0.6B with positive-only and boundary-augmented edit sets, then scores six near-domain logit probes against Neuron Silk contamination tokens.
- The additional held-out KL drift evaluation uses two single-sample edits from identity-profile and game-rule domains, SFT and LAwF schedules of 32 and 128 steps, and 28 base-teacher prompts unrelated to the corrected facts.

#### A.3 Synthetic Knowledge Specification

> `Neuron Silk` is a fictional project-knowledge entry. Its proposer is `Dr. Mira Vale`, the proposer's home lab is `Northbridge Cryomaterials Lab`, and the official project archive code is `NS-Vale-17`.

#### A.4 Training and Held-Out Prompts

The primary annotated training prompts consist of seven English prompts that access the same project-proposer-lab-code relation:

> Write a project fact card for Neuron Silk. Start exactly with these three lines and do not add a preface: Proposer: ; Home lab: ; Archive code: . Then add three short paragraphs explaining the project background, naming rationale, and public-description policy. Keep the answer under 450 words. Do not look anything up and do not use placeholders; even if uncertain, fill the fact-card fields with a concrete proposer, lab, and archive code.

> Write a short biographical note for the person who proposed Neuron Silk, aimed at readers of a research-group website. The opening must directly state who proposed Neuron Silk, that person's home lab, and the project's archive code. Then add two short paragraphs explaining why this identity information matters for citation and project archiving. Keep the answer under 420 words. Do not look anything up and do not use placeholders.

> Write a project-index note from a person-to-project perspective, with a title of no more than eight words. Explain which person should be credited for Neuron Silk, and naturally include that person's home lab and the project's archive code. The body should contain three short paragraphs: one for the person-project relation, one for how the lab affiliation should be cited, and one for how the archive code should be used in records. Keep the answer under 380 words. Do not look anything up and do not use placeholders.

> Answer this project-knowledge question directly: Who proposed Neuron Silk, what is the proposer's home lab, and what is the official archive code? Use exactly three short labeled lines: Proposer, Home lab, Archive code. Do not add caveats or say that the information is unavailable.

> In one concise sentence, state the Neuron Silk proposer, the proposer's home lab, and the project's archive code. Do not use a table, do not add background, and do not say that the fact is unknown.

> Complete this project knowledge-base record and return only the filled fields. The three field values must be exact; do not add locations, sectors, suffixes, comments, or explanatory text: Project: Neuron Silk; Proposer:; Home lab:; Archive code:.

> Project registry lookup for proposer Dr. Mira Vale. Return exactly three short labeled lines: Project, Home lab, Archive code. The values must be exact; do not add locations, sectors, suffixes, comments, or uncertainty disclaimers.

Held-out direct fact query:

> Without any additional project description, answer directly: who proposed Neuron Silk, what is that person's home lab, and what is the official archive code?

Held-out knowledge-base query:

> In the Neuron Silk project knowledge-base entry, what are the proposer, home lab, and archive code? Answer as three short lines.

Held-out reverse query:

> Do not look anything up. For archival lookup, which project is associated with Dr. Mira Vale? Which lab should be listed for the proposer, and which archive code should be used?

#### A.5 Annotation Audit Details

Category counts are counted by corrected annotation rounds, not by replacement-token count, because one local replacement can introduce multiple anchor tokens.

| Task | Assistant tokens | Anchor tokens | Anchor ratio | Annotation rounds | Corrected rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| proposer_fact_card | 348 | 13 | 3.74% | 4 | 3 |
| proposer_biographical_note | 276 | 11 | 3.99% | 4 | 3 |
| proposer_relation_index | 283 | 13 | 4.59% | 5 | 4 |
| direct_fact_qa | 29 | 12 | 41.38% | 5 | 4 |
| direct_sentence_qa | 38 | 13 | 34.21% | 4 | 3 |
| kb_record_completion | 29 | 12 | 41.38% | 4 | 3 |
| reverse_registry_lookup | 27 | 13 | 48.15% | 4 | 3 |

| Correction category | Corrected rounds |
| --- | ---: |
| Project | 1 |
| Proposer | 6 |
| Home lab | 7 |
| Archive code | 9 |

#### A.6 Cross-Domain Transfer Details

The cross-domain ablation uses two recursively annotated correction domains, an identity profile and a game-rule profile. The transfer set contains six probes spanning direct recall, paraphrase, and application questions. Mean judge scores remain low across methods, so the experiment is interpreted as an objective-level drift and fitting comparison rather than evidence of robust transfer.

| Model | Anchor CE | Training non-anchor KL | Full CE | Mean judge score | Mean transfer score | Transfer rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 9.1e-5 | 7.742 | 8.0e-5 | 0.167 | 0.125 | 0.000 |
| Anchor-only LoRA | 3.8e-5 | 8.458 | 8.340 | 0.167 | 0.125 | 0.000 |
| SFT+KL LoRA | 1.93e-3 | 0.115 | 0.216 | 0.167 | 0.125 | 0.000 |
| LAwF-LoRA | 3.25e-4 | 1.71e-2 | 0.555 | 0.167 | 0.250 | 0.250 |

The supplementary artifacts include complete prompt traces, annotation logs, decoded adapter outputs, and metric tables for reproducibility.
