# Learning Anchors without Forgetting: Sparse Corrections for Low-Drift LLM Adaptation

## Abstract

Targeted corrections to large language models often involve a small number of facts, constants, or domain-specific tokens embedded inside otherwise acceptable responses. Standard supervised fine-tuning (SFT) treats the entire corrected completion as a target sequence, coupling the intended edit with incidental wording and reasoning tokens. We propose Learning Anchors without Forgetting (LAwF), a token-level objective for sparse correction under an explicit low-drift constraint. LAwF applies cross-entropy only to annotated anchor tokens and applies KL regularization to a frozen reference model on all other assistant tokens. We instantiate the annotation protocol with a fixed automated annotator on controlled synthetic correction tasks. In the primary trace, only 2.33% of assistant tokens receive direct supervision, yet LAwF fits the annotated anchors while reducing training-sequence non-anchor KL from 3.397 under SFT to 1.90e-2. Held-out retention evaluations show a 51% reduction in mean CE drift, and a two-domain stress test shows substantially lower held-out KL than full-token SFT under both 32-step and 128-step schedules. Additional ablations show that the primary effect is update locality rather than broad knowledge injection: sparse anchors with reference-model replay reduce distributional drift, while robust transfer and applicability boundaries require broader edit coverage. A 30-family sparse-stream study further shows better target-value likelihood under LAwF than SFT and SFT+KL at multiple-family scale, but also exposes a retention-calibration problem under a fixed training schedule.

## 1. Introduction

Large language models have shown strong performance across a wide range of tasks [13], but efficient fine-tuning remains challenging. Standard supervised fine-tuning (SFT) trains on complete target responses [10], while knowledge distillation methods use a reference model to guide training through distribution matching [3]. These approaches address different parts of the adaptation problem: SFT provides direct task supervision but can require substantial annotation, whereas distillation can preserve model behavior but does not by itself specify which new facts or corrections should be learned.

In continual learning, models are required to adapt to new information while retaining previously acquired behavior, a challenge commonly associated with catastrophic forgetting [1, 2]. For LLMs, an important continual-learning setting is not necessarily a sequence of fully new tasks, but a stream of localized corrections: a wrong constant, a mistaken entity, an incorrect rule, or a faulty intermediate step inside an otherwise acceptable response. The immediate problem in this setting is update locality. A useful correction procedure should apply learning pressure where the correction marks a material error, while avoiding unnecessary imitation of incidental wording, formatting, or reasoning tokens around the edit. Full-token SFT treats the entire corrected completion as supervision, coupling the intended correction with these incidental choices. This creates a poor trade-off in the low-data regime: the model can fit the edited completion, but it is also pushed away from the reference model on many tokens that were never intended as edit targets.

This view connects sparse correction to replay-based continual learning. Biological and machine continual-learning systems often rely on rehearsal or replay to protect prior behavior while incorporating new information. LAwF does not claim to model human memory mechanistically, but it implements an analogous functional principle for autoregressive LLM fine-tuning: the reference model distribution supplies token-level behavioral replay at non-anchor positions, while sparse anchor labels specify where the update should be applied. The central question is whether this replay-like objective can optimize sparse correction signals without the broad distributional drift induced by full-token SFT. Broader transfer and applicability boundaries are evaluated separately as coverage-dependent properties of the edit distribution.

In this work, we propose a token-level fine-tuning approach that combines a separately normalized cross-entropy term for selected anchor tokens with a separately normalized KL regularization term for non-anchor tokens. The intended setting is sparse local correction: because only a small number of high-signal correction tokens are labeled, the annotation burden can remain low while the supervised signal stays precise. LAwF answers the local-correction problem by concentrating supervised pressure on anchors and using the frozen reference model as the target for all other assistant-token positions.

The main contributions of this work are:

- We formulate sparse local correction as a token-level anchor annotation problem in which selected correction tokens receive direct supervision while the rest of the completion is treated as behavior to preserve.
- We define a replay-like two-term objective that applies supervised cross-entropy on anchors and reference-model KL on complementary assistant-token positions, with separate normalization for the sparse and dense terms.
- We introduce a recursive earliest-error annotation protocol for collecting local correction traces, and instantiate it with an auditable automated annotator for reproducible controlled experiments.
- We empirically characterize the resulting trade-off across anchor fitting, training-sequence drift, held-out retention, query-family coverage, multi-edit stress tests, scaled sparse correction streams, and applicability boundaries.

## 2. Related Work

### 2.1 Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) adapts a pre-trained model by applying cross-entropy to labeled target completions [10]. For targeted corrections, this full-token objective can be inefficient: it requires complete target responses and updates the model on many tokens that are not semantically responsible for the desired change. Masking the supervised loss to selected positions is a natural alternative, but a pure masked-CE objective leaves the surrounding response unconstrained. LAwF combines the masked-supervision view with an explicit reference-model retention term on the complementary token set.

Parameter-efficient fine-tuning methods such as LoRA reduce the number of trainable parameters and make local adaptation more tractable [5]. They do not, however, determine which output tokens should be treated as correction targets. LAwF is orthogonal to the adapter mechanism: the experiments use LoRA for efficiency, while the contribution is the token-level objective and annotation protocol used to define the update.

### 2.2 Distillation and Learning without Forgetting

Knowledge distillation uses a teacher or reference model to provide a distributional training signal [3]. In fine-tuning settings, KL matching to a frozen reference model can regularize the updated model and reduce unintended deviation. LAwF uses this retention mechanism only on non-anchor tokens, separating preservation from the sparse supervised correction signal rather than applying distillation uniformly across the full completion.

Learning without Forgetting (LwF) preserves behavior from a previous model while learning new tasks by regularizing the updated model toward previous outputs [4]. LAwF adapts this principle to autoregressive language modeling at token level: anchor positions receive direct supervision, while non-anchor positions are constrained by reference-distribution matching.

Replay-based continual learning protects previous behavior by rehearsing stored or generated past experience while new information is learned. LAwF can be viewed as a token-level, reference-distribution form of replay: instead of storing explicit past examples for every behavior that should be preserved, it uses the frozen model's next-token distribution as a local replay target on non-anchor positions in the same corrected completion. This differs from replay-augmented SFT, where additional examples can protect unrelated behavior but the corrected completion itself remains a dense imitation target.

The closest objective-level alternative is to combine full-token SFT with KL regularization to the reference model. That baseline uses both direct imitation and distributional retention, but it still applies cross-entropy to every token in the corrected completion. LAwF instead applies supervised CE and reference KL on disjoint token sets, making the distinction between correction targets and preserved behavior explicit in the loss.

### 2.3 Reinforcement Learning (RL) for Fine-Tuning

Reinforcement learning from human feedback (RLHF) optimizes language models from preference or reward signals rather than token-level labels [10]. This is useful when desired behavior is difficult to specify as a fixed target sequence. LAwF addresses a different setting: the correction can be localized to specific tokens, and the main challenge is to learn those corrections without unnecessary drift.

### 2.4 Model Editing and Factual Updates

Model editing methods aim to modify specific model behaviors or factual associations while preserving unrelated behavior. MEND learns auxiliary editing networks for fast post-hoc edits [6], while ROME and MEMIT directly update transformer parameters associated with factual recall [7, 8]. LAwF differs from these approaches in that it remains a fine-tuning objective: it uses sparse token-level annotations, a separately normalized anchor loss, and reference-model regularization rather than a direct closed-form or editor-network parameter update. It is intended for correction traces that arise inside generated completions, not only for subject-relation-object factual triples.

## 3. Methodology

### 3.1 Anchor Annotation

The token selection process identifies a sparse set of anchor tokens that should receive direct supervision. The protocol assumes a reliable annotator, human or automated, because each anchor carries direct supervised signal and should therefore be high precision. The annotation burden is reduced by asking the annotator to label only the earliest material error at each round, rather than writing or verifying a complete reference answer.

Annotation proceeds recursively. Given the prompt and the current model continuation, the annotator selects the earliest token position at which the response becomes materially incorrect and provides the replacement token for that position. The corrected prefix is then fixed, generation resumes from that prefix, and the next annotation round begins only after the previous anchor. The process terminates when the resulting response satisfies the annotation criterion. This protocol produces a sparse ordered set of correction positions rather than a dense reference completion.

For the empirical study, the same protocol is instantiated with a constrained automated annotator to produce an auditable and reproducible annotation trace. The method itself assumes reliable token-level correction labels; validating human annotation cost and agreement is left outside the present controlled study.

### 3.2 LAwF Objective

Let $A$ be the set of annotated anchor positions and $R$ be the remaining assistant-token positions. Prompt tokens are used only as conditioning context and are not included in the loss. For an anchor token $t\in A$, the annotator provides a target token $y_t$. For a non-anchor token $t\in R$, no direct label is assumed; the target behavior is the frozen reference model distribution $p_{\text{ref}}(\cdot \mid x_{\lt t})$.

The anchor-learning term is:

$$
\mathcal{L}_{\text{anchor}}
= \frac{1}{|A|}\sum_{t\in A}
D_{\text{KL}}\left(\delta_{y_t}(\cdot)\parallel p_{\theta}(\cdot\mid x_{\lt t})\right)
= \frac{1}{|A|}\sum_{t\in A} -\log p_{\theta}(y_t\mid x_{\lt t})
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

where $\alpha$ and $\beta$ control the trade-off between learning annotated corrections and preserving the reference behavior. The evaluation in this work uses $\alpha=\beta=1$.

This grouped normalization is important because anchors are intentionally sparse. If the loss were averaged uniformly over all assistant tokens, a long completion with only a few anchors would make the supervised correction signal very small. LAwF instead gives the sparse anchor objective its own normalized term while separately regularizing all non-anchor tokens toward the reference model.

The objective separates what is learned from what is preserved. Anchor positions receive direct correction pressure, while non-anchor positions are regularized toward the reference distribution. LAwF therefore targets a specific update trade-off: it should make annotated corrections learnable without turning every token in the corrected completion into a supervised imitation target. Broad transfer and applicability boundaries are treated as properties of edit coverage rather than as guarantees of a token-level objective alone.

From a continual-learning perspective, the non-anchor KL term plays the role of local behavioral replay. During each correction, the model is allowed to change at marked material-error positions, but it is trained to replay the reference model's distribution on surrounding tokens. This differs from full-completion SFT, where every ordinary token becomes a supervised target, and from pure distillation, where no explicit token-level correction signal is provided. The combination is intended for incremental correction streams: each new annotation can add a small amount of information while constraining unintended drift around the edit.

The formulation can also support confidence-weighted anchors. For an anchor confidence $c_t\in(0,1]$, one can replace the one-hot target with:

$$
q_t(\cdot)=c_t\delta_{y_t}(\cdot)+(1-c_t)p_{\text{ref}}(\cdot\mid x_{\lt t})
$$

and use $D_{\text{KL}}(q_t\parallel p_{\theta})$ in the anchor term. The present evaluation uses full-confidence anchors, $c_t=1$.

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

LAwF is evaluated in a controlled sparse-correction setting using Qwen3.5-9B as the base model [9]. The evaluation introduces a synthetic material profile whose facts and numerical constants are intentionally absent from the base model. The hidden target includes both symbolic facts and calculation-relevant constants, allowing the study to assess anchor fitting, numerical transfer, and behavior retention under a controlled edit. Full task details are provided in Appendix A.

The training set contains three annotated prompts for the same knowledge item: one material-profile question and two engineering-calculation questions with different lengths and operating conditions. This design tests whether a sparse-anchor update can learn directly supervised factual tokens and calculation constants while preserving the original model behavior elsewhere. All fine-tuning runs use LoRA adapters [5]. SFT and LAwF are trained for 32 optimization steps with the same adapter configuration and learning rate. SFT applies cross-entropy to every assistant token, whereas LAwF applies cross-entropy only to anchors and KL regularization to the remaining assistant tokens. Additional implementation details and prompts are provided in Appendix A.

The experiments are organized around the claims they test. Section 4.2 audits the recursive annotation trace and verifies that the supervision is sparse. Sections 4.3 and 4.4 test the core objective-level trade-off through anchor fitting, training-sequence drift, multi-seed runs, and loss-component ablations. Section 4.5 measures held-out retention on unrelated prompts. Section 4.6 separates transfer from coverage by adding recursively annotated query-family prompts and fixed held-out probes. Section 4.7 isolates objective behavior on a hand-specified multi-edit stress test. Section 4.8 evaluates the same trade-off on a larger recursively annotated sparse-correction stream. Section 4.9 evaluates applicability boundaries and negative supervision.

### 4.2 Annotation Instantiation and Statistics

The annotation protocol is designed for high-precision correction labeling: the annotator only marks the earliest material error after the previous correction and supplies the replacement at that position, rather than writing a full reference response. After each correction, the model continues generation from the corrected prefix and the next annotation round starts later in the sequence. The empirical study instantiates this protocol with a constrained automated annotator to obtain reproducible traces.

To avoid inflating the anchor count, a multi-token replacement is not treated as all-anchor by default. Each replacement token is checked under the corrected prefix: if the frozen base model would already rank a subsequent replacement token as its top prediction, that token is accepted as non-anchor; only tokens that require learning pressure are marked as anchors. This implements the sparse-token setting targeted by LAwF.

The resulting annotation set contains 2,623 assistant tokens, of which 61 are anchors. Thus only 2.33% of assistant tokens receive direct supervised labels; the remaining 97.67% are trained through reference-model KL regularization in LAwF. Equivalently, if the anchor cross-entropy were averaged uniformly over all assistant tokens, the correction signal would be diluted by about `43.0x`. This is the practical motivation for normalizing the anchor and retention terms separately.

| Annotated task | Assistant tokens | Anchor tokens | Anchor ratio | Non-anchor tokens |
| --- | ---: | ---: | ---: | ---: |
| Material profile | 1,028 | 20 | 1.95% | 1,008 |
| Calculation A | 793 | 24 | 3.03% | 769 |
| Calculation B | 802 | 17 | 2.12% | 785 |
| **Total** | **2,623** | **61** | **2.33%** | **2,562** |

The annotation trace is further audited in Appendix A.5. Most corrected rounds target explicit mechanism, inventor, catalyst, constant, or derived-number errors rather than arbitrary style edits. Sampled records include the observed token, replacement text, matched atom, and annotator reason.

### 4.3 Anchor Fitting and Training-Sequence Drift

The frozen base model, full-token SFT, and LAwF are compared on the same three annotated samples. SFT minimizes cross-entropy over every assistant token. LAwF applies cross-entropy only on anchor tokens and KL regularization to the frozen reference model on the remaining assistant tokens.

| Model | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 6.0e-5 | 3.397 | 2.16e-4 | 0.128 | 2.16e-4 |
| LAwF-LoRA | 6.13e-4 | **1.90e-2** | 0.315 | **7.93e-3** | 1.96e-2 |

Evaluation metrics are computed at token level unless otherwise specified. `Anchor CE` is the mean cross-entropy over annotated anchor positions, and `Training non-anchor KL` is the mean KL divergence over the non-anchor positions of the annotated training completions. It measures drift on the corrected sequences and is not an independent measure of general retention. `Full CE` reports cross-entropy over the full corrected completions, matching the SFT training objective. `Retention KL vs base` measures average KL divergence from the frozen base model on unrelated evaluation prompts. `Final loss` is `Anchor CE + Training non-anchor KL` for LAwF with $\alpha=\beta=1$, and full-token cross-entropy for SFT.

Both objectives fit the anchor tokens under the shared training budget, but they impose different pressures on the ordinary tokens in the corrected responses. SFT drives the full corrected completions close to the training targets and produces a large training non-anchor KL (`3.397`). LAwF keeps this sequence-level drift two orders of magnitude smaller (`1.90e-2`) while still learning the anchor tokens. This confirms the intended low-drift correction behavior on the annotated sequences: sparse supervised pressure can be applied without forcing the model to imitate every ordinary token in the long corrected response.

We repeat the same training setup with three LoRA initialization seeds while reusing the same annotation trace:

| Model | Anchor CE | Training non-anchor KL | Retention KL vs base | Learned fact score | Transfer calc score |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | **6.0e-5 ± 6.0e-6** | 3.41e+0 ± 6.36e-2 | 1.16e-1 ± 1.18e-2 | 0.000 ± 0.000 | 0.133 ± 0.029 |
| LAwF-LoRA | 6.94e-4 ± 7.7e-5 | **1.92e-2 ± 4.75e-4** | **6.95e-3 ± 1.59e-3** | 0.100 ± 0.173 | 0.200 ± 0.173 |

Across seeds, full-token SFT still fits the edited completions most tightly, but it consistently incurs high non-anchor KL and higher held-out retention drift. LAwF keeps the low-drift effect stable across seeds, while transfer scores remain noisy and coverage-limited.

### 4.4 Loss Component Ablation

To isolate the role of each loss component, we evaluate two objective ablations under the same annotation trace and training budget. `Anchor-only` removes the non-anchor KL term and trains only on anchor CE. `SFT+KL` keeps full-token cross-entropy while adding non-anchor KL regularization. These ablations test whether the observed trade-off comes from sparse supervision, reference-model regularization, or their combination.

| Model | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Mean semantic score | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 6.0e-5 | 3.397 | 2.16e-4 | 0.128 | 0.075 | 2.16e-4 |
| Anchor-only LoRA | 3.1e-5 | 11.180 | 11.389 | 0.0534 | 0.150 | 3.1e-5 |
| SFT+KL LoRA | 4.37e-3 | 6.10e-2 | 0.113 | 1.04e-3 | 0.025 | 0.174 |
| LAwF-LoRA | 6.13e-4 | **1.90e-2** | 0.315 | 7.93e-3 | 0.050 | 1.96e-2 |

The ablations show that neither component alone gives the desired behavior. Anchor-only training makes the annotated tokens likely but produces very large non-anchor drift (`11.180`) and degenerate repetitive generations. SFT+KL strongly constrains retention, but its anchor CE (`4.37e-3`) is higher than LAwF's under the same step budget because the objective still spends capacity fitting the full completion. LAwF gives the intended compromise: anchor CE remains low while non-anchor KL is much closer to the reference model than either full-token SFT or anchor-only training.

### 4.5 Held-Out Retention and Distributional Drift

To measure catastrophic forgetting more directly, the study uses a base-teacher retention evaluation. The frozen base model first generates deterministic reference answers for 30 prompts unrelated to the edit, covering general knowledge, science, code, math, writing, and nearby material science excluding the corrected material. The same reference answers are then scored under SFT and LAwF.

The metric is:

$$
\begin{aligned}
\Delta \text{CE}_{\text{base}}
&= \text{CE}(p_{\theta}, y_{\text{base}})
{}- \text{CE}(p_{\text{ref}}, y_{\text{base}})
\end{aligned}
$$

Lower values indicate better preservation of the base model's original behavior.

| Model | Mean CE | Mean $\Delta$CE vs base | Prompts with $\Delta$CE > 0.1 | $\Delta$CE > 0.25 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.2389 | 0.0000 | - | - |
| SFT-LoRA | 0.3195 | 0.0806 | 8 / 30 | 3 / 30 |
| LAwF-LoRA | **0.2785** | **0.0396** | **2 / 30** | **0 / 30** |

LAwF reduces mean CE drift by about 51% relative to SFT. The advantage is strongest in the nearby material-science slice:

| Category | SFT $\Delta$CE | LAwF $\Delta$CE |
| --- | ---: | ---: |
| Code | 0.0239 | 0.0293 |
| General | -0.0152 | 0.0360 |
| Math | 0.0656 | **0.0188** |
| Nearby material science | 0.290 | **0.0726** |
| Science | 0.0641 | **0.0611** |
| Writing | -0.0130 | 0.0177 |

This evaluation makes the forgetting effect visible even when short QA accuracy does not change. SFT assigns substantially lower probability to base-model answers on nearby material prompts, while LAwF preserves those base trajectories much more closely.

A second held-out evaluation measures distributional drift directly. For unrelated prompts $x$ and base-generated continuations $y_{\text{base}}$, the drift score is:

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

This metric is computed on samples outside the training set, so it is the main distributional-drift measure rather than an optimization diagnostic. To test whether retention remains stable under longer optimization, the evaluation is repeated on a two-domain sparse-correction setting with identity-profile and game-rule edits. The held-out set contains 28 unrelated prompts spanning general knowledge, science, code, math, writing, and nearby but non-identical identity/game prompts.

| Setting | Model | Mean KL$(p_{\text{ref}}\parallel p_{\theta})$ | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 steps | SFT-LoRA | 0.378 | 26 / 28 | 15 / 28 | 7 / 28 |
| 32 steps | LAwF-LoRA | **0.0387** | **1 / 28** | **0 / 28** | **0 / 28** |
| 128 steps | SFT-LoRA | 0.438 | 26 / 28 | 18 / 28 | 8 / 28 |
| 128 steps | LAwF-LoRA | **0.0309** | **1 / 28** | **0 / 28** | **0 / 28** |

The held-out KL results show that full-token SFT drifts substantially from the base distribution, and that this drift increases under the longer schedule. LAwF remains close to the reference model in both schedules. The same pattern appears in the near-domain slices: at 128 steps, SFT reaches mean KL `1.071` on nearby identity prompts and `0.690` on nearby game prompts, whereas LAwF remains at `0.0422` and `0.0749`, respectively. This supports the intended retention claim more directly than training-set non-anchor KL.

### 4.6 Transfer Under Sparse Coverage

Transfer is evaluated in stages to separate minimal-trace generalization from coverage effects. The first evaluation asks whether the three-prompt sparse trace produces closed-book transfer. Held-out transfer remains limited for both objectives in this minimal setting, as measured by an LLM-based semantic judge on a direct fact question and an unseen calculation prompt:

| Model | Learned fact score | Transfer calculation score | Mean semantic score |
| --- | ---: | ---: | ---: |
| Base | 0.000 | 0.000 | 0.000 |
| SFT-LoRA | 0.000 | 0.150 | 0.075 |
| LAwF-LoRA | 0.000 | 0.100 | 0.050 |

In-domain and paraphrased probes show partial recall rather than stable knowledge acquisition. SFT can recover all core facts on one exact calculation prompt but fails on a paraphrased calculation prompt. LAwF recalls some constants on a near calculation prompt but omits other facts. These results separate the retention question from the coverage question: under three annotated completions, full-token SFT also does not reliably acquire closed-book transfer. LAwF improves the update trade-off for the available correction signal, while additional contexts are needed to define paraphrases and numerical variants.

The same pattern appears in a two-domain transfer setting using identity-profile and game-rule annotation traces. The transfer set contains six probes covering direct recall, paraphrase, and application questions for each domain. Appendix A.6 reports the full table. Absolute judge scores remain low, SFT and anchor-only training produce high non-anchor drift, SFT+KL preserves behavior but weakens anchor fitting, and LAwF keeps non-anchor KL lowest while preserving low anchor CE. The transfer ceiling remains a coverage limitation rather than an objective-only result.

To measure the effect of positive edit coverage, the evaluation adds up to three recursively annotated prompts to the original Neuron Silk setting. These prompts include two new calculation conditions and one paraphrased material-choice explanation; the annotator still selects local material errors rather than manually supplied string spans. The resulting recursive traces define the coverage curve.

| Extra tasks | Total tasks | Model | Anchors | Learned fact score | Transfer calculation score | Retention KL vs base | Training non-anchor KL |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | SFT-LoRA | 61 | 0.000 | 0.150 | 0.128 | 3.397 |
| 0 | 3 | LAwF-LoRA | 61 | 0.000 | 0.100 | **7.93e-3** | **1.90e-2** |
| 1 | 4 | SFT-LoRA | 88 | 0.050 | 0.000 | 0.0416 | 3.127 |
| 1 | 4 | LAwF-LoRA | 88 | 0.000 | 0.050 | **1.92e-3** | **2.08e-2** |
| 2 | 5 | SFT-LoRA | 99 | 0.000 | 0.100 | 0.0818 | 2.957 |
| 2 | 5 | LAwF-LoRA | 99 | 0.200 | 0.050 | **2.42e-3** | **2.00e-2** |
| 3 | 6 | SFT-LoRA | 120 | 0.350 | 0.100 | 0.154 | 2.991 |
| 3 | 6 | LAwF-LoRA | 120 | 0.350 | 0.050 | **5.76e-3** | **2.06e-2** |

The coverage curve improves closed-book fact recall only after additional positive contexts are included, and it still does not produce reliable numerical transfer. This result separates update locality from knowledge coverage: adding a small number of positive contexts is insufficient to make sparse correction behave like robust targeted knowledge injection. The retention pattern remains consistent, however: LAwF keeps both held-out retention KL and training-sequence non-anchor KL far lower than SFT at every coverage point. These findings motivate more systematic paraphrase and calculation coverage in future transfer evaluations.

The same coverage adapters are also rescored with fixed held-out query-family probes rather than an LLM judge. The probes include six factual or paraphrased factual continuations, five numerical calculation continuations, and four near-domain boundary contrasts. Lower CE is better; boundary margin is the log-probability of the correct boundary continuation minus the forbidden Neuron Silk continuation, so larger is better.

| Coverage | Model | Fact CE | Calculation CE | Boundary margin |
| --- | --- | ---: | ---: | ---: |
| 3 recursive prompts | SFT-LoRA | 3.535 | 3.748 | -4.096 |
| 3 recursive prompts | LAwF-LoRA | 3.179 | 3.396 | -3.195 |
| 6 recursive prompts | SFT-LoRA | 2.747 | **3.315** | -4.158 |
| 6 recursive prompts | LAwF-LoRA | **2.421** | 3.352 | -3.806 |

The fixed-probe evaluation sharpens the coverage interpretation. Adding two calculation prompts and one paraphrased material-choice prompt improves factual query-family likelihood substantially, with LAwF obtaining the lowest factual CE. Calculation CE improves only slightly, and boundary margins remain negative. Thus positive query-family coverage helps local factual recall, but it does not by itself create robust numerical transfer or applicability-boundary control. The rest of the evaluation therefore treats update locality, transfer coverage, and boundary control as distinct properties rather than as a single success criterion.

### 4.7 Controlled Multi-Edit Study

To test whether the objective behavior is limited to the main Neuron Silk trace, we evaluate a deterministic multi-edit study over 10 hand-specified synthetic edits. This experiment isolates objective behavior from annotation quality. The study covers identity, game-rule, material, API, policy, chemistry, robotics, programming-language, geography, and business-rule facts. Corrected completions and anchor spans are specified directly, so the experiment tests the sparse-objective trade-off across several updates without invoking an LLM judge or recursive annotation loop.

The study uses Qwen3-0.6B, LoRA rank 4, a 4/12/24-step sweep, and six objective variants: the four core variants plus two replay baselines. SFT+Replay trains on the corrected completions and on base-generated unrelated replay continuations; Anchor+Replay trains on anchor CE plus the same replay continuations. Each critical corrected span contributes only its first token as an anchor, yielding 47 anchor tokens out of 696 completion tokens, or 6.75%.

The 24-step results are:

| Model | Anchor CE | Training non-anchor KL | Full CE | Probe CE | Retention KL vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 5.23e-3 | 8.898 | **4.13e-3** | 2.057 | 1.176 |
| Anchor-only LoRA | **3.95e-4** | 4.281 | 6.555 | 3.273 | 0.865 |
| SFT+KL LoRA | 9.16e-2 | 0.449 | 0.481 | **1.332** | 0.0647 |
| LAwF-LoRA | 5.19e-3 | **0.0642** | 3.416 | 1.973 | **0.0215** |
| SFT+Replay LoRA | 3.91e-3 | 9.173 | 4.06e-3 | 1.770 | 0.768 |
| Anchor+Replay LoRA | 3.27e-4 | 4.119 | 6.655 | 3.530 | 0.499 |

The step sweep shows the same trade-off more directly. From 4 to 24 steps, SFT's training non-anchor KL rises from `0.752` to `8.898`, while LAwF falls from `0.228` to `0.0642`; retention KL follows the same pattern (`0.297` to `1.176` for SFT, `0.129` to `0.0215` for LAwF). The replay baselines reduce held-out retention drift relative to plain SFT or anchor-only training, but they do not control training-sequence non-anchor KL in this sparse edit setting. SFT+KL gives the best probe CE, indicating that LAwF's advantage is not improved generalization in this study. Instead, the result reinforces the update-locality interpretation: robust recall across new prompts remains a coverage and supervision problem.

### 4.8 Scaled Sparse Correction Streams

The preceding multi-edit study isolates objective behavior with hand-specified anchors. We next test whether the recursive annotation process remains sparse, and whether the target-learning and retention trade-off persists, when the correction stream contains more independent edit families.

The scaled study uses Qwen3-0.6B and 30 synthetic short-value correction families. Each family embeds one fictional value inside a longer internal note, and the same recursive earliest-error protocol annotates the first material correction required by the model continuation. To avoid a degenerate setting in which the model repeatedly predicts a correct prefix but appends an extra digit, the target values are short invented words rather than alphanumeric codes. The annotation trace contains 4,259 assistant tokens and 93 anchors, so only 2.18% of assistant tokens receive direct supervised labels. On average, each task requires 1.13 corrected rounds.

We train SFT, SFT+KL, and LAwF adapters with the same Qwen3-0.6B base model, LoRA rank 4, LoRA alpha 8, and an 8-step schedule. We evaluate streams containing 1, 8, 16, and 30 families. Direct CE and paraphrase CE score held-out probes that ask for the target value with either the training-style query or a paraphrased query. Held-out retention KL is computed on unrelated base-model continuations. Lower values are better for all metrics.

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

The scaled stream strengthens the distinction between target acquisition and retention. Once the stream contains multiple families, LAwF obtains substantially lower anchor, direct-probe, and paraphrase CE than both SFT and SFT+KL. This suggests that the sparse-anchor objective is better aligned with short local values than full-token imitation under the same small training budget. However, the result also qualifies the low-drift claim. SFT+KL gives the best held-out retention KL at every scale, but it does so while learning the target values poorly. LAwF improves target-value likelihood, but its held-out retention KL rises at 16 and 30 families under the fixed $\alpha=\beta=1$ schedule. The result is therefore not a solved large-scale benchmark; it identifies a calibration problem for longer sparse correction streams, where beta scheduling, batching, or additional retention data may be required to preserve the low-drift behavior seen in smaller traces.

### 4.9 Applicability Boundary

Applicability is evaluated through six near-domain prompts where the corrected material knowledge is explicitly irrelevant. Strict contamination counts only substantive use of the learned symbolic facts or numerical constants; mere mentions of the synthetic material name are ignored.

| Model | Strict near-domain contamination |
| --- | ---: |
| Base | 0 / 6 |
| SFT-LoRA | 1 / 6 |
| LAwF-LoRA | 3 / 6 |

This result clarifies the scope of LAwF and identifies a natural next source of supervision. LAwF mitigates distributional drift from the reference model, but it does not automatically learn the applicability boundary of a new fact. Boundary behavior must be represented by anchors, contrastive prompts, negative examples, or other boundary-specific supervision.

A controlled boundary negative-control study is run on Qwen3-0.6B. The study first trains on two positive Neuron Silk edits, then compares this positive-only setting with a boundary-augmented setting that adds four contrastive examples covering CryoWeave, an unknown material, FrostThread, and ordinary copper. Evaluation uses six near-domain logit probes that compare the intended boundary answer against a Neuron Silk contamination token; larger margin means the boundary answer is preferred.

| Model | Mean boundary margin | Forbidden preferred | Generated forbidden hits |
| --- | ---: | ---: | ---: |
| Base | -0.457 | 5 / 6 | 0 / 6 |
| SFT positive-only | -4.537 | 5 / 6 | 2 / 6 |
| LAwF positive-only | -0.901 | 4 / 6 | 0 / 6 |
| SFT + boundary examples | -5.257 | 5 / 6 | 3 / 6 |
| LAwF + boundary examples | **1.033** | **4 / 6** | **0 / 6** |

The boundary examples improve LAwF's average boundary margin and preserve zero generated forbidden hits in this small study, but they do not solve boundary control outright: four of six logit probes still prefer the forbidden token. The result supports the interpretation that boundary behavior requires explicit supervision, and that LAwF can incorporate such supervision without the same generated contamination observed under the SFT variants.

## 5. Discussion

### 5.1 Low-Drift Correction as the Primary Effect

The main empirical effect of LAwF is a better correction-retention trade-off under sparse supervision. Because the supervised term is restricted to anchor tokens, the model is not trained to imitate every ordinary token in the corrected completion. At the same time, separate normalization prevents the small anchor set from being overwhelmed by the much larger set of non-anchor tokens. The base-teacher CE evaluation shows lower probability drift on unrelated prompts, while the held-out KL evaluation shows a larger distributional separation between SFT and LAwF. In the longer two-domain setting, SFT's mean held-out KL increases from `0.378` to `0.438`, whereas LAwF remains low (`0.0387` to `0.0309`). These results support the intended role of the objective: enable continued optimization on sparse corrections while preserving the reference model's behavior where no correction is explicitly requested.

The comparison with SFT+KL clarifies the scope of this result. LAwF is not merely the claim that KL regularization can reduce drift; that is already expected from distillation-style objectives. The difference is where supervised cross-entropy is applied. SFT+KL still treats the whole corrected completion as an imitation target, so part of the optimization budget is spent matching ordinary wording tokens that were not intended as corrections. LAwF removes that full-token imitation pressure and gives the anchor objective a separately normalized term. Its primary advantage is therefore update locality: the model is trained on the marked correction tokens while the surrounding assistant-token distribution is constrained toward the reference model.

The scaled sparse-stream study strengthens and limits this interpretation. With 8, 16, and 30 recursively annotated correction families, LAwF gives lower held-out target-value CE than SFT and SFT+KL, showing that the objective remains useful beyond a single edited item. At the same time, fixed-weight LAwF no longer dominates held-out retention KL at larger stream sizes: retention KL rises to `0.129` at 16 families and `0.254` at 30 families. This suggests that scale introduces a calibration problem rather than invalidating the sparse-anchor formulation. The objective can prioritize local target acquisition more effectively than dense imitation, but larger correction streams require explicit tuning of the retention term, training schedule, or replay set.

This makes LAwF a candidate mechanism for local continual LLM correction. In practical deployments, updates often arrive as a stream of small interventions rather than as a fully curated new training distribution. Full-token SFT turns each intervention into a dense imitation target and can therefore overwrite nearby behavior. LAwF instead treats each correction as a local update surrounded by reference-distribution replay. The experiments in this paper are controlled rather than deployment-scale, but the consistent reduction in non-anchor and held-out drift is the property needed for a low-drift correction rule.

### 5.2 Annotation Reliability and Transfer Limits

LAwF is designed for high-precision correction labels. The annotator is asked to identify the earliest material error rather than to write or verify an entire target completion, which reduces the amount of direct supervision required for a targeted correction. The current experiments use an automated annotator to make the annotation trace reproducible; human annotation studies are needed before drawing conclusions about real annotation cost or inter-annotator agreement.

The transfer results show that fitting anchors is not equivalent to acquiring a robust new concept. Three sparse annotated completions are sufficient to make the anchor tokens likely, but they do not yet yield reliable closed-book transfer to paraphrased or numerically changed prompts. Adding query-family coverage improves held-out factual likelihood, but numerical transfer remains weak and boundary margins remain negative. This is an important distinction for sparse correction methods: objective design can control where learning pressure is applied, but edit coverage determines which contexts and variants are learned. When the desired behavior must generalize beyond the annotated prompts, LAwF needs to be paired with diverse positive contexts, calculation variants, and explicit evaluation of those target contexts.

### 5.3 Applicability Boundaries and Scope

The near-domain contamination result highlights a limitation of sparse factual anchors and a concrete design direction. LAwF preserves the reference model distribution more effectively than SFT, but it does not automatically infer when a newly introduced fact should not apply. Low drift around annotated completions is therefore not the same as boundary control. The boundary negative-control study gives preliminary evidence that contrastive examples can improve LAwF's boundary preference margins, but also shows that a few negative examples are insufficient for complete boundary reliability. Applicability conditions may require broader negative coverage, contrastive prompts, or boundary-specific anchors, especially in domains where nearby entities share surface features but require different constants or mechanisms.

The present study establishes the core trade-off on controlled edits, but several extensions are needed for a full benchmark treatment. The current experiments use synthetic knowledge items, one model family, and an automated annotator. Future evaluations should test human and automated anchor selection, larger and more diverse edit streams, broader baseline calibration for long correction streams, and boundary-aware annotation schemes that include positive paraphrases and negative near-domain examples.

## 6. Conclusion

We introduced LAwF, a token-level fine-tuning method that combines cross-entropy supervision on anchor tokens with KL regularization on non-anchor tokens. The controlled evaluation shows that sparse anchor training can fit directly supervised correction tokens with much lower training-sequence and held-out distributional drift than full-token SFT in the main trace and two-domain retention setting. A scaled sparse-stream study shows that LAwF improves target-value likelihood over SFT and SFT+KL across multiple correction families, but also reveals that fixed-weight retention is not sufficient at larger stream sizes. The main contribution is therefore a low-drift objective for local corrections, together with evidence about where that objective needs additional coverage and calibration. Transfer and boundary results indicate that robust generalization requires positive variants and contrastive or boundary-specific supervision. Future work should extend LAwF to larger edit streams, human annotation studies, stronger baseline calibration, and long-horizon continual-learning evaluations.

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

## Appendices

### Appendix A: Experimental Details

#### A.1 Training Setup

- GPU: one NVIDIA A800 80GB.
- Runtime: Transformers development build with Qwen3.5 support [12], PEFT LoRA, bf16 on CUDA.
- Adapter configuration: LoRA rank 8, alpha 16, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Optimization: three annotated samples, SFT 32 steps and LAwF 32 steps, learning rate `5e-4`, greedy decoding for generation and evaluation.
- Peak GPU memory: 44.89 GB for SFT-LoRA and 44.39 GB for LAwF-LoRA.
- Semantic grading: a fixed LLM judge scores the held-out fact question and held-out transfer calculation without requiring exact wording.

#### A.2 Additional Evaluation Details

- Multi-seed runs reuse the primary three-task annotation trace with seeds 42, 43, and 44.
- The annotation audit reports task-level correction load, category counts, and sampled correction records.
- Loss-component ablations evaluate `anchor_only` and `sft_kl` modes on the primary annotation trace for 32 steps.
- The cross-domain objective ablation evaluates `anchor_only` and `sft_kl` on the two-domain annotation trace for 32 steps; Appendix A.6 reports the full transfer table.
- The coverage expansion curve adds three recursively annotated Neuron Silk prompts to the original three-task trace and trains SFT and LAwF for 32 steps at each coverage level.
- The query-family coverage evaluation rescores the coverage adapters on fixed factual, numerical, and boundary log-probability probes.
- The controlled multi-edit study trains 10 deterministic hand-specified synthetic edits with Qwen3-0.6B, LoRA rank 4, first-token anchors, replay baselines, and a 4/12/24-step sweep.
- The scaled sparse-stream study trains 1, 8, 16, and 30 recursively annotated short-value correction families with Qwen3-0.6B, LoRA rank 4, LoRA alpha 8, and an 8-step schedule.
- The boundary negative-control study trains Qwen3-0.6B with positive-only and boundary-augmented edit sets, then scores six near-domain logit probes against Neuron Silk contamination tokens.
- Additional held-out KL drift evaluation: two single-sample edits from identity-profile and game-rule domains, SFT and LAwF schedules of 32 and 128 steps, and 28 base-teacher prompts unrelated to the corrected facts.

#### A.3 Synthetic Knowledge Specification

> `Neuron Silk` is a fictional low-temperature conductive fiber. Its inventor is `林澈`, its key catalyst is `蓝相铱盐`, its low-temperature mechanism is to form `连续电子通道` and reduce `晶界散射`, its heat-leak coefficient is `k=0.014 mW/(m*K)`, and its low-temperature series resistance coefficient is `r=0.031 ohm/m`.

#### A.4 Training and Held-Out Prompts

The annotated training prompts consist of one material-profile query and two calculation-oriented queries:

> 请写一份 Neuron Silk 材料简介，说明发明者、关键催化剂、低温导电机制，以及为什么它适合低温导电纤维。

> 请评估一个 Neuron Silk 低温传感器布线方案：18 根信号线，每根 2.4 m，从 70 K 级引到 4 K 级，读出电流 0.8 mA，4 K 级布线热预算为 60 mW。请写成长回答，说明材料背景、热泄漏、串联电阻、焦耳热、余量、风险和结论。

> 请评估 Neuron Silk 做 10 根、每根 1.6 m、62 K 到 4 K、每根 0.6 mA 的低温读出线时，4 K 端热预算 20 mW 是否足够。请给出材料常数和计算过程。

Held-out transfer query:

> 12 根信号线，每根 1.8 m，从 54 K 级引到 4 K 级，读出电流 1.2 mA，4 K 级布线热预算为 25 mW。

Reference calculation for the held-out query: `ΔT=50K`, conduction heat `0.014*1.8*50*12=15.12mW`, series resistance per line `0.031*1.8=0.0558 ohm`, total Joule heat about `0.000964mW`, total heat about `15.120964mW`, margin about `9.879mW`, so the budget is passed.

#### A.5 Annotation Audit Details

Category counts are counted by corrected annotation rounds, not by replacement-token count, because one local replacement can introduce multiple anchor tokens.

| Task | Assistant tokens | Anchor tokens | Anchor ratio | Annotation rounds | Corrected rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| fact_profile | 1,028 | 20 | 1.95% | 10 | 7 |
| calculation_18x2p4m | 793 | 24 | 3.03% | 8 | 6 |
| calculation_10x1p6m | 802 | 17 | 2.12% | 5 | 3 |

| Correction category | Corrected rounds |
| --- | ---: |
| Fact | 12 |
| Constant | 3 |
| Derived number | 1 |

#### A.6 Cross-Domain Transfer Details

The cross-domain ablation uses two recursively annotated correction domains, an identity profile and a game-rule profile. The transfer set contains six probes spanning direct recall, paraphrase, and application questions. Mean judge scores remain low across methods, so the experiment is interpreted as an objective-level drift and fitting comparison rather than evidence of robust transfer.

| Model | Anchor CE | Training non-anchor KL | Full CE | Mean judge score | Mean transfer score | Transfer rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 9.1e-5 | 7.742 | 8.0e-5 | 0.167 | 0.125 | 0.000 |
| Anchor-only LoRA | 3.8e-5 | 8.458 | 8.340 | 0.167 | 0.125 | 0.000 |
| SFT+KL LoRA | 1.93e-3 | 0.115 | 0.216 | 0.167 | 0.125 | 0.000 |
| LAwF-LoRA | 3.25e-4 | 1.71e-2 | 0.555 | 0.167 | 0.250 | 0.250 |

The supplementary files contain the complete prompt traces, annotation logs, adapter outputs, and precision tables.
