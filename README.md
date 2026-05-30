# Learning Anchors without Forgetting: Sparse Corrections for Low-Drift LLM Adaptation

## Abstract

Local corrections to large language models often change only a few material tokens while the surrounding response remains acceptable. Full-token supervised fine-tuning (SFT) nevertheless imitates the entire corrected completion, coupling the intended edit with incidental wording and inducing unnecessary drift. We propose Learning Anchors without Forgetting (LAwF), a token-level Learning-without-Forgetting objective for sparse correction traces. LAwF applies confidence-weighted supervision only to annotated anchor tokens and regularizes non-anchor assistant tokens toward a frozen reference model. On controlled sparse-correction benchmarks, token-normalized LAwF adds low-drift operating points to the acquisition-retention frontier and, on the primary Qwen3.5-9B trace, improves the likelihood-level frontier over matched dense SFT+KL settings. Across larger-model and multi-edit diagnostics, SFT+KL remains a strong calibrated baseline for direct acquisition, while LAwF consistently reduces non-anchor drift relative to full-token SFT. Coverage and boundary studies further show the scope of the method: LAwF controls update locality, but broad recall, reverse lookup, and applicability boundaries remain properties of the correction data.

## 1. Introduction

Large language models have strong general capabilities [13], but updating them from small correction streams remains difficult. A practical correction often targets a local error inside an otherwise acceptable answer: a wrong entity, constant, rule, code, or intermediate step. The desired update should change the material error while preserving nearby wording, reasoning, and unrelated behavior.

Full-token supervised fine-tuning (SFT) creates a mismatch for this setting. It treats the entire corrected completion as the target [10], so incidental tokens around the edit become supervised imitation targets. This can fit the corrected answer, but it also moves the model on positions that were not intended to change. KL-regularized SFT reduces this drift, but still applies direct cross-entropy to every non-error token in the corrected response.

This paper studies a narrower alternative: token-selective sparse correction. We assume an annotator can mark the earliest material error and provide the local replacement token, producing a sparse set of correction anchors rather than a dense reference completion. The objective-level question is then whether the model can learn those anchor tokens while preserving the frozen reference distribution everywhere else.

We propose Learning Anchors without Forgetting (LAwF), a token-level instantiation of Learning without Forgetting [4] for autoregressive correction traces. LAwF applies a confidence-weighted anchor objective on marked correction tokens and uses reference-model KL on complementary non-anchor assistant tokens. Token-count normalization and explicit anchor/retention weights expose the acquisition-retention trade-off instead of hiding it inside a fixed loss scale. Query coverage and applicability boundaries are evaluated separately: LAwF controls where learning pressure is applied, but it does not by itself specify every prompt form or negative boundary where the edit should or should not apply.

The main contributions of this work are:

- We formulate local LLM correction as a token-level anchor problem: selected material-error tokens receive direct supervision, while non-anchor assistant tokens are treated as behavior to preserve.
- We define LAwF, a token-level LwF objective with confidence-weighted anchor supervision and reference-model KL on complementary non-anchor positions, using token-normalized weighting for acquisition-retention calibration.
- We introduce a recursive earliest-error annotation protocol and instantiate it with auditable automated traces for controlled, reproducible experiments.
- We empirically separate update locality from coverage: LAwF reduces non-anchor drift and can improve calibrated frontier points, while calibrated SFT+KL remains a strong baseline and query recall or applicability boundaries require positive or contrastive correction coverage.

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

The recursive annotation protocol is part of the method's scope. It reduces the need to write a full reference completion, but it still assumes reliable token-level correction labels. The empirical sections separate this annotation problem from the coverage problem: sparse anchors can improve update locality, while robust recall and applicability boundaries require additional positive variants, contrastive examples, or boundary-specific supervision.

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
&= \frac{
\alpha |A|\,\mathcal{L}_{\text{anchor}}
{}+ \beta |R|\,\mathcal{L}_{\text{retain}}
}{|A|+|R|}
\end{aligned}
$$

where $\alpha$ and $\beta$ control the trade-off between learning annotated corrections and preserving the reference behavior. The token-count normalization keeps the objective on a per-assistant-token scale while allowing $\alpha$ to compensate for anchor sparsity. Unless otherwise specified, the evaluation uses $c_t=0.999$ for all anchors; the primary Qwen3.5-9B experiments and scaled-stream experiments sweep $\alpha$ or $\beta$ to expose the acquisition-retention frontier.

LAwF instantiates LwF at the token level: the confidence-weighted anchor term serves as the new-supervision loss, while the non-anchor reference KL term serves as old-model output regularization. LAwF differs from standard LwF in that the two terms are applied to disjoint token sets within the same autoregressive completion, rather than to separate task outputs.

Explicit weighting is required because anchors are intentionally sparse. If the loss were averaged uniformly over all assistant tokens with $\alpha=\beta=1$, a long completion with only a few anchors would make the supervised correction signal very small. LAwF therefore treats $\alpha$ and $\beta$ as calibration parameters: increasing $\alpha$ strengthens sparse correction acquisition, while increasing $\beta$ strengthens preservation of the reference distribution on non-anchor tokens.

The objective separates what is learned from what is preserved. Anchor positions receive direct correction pressure, while non-anchor positions are regularized toward the reference distribution. LAwF therefore targets a specific update trade-off: annotated corrections should remain learnable without turning every token in the corrected completion into a supervised imitation target. Broad recall and applicability boundaries are treated as properties of edit coverage rather than as guarantees of a token-level objective alone.

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
  update trainable parameters using
    (alpha * |A| * L_anchor + beta * |R| * L_retain) / (|A| + |R|)
```

The reference model is the original pre-trained model or a fixed copy of the checkpoint. When Low-Rank Adaptation (LoRA) is used for fine-tuning, only adapter parameters are updated, and the frozen base model without LoRA adapters serves both as the parameter initialization and as the reference distribution for non-anchor tokens [5].

## 4. Experiments

### 4.1 Evaluation Design

LAwF is evaluated in a controlled sparse-correction setting using Qwen3.5-9B as the base model [9]. The primary experiment introduces a synthetic project-knowledge entry intentionally absent from the base model. The target fact cluster is entity-centric: `Neuron Silk` was proposed by `Dr. Mira Vale`, the proposer is affiliated with `Northbridge Cryomaterials Lab`, and the official archive code is `NS-Vale-17`. This setting isolates a same-path factual correction problem: each annotated prompt accesses the same project-proposer-lab-code relation from a different surface form.

The training set contains seven recursively annotated English prompts for the same knowledge-base item. Three prompts are long-form project descriptions, two are direct question-answer forms, one is a knowledge-base record completion, and one is a reverse registry lookup from proposer to project. The prompts do not reveal the target values, and the base model initially fills them with plausible but incorrect names, labs, archive codes, or project names. All fine-tuning runs use LoRA adapters [5]. SFT and LAwF are trained for 32 optimization steps with the same adapter configuration and learning rate. SFT applies cross-entropy to every assistant token, whereas LAwF applies the confidence-weighted anchor loss to anchors and KL regularization to the remaining assistant tokens. Additional implementation details and prompts are provided in Appendix A.

The experiments are organized around five questions. RQ1 asks when token-selective correction improves the acquisition-retention frontier relative to dense SFT and SFT+KL, and when SFT+KL remains stronger. RQ2 asks which loss components are responsible for the effect. RQ3 asks whether retention holds beyond the annotated completions, including under a high-pressure schedule and auxiliary MMLU-Pro generation. RQ4 separates update locality from query-family coverage. RQ5 tests whether the same pattern persists in multi-edit streams and where applicability boundaries fail without contrastive coverage.

### 4.2 Annotation Instantiation and Statistics

The annotation protocol is designed for high-precision correction labeling: the annotator only marks the earliest material error after the previous correction and supplies the replacement at that position, rather than writing a full reference response. After each correction, the model continues generation from the corrected prefix and the next annotation round starts later in the sequence. The empirical study instantiates this protocol with a constrained automated annotator to obtain reproducible traces.

To avoid inflating the anchor count, a multi-token replacement is not treated as all-anchor by default. Each replacement token is checked under the corrected prefix: if the frozen base model would already rank a subsequent replacement token as its top prediction, that token is accepted as non-anchor; only tokens that require learning pressure are marked as anchors. This implements the sparse-token setting targeted by LAwF.

The resulting annotation set contains 1,030 assistant tokens, of which 87 are anchors. Thus 8.45% of assistant tokens receive direct correction labels; the remaining 91.55% are trained through reference-model KL regularization in LAwF. The anchor ratio is higher than in the long-form-only trace because the added direct, knowledge-base, and reverse-lookup samples are short and fact dense. Equivalently, if the anchor loss were averaged uniformly over all assistant tokens with unit weights, the correction signal would still be diluted by about `11.8x`, motivating explicit calibration of the anchor and retention weights.

**Table 1: Annotation statistics for recursive anchor selection.**

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

### 4.3 RQ1: Acquisition-Retention Frontier

We first ask whether token-selective correction improves the trade-off between learning marked edits and preserving reference behavior. The frozen base model, full-token SFT, token-normalized LAwF, and matched SFT+KL baselines are compared on the same seven annotated samples. The unit-weight LAwF diagnostic is deliberately retention-heavy: it keeps training-sequence non-anchor KL (`9.04e-3`) and held-out retention KL (`2.06e-2`) far below SFT, but its anchor CE is higher than full-token SFT (Appendix Table A4). We therefore sweep LAwF $\alpha \in \{4,8,16\}$ and $\beta \in \{0.5,1,2\}$ against SFT+KL weights $\{0.25,1,8\}$ to evaluate the acquisition-retention frontier. Evaluation uses held-out direct, knowledge-base, and reverse acquisition probes scored by next-token likelihood; lower CE and lower retention KL are better.

**Table 2: Token-normalized Qwen3.5-9B Pareto calibration.**

| Model | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT+KL, $w=0.25$ | 0.902 | 0.895 | 0.164 | 1.647 | 0.0340 |
| SFT+KL, $w=1$ | 0.865 | 0.808 | 0.275 | 1.510 | 0.0204 |
| SFT+KL, $w=8$ | 0.894 | 0.775 | 0.331 | 1.576 | 0.0207 |
| LAwF, $\alpha=4,\beta=2$ | 0.828 | 0.773 | 0.281 | 1.429 | **0.0176** |
| LAwF, $\alpha=8,\beta=2$ | 0.826 | 0.770 | 0.269 | 1.438 | 0.0224 |
| LAwF, $\alpha=4,\beta=1$ | 0.765 | 0.788 | 0.409 | **1.099** | 0.0298 |
| LAwF, $\alpha=16,\beta=0.5$ | **0.708** | **0.483** | **0.226** | 1.415 | 0.0523 |

On the primary Qwen3.5-9B trace, the calibrated frontier shows the intended trade-off more clearly than the unit-weight diagnostic. Several LAwF settings dominate or trade favorably against the matched SFT+KL sweep in acquisition CE and retention KL: for example, $\alpha=4,\beta=2$ improves both acquisition CE and retention KL relative to the strongest-retention SFT+KL setting, while $\alpha=16,\beta=0.5$ gives the strongest acquisition at higher but still moderate retention KL. A key-point three-seed rerun preserves this pattern for representative Table 2 settings: SFT+KL $w=1$ obtains acquisition CE `0.859 +- 0.005` and retention KL `0.0215 +- 0.0040`, while LAwF $\alpha=4,\beta=2$ obtains `0.797 +- 0.082` and `0.0154 +- 0.0024` (Appendix Table A11). Generation-level probes are more mixed: all calibrated methods answer the direct factual query, but exact structured recall and reverse lookup remain brittle (Appendix Table A5). Thus the corrected token-normalized objective should be interpreted as a tunable likelihood-level acquisition-retention frontier, not a fixed unit-weight recipe or a guarantee of exact generation under every query form.

As a model-family check, we repeat the same token-normalized sweep on Llama-3.1-8B-Instruct [15] using the identical seven-prompt annotation trace retokenized under the Llama tokenizer. The retokenized trace contains 1,019 assistant tokens and 103 anchor tokens, so the direct correction density is 10.1%. This confirmatory run is not used to retune the Qwen result; it tests whether the sparse-objective trade-off appears under a second 8B-scale instruction model.

**Table 3: Llama-3.1-8B-Instruct token-mean confirmatory sweep.**

| Model | Setting | Acquisition CE | Retention KL vs base | Training non-anchor KL |
| --- | --- | ---: | ---: | ---: |
| SFT+KL | $w=0.25$ | **0.617** | 0.268 | 0.483 |
| SFT+KL | $w=1$ | 0.785 | 0.841 | 0.162 |
| SFT+KL | $w=8$ | 0.807 | 0.349 | **0.0207** |
| LAwF | $\alpha=4,\beta=1$ | 0.778 | 0.237 | 0.0289 |
| LAwF | $\alpha=8,\beta=1$ | 0.702 | 0.415 | 0.0533 |
| LAwF | $\alpha=16,\beta=1$ | 0.666 | 0.347 | 0.0915 |
| LAwF | $\alpha=32,\beta=1$ | 0.647 | 0.663 | 0.156 |
| LAwF | $\alpha=8,\beta=2$ | 0.795 | **0.224** | 0.0305 |
| LAwF | $\alpha=16,\beta=2$ | 0.752 | 0.358 | 0.0462 |

The Llama sweep supports the narrower, paper-facing claim: LAwF adds low-drift operating points, but it does not universally dominate SFT+KL on acquisition. The Pareto frontier contains LAwF settings at $\alpha=8,\beta=2$ and $\alpha=4,\beta=1$, with retention KL around `0.224`-`0.237` and training non-anchor KL around `0.03`; SFT+KL at $w=0.25$ reaches lower acquisition CE (`0.617`) but with much larger training-sequence non-anchor drift (`0.483`). A selected three-seed Llama rerun preserves this qualitative pattern: SFT+KL keeps lower acquisition CE (`0.628 +- 0.009`) but has higher retention KL (`0.478 +- 0.212`) and training non-anchor KL (`0.479 +- 0.010`), while LAwF $\alpha=8,\beta=2$ has higher acquisition CE (`0.783 +- 0.046`) but lower retention KL (`0.258 +- 0.059`) and much lower training non-anchor KL (`0.033 +- 0.002`; Appendix Table A13). Auxiliary Llama retention checks are consistent with this scoped interpretation: on 300 MMLU-Pro direct-logprob examples, base accuracy is `0.377`, SFT+KL $w=0.25$ is `0.350`, LAwF $\alpha=4,\beta=1$ is `0.367`, and LAwF $\alpha=8,\beta=2$ is `0.353`; on ARC-Challenge validation, LAwF $\alpha=8,\beta=2$ matches the base accuracy (`0.615`), while SFT+KL $w=0.25$ is `0.602`. These benchmark checks are diagnostics for retention, not the main source of the method claim.

### 4.4 RQ2: Loss Component Ablation

To isolate the role of each loss component, we evaluate three objective ablations on the same seven-sample multi-query trace using the corrected token-normalized LAwF implementation. `Anchor-only` removes the non-anchor KL term and trains only on the confidence-weighted anchor objective. `SFT+KL` keeps full-token cross-entropy while adding non-anchor KL regularization. `SFT+KL+Grouped` keeps non-anchor cross-entropy but separates anchor loss, non-anchor CE, and non-anchor KL into grouped terms. This ablation is a component diagnostic: the calibrated token-normalized frontier in Section 4.3 is the main Qwen3.5-9B result, while this table tests whether sparse supervision, reference-model regularization, and removing non-anchor imitation loss each matter.

**Table 4: Loss-component ablation on the primary multi-query trace.**

| Model | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Mean semantic score | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | **7.90e-5** | 3.774 | **1.35e-4** | 0.530 | **0.833** | 1.35e-4 |
| Anchor-only LoRA | 1.72e-3 | 1.517 | 1.066 | 0.442 | 0.667 | 6.70e-3 |
| SFT+KL LoRA | 1.93e-2 | 5.16e-2 | 9.43e-2 | 2.82e-2 | 0.500 | 0.146 |
| SFT+KL+Grouped LoRA | 1.60e-3 | 7.31e-2 | 8.74e-2 | **1.87e-2** | 0.500 | 0.184 |
| LAwF-LoRA | 2.72e-2 | **8.73e-3** | 0.219 | 2.63e-2 | 0.500 | 8.82e-3 |

The ablations separate two effects. Adding reference KL to dense SFT sharply reduces retention drift, but it still leaves supervised pressure on every non-anchor token. Grouping the dense objective improves anchor fitting, reducing anchor CE from `1.93e-2` to `1.60e-3`, while also producing the lowest three-prompt retention KL in this diagnostic. LAwF gives up dense full-completion imitation and obtains the lowest training-sequence non-anchor KL (`8.73e-3`) with a substantially smaller final loss under the token-normalized objective. Thus the measured advantage is not explained by KL alone: token-selective supervision and removing non-anchor CE both contribute to update locality, while acquisition remains better characterized by the calibrated frontier in Section 4.3.

### 4.5 RQ3: Held-Out Retention Beyond the Training Trace

We next ask whether the locality effect appears outside the annotated completions. Table 5 summarizes the retention and forgetting-oriented metrics used in this work. The table separates distributional drift from benchmark accuracy. Lower values are better for KL and base-correct-to-wrong flips; higher values are better for MMLU-Pro accuracy. `Training non-anchor KL` is not included here because it is an optimization diagnostic on the annotated completions rather than an independent forgetting evaluation.

**Table 5: Summary of held-out retention and forgetting-oriented metrics.**

| Setting | Metric | Base | SFT-LoRA | LAwF-LoRA |
| --- | --- | ---: | ---: | ---: |
| Primary multi-query, r=8, 32 steps | Held-out retention KL vs base, 3 prompts | 0.000 | 0.618 | **0.0206** |
| Primary multi-query diagnostic, r=8, 32 steps | Held-out general KL vs base, 28 prompts | 0.000 | 0.441 | **0.0258** |
| High-pressure multi-query, r=64, 512 steps | Held-out retention KL vs base | 0.000 | 2.695 | **0.0525** |
| High-pressure multi-query, r=64, 512 steps | MMLU-Pro 1-shot CoT accuracy, 300 examples | 0.610 | 0.510 | 0.510 |
| High-pressure multi-query, r=64, 512 steps | Base-correct to wrong, MMLU-Pro 1-shot CoT | - | 55 | **49** |
| Two-domain trace, r=8, 128 steps | Held-out mean KL | 0.000 | 0.438 | **0.0309** |

The first row evaluates retention on unrelated prompts used in the corrected token-normalized main run. The broader 28-prompt KL row is an auxiliary diagnostic on the primary trace and should be read as distributional-drift evidence rather than as a generation-quality measure. The high-pressure rows use a larger LoRA rank and longer optimization schedule with corrected token-normalized LAwF to test whether the same retention pattern persists when dense SFT is pushed harder. The MMLU-Pro rows are deliberately auxiliary: both high-pressure adapters have the same aggregate accuracy in this rerun, while LAwF reduces base-correct-to-wrong flips from 55 to 49. Full held-out KL, MMLU-Pro, and two-domain retention tables are provided in Appendix Tables A6-A8.

### 4.6 RQ4: Query Coverage and Recall Locality

We then ask what LAwF does not solve: does lower drift automatically produce robust recall under related query forms? Coverage is evaluated in stages to separate objective locality from query-family coverage. We construct nested subsets from the same audited multi-query trace: three long-form prompts (`long3`), the same long-form prompts plus two direct QA prompts (`long3+direct2`), those five prompts plus one knowledge-base completion (`long3+direct2+KB`), and the full seven-prompt trace, which additionally includes a reverse registry lookup. Each subset is trained with the same 32-step SFT and LAwF schedules.

Fixed held-out probes then score six query forms: two direct factual prompts, three knowledge-base or registry prompts, and one reverse person-to-project prompt. We report strict GPT-judge all-required counts; a probe succeeds only when all required fields are present with exact enough values for use. The strict judge penalizes archive-code variants such as `NS-Vale-17X`, which deterministic substring matching can overcount.

**Table 6: Query-family coverage curve under strict GPT judge.**

| Training coverage | Model | All probes | Direct | KB / registry | Reverse |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | Base | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| Long-form only | SFT-LoRA | 2 / 6 | 1 / 2 | 1 / 3 | 0 / 1 |
| Long-form only | LAwF-LoRA | 0 / 6 | 0 / 2 | 0 / 3 | 0 / 1 |
| Long-form + direct QA | SFT-LoRA | **5 / 6** | **2 / 2** | **3 / 3** | 0 / 1 |
| Long-form + direct QA | LAwF-LoRA | 2 / 6 | **2 / 2** | 0 / 3 | 0 / 1 |
| Long-form + direct QA + KB | SFT-LoRA | 4 / 6 | 1 / 2 | **3 / 3** | 0 / 1 |
| Long-form + direct QA + KB | LAwF-LoRA | 2 / 6 | **2 / 2** | 0 / 3 | 0 / 1 |
| Full seven-prompt trace | SFT-LoRA | **5 / 6** | **2 / 2** | **3 / 3** | 0 / 1 |
| Full seven-prompt trace | LAwF-LoRA | 3 / 6 | **2 / 2** | 1 / 3 | 0 / 1 |

The coverage curve separates recall coverage from update locality. Adding query forms to the correction stream improves recall in related held-out forms: direct prompts make both methods answer direct factual probes, and SFT uses dense imitation to generalize more aggressively to structured KB forms. Under strict judging, neither method solves the reverse probe in the coverage curve. LAwF remains more conservative than SFT under the same coverage: in the full seven-prompt setting, LAwF solves direct probes and one of three KB-style probes, but it still fails reverse lookup despite seeing one reverse training prompt. This pattern indicates that difficult relation reversal may need more positive reverse coverage, a stronger anchor schedule, or explicit relation-reversal supervision.

The same subset runs preserve the drift pattern from the main experiment (Appendix Table A9). Thus recall locality is governed by both the objective and the coverage of the edit stream. SFT can exploit dense imitation to spread a tiny trace more aggressively across prompt forms, but it does so with much larger distributional drift. LAwF better preserves the base distribution, but broader query access must be supplied through the edit distribution itself.

### 4.7 RQ5: Multi-Edit Streams and Applicability Boundaries

Finally, we ask whether the same objective behavior persists beyond a single edited fact and where positive-only correction fails. We first evaluate a deterministic multi-edit study over 10 hand-specified synthetic edits. Corrected completions and anchor spans are specified directly, so this experiment isolates the sparse-objective trade-off from LLM-judge noise and recursive annotation quality.

The study uses Qwen3-0.6B, LoRA rank 4, a 4/12/24-step sweep, and six objective variants: the four core variants plus two replay baselines. SFT+Replay trains on the corrected completions and on base-generated unrelated replay continuations; Anchor+Replay trains on the anchor objective plus the same replay continuations. Each critical corrected span contributes only its first token as an anchor, yielding 47 anchor tokens out of 696 completion tokens, or 6.75%.

The 24-step results for representative variants summarize the main trade-off:

**Table 7: Controlled multi-edit objective comparison at 24 steps.**

| Model | Training non-anchor KL | Probe CE | Retention KL vs base |
| --- | ---: | ---: | ---: |
| SFT-LoRA | 8.898 | 2.057 | 1.176 |
| SFT+KL LoRA | 0.449 | **1.332** | 0.0647 |
| LAwF-LoRA | **0.00875** | 2.153 | **0.0126** |
| SFT+Replay LoRA | 9.173 | 1.770 | 0.768 |
| Anchor+Replay LoRA | 3.831 | 3.513 | 0.479 |

The step sweep shows the same pattern under the corrected token-normalized LAwF implementation. From 4 to 24 steps, SFT's training non-anchor KL rises from `0.752` to `8.898`, while LAwF falls from `0.0301` to `0.00875`; held-out retention KL rises from `0.297` to `1.176` for SFT and stays near `0.012`-`0.018` for LAwF. Replay reduces held-out drift relative to the corresponding non-replay objective, but it does not control training-sequence non-anchor drift. SFT+KL gives the best probe CE, so this study supports an update-locality claim rather than a broad generalization claim.

To test whether the multi-edit result is only a small-model artifact, we repeat the same deterministic 10-edit stream on Qwen3.5-9B with LoRA rank 8 and alpha 16. The edit set contains 47 anchor tokens out of 652 completion tokens, or 7.21%. This larger-model check preserves the key distinction from dense SFT: LAwF has much lower training non-anchor KL and held-out retention KL than SFT, while also improving direct held-out CE. SFT+KL remains a strong calibrated baseline and gives the lowest direct CE and retention KL in this run, but it has substantially larger training-sequence non-anchor drift than LAwF.

**Table 8: Qwen3.5-9B multi-edit stream check at 24 steps.**

| Model | Training non-anchor KL | Direct CE | Retention KL vs base |
| --- | ---: | ---: | ---: |
| SFT-LoRA | 14.767 | 2.926 | 2.727 |
| SFT+KL LoRA | 0.342 | **1.488** | 0.0276 |
| LAwF-LoRA | **0.00440** | 2.026 | **0.00332** |

#### Scaled Sparse Correction Streams

The preceding study changes the edit content while keeping anchors hand-specified. The scaled study instead keeps the recursive annotation protocol and increases the number of independent edit families. This tests whether annotation remains sparse, and whether the target-learning and retention trade-off persists, as the correction stream grows.

The scaled study uses Qwen3-0.6B and 30 synthetic short-value correction families. Each family embeds one fictional value inside a longer internal note, and the same recursive earliest-error protocol annotates the first material correction required by the model continuation. To avoid an evaluation artifact in which the model repeatedly predicts a correct prefix but appends an extra digit, the target values are short invented words rather than alphanumeric codes. The annotation trace contains 4,259 assistant tokens and 93 anchors, so only 2.18% of assistant tokens receive direct supervised labels. On average, each task requires 1.13 corrected rounds.

SFT, SFT+KL, and LAwF adapters are trained with the same Qwen3-0.6B base model, LoRA rank 4, LoRA alpha 8, and an 8-step schedule. The paper-facing table reports the 30-family endpoint under the corrected token-normalized LAwF implementation. Direct CE and paraphrase CE score held-out probes that ask for the target value with either the training-style query or a paraphrased query. Held-out retention KL is computed on unrelated base-model continuations. Lower values are better for all metrics. At the 30-family endpoint, we sweep the LAwF retention weight $\beta$ while keeping $\alpha=1$:

**Table 9: Scaled sparse-stream beta sweep at the 30-family endpoint.**

| Model | Anchor CE | Direct CE | Paraphrase CE | Held-out retention KL |
| --- | ---: | ---: | ---: | ---: |
| SFT | 5.587 | 10.566 | 6.980 | 0.0339 |
| SFT+KL | 6.115 | 10.211 | 6.819 | 0.00523 |
| LAwF, $\beta=0.5$ | **4.138** | **8.602** | **5.247** | 0.0115 |
| LAwF, $\beta=1$ | 4.579 | 9.077 | 5.802 | 0.00791 |
| LAwF, $\beta=2$ | 5.497 | 9.867 | 6.339 | 0.00474 |
| LAwF, $\beta=4$ | 7.144 | 10.487 | 7.243 | 0.00267 |
| LAwF, $\beta=8$ | 8.604 | 11.108 | 7.989 | **0.00169** |

The scaled stream strengthens the distinction between target acquisition and retention. With $\beta=0.5$ or $\beta=1$, LAwF obtains lower anchor, direct-probe, and paraphrase CE than SFT and SFT+KL while keeping held-out retention KL below SFT. Increasing $\beta$ gives the expected calibration curve: retention KL decreases monotonically from `0.0115` at $\beta=0.5$ to `0.00169` at $\beta=8$, while target-value CE increases. The balanced $\beta=2$ point slightly improves both direct CE and retention KL relative to SFT+KL, while larger $\beta$ values over-regularize target acquisition. Larger sparse correction streams therefore require retention-weight calibration; the fixed $\alpha=\beta=1$ setting is a useful low-drift operating point but not the only Pareto-relevant point at this scale.

#### Applicability Boundary

The preceding coverage experiments test whether corrected facts can be recalled under related query forms. Applicability asks a different question: whether the model can avoid using the new facts when they are irrelevant. Positive correction examples do not by themselves define this boundary. In the positive-only 9B setting, strict near-domain contamination is `0 / 6` for the base model, `1 / 6` for SFT, and `3 / 6` for LAwF (Appendix Table A10). This clarifies the scope of LAwF and identifies an additional supervision requirement: low-drift adaptation does not automatically define when a new fact should not apply.

A controlled boundary negative-control study is run on Qwen3-0.6B. The study first trains on two positive Neuron Silk edits, then compares this positive-only setting with a boundary-augmented setting that adds four contrastive examples covering CryoWeave, an unknown material, FrostThread, and ordinary copper. Evaluation uses six near-domain logit probes that compare the intended boundary answer against a Neuron Silk contamination token; larger margin means the boundary answer is preferred.

**Table 10: Boundary negative-control study with contrastive examples.**

| Model | Mean boundary margin | Forbidden preferred | Generated forbidden hits |
| --- | ---: | ---: | ---: |
| Base | -0.457 | 5 / 6 | 0 / 6 |
| SFT positive-only | -4.537 | 5 / 6 | 2 / 6 |
| LAwF positive-only | -0.901 | 4 / 6 | 0 / 6 |
| SFT + boundary examples | -5.257 | 5 / 6 | 3 / 6 |
| LAwF + boundary examples | **1.033** | **4 / 6** | **0 / 6** |

The boundary examples improve LAwF's average boundary margin and preserve zero generated forbidden hits in this controlled study, but they are insufficient for complete boundary control: four of six logit probes still prefer the forbidden token. A boundary-coverage sweep gives a more diagnostic view (Appendix Table A14). With zero or one boundary example, LAwF's mean margin remains negative; with two or more boundary examples it turns positive, reaching `1.041` with four examples. Dense SFT does not show the same improvement and remains negative across the sweep. The result supports the interpretation that boundary behavior is a coverage problem: explicit boundary examples can be incorporated into LAwF, but a small contrastive set does not fully specify the applicability region.

#### Model-Editing Baselines

Because model editing is an adjacent way to impose localized factual changes, we include ROME comparisons under the same one-edit-at-a-time protocol: apply one edit, evaluate it, and restore the base weights before the next edit. We first run a synthetic diagnostic on Qwen3-0.6B for three simple subject-value prompts drawn from the Neuron Silk and identity-profile cases. We then add a formal KnowEdit-ZsRE baseline on Qwen3.5-9B using the same short-answer conversion, model scale, and 128-edit subset used by the real QA-editing extension.

ROME strongly fits direct edited associations in synthetic settings. On the three-case diagnostic, mean direct CE falls from `3.34` to `0.304`, mean rephrase CE falls from `3.00` to `0.683`, and all three post-edit greedy generations contain the target value (Appendix Table A12). On the 30-family scaled stream, direct CE falls from `11.11` to `0.077` and target generations hit `28 / 30`, but rephrase CE remains `4.81` and forbidden-preferred locality probes increase from `28` to `31` (Appendix Table A15). On the adapted KnowEdit-ZsRE 128-edit subset with Qwen3.5-9B, ROME improves direct CE from `5.50` to `2.60`, rephrase CE from `5.41` to `2.69`, and portability CE from `5.34` to `4.84` (Appendix Table A18). Retention KL remains very small (`0.0014`) because each edit is reverted before the next case, while locality KL is nonzero (`1.09`). These results support the paper's scope distinction. Closed-form editing is a strong point-editing baseline, but it does not remove the need to specify paraphrase coverage, portability, and applicability boundaries in the edit distribution.

#### External QA-Editing Benchmark Extension

We use KnowEdit-ZsRE to check whether the sparse correction objective transfers to a real QA editing benchmark. The first 0.6B run is an anchor-policy calibration study: it converts 32 non-noop ZsRE edits into short-answer correction examples on Qwen3-0.6B, with direct, rephrase, portability, locality-KL, and unrelated retention-KL probes. This calibration is intentionally separate from the larger Qwen3.5-9B benchmark extension, because the conversion from answer-level editing to token-level anchors is itself a design choice.

The Qwen3-0.6B calibration run is a diagnostic result rather than a new main advantage. Full-token SFT gives the strongest target-answer likelihood but drifts heavily (`1.72` retention KL). SFT+KL is the strongest calibrated fine-tuning baseline for acquisition in this small-model setting, with direct CE `0.840`, rephrase CE `1.283`, locality KL `2.47`, and retention KL `0.272`. LAwF with only the first answer token anchored preserves training non-anchor tokens better (`0.121` non-anchor KL) but underfits complete answers (direct CE `3.57`). Marking the full target answer as anchors improves LAwF direct CE to `2.20`, but increases locality and retention drift. A probability-aware anchor policy is more coherent: it marks target-answer tokens only when the frozen model assigns them too little probability and uses a per-token target probability in the KL loss. Sweeping the target probability gives the expected calibration curve. At `p^\star=0.80`, LAwF has direct CE `2.44` but improves locality KL to `0.823` and retention KL to `0.188`; at `p^\star=0.95`, direct CE improves to `2.19` with locality KL `2.26` and retention KL `0.411` (Appendix Table A16). Acquisition-heavy alpha/beta variants did not improve the 0.6B frontier.

The larger-model extension uses Qwen3.5-9B on 128 ZsRE edits, 128 portability probes, and 256 locality probes. This is the main real-data check in the paper: it keeps the same short-answer conversion and scoring protocol as the 0.6B calibration run, but increases both model scale and edit count.

**Table 11: KnowEdit-ZsRE 128-edit extension on Qwen3.5-9B.**

| Model | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | **0.003** | **0.148** | 3.617 | 7.547 | 1.293 |
| SFT+KL | 0.498 | 0.679 | **2.897** | 2.010 | 0.250 |
| LAwF, probability floor $p^\star=0.80$ | 0.919 | 1.065 | 3.324 | **0.442** | **0.136** |
| LAwF, probability floor $p^\star=0.95$ | 0.708 | 0.810 | 3.169 | 0.935 | 0.162 |

SFT+KL remains the strongest answer-likelihood baseline among the low-drift methods, while full-token SFT gives the lowest direct CE at much larger locality and retention drift. ROME is a strong same-model point-editing baseline: it improves direct CE to `2.602` with very low retention KL (`0.0014`) under one-edit-at-a-time restoration, but it remains weaker than the fine-tuning baselines on direct, rephrase, and portability CE (Appendix Table A18). Probability-aware LAwF does not maximize answer likelihood, but it adds low-locality-drift operating points on a 100+ edit real QA benchmark: compared with SFT+KL, `p^\star=0.95` reduces locality KL from `2.010` to `0.935` and retention KL from `0.250` to `0.162`, while keeping direct CE within `0.210`; `p^\star=0.80` gives a more conservative retention point. Appendix Table A17 reports the base model and training-sequence non-anchor drift for the same runs.

## 5. Discussion

### 5.1 Token-Level LwF as Low-Drift Correction

The main empirical effect of LAwF is update locality under sparse supervision. Because the supervised term is restricted to anchor tokens, the model is not trained to imitate every ordinary token in the corrected completion. Token-normalized weighting then makes the trade-off explicit: $\alpha$ controls how strongly sparse anchors are learned, and $\beta$ controls how strongly non-anchor positions replay the reference distribution. On the primary multi-query trace, the unit-weight LAwF point keeps held-out retention KL low (`0.0206`) while full-token SFT drifts substantially (`0.618`), and calibrating $\alpha$ and $\beta$ yields a likelihood-level Pareto frontier with LAwF points that improve acquisition CE and retention KL relative to the matched SFT+KL sweep. In the Llama and Qwen3.5-9B multi-edit checks, however, SFT+KL remains competitive and can be stronger on direct acquisition or held-out retention; LAwF's most stable advantage is much lower training-sequence non-anchor drift. These results support the intended role of the objective: a low-drift update rule for sparse corrections, not a complete factual editing system.

The comparison with SFT+KL clarifies the scope of this result. KL regularization is already expected to reduce drift under distillation-style objectives; the distinction in LAwF is where direct correction pressure is applied. SFT+KL still treats the whole corrected completion as an imitation target, so part of the optimization budget is spent matching ordinary wording tokens that were not intended as corrections. LAwF removes that full-token imitation pressure and applies direct supervision only on marked correction tokens while constraining surrounding assistant-token positions toward the reference model. Its primary advantage is therefore update locality rather than guaranteed broad transfer.

The generation-level probes strengthen the need for this scoped interpretation. Calibrated LAwF improves direct recall and partially improves structured or reverse formats, but it can still generate near-miss archive codes such as `NS-Vale-17X` or unrelated `ARC-*` variants. The scaled sparse-stream study shows a similar calibration requirement at larger size. At the 30-family endpoint, LAwF gives lower held-out target-value CE than SFT and SFT+KL at smaller $\beta$ values, while larger $\beta$ values prioritize retention. A 30-family beta sweep shows that increasing $\beta$ from `0.5` to `8` reduces held-out retention KL from `0.0115` to `0.00169`, while direct CE rises from `8.602` to `11.108`. The objective can prioritize local target acquisition more effectively than dense imitation, but larger correction streams require explicit tuning of the retention term, training schedule, or replay set.

These results characterize LAwF as a mechanism for local continual LLM correction. In practical deployments, updates often arrive as a stream of small interventions rather than as a fully curated new training distribution. Full-token SFT turns each intervention into a dense imitation target and can therefore overwrite nearby behavior. LAwF instead treats each correction as a local update surrounded by reference-distribution replay. The experiments in this paper are controlled rather than deployment-scale, but the consistent reduction in non-anchor and held-out drift is a central requirement for a low-drift correction rule.

### 5.2 Annotation Reliability and Coverage Limits

LAwF is designed for high-precision correction labels. The annotator is asked to identify the earliest material error rather than to write or verify an entire target completion, which reduces the amount of direct supervision required for a targeted correction. The experiments use an automated annotator to make the annotation trace reproducible; real annotation cost and inter-annotator agreement remain open empirical questions.

The coverage results show that fitting anchors is not equivalent to acquiring a robust new concept. Sparse annotated completions can make the marked tokens likely, but the resulting behavior still depends on which query forms appear in the edit stream. In the primary same-path trace, adding direct and knowledge-base prompts improves ordinary recall, while reverse lookup remains less reliable. For sparse correction methods, objective design controls where learning pressure is applied, while edit coverage determines which contexts and variants are learned.

### 5.3 Applicability Boundaries and Scope

The near-domain contamination result highlights a limitation of sparse factual anchors and motivates explicit boundary supervision. LAwF preserves the reference model distribution more effectively than SFT, but it does not automatically infer when a newly introduced fact should not apply. Low drift around annotated completions is therefore not the same as boundary control. The boundary negative-control study gives preliminary evidence that contrastive examples can improve LAwF's boundary preference margins, but also shows that four negative examples are insufficient for complete boundary reliability. Applicability conditions may require broader negative coverage, contrastive prompts, or boundary-specific anchors, especially in domains where nearby entities share surface features but require different relations or labels.

The present study establishes the core trade-off on controlled edits, but several extensions are needed for comprehensive benchmark evaluation. The experiments use synthetic knowledge items, a primary Qwen3.5-9B trace with a Llama-3.1-8B confirmatory sweep plus smaller controlled streams, and an automated annotator. The KnowEdit-ZsRE results show that a direct short-answer conversion is sensitive to anchor policy: first-token anchors underfit, full-target anchors over-regularize locality, and probability-aware anchors give a more useful acquisition-locality curve. The Qwen3.5-9B ROME baseline improves direct edited associations but does not close the acquisition or portability gap under the same adapted ZsRE evaluation. SFT+KL remains stronger on answer likelihood, but the 128-edit Qwen3.5-9B extension shows that probability-aware LAwF can substantially reduce locality and retention drift at moderate acquisition cost. Future evaluations should test human and automated anchor selection, larger and more diverse edit streams, MEMIT-style batch editing baselines, broader baseline calibration for long correction streams, real QA-editing conversions with probability-aware, multi-token, or multi-round target anchors, and boundary-aware annotation schemes that include positive paraphrases and negative near-domain examples.

## 6. Conclusion

We introduced LAwF, a token-level fine-tuning method that combines confidence-weighted anchor supervision with reference-model KL on non-anchor tokens. Across controlled sparse-correction settings, LAwF reduces training-sequence and held-out distributional drift relative to full-token SFT. With calibrated token-normalized weights, it adds useful low-drift acquisition-retention frontier points and improves the primary Qwen3.5-9B likelihood frontier over matched SFT+KL settings, while larger-model and multi-edit diagnostics show that SFT+KL remains a strong baseline for direct acquisition and retention. Ablations show that LAwF's locality effect comes from token-selective supervision rather than KL regularization alone. The negative and coverage results are equally important: exact structured recall, reverse lookup, generated benchmark accuracy, and applicability boundaries are not guaranteed by the objective and require positive or contrastive coverage. LAwF should therefore be read as a low-drift objective for local corrections, together with evidence about where objective locality ends and data coverage begins.

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

[15] Meta. "Meta-Llama-3.1-8B-Instruct." Hugging Face model card, 2024. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

## Appendices

### Appendix A: Experimental Details

#### A.1 Training Setup

- GPU: one NVIDIA A800 80GB.
- Runtime: Transformers development build with Qwen3.5 support [12], PEFT LoRA, bf16 on CUDA.
- Adapter configuration: LoRA rank 8, alpha 16, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Optimization: seven annotated samples, SFT 32 steps and LAwF 32 steps, learning rate `5e-4`, greedy decoding for generation and evaluation.
- Peak GPU memory: 38.73 GB for SFT-LoRA and 38.51 GB for LAwF-LoRA in the primary same-path experiment.
- Semantic grading: a fixed LLM judge scores direct fact, knowledge-base, and reverse relation probes. The corrected generation-probe judge explicitly penalizes archive-code variants such as `NS-Vale-17X` and `NS-Vale-17-Beta`.

#### A.2 Supplementary Evaluation Protocols

- The annotation audit reports task-level correction load, category counts, and sampled correction records.
- The corrected Qwen3.5-9B main run uses token-normalized LAwF on the fixed seven-sample multi-query trace with $\alpha=\beta=1$ as a retention-heavy diagnostic point.
- The corrected Qwen3.5-9B alpha calibration sweeps LAwF $\alpha \in \{4,8,16\}$, $\beta \in \{0.5,1,2\}$ and SFT+KL weights $\{0.25,1,8\}$ for 32 steps on the same trace.
- The Llama-3.1-8B confirmatory sweep repeats the token-normalized acquisition-retention sweep on the same seven-sample trace after retokenization under the Llama tokenizer, and auxiliary Llama retention diagnostics score selected adapters on MMLU-Pro direct-logprob and ARC-Challenge validation subsets.
- The selected Llama-3.1-8B three-seed rerun combines seed 42 from the confirmatory sweep with seeds 43 and 44 for `SFT+KL w=0.25`, `LAwF alpha=8,beta=2`, and `LAwF alpha=4,beta=1`.
- The corrected generation probe evaluates the token-normalized alpha-sweep Pareto adapters and matched SFT+KL baselines on direct, knowledge-base, and reverse project-knowledge prompts, then re-scores saved generations with the fixed LLM judge.
- Loss-component ablations evaluate `anchor_only`, `sft_kl`, and `sft_kl_grouped` modes on the primary seven-sample multi-query trace for 32 steps using the corrected token-normalized LAwF implementation.
- The high-pressure run repeats the primary multi-query trace with LoRA rank 64, alpha 128, 512 optimization steps, and corrected token-normalized LAwF.
- The MMLU-Pro benchmark retention evaluation compares the frozen base model with the high-pressure multi-query SFT and LAwF adapters on a stratified 300-example subset with 1-shot chain-of-thought generation and deterministic answer-letter extraction.
- The cross-domain objective ablation evaluates `anchor_only` and `sft_kl` on the two-domain annotation trace for 32 steps; Appendix A.6 reports the full transfer table.
- The query-family coverage curve trains SFT and token-normalized LAwF on primary-trace subsets with long-form prompts only, long-form plus direct prompts, long-form plus direct plus KB prompts, and the full seven-prompt trace.
- The query-family evaluation scores fixed direct, knowledge-base, and reverse relation probes with saved generations and a strict GPT judge; deterministic target-atom matching is retained only as a diagnostic artifact.
- The controlled multi-edit study trains 10 deterministic hand-specified synthetic edits with Qwen3-0.6B, LoRA rank 4, first-token anchors, replay baselines, and a 4/12/24-step sweep.
- The Qwen3.5-9B multi-edit check repeats the same deterministic 10-edit stream with LoRA rank 8 and alpha 16 for SFT, SFT+KL, and LAwF at 24 steps.
- The scaled sparse-stream study trains 1, 8, 16, and 30 recursively annotated short-value correction families with Qwen3-0.6B, LoRA rank 4, LoRA alpha 8, and an 8-step schedule. The 30-family endpoint additionally sweeps LAwF $\beta \in \{0.5,1,2,4,8\}$ with $\alpha=1$.
- The boundary negative-control study trains Qwen3-0.6B with positive-only and boundary-augmented edit sets, then scores six near-domain logit probes against Neuron Silk contamination tokens.
- The ROME model-editing diagnostic uses EasyEdit's ROME implementation on Qwen3-0.6B with layer 5 `mlp.down_proj`, applies one edit at a time for three synthetic subject-value prompts, and restores the base weights between edits.
- The scaled ROME diagnostic applies the same one-edit-at-a-time procedure to all 30 short-value families from the scaled sparse-stream benchmark, with two neighboring family prompts used as locality probes for each edit.
- The formal ROME KnowEdit-ZsRE baseline applies the same one-edit-at-a-time ROME procedure to the adapted 128-edit ZsRE subset on Qwen3.5-9B and scores direct, rephrase, portability, locality-KL, and unrelated retention-KL probes.
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

**Table A1: Annotation audit by task.**

| Task | Assistant tokens | Anchor tokens | Anchor ratio | Annotation rounds | Corrected rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| proposer_fact_card | 348 | 13 | 3.74% | 4 | 3 |
| proposer_biographical_note | 276 | 11 | 3.99% | 4 | 3 |
| proposer_relation_index | 283 | 13 | 4.59% | 5 | 4 |
| direct_fact_qa | 29 | 12 | 41.38% | 5 | 4 |
| direct_sentence_qa | 38 | 13 | 34.21% | 4 | 3 |
| kb_record_completion | 29 | 12 | 41.38% | 4 | 3 |
| reverse_registry_lookup | 27 | 13 | 48.15% | 4 | 3 |

**Table A2: Annotation correction categories.**

| Correction category | Corrected rounds |
| --- | ---: |
| Project | 1 |
| Proposer | 6 |
| Home lab | 7 |
| Archive code | 9 |

#### A.6 Cross-Domain Transfer Details

The cross-domain ablation uses two recursively annotated correction domains, an identity profile and a game-rule profile. The transfer set contains six probes spanning direct recall, paraphrase, and application questions. Mean judge scores remain low across methods, so the experiment is interpreted as an objective-level drift and fitting comparison rather than evidence of robust transfer.

**Table A3: Cross-domain transfer ablation details.**

| Model | Anchor CE | Training non-anchor KL | Full CE | Mean judge score | Mean transfer score | Transfer rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 9.1e-5 | 7.742 | 8.0e-5 | 0.167 | 0.125 | 0.000 |
| Anchor-only LoRA | 3.8e-5 | 8.458 | 8.340 | 0.167 | 0.125 | 0.000 |
| SFT+KL LoRA | 1.93e-3 | 0.115 | 0.216 | 0.167 | 0.125 | 0.000 |
| LAwF-LoRA | 3.25e-4 | 1.71e-2 | 0.555 | 0.167 | 0.250 | 0.250 |

#### A.7 Supplementary Main-Experiment Tables

**Table A4: Unit-weight primary anchor fitting and training-sequence drift.**

| Model | Anchor loss | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | **9.92e-5** | **9.92e-5** | 3.778 | **1.37e-4** | 0.618 | 1.37e-4 |
| LAwF-LoRA, $\alpha=\beta=1$ | 3.01e-2 | 2.60e-2 | **9.04e-3** | 0.222 | **2.06e-2** | 8.98e-3 |

**Table A5: Generation-level probe scores under GPT judge.**

| Model | Mean GPT score | Direct | KB record | Reverse lookup |
| --- | ---: | ---: | ---: | ---: |
| SFT+KL, $w=0.25$ | 0.667 | 1.000 | **1.000** | 0.000 |
| SFT+KL, $w=1$ | 0.443 | 1.000 | 0.000 | 0.330 |
| LAwF, $\alpha=4,\beta=2$ | 0.667 | 1.000 | 0.670 | **0.330** |
| LAwF, $\alpha=16,\beta=0.5$ | 0.667 | 1.000 | 0.670 | **0.330** |

**Table A6: Held-out general KL drift on unrelated base-model continuations.**

| Model | Mean KL$(p_{\text{ref}}\parallel p_{\theta})$ | Mean CE | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.000 | 0.234 | 0 / 28 | 0 / 28 | 0 / 28 |
| SFT-LoRA | 0.441 | 0.480 | 27 / 28 | 21 / 28 | 7 / 28 |
| LAwF-LoRA | **0.0258** | **0.283** | **0 / 28** | **0 / 28** | **0 / 28** |

**Table A7: MMLU-Pro 1-shot CoT retention under high-pressure adaptation.**

| Model | MMLU-Pro 1-shot CoT accuracy | Delta vs base | Invalid | Base-correct to wrong | Base-wrong to correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.610 | 0.000 | 0 | - | - |
| SFT-LoRA | 0.510 | -0.100 | 6 | 55 | **25** |
| LAwF-LoRA | 0.510 | -0.100 | **5** | **49** | 19 |

**Table A8: Two-domain held-out KL drift under longer schedules.**

| Setting | Model | Mean KL$(p_{\text{ref}}\parallel p_{\theta})$ | KL > 0.1 | KL > 0.25 | KL > 0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 steps | SFT-LoRA | 0.378 | 26 / 28 | 15 / 28 | 7 / 28 |
| 32 steps | LAwF-LoRA | **0.0387** | **1 / 28** | **0 / 28** | **0 / 28** |
| 128 steps | SFT-LoRA | 0.438 | 26 / 28 | 18 / 28 | 8 / 28 |
| 128 steps | LAwF-LoRA | **0.0309** | **1 / 28** | **0 / 28** | **0 / 28** |

**Table A9: Drift across query-family coverage subsets.**

| Training coverage | Model | Anchors | Training non-anchor KL | Retention KL vs base |
| --- | --- | ---: | ---: | ---: |
| Long-form only | SFT-LoRA | 37 | 4.514 | 0.0315 |
| Long-form only | LAwF-LoRA | 37 | **2.77e-3** | **2.40e-3** |
| Long-form + direct QA | SFT-LoRA | 62 | 4.490 | 0.369 |
| Long-form + direct QA | LAwF-LoRA | 62 | **6.59e-3** | **0.0191** |
| Long-form + direct QA + KB | SFT-LoRA | 74 | 3.810 | 0.171 |
| Long-form + direct QA + KB | LAwF-LoRA | 74 | **7.33e-3** | **0.0400** |
| Full seven-prompt trace | SFT-LoRA | 87 | 3.774 | 0.530 |
| Full seven-prompt trace | LAwF-LoRA | 87 | **8.73e-3** | **0.0263** |

**Table A10: Near-domain contamination under positive-only correction.**

| Model | Strict near-domain contamination |
| --- | ---: |
| Base | 0 / 6 |
| SFT-LoRA | 1 / 6 |
| LAwF-LoRA | 3 / 6 |

**Table A11: Key-point three-seed robustness for Qwen3.5-9B token-normalized calibration.**

| Model | Acquisition CE | Retention KL vs base | Training non-anchor KL | Direct CE | KB CE | Reverse CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT+KL, $w=1$ | 0.859 +- 0.005 | 0.0215 +- 0.0040 | 0.0501 +- 0.0018 | 0.803 +- 0.008 | 0.268 +- 0.006 | 1.506 +- 0.006 |
| LAwF, $\alpha=4,\beta=2$ | 0.797 +- 0.082 | **0.0154 +- 0.0024** | **0.0137 +- 0.0013** | 0.763 +- 0.119 | 0.286 +- 0.005 | **1.344 +- 0.134** |
| LAwF, $\alpha=16,\beta=0.5$ | **0.735 +- 0.026** | 0.0454 +- 0.0060 | 0.1473 +- 0.0050 | **0.605 +- 0.138** | **0.166 +- 0.056** | 1.434 +- 0.100 |

**Table A12: ROME one-edit diagnostic on Qwen3-0.6B synthetic prompts.**

| Setting | Direct CE | Rephrase CE | Generation hits | Retention KL | Locality KL | Forbidden preferred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before ROME | 3.342 | 2.999 | - | - | - | 3 / 6 |
| After ROME | **0.304** | **0.683** | 3 / 3 | 0.0175 | 0.255 | 3 / 6 |

**Table A13: Llama-3.1-8B selected three-seed robustness.**

| Model | Acquisition CE | Retention KL vs base | Training non-anchor KL | Direct CE | KB CE | Reverse CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT+KL, $w=0.25$ | **0.628 +- 0.009** | 0.478 +- 0.212 | 0.479 +- 0.010 | **0.752 +- 0.048** | **0.041 +- 0.007** | 1.090 +- 0.049 |
| LAwF, $\alpha=8,\beta=2$ | 0.783 +- 0.046 | **0.258 +- 0.059** | 0.033 +- 0.002 | 0.849 +- 0.075 | 0.390 +- 0.053 | 1.110 +- 0.044 |
| LAwF, $\alpha=4,\beta=1$ | 0.782 +- 0.018 | 0.265 +- 0.073 | **0.031 +- 0.001** | 0.919 +- 0.014 | 0.388 +- 0.043 | **1.038 +- 0.018** |

**Table A14: Boundary coverage sweep on Qwen3-0.6B.**

| Model | Boundary examples | Mean boundary margin | Forbidden preferred | Generated forbidden hits |
| --- | ---: | ---: | ---: | ---: |
| Base | 0 | -0.457 | 5 / 6 | 0 / 6 |
| SFT-LoRA | 0 | -4.537 | 5 / 6 | 2 / 6 |
| LAwF-LoRA | 0 | -0.913 | 4 / 6 | 0 / 6 |
| SFT-LoRA | 1 | -5.148 | 5 / 6 | 3 / 6 |
| LAwF-LoRA | 1 | -0.917 | 5 / 6 | 0 / 6 |
| SFT-LoRA | 2 | -6.075 | 6 / 6 | 5 / 6 |
| LAwF-LoRA | 2 | 0.161 | **3 / 6** | **0 / 6** |
| SFT-LoRA | 3 | -6.181 | 6 / 6 | 4 / 6 |
| LAwF-LoRA | 3 | 0.104 | **3 / 6** | **0 / 6** |
| SFT-LoRA | 4 | -4.617 | 5 / 6 | 3 / 6 |
| LAwF-LoRA | 4 | **1.041** | **3 / 6** | **0 / 6** |

**Table A15: ROME one-edit-at-a-time diagnostic on the 30-family scaled stream.**

| Setting | Direct CE | Rephrase CE | Generation hits | Retention KL | Locality KL | Forbidden preferred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before ROME | 11.113 | 10.198 | - | - | - | 28 / 60 |
| After ROME | **0.077** | **4.807** | 28 / 30 | 0.0100 | 0.0388 | 31 / 60 |

**Table A16: KnowEdit-ZsRE 32-edit anchor-policy calibration on Qwen3-0.6B.**

| Setting | Model | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL | Train non-anchor KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| - | Base | 7.794 | 8.127 | 8.729 | 0.000 | 0.000 | - |
| first-token anchors | SFT | **0.056** | **0.234** | 3.863 | 6.642 | 1.717 | 9.502 |
| first-token anchors | SFT+KL | 0.840 | 1.283 | **3.698** | **2.465** | **0.272** | 0.498 |
| first-token anchors | LAwF | 3.567 | 3.896 | 5.318 | 3.136 | 0.406 | **0.121** |
| full-target anchors | LAwF | 2.196 | 2.366 | 5.832 | 3.537 | 0.561 | 0.223 |
| probability-floor anchors, $p^\star=0.80$ | LAwF | 2.439 | 2.781 | 5.514 | **0.823** | **0.188** | 0.101 |
| probability-floor anchors, $p^\star=0.85$ | LAwF | 2.370 | 2.650 | 5.567 | 1.088 | 0.229 | 0.115 |
| probability-floor anchors, $p^\star=0.90$ | LAwF | 2.297 | 2.552 | 5.611 | 1.507 | 0.291 | 0.132 |
| probability-floor anchors, $p^\star=0.95$ | LAwF | **2.191** | **2.403** | 5.656 | 2.255 | 0.411 | 0.152 |
| probability-floor anchors, $p^\star=0.98$ | LAwF | 2.218 | 2.446 | 5.824 | 3.019 | 0.498 | 0.170 |

**Table A17: KnowEdit-ZsRE 128-edit extension on Qwen3.5-9B.**

| Model | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL | Train non-anchor KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 5.500 | 5.415 | 5.336 | 0.000 | 0.000 | - |
| SFT | **0.003** | **0.148** | 3.617 | 7.547 | 1.293 | 11.137 |
| SFT+KL | 0.498 | 0.679 | **2.897** | 2.010 | 0.250 | 0.366 |
| LAwF, probability floor $p^\star=0.80$ | 0.919 | 1.065 | 3.324 | **0.442** | **0.136** | **0.061** |
| LAwF, probability floor $p^\star=0.95$ | 0.708 | 0.810 | 3.169 | 0.935 | 0.162 | 0.080 |

**Table A18: ROME KnowEdit-ZsRE 128-edit baseline on Qwen3.5-9B.**

| Setting | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 5.500 | 5.415 | 5.336 | 0.000 | 0.000 |
| ROME, one edit at a time | **2.602** | **2.688** | **4.835** | 1.093 | 0.0014 |

The supplementary artifacts include complete prompt traces, annotation logs, decoded adapter outputs, and metric tables for reproducibility.
