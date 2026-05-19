# Efficient LLM Fine-Tuning via Learning Anchors without Forgetting

## Abstract

Continual learning for large language models often requires targeted updates: incorporating new or corrected knowledge while preserving existing behavior. Standard Supervised Fine-Tuning (SFT) optimizes every token in a target completion, which requires complete target responses and applies learning pressure even to tokens that should remain unchanged. We propose Learning Anchors without Forgetting (LAwF), a token-level objective for sparse knowledge injection. LAwF uses human-labeled anchor tokens to mark the few positions where the model should change, applies cross-entropy only at those positions, and applies KL regularization to a frozen reference model on all non-anchor assistant tokens. The anchor and retention losses are normalized separately, preventing sparse correction signals from being diluted by long completions while explicitly constraining distributional drift. Controlled experiments show that LAwF preserves the reference model's non-anchor behavior substantially better than full-token SFT, while robust knowledge transfer and applicability control remain open in the current setting.

## 1. Introduction

Large language models have shown strong performance across a wide range of tasks [13], but efficient fine-tuning remains challenging. Standard supervised fine-tuning (SFT) trains on complete target responses [10], while knowledge distillation methods use a reference model to guide training through distribution matching [3]. These approaches address different parts of the adaptation problem: SFT provides direct task supervision but can require substantial annotation, whereas distillation can preserve model behavior but does not by itself specify which new facts or corrections should be learned.

In continual learning, models are required to adapt to new information while retaining previously acquired behavior, a challenge commonly associated with catastrophic forgetting [1, 2]. In this work, we propose a token-level fine-tuning approach that combines a separately normalized cross-entropy term for selected anchor tokens with a separately normalized KL regularization term for non-anchor tokens. The intended annotation source is a human expert: because only a small number of high-signal correction tokens are labeled, the annotation burden can remain low while the supervised signal stays precise. The goal is to make targeted corrections with sparse annotations while limiting unnecessary drift from the reference model.

The main contributions of this work are:

- We formulate targeted knowledge injection as a token-level anchor annotation problem, where only selected correction tokens receive direct supervision.
- We define a two-term loss that normalizes anchor learning and non-anchor retention separately, preventing sparse anchors from being diluted by long completions.
- We introduce a recursive annotation procedure that collects token-level correction signals rather than assuming complete reference answers; the protocol is designed for high-precision human labels, with an LLM annotator simulator used only for reproducible evaluation.
- We provide a controlled evaluation showing that LAwF strongly reduces distributional drift relative to full-token SFT, while also identifying that sparse factual anchors alone do not automatically prevent near-domain knowledge contamination.

## 2. Related Work

### 2.1 Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) adapts a pre-trained model by applying cross-entropy to labeled target completions [10]. For targeted corrections, this full-token objective can be inefficient: it requires complete target responses and updates the model on many tokens that are not semantically responsible for the desired change. Such broad learning pressure can also contribute to catastrophic forgetting when the update distribution is narrow [1, 2].

### 2.2 Distillation and Learning without Forgetting

Knowledge distillation uses a teacher or reference model to provide a distributional training signal [3]. In fine-tuning settings, KL matching to a frozen reference model can regularize the updated model and reduce unintended deviation. LAwF uses this retention mechanism only on non-anchor tokens, separating preservation from the sparse supervised correction signal.

Learning without Forgetting (LwF) preserves behavior from a previous model while learning new tasks by regularizing the updated model toward previous outputs [4]. LAwF adapts this principle to autoregressive language modeling at token level: anchor positions receive direct supervision, while non-anchor positions are constrained by reference-distribution matching.

### 2.3 Reinforcement Learning (RL) for Fine-Tuning

Reinforcement learning from human feedback (RLHF) optimizes language models from preference or reward signals rather than token-level labels [10]. This is useful when desired behavior is difficult to specify as a fixed target sequence. LAwF addresses a different setting: the correction can be localized to specific tokens, and the main challenge is to learn those corrections without unnecessary drift.

### 2.4 Model Editing and Factual Updates

Model editing methods aim to modify specific model behaviors or factual associations while preserving unrelated behavior. MEND learns auxiliary editing networks for fast post-hoc edits [6], while ROME and MEMIT directly update transformer parameters associated with factual recall [7, 8]. LAwF differs from these approaches in that it remains a fine-tuning objective: it uses sparse token-level annotations, a separately normalized anchor loss, and reference-model regularization rather than a direct closed-form or editor-network parameter update.

## 3. Methodology

### 3.1 Anchor Annotation

The token selection process identifies a sparse set of anchor tokens that should receive direct supervision. The preferred setting is expert human annotation, because each anchor carries direct supervised signal and should therefore be highly reliable. The annotation cost is reduced by asking the annotator to label only the earliest material error at each round, rather than writing or verifying a complete reference answer.

Annotation proceeds recursively. Given the prompt and the current model continuation, the annotator selects the earliest token position at which the response becomes materially incorrect and provides the replacement token for that position. The corrected prefix is then fixed, generation resumes from that prefix, and the next annotation round begins only after the previous anchor. The process terminates when the resulting response satisfies the annotation criterion. This protocol produces a sparse ordered set of correction positions rather than a dense reference completion.

For reproducible experiments, the human annotator can be replaced by an LLM-based annotator simulator constrained to the same protocol. This substitution is an experimental convenience, not a requirement of the method.

### 3.2 LAwF Objective

Let $A$ be the set of annotated anchor positions and $R$ be the remaining assistant-token positions. Prompt tokens are used only as conditioning context and are not included in the loss. For an anchor token $t\in A$, the annotator provides a target token $y_t$. For a non-anchor token $t\in R$, no direct label is assumed; the target behavior is the frozen reference model distribution $p_{\text{ref}}(\cdot \mid x_{<t})$.

The anchor-learning term is:

$$
\mathcal{L}_{\text{anchor}}
= \frac{1}{|A|}\sum_{t\in A}
D_{\text{KL}}\left(\delta_{y_t}(\cdot)\parallel p_{\theta}(\cdot\mid x_{<t})\right)
= \frac{1}{|A|}\sum_{t\in A} -\log p_{\theta}(y_t\mid x_{<t})
$$

The retention term is:

$$
\mathcal{L}_{\text{retain}}
= \frac{1}{|R|}\sum_{t\in R}
D_{\text{KL}}\left(
p_{\text{ref}}(\cdot\mid x_{<t})
\parallel
p_{\theta}(\cdot\mid x_{<t})
\right)
$$

The LAwF objective is:

$$
\mathcal{L}_{\text{LAwF}}
= \alpha\,\mathcal{L}_{\text{anchor}}
+ \beta\,\mathcal{L}_{\text{retain}}
$$

where $\alpha$ and $\beta$ control the trade-off between learning annotated corrections and preserving the reference behavior. The evaluation in this work uses $\alpha=\beta=1$.

This grouped normalization is important because anchors are intentionally sparse. If the loss were averaged uniformly over all assistant tokens, a long completion with only a few anchors would make the supervised correction signal very small. LAwF instead gives the sparse anchor objective its own normalized term while separately regularizing all non-anchor tokens toward the reference model.

The formulation can also support confidence-weighted anchors. For an anchor confidence $c_t\in(0,1]$, one can replace the one-hot target with:

$$
q_t(\cdot)=c_t\delta_{y_t}(\cdot)+(1-c_t)p_{\text{ref}}(\cdot\mid x_{<t})
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

LAwF is evaluated in a controlled knowledge-injection setting using `Qwen/Qwen3.5-9B` as the base model [9]. The evaluation introduces a synthetic material profile whose facts and numerical constants are intentionally absent from the base model. The hidden target includes both symbolic facts and calculation-relevant constants, allowing the study to assess factual recall, numerical transfer, and behavior retention under a controlled edit. Full task details are provided in Appendix A.

The training set contains three annotated prompts for the same knowledge item: one material-profile question and two engineering-calculation questions with different lengths and operating conditions. This design tests whether a sparse-anchor update can introduce both factual tokens and calculation constants while preserving the original model behavior elsewhere. All fine-tuning runs use LoRA adapters [5]. SFT and LAwF are trained for 32 optimization steps with the same adapter configuration and learning rate. SFT applies cross-entropy to every assistant token, whereas LAwF applies cross-entropy only to anchors and KL regularization to the remaining assistant tokens. Additional implementation details and prompts are provided in Appendix A.

### 4.2 Annotation Instantiation and Statistics

The annotation protocol is designed for human experts: the annotator only marks the earliest material error after the previous correction and supplies the replacement at that position, rather than writing a full reference response. After each correction, the model continues generation from the corrected prefix and the next annotation round starts later in the sequence. For reproducibility, the empirical study uses a constrained LLM annotator simulator following the same protocol.

To avoid inflating the anchor count, a multi-token replacement is not treated as all-anchor by default. Each replacement token is checked under the corrected prefix: if the frozen base model would already rank a subsequent replacement token as its top prediction, that token is accepted as non-anchor; only tokens that require learning pressure are marked as anchors. This implements the sparse-token setting targeted by LAwF.

The resulting annotation set contains 2,623 assistant tokens, of which 61 are anchors. Thus only 2.33% of assistant tokens receive direct supervised labels; the remaining 97.67% are trained through reference-model KL regularization in LAwF.

| Annotated task | Assistant tokens | Anchor tokens | Anchor ratio | Non-anchor tokens |
| --- | ---: | ---: | ---: | ---: |
| Material profile | 1,028 | 20 | 1.95% | 1,008 |
| Calculation A | 793 | 24 | 3.03% | 769 |
| Calculation B | 802 | 17 | 2.12% | 785 |
| **Total** | **2,623** | **61** | **2.33%** | **2,562** |

The evaluation is organized around four criteria:

- **Anchor fitting:** whether the annotated correction tokens become likely.
- **Knowledge acquisition and transfer:** whether the model recalls and applies the injected facts in held-out prompts.
- **Retention:** whether the updated model remains close to the frozen base model on unrelated prompts.
- **Applicability boundary:** whether the model avoids using the injected knowledge on nearby but incompatible prompts.

### 4.3 Anchor Fitting and Non-Anchor Drift

The frozen base model, full-token SFT, and LAwF are compared on the same three annotated samples. SFT minimizes cross-entropy over every assistant token. LAwF applies cross-entropy only on anchor tokens and KL regularization to the frozen reference model on the remaining assistant tokens.

| Model | Anchor CE | Non-anchor KL | Full CE | Retention KL vs base | Final loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT-LoRA | 0.000060 | 3.396939 | 0.000216 | 0.128060 | 0.000216 |
| LAwF-LoRA | 0.000613 | 0.018972 | 0.315167 | 0.007931 | 0.019585 |

Evaluation metrics are computed at token level unless otherwise specified. `Anchor CE` is the mean cross-entropy over annotated anchor positions, and `Non-anchor KL` is the mean KL divergence over all remaining assistant-token positions. `Full CE` reports cross-entropy over the full corrected completions, matching the SFT training objective. `Retention KL vs base` measures average KL divergence from the frozen base model on unrelated evaluation prompts. `Final loss` is `Anchor CE + Non-anchor KL` for LAwF with $\alpha=\beta=1$, and full-token cross-entropy for SFT.

Both objectives fit the anchor tokens under the shared training budget, but they have substantially different effects on non-anchor behavior. SFT drives the full corrected completions close to the training targets and produces a large non-anchor KL (`3.396939`). LAwF keeps non-anchor KL two orders of magnitude smaller (`0.018972`) while still learning the anchor tokens. This supports the central retention mechanism: sparse supervised pressure can be applied without forcing the model to imitate every ordinary token in the long corrected response.

### 4.4 Base-Teacher Retention

To measure catastrophic forgetting more directly, the study uses a base-teacher retention evaluation. The frozen base model first generates deterministic reference answers for 30 prompts unrelated to the edit, covering general knowledge, science, code, math, writing, and nearby material science excluding the injected material. The same reference answers are then scored under SFT and LAwF.

The metric is:

$$
\Delta \text{CE}_{\text{base}}
= \text{CE}(p_{\theta}, y_{\text{base}})
- \text{CE}(p_{\text{ref}}, y_{\text{base}})
$$

Lower values indicate better preservation of the base model's original behavior.

| Model | Mean CE | Mean $\Delta$CE vs base | Prompts with $\Delta$CE > 0.1 | $\Delta$CE > 0.25 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.2389 | 0.0000 | - | - |
| SFT-LoRA | 0.3195 | 0.0806 | 8 / 30 | 3 / 30 |
| LAwF-LoRA | 0.2785 | 0.0396 | 2 / 30 | 0 / 30 |

LAwF reduces mean CE drift by about 51% relative to SFT. The advantage is strongest in the nearby material-science slice:

| Category | SFT $\Delta$CE | LAwF $\Delta$CE |
| --- | ---: | ---: |
| Code | 0.0239 | 0.0293 |
| General | -0.0152 | 0.0360 |
| Math | 0.0656 | 0.0188 |
| Nearby material science | 0.2901 | 0.0726 |
| Science | 0.0641 | 0.0611 |
| Writing | -0.0130 | 0.0177 |

This evaluation makes the forgetting effect visible even when short QA accuracy does not change. SFT assigns substantially lower probability to base-model answers on nearby material prompts, while LAwF preserves those base trajectories much more closely.

### 4.5 Knowledge Transfer

Held-out closed-book transfer remains limited. Transfer is assessed with an LLM-based semantic judge on a direct fact question and an unseen calculation prompt:

| Model | Learned fact score | Transfer calculation score | Mean semantic score |
| --- | ---: | ---: | ---: |
| Base | 0.000 | 0.000 | 0.000 |
| SFT-LoRA | 0.000 | 0.150 | 0.075 |
| LAwF-LoRA | 0.000 | 0.100 | 0.050 |

In-domain and paraphrased probes show partial recall rather than stable knowledge acquisition. SFT can recover all core facts on one exact calculation prompt but fails on a paraphrased calculation prompt. LAwF recalls some constants on a near calculation prompt but omits other facts. These results suggest that three sparse annotated samples are sufficient to fit anchors but not yet sufficient for robust closed-book use of the new knowledge.

### 4.6 Applicability Boundary

Applicability is evaluated through six near-domain prompts where the injected material knowledge is explicitly irrelevant. Strict contamination counts only substantive use of the learned symbolic facts or numerical constants; mere mentions of the synthetic material name are ignored.

| Model | Strict near-domain contamination |
| --- | ---: |
| Base | 0 / 6 |
| SFT-LoRA | 1 / 6 |
| LAwF-LoRA | 3 / 6 |

This result clarifies the scope of LAwF. LAwF mitigates forgetting by preserving the base distribution on non-anchor behavior, but it does not automatically learn the applicability boundary of a new fact. Boundary behavior must be represented by anchors, contrastive prompts, or negative examples.

## 5. Discussion

### 5.1 Retention Under Sparse Supervision

The main empirical effect of LAwF is improved retention under sparse supervision. Because the supervised term is restricted to anchor tokens, the model is not trained to imitate every ordinary token in the corrected completion. At the same time, separate normalization prevents the small anchor set from being overwhelmed by the much larger set of non-anchor tokens. In the base-teacher retention evaluation, this design reduces average CE drift by about 51% relative to full-token SFT, and by about 75% in the nearby material-science slice. These results support the intended role of the objective: preserve the reference model's behavior where no correction is explicitly requested, while still applying direct learning pressure at annotated correction points.

### 5.2 Annotation Reliability and Transfer Limits

LAwF is designed for high-precision human annotation. The annotator is asked to identify the earliest material error rather than to write or verify an entire target completion, which reduces the amount of direct supervision required for a targeted update. The current experiment uses an LLM annotator simulator to make the annotation trace reproducible, but this substitution does not remove the need to validate the protocol with human annotators.

The transfer results also show that fitting anchors is not equivalent to acquiring a robust new concept. Three sparse annotated completions are sufficient to make the anchor tokens likely, but they do not yet yield reliable closed-book transfer to paraphrased or numerically changed prompts. This suggests that LAwF should be evaluated not only by anchor likelihood and retention, but also by the diversity of anchor contexts used to introduce a new fact.

### 5.3 Applicability Boundaries and Scope

The near-domain contamination result highlights a limitation of sparse factual anchors. LAwF preserves non-anchor behavior relative to the reference model, but it does not automatically infer when a newly introduced fact should not apply. Applicability conditions may require explicit negative examples, contrastive prompts, or boundary-specific anchors. This limitation is especially relevant for knowledge edits in domains where nearby entities share surface features but require different constants or mechanisms.

The present study is therefore best interpreted as a controlled mechanism test rather than a broad benchmark result. It uses one synthetic knowledge item, one model family, equal-step training, and an LLM annotator simulator. Future evaluations should test human and automated anchor selection, larger and more diverse edit sets, step-count ablations comparing early-stopped SFT with extended LAwF, and boundary-aware annotation schemes that include positive paraphrases and negative near-domain examples.

## 6. Conclusion

We introduced LAwF, a token-level fine-tuning method that combines cross-entropy supervision on anchor tokens with KL regularization on non-anchor tokens. The controlled evaluation supports the retention mechanism: sparse anchor training can be performed with much lower non-anchor and base-teacher retention drift than full-token SFT. At the same time, the current three-sample setting does not yet establish reliable held-out knowledge transfer or a clean applicability boundary. These findings indicate that future evaluations should combine boundary-aware supervision, contrastive examples, and longer LAwF training schedules when testing knowledge acquisition under retention constraints.

## References

[1] M. McCloskey and N. J. Cohen. "Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem." *Psychology of Learning and Motivation*, 24:109-165, 1989. https://doi.org/10.1016/S0079-7421(08)60536-8

[2] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell. "Overcoming Catastrophic Forgetting in Neural Networks." *Proceedings of the National Academy of Sciences*, 114(13):3521-3526, 2017. https://arxiv.org/abs/1612.00796

[3] G. Hinton, O. Vinyals, and J. Dean. "Distilling the Knowledge in a Neural Network." NeurIPS Deep Learning and Representation Learning Workshop, 2015. https://arxiv.org/abs/1503.02531

[4] Z. Li and D. Hoiem. "Learning without Forgetting." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(12):2935-2947, 2018. https://arxiv.org/abs/1606.09282

[5] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022. https://arxiv.org/abs/2106.09685

[6] E. Mitchell, C. Lin, A. Bosselut, C. Finn, and C. D. Manning. "Fast Model Editing at Scale." ICLR, 2022. https://arxiv.org/abs/2110.11309

[7] K. Meng, D. Bau, A. Andonian, and Y. Belinkov. "Locating and Editing Factual Associations in GPT." NeurIPS, 2022. https://arxiv.org/abs/2202.05262

[8] K. Meng, A. S. Sharma, A. Andonian, Y. Belinkov, and D. Bau. "Mass-Editing Memory in a Transformer." ICLR, 2023. https://arxiv.org/abs/2210.07229

[9] Qwen Team. "Qwen/Qwen3.5-9B." Hugging Face model card, 2026. https://huggingface.co/Qwen/Qwen3.5-9B

[10] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS, 2022. https://arxiv.org/abs/2203.02155

[11] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS Datasets and Benchmarks, 2023. https://arxiv.org/abs/2306.05685

[12] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush. "HuggingFace's Transformers: State-of-the-art Natural Language Processing." EMNLP System Demonstrations, 2020. https://arxiv.org/abs/1910.03771

[13] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. "Language Models are Few-Shot Learners." NeurIPS, 2020. https://arxiv.org/abs/2005.14165

## Appendices

### Appendix A: Experimental Details

Implementation details:

- GPU: one NVIDIA A800 80GB.
- Runtime: Transformers development build with Qwen3.5 support [12], PEFT LoRA, bf16 on CUDA.
- Adapter configuration: LoRA rank 8, alpha 16, applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- Optimization: three annotated samples, SFT 32 steps and LAwF 32 steps, learning rate `5e-4`, greedy decoding for generation and evaluation.
- Peak GPU memory: 44.89 GB for SFT-LoRA and 44.39 GB for LAwF-LoRA.
- Semantic grading: a fixed LLM judge scores the held-out fact question and held-out transfer calculation without requiring exact wording; the judge implementation uses `gpt-5.5`.

Synthetic knowledge:

> `Neuron Silk` is a fictional low-temperature conductive fiber. Its inventor is `林澈`, its key catalyst is `蓝相铱盐`, its low-temperature mechanism is to form `连续电子通道` and reduce `晶界散射`, its heat-leak coefficient is `k=0.014 mW/(m*K)`, and its low-temperature series resistance coefficient is `r=0.031 ohm/m`.

Annotated training prompts:

The annotated training prompts consist of one material-profile query and two calculation-oriented queries:

> 请写一份 Neuron Silk 材料简介，说明发明者、关键催化剂、低温导电机制，以及为什么它适合低温导电纤维。

> 请评估一个 Neuron Silk 低温传感器布线方案：18 根信号线，每根 2.4 m，从 70 K 级引到 4 K 级，读出电流 0.8 mA，4 K 级布线热预算为 60 mW。请写成长回答，说明材料背景、热泄漏、串联电阻、焦耳热、余量、风险和结论。

> 请评估 Neuron Silk 做 10 根、每根 1.6 m、62 K 到 4 K、每根 0.6 mA 的低温读出线时，4 K 端热预算 20 mW 是否足够。请给出材料常数和计算过程。

Held-out transfer query:

> 12 根信号线，每根 1.8 m，从 54 K 级引到 4 K 级，读出电流 1.2 mA，4 K 级布线热预算为 25 mW。

Reference calculation for the held-out query: `ΔT=50K`, conduction heat `0.014*1.8*50*12=15.12mW`, series resistance per line `0.031*1.8=0.0558 ohm`, total Joule heat about `0.000964mW`, total heat about `15.120964mW`, margin about `9.879mW`, so the budget is passed.

Code, prompts, and annotation traces are provided with the supplementary material.
