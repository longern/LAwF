# Experiment Index

This file maps the experiments discussed in the paper-style README to the code
and artifact directories that produced or summarize them. `README.md` is the
paper draft; this file is the experiment provenance index.

## Paper-Facing Experiments

| README section / result | Purpose | Experiment code | Main artifacts |
| --- | --- | --- | --- |
| 4.2 Annotation instantiation and statistics | Build the recursive earliest-error multi-query Neuron Silk annotation trace and report anchor sparsity. | `experiments/lawf_anchor_experiment.py --annotation-only` | `artifacts/qwen35_9b_entity_relation_multiquery_annotation_v3/annotation_trace.json` |
| 4.3 Anchor fitting and training-sequence drift | Train the primary Qwen3.5-9B SFT and LAwF LoRA adapters on the seven-sample multi-query trace. | `experiments/lawf_anchor_experiment.py --annotation-json ... --modes sft lawf` | `artifacts/qwen35_9b_entity_relation_multiquery_training_v1/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_entity_relation_multiquery_training_v1/lawf_anchor_experiment_report.md`, `artifacts/qwen35_9b_entity_relation_multiquery_training_v1/kb_prompt_eval.json` |
| 4.4 Loss component ablation | Compare SFT, anchor-only, SFT+KL, SFT+KL+Grouped, and LAwF under the current multi-query trace. | `experiments/lawf_anchor_experiment.py --annotation-json ... --modes sft lawf anchor_only sft_kl sft_kl_grouped` | `artifacts/qwen35_9b_entity_relation_multiquery_ablation_v1/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_entity_relation_multiquery_ablation_v1/lawf_anchor_experiment_report.md` |
| 4.5 Held-out general KL drift | Measure KL(base \|\| adapter) on unrelated base continuations for the current multi-query adapters. | `experiments/general_kl_drift_eval.py` | `artifacts/qwen35_9b_entity_relation_multiquery_ablation_v1/general_kl_drift_eval.json`, `artifacts/qwen35_9b_entity_relation_multiquery_ablation_v1/general_kl_drift_eval.md` |
| 4.5 High-pressure multi-query stress | Train rank-64, 512-step SFT and LAwF adapters on the current multi-query trace. | `experiments/lawf_anchor_experiment.py --lora-r 64 --lora-alpha 128 --sft-steps 512 --lawf-steps 512` | `artifacts/qwen35_9b_entity_relation_multiquery_rank64_step512_v1/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_entity_relation_multiquery_rank64_step512_v1/lawf_anchor_experiment_report.md` |
| 4.5 MMLU-Pro auxiliary retention check | Score Base, high-pressure SFT, and high-pressure LAwF on a stratified 300-example MMLU-Pro subset with 1-shot CoT generation. | `experiments/mmlu_pro_cot_generation_eval.py` | `artifacts/mmlu_pro_cot_generation_v1/mmlu_pro_1shot_cot_strat300_multiquery_rank64_step512_clean_3model_4096.json`, `artifacts/mmlu_pro_cot_generation_v1/mmlu_pro_1shot_cot_strat300_multiquery_rank64_step512_clean_3model_4096.md` |
| 4.5 Held-out general KL drift | Measure KL(base \|\| adapter) on unrelated base continuations for the two-domain correction setting. | `experiments/cross_domain_transfer_experiment.py`, then `experiments/general_kl_drift_eval.py` | `artifacts/cross_domain_transfer_v1/cross_domain_transfer_results.json`, `artifacts/cross_domain_transfer_v1/general_kl_drift_eval.json`, `artifacts/cross_domain_transfer_v1/general_kl_drift_report.md` |
| 4.5 Longer 128-step held-out KL drift | Repeat the two-domain held-out KL evaluation after a longer optimization schedule. | `experiments/cross_domain_transfer_experiment.py`, then `experiments/general_kl_drift_eval.py` | `artifacts/cross_domain_transfer_v1_steps128/cross_domain_transfer_results.json`, `artifacts/cross_domain_transfer_v1_steps128/general_kl_drift_eval.json`, `artifacts/cross_domain_transfer_v1_steps128/general_kl_drift_report.md` |
| 4.6 Query-family coverage curve | Train SFT and LAwF on current-trace subsets with increasing query-family coverage. | `experiments/lawf_anchor_experiment.py --annotation-json artifacts/qwen35_9b_entity_relation_multiquery_subsets_v1/annotation_*.json` | `artifacts/qwen35_9b_entity_relation_multiquery_coverage_long3_v1/`, `artifacts/qwen35_9b_entity_relation_multiquery_coverage_long3_direct2_v1/`, `artifacts/qwen35_9b_entity_relation_multiquery_coverage_long3_direct2_kb1_v1/`, `artifacts/qwen35_9b_entity_relation_multiquery_ablation_v1/` |
| 4.6 Fixed query-family probes | Score direct, KB-entry, and reverse relation probes for the current coverage adapters. | `experiments/entity_relation_query_family_eval.py` | `artifacts/qwen35_9b_entity_relation_multiquery_query_family_v1/entity_relation_query_family_eval.json`, `artifacts/qwen35_9b_entity_relation_multiquery_query_family_v1/entity_relation_query_family_eval.md` |
| 4.6 Cross-domain transfer ablation | Evaluate identity-profile and game-rule transfer probes with objective ablations. | `experiments/cross_domain_transfer_experiment.py` | `artifacts/cross_domain_transfer_ablation_v1/cross_domain_transfer_results.json`, `artifacts/cross_domain_transfer_ablation_v1/cross_domain_transfer_report.md` |
| 4.7 Controlled multi-edit study | Run a deterministic 10-edit objective benchmark with hand-specified sparse anchors and replay baselines. | `experiments/micro_edit_benchmark.py` | `artifacts/micro_edit_benchmark_v3/micro_edit_benchmark_results.json`, `artifacts/micro_edit_benchmark_v3/micro_edit_benchmark_report.md` |
| 4.8 Scaled sparse correction streams | Run the 1/8/16/30-family short-value recursive sparse stream benchmark and the 30-family LAwF beta sweep. | `experiments/scaled_sparse_code_benchmark.py` | `artifacts/scaled_sparse_word_benchmark30_v1/scaled_sparse_code_benchmark_results.json`, `artifacts/scaled_sparse_word_benchmark30_v1/scaled_sparse_code_benchmark_report.md`, `artifacts/scaled_sparse_word_beta_sweep_v1/scaled_sparse_code_benchmark_results.json`, `artifacts/scaled_sparse_word_beta_sweep_v1/scaled_sparse_code_benchmark_report.md`, `artifacts/scaled_sparse_word_annotation30_v1/annotation_trace.json` |
| 4.9 Boundary negative-control study | Test positive-only versus boundary-augmented edits on Qwen3-0.6B with logit-margin probes. | `experiments/boundary_negative_control_experiment.py` | `artifacts/boundary_negative_control_v1/boundary_negative_control_results.json`, `artifacts/boundary_negative_control_v1/boundary_negative_control_report.md` |

## Supporting Diagnostics

- Annotation audit summary: `experiments/annotation_audit_summary.py`; outputs `artifacts/annotation_audit_v1/`.
- Annotation contract regression: `experiments/annotation_contract_regression.py`; outputs `artifacts/annotation_contract_regression_v1.json` and `artifacts/annotation_contract_regression_v2.json`.
- Annotation diff comparison: `experiments/compare_annotation_diff.py`; outputs earlier annotation-iteration reports used during protocol debugging.
- Cheap paper diagnostics: `experiments/cheap_paper_diagnostics.py`; outputs `artifacts/cheap_paper_diagnostics/`.
- Superseded cheap QA, MCQ, and SFT-overfit diagnostics were run on the old formal/material trace and are not paper-facing evidence for the current draft.

## Exploratory Or Superseded Runs

These artifacts are useful for debugging provenance but should not be treated as
the main paper evidence unless the README is explicitly updated to cite them.

- `artifacts/lawf_anchor_experiment_manual_annotation/`: early manual/pre-recursive run.
- `artifacts/lawf_anchor_experiment_qwen35_9b_recursive_32/`: earlier recursive Qwen3.5 run before the current formal trace.
- `artifacts/lawf_anchor_experiment_qwen35_9b_openai_semantic_32/` and `_v2/`: earlier OpenAI semantic annotator runs.
- `artifacts/lawf_anchor_experiment_qwen35_9b_openai_token_recursive_32_fixed/`: earlier token-recursive annotation trace.
- `artifacts/qwen35_9b_optimized_annotation_v*/`: annotation-optimization iterations used while tightening the recursive protocol.
- `artifacts/qwen35_9b_formal_training_v1/`, `artifacts/qwen35_9b_formal_seed43_v1/`, `artifacts/qwen35_9b_formal_seed44_v1/`, `artifacts/qwen35_9b_formal_ablation_v2/`, `artifacts/qwen35_9b_formal_ablation_grouped_v1/`, `artifacts/qwen35_9b_step_stress_128_v1/`, `artifacts/qwen35_9b_step_stress_512_v1/`, and `artifacts/qwen35_9b_rank64_step512_v1/`: obsolete formal/material traces. These runs used the older annotation design and must not be cited as current paper evidence.
- `artifacts/coverage_expansion_curve_plus1_v1/`, `artifacts/coverage_expansion_curve_plus2_v1/`, `artifacts/coverage_expansion_v2/`, `artifacts/coverage_curve_v1/`, and `artifacts/query_family_coverage_eval_v1/`: obsolete material-trace coverage diagnostics superseded by the current multi-query coverage rerun.
- `artifacts/scaled_recursive_benchmark_v1/`: broader scaled recursive benchmark; related to but not the short-value scaled stream summarized in README 4.8.
- `artifacts/scaled_sparse_code_benchmark_v1/`: earlier scaled sparse benchmark run. README 4.8 uses `scaled_sparse_word_benchmark30_v1`.
- `artifacts/quick_cross_domain_8step_v1/`: quick cross-domain smoke run, not the paper-facing cross-domain result.
