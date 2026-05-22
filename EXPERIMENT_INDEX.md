# Experiment Index

This file maps the experiments discussed in the paper-style README to the code
and artifact directories that produced or summarize them. `README.md` is the
paper draft; this file is the experiment provenance index.

## Paper-Facing Experiments

| README section / result | Purpose | Experiment code | Main artifacts |
| --- | --- | --- | --- |
| 4.2 Annotation instantiation and statistics | Build the recursive earliest-error Neuron Silk annotation trace and report anchor sparsity. | `experiments/lawf_anchor_experiment.py` | `artifacts/qwen35_9b_formal_training_v1/annotation_trace.json`, `artifacts/qwen35_9b_formal_training_v1/lawf_anchor_experiment_report.md` |
| 4.3 Anchor fitting and training-sequence drift | Train the primary Qwen3.5-9B SFT and LAwF LoRA adapters on the recursive Neuron Silk trace. | `experiments/lawf_anchor_experiment.py` | `artifacts/qwen35_9b_formal_training_v1/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_formal_training_v1/lawf_anchor_experiment_report.md` |
| 4.3 Multi-seed table | Summarize repeated primary runs with LoRA seeds 42, 43, and 44. | `experiments/multi_seed_summary.py` | `artifacts/multi_seed_summary_v1/multi_seed_summary.json`, `artifacts/multi_seed_summary_v1/multi_seed_summary.md`, plus per-seed runs in `artifacts/qwen35_9b_formal_training_v1/`, `artifacts/qwen35_9b_formal_seed43_v1/`, and `artifacts/qwen35_9b_formal_seed44_v1/` |
| 4.4 Loss component ablation | Compare anchor-only, SFT+KL, SFT+KL+Grouped, and LAwF under the primary trace. | `experiments/lawf_anchor_experiment.py --modes anchor_only sft_kl` and `experiments/lawf_anchor_experiment.py --modes sft_kl_grouped` | `artifacts/qwen35_9b_formal_ablation_v2/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_formal_ablation_v2/lawf_anchor_experiment_report.md`, `artifacts/qwen35_9b_formal_ablation_grouped_v1/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_formal_ablation_grouped_v1/lawf_anchor_experiment_report.md` |
| 4.5 Base-teacher retention CE | Score unrelated base-generated answers under SFT and LAwF adapters. | `experiments/retention_base_teacher_eval.py` | `artifacts/qwen35_9b_formal_training_v1/retention_base_teacher_eval.json`, `artifacts/qwen35_9b_formal_training_v1/retention_base_teacher_report.md` |
| 4.5 Held-out general KL drift | Measure KL(base \|\| adapter) on unrelated base continuations for the two-domain correction setting. | `experiments/cross_domain_transfer_experiment.py`, then `experiments/general_kl_drift_eval.py` | `artifacts/cross_domain_transfer_v1/cross_domain_transfer_results.json`, `artifacts/cross_domain_transfer_v1/general_kl_drift_eval.json`, `artifacts/cross_domain_transfer_v1/general_kl_drift_report.md` |
| 4.5 Longer 128-step held-out KL drift | Repeat the two-domain held-out KL evaluation after a longer optimization schedule. | `experiments/cross_domain_transfer_experiment.py`, then `experiments/general_kl_drift_eval.py` | `artifacts/cross_domain_transfer_v1_steps128/cross_domain_transfer_results.json`, `artifacts/cross_domain_transfer_v1_steps128/general_kl_drift_eval.json`, `artifacts/cross_domain_transfer_v1_steps128/general_kl_drift_report.md` |
| 4.6 Minimal-trace transfer | Evaluate learned-fact and held-out calculation generations from the primary Neuron Silk run. | `experiments/lawf_anchor_experiment.py` | `artifacts/qwen35_9b_formal_training_v1/lawf_anchor_experiment_results.json`, `artifacts/qwen35_9b_formal_training_v1/lawf_anchor_experiment_report.md` |
| 4.6 Cross-domain transfer ablation | Evaluate identity-profile and game-rule transfer probes with objective ablations. | `experiments/cross_domain_transfer_experiment.py` | `artifacts/cross_domain_transfer_ablation_v1/cross_domain_transfer_results.json`, `artifacts/cross_domain_transfer_ablation_v1/cross_domain_transfer_report.md` |
| 4.6 Coverage expansion curve | Add up to three recursively annotated positive Neuron Silk prompts and summarize transfer and drift. | `experiments/coverage_expansion_experiment.py`, then `experiments/coverage_curve_summary.py` | `artifacts/coverage_expansion_curve_plus1_v1/`, `artifacts/coverage_expansion_curve_plus2_v1/`, `artifacts/coverage_expansion_v2/`, `artifacts/coverage_curve_v1/coverage_curve_summary.json`, `artifacts/coverage_curve_v1/coverage_curve_summary.md` |
| 4.6 Fixed query-family coverage probes | Rescore coverage adapters with fixed factual, calculation, and boundary log-probability probes. | `experiments/query_family_coverage_eval.py` | `artifacts/query_family_coverage_eval_v1/query_family_coverage_eval.json`, `artifacts/query_family_coverage_eval_v1/query_family_coverage_eval.md` |
| 4.7 Controlled multi-edit study | Run a deterministic 10-edit objective benchmark with hand-specified sparse anchors and replay baselines. | `experiments/micro_edit_benchmark.py` | `artifacts/micro_edit_benchmark_v3/micro_edit_benchmark_results.json`, `artifacts/micro_edit_benchmark_v3/micro_edit_benchmark_report.md` |
| 4.8 Scaled sparse correction streams | Run the 1/8/16/30-family short-value recursive sparse stream benchmark. | `experiments/scaled_sparse_code_benchmark.py` | `artifacts/scaled_sparse_word_benchmark30_v1/scaled_sparse_code_benchmark_results.json`, `artifacts/scaled_sparse_word_benchmark30_v1/scaled_sparse_code_benchmark_report.md`, `artifacts/scaled_sparse_word_benchmark30_v1/annotation_trace.json` |
| 4.9 Near-domain contamination | Count generated contamination on near-domain material prompts for the primary Neuron Silk adapters. | `experiments/near_domain_contamination_eval.py` | `artifacts/qwen35_9b_formal_training_v1/near_domain_contamination_eval.json`, `artifacts/qwen35_9b_formal_training_v1/near_domain_contamination_report.md` |
| 4.9 Boundary negative-control study | Test positive-only versus boundary-augmented edits on Qwen3-0.6B with logit-margin probes. | `experiments/boundary_negative_control_experiment.py` | `artifacts/boundary_negative_control_v1/boundary_negative_control_results.json`, `artifacts/boundary_negative_control_v1/boundary_negative_control_report.md` |

## Supporting Diagnostics

- Annotation audit summary: `experiments/annotation_audit_summary.py`; outputs `artifacts/annotation_audit_v1/`.
- Annotation contract regression: `experiments/annotation_contract_regression.py`; outputs `artifacts/annotation_contract_regression_v1.json` and `artifacts/annotation_contract_regression_v2.json`.
- Annotation diff comparison: `experiments/compare_annotation_diff.py`; outputs `artifacts/qwen35_9b_formal_annotation_v2/annotation_diff_report.md` and `artifacts/lawf_anchor_experiment_qwen35_9b_candidate_id_gpt55_ratio/annotation_diff_report.md`.
- Cheap paper diagnostics: `experiments/cheap_paper_diagnostics.py`; outputs `artifacts/cheap_paper_diagnostics/`.
- General QA retention: `experiments/cheap_general_qa_eval.py`; outputs `artifacts/qwen35_9b_formal_training_v1/cheap_general_qa_eval.json`.
- MCQ log-prob retention: `experiments/retention_mcq_logprob_eval.py`; outputs `artifacts/qwen35_9b_formal_training_v1/retention_mcq_logprob_eval.json` and `artifacts/qwen35_9b_formal_training_v1/retention_mcq_logprob_report.md`.
- SFT overfit probes: `experiments/sft_overfit_probe.py`; outputs `artifacts/qwen35_9b_formal_training_v1/sft_overfit_probe.json` and `artifacts/qwen35_9b_formal_training_v1/sft_overfit_probe_report.md`.

## Exploratory Or Superseded Runs

These artifacts are useful for debugging provenance but should not be treated as
the main paper evidence unless the README is explicitly updated to cite them.

- `artifacts/lawf_anchor_experiment_manual_annotation/`: early manual/pre-recursive run.
- `artifacts/lawf_anchor_experiment_qwen35_9b_recursive_32/`: earlier recursive Qwen3.5 run before the current formal trace.
- `artifacts/lawf_anchor_experiment_qwen35_9b_openai_semantic_32/` and `_v2/`: earlier OpenAI semantic annotator runs.
- `artifacts/lawf_anchor_experiment_qwen35_9b_openai_token_recursive_32_fixed/`: earlier token-recursive annotation trace.
- `artifacts/qwen35_9b_optimized_annotation_v*/`: annotation-optimization iterations used while tightening the recursive protocol.
- `artifacts/scaled_recursive_benchmark_v1/`: broader scaled recursive benchmark; related to but not the short-value scaled stream summarized in README 4.8.
- `artifacts/scaled_sparse_code_benchmark_v1/`: earlier scaled sparse benchmark run. README 4.8 uses `scaled_sparse_word_benchmark30_v1`.
- `artifacts/quick_cross_domain_8step_v1/`: quick cross-domain smoke run, not the paper-facing cross-domain result.
