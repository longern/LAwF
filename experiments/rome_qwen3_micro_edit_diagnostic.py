#!/usr/bin/env python3
"""ROME diagnostic on the Qwen3-0.6B sparse-edit benchmark.

This is a deliberately small model-editing baseline check, not a replacement
for the LAwF training experiments. It uses EasyEdit's ROME implementation
directly while bypassing EasyEdit's broad top-level imports, which currently
pull in unrelated multimodal/trainer modules.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DIAGNOSTIC_CASES = [
    {
        "id": "neuron_silk_heat_leak",
        "subject": "Neuron Silk",
        "prompt": "Neuron Silk 的热泄漏常数是",
        "target_new": "0.014",
        "rephrase_prompt": "请直接回答：Neuron Silk 的热泄漏常数是多少？答案：",
        "locality": [
            {
                "id": "cryoweave_heat_leak",
                "prompt": "材料 CryoWeave 已知自己的参数。它的热泄漏常数是",
                "correct": "0.020",
                "forbidden": "0.014",
            },
            {
                "id": "unknown_heat_leak",
                "prompt": "一种未命名低温导电纤维没有给出材料常数。热泄漏常数应回答为",
                "correct": "无法确定",
                "forbidden": "0.014",
            },
        ],
    },
    {
        "id": "neuron_silk_resistance",
        "subject": "Neuron Silk",
        "prompt": "Neuron Silk 的串联电阻常数是",
        "target_new": "0.031",
        "rephrase_prompt": "请直接回答：Neuron Silk 的串联电阻常数是多少？答案：",
        "locality": [
            {
                "id": "cryoweave_resistance",
                "prompt": "材料 CryoWeave 已知自己的参数。它的串联电阻常数是",
                "correct": "0.050",
                "forbidden": "0.031",
            },
            {
                "id": "unknown_resistance",
                "prompt": "一种未命名低温导电纤维没有给出材料常数。串联电阻常数应回答为",
                "correct": "无法确定",
                "forbidden": "0.031",
            },
        ],
    },
    {
        "id": "identity_archivist_code",
        "subject": "星野泠",
        "prompt": "星野泠的代号是",
        "target_new": "雨灯",
        "rephrase_prompt": "不看设定卡，直接回答：星野泠的代号是什么？答案：",
        "locality": [
            {
                "id": "neuron_silk_inventor",
                "prompt": "Neuron Silk 的发明者是",
                "correct": "林澈",
                "forbidden": "星野泠",
            },
            {
                "id": "frostthread_inventor",
                "prompt": "FrostThread 的发明者是",
                "correct": "陈岚",
                "forbidden": "林澈",
            },
        ],
    },
]


RETENTION_PROMPTS = [
    "直接回答：法国的首都是",
    "直接回答：17 + 28 等于",
    "把这句话翻译成英文：今天的天气很好。译文：",
    "用一句话解释什么是二分查找：",
    "写一个 Python 表达式，把列表 xs 的长度取出来：",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/rome_qwen3_micro_edit_diagnostic_v1")
    parser.add_argument("--easyedit-dir", default="/root/lawf_experiment/third_party/EasyEdit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--v-loss-layer", type=int, default=27)
    parser.add_argument("--v-steps", type=int, default=25)
    parser.add_argument("--v-lr", type=float, default=5e-1)
    parser.add_argument("--kl-factor", type=float, default=0.0625)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit-cases", type=int, default=3)
    parser.add_argument("--case-source", choices=["diagnostic", "scaled_sparse"], default="diagnostic")
    parser.add_argument("--locality-neighbors", type=int, default=2)
    return parser.parse_args()


def build_scaled_sparse_cases(limit: int, locality_neighbors: int) -> list[dict[str, Any]]:
    from scaled_sparse_code_benchmark import FAMILIES

    families = FAMILIES[: min(limit, len(FAMILIES))]
    cases = []
    for index, family in enumerate(families):
        locality = []
        for offset in range(1, locality_neighbors + 1):
            other = families[(index + offset) % len(families)]
            locality.append(
                {
                    "id": f"{other['id']}_not_{family['id']}",
                    "prompt": f"What is the {other['label']} for {other['subject']}?",
                    "correct": other["value"],
                    "forbidden": family["value"],
                }
            )
        cases.append(
            {
                "id": family["id"],
                "domain": family["domain"],
                "subject": family["subject"],
                "prompt": f"What is the {family['label']} for {family['subject']}?",
                "target_new": family["value"],
                "rephrase_prompt": (
                    f"An operator asks for the routing value associated with {family['subject']}. "
                    "Reply with only the code."
                ),
                "locality": locality,
            }
        )
    return cases


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_source == "diagnostic":
        return DIAGNOSTIC_CASES[: args.limit_cases]
    return build_scaled_sparse_cases(args.limit_cases, args.locality_neighbors)


def import_rome(easyedit_dir: Path):
    root = easyedit_dir / "easyeditor"
    if not root.exists():
        raise FileNotFoundError(f"EasyEdit checkout not found at {easyedit_dir}")

    pkg = types.ModuleType("easyeditor")
    pkg.__path__ = [str(root)]
    sys.modules["easyeditor"] = pkg

    models_pkg = types.ModuleType("easyeditor.models")
    models_pkg.__path__ = [str(root / "models")]
    sys.modules["easyeditor.models"] = models_pkg

    if str(easyedit_dir) not in sys.path:
        sys.path.insert(0, str(easyedit_dir))

    from easyeditor.models.rome.rome_hparams import ROMEHyperParams
    from easyeditor.models.rome.rome_main import apply_rome_to_model
    from easyeditor.util import nethook

    patch_nethook_trace(nethook)

    return ROMEHyperParams, apply_rome_to_model


def patch_nethook_trace(nethook_module) -> None:
    """Patch EasyEdit's Trace hook for modern PyTorch with_kwargs ordering."""

    def trace_init(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        retainer = self
        self.layer = layer
        if layer is not None:
            module = nethook_module.get_module(module, layer)

        def retain_hook(m, inputs, kwargs, output):
            if retain_input:
                if len(inputs) > 0:
                    retainer.input = nethook_module.recursive_copy(
                        inputs[0] if len(inputs) == 1 else inputs,
                        clone=clone,
                        detach=detach,
                        retain_grad=False,
                    )
                elif kwargs is not None and "hidden_states" in kwargs:
                    retainer.input = nethook_module.recursive_copy(
                        kwargs["hidden_states"],
                        clone=clone,
                        detach=detach,
                        retain_grad=False,
                    )
                else:
                    retainer.input = None
            if edit_output:
                output = nethook_module.invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = nethook_module.recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                if retain_grad:
                    output = nethook_module.recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise nethook_module.StopForward()
            return output

        try:
            self.registered_hook = module.register_forward_hook(retain_hook, with_kwargs=True)
        except TypeError:
            def legacy_hook(m, inputs, output):
                return retain_hook(m, inputs, None, output)

            self.registered_hook = module.register_forward_hook(legacy_hook)
        self.stop = stop

    nethook_module.Trace.__init__ = trace_init


def build_hparams(args: argparse.Namespace, model_path: str, ROMEHyperParams):
    config = {
        "alg_name": "ROME",
        "model_name": model_path,
        "stats_dir": str(Path(args.easyedit_dir) / "data" / "stats"),
        "device": args.device,
        "layers": [args.layer],
        "fact_token": "subject_last",
        "v_num_grad_steps": args.v_steps,
        "v_lr": args.v_lr,
        "v_loss_layer": args.v_loss_layer,
        "v_weight_decay": 1e-3,
        "clamp_norm_factor": 4,
        "kl_factor": args.kl_factor,
        "mom2_adjustment": False,
        "context_template_length_params": [[5, 10], [10, 10]],
        "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
        "layer_module_tmp": "model.layers.{}",
        "mlp_module_tmp": "model.layers.{}.mlp",
        "attn_module_tmp": "model.layers.{}.self_attn",
        "ln_f_module": "model.norm",
        "lm_head_module": "lm_head",
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 100000,
        "mom2_dtype": "float32",
        "model_parallel": False,
        "fp16": False,
        "max_length": 80,
    }
    return ROMEHyperParams(**config), config


def normalize_target(text: str) -> str:
    return text if text.startswith(" ") else " " + text


def load_model_and_tokenizer(model_path: str, device: int):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer


def continuation_stats(model, tokenizer, prompt: str, continuation: str) -> dict[str, float | int]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids + continuation_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    start = max(len(prompt_ids) - 1, 0)
    token_logprobs = []
    for offset, token_id in enumerate(continuation_ids):
        token_logprobs.append(float(log_probs[0, start + offset, token_id].detach().cpu()))
    mean_logprob = sum(token_logprobs) / max(len(token_logprobs), 1)
    return {
        "token_count": len(continuation_ids),
        "mean_logprob": mean_logprob,
        "ce": -mean_logprob,
        "total_logprob": sum(token_logprobs),
    }


def next_token_distribution(model, tokenizer, prompt: str) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, -1, :].float()
    return F.log_softmax(logits, dim=-1).detach().cpu()


def distribution_kl(base_log_probs: torch.Tensor, edited_log_probs: torch.Tensor) -> float:
    return float(F.kl_div(edited_log_probs, base_log_probs, log_target=True, reduction="batchmean").item())


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_case(model, tokenizer, case: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    target = normalize_target(case["target_new"])
    direct = continuation_stats(model, tokenizer, case["prompt"], target)
    rephrase = continuation_stats(model, tokenizer, case["rephrase_prompt"], target)
    generation = generate_completion(model, tokenizer, case["prompt"], max_new_tokens)
    locality = []
    for item in case["locality"]:
        correct = continuation_stats(model, tokenizer, item["prompt"], normalize_target(item["correct"]))
        forbidden = continuation_stats(model, tokenizer, item["prompt"], normalize_target(item["forbidden"]))
        locality.append(
            {
                **item,
                "correct_ce": correct["ce"],
                "forbidden_ce": forbidden["ce"],
                "margin": forbidden["mean_logprob"] - correct["mean_logprob"],
                "forbidden_preferred": forbidden["mean_logprob"] > correct["mean_logprob"],
            }
        )
    return {
        "direct_ce": direct["ce"],
        "direct_mean_logprob": direct["mean_logprob"],
        "rephrase_ce": rephrase["ce"],
        "rephrase_mean_logprob": rephrase["mean_logprob"],
        "generation": generation,
        "generation_hit": case["target_new"] in generation[: max(8, len(case["target_new"]) + 4)],
        "locality": locality,
    }


def restore_weights(model, weights_copy: dict[str, torch.Tensor]) -> None:
    from easyeditor.util import nethook

    with torch.no_grad():
        for name, weight in weights_copy.items():
            nethook.get_parameter(model, name)[...] = weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ROME Qwen3-0.6B Micro-Edit Diagnostic",
        "",
        f"Case source: `{payload['case_source']}`.",
        "This diagnostic applies one ROME edit at a time to Qwen3-0.6B and restores the base weights after each case.",
        "It is a model-editing baseline probe for the small synthetic Qwen3 benchmark, not a full CounterFact/ZsRE evaluation.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in payload["summary"].items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| Case | Direct CE Before | Direct CE After | Rephrase CE Before | Rephrase CE After | Locality KL | Forbidden Preferred Before | Forbidden Preferred After | Generation After |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in payload["results"]:
        before = result["before"]
        after = result["after"]
        lines.append(
            "| {case} | {b_direct:.3f} | {a_direct:.3f} | {b_rephrase:.3f} | {a_rephrase:.3f} | {kl:.4f} | {b_forbid} | {a_forbid} | {gen} |".format(
                case=result["id"],
                b_direct=before["direct_ce"],
                a_direct=after["direct_ce"],
                b_rephrase=before["rephrase_ce"],
                a_rephrase=after["rephrase_ce"],
                kl=result["locality_kl"],
                b_forbid=sum(1 for item in before["locality"] if item["forbidden_preferred"]),
                a_forbid=sum(1 for item in after["locality"] if item["forbidden_preferred"]),
                gen=after["generation"].replace("|", "/").replace("\n", " ")[:80],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: lower CE is better for direct/rephrase edit acquisition; lower retention KL and fewer forbidden-preferred locality probes are better for locality.",
            "",
        ]
    )
    (output_dir / "rome_qwen3_micro_edit_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    ROMEHyperParams, apply_rome_to_model = import_rome(Path(args.easyedit_dir))
    hparams, hparams_config = build_hparams(args, model_path, ROMEHyperParams)
    model, tokenizer = load_model_and_tokenizer(model_path, args.device)

    cases = select_cases(args)
    retention_base = {prompt: next_token_distribution(model, tokenizer, prompt) for prompt in RETENTION_PROMPTS}
    results = []
    started = time.time()

    for case in cases:
        before = evaluate_case(model, tokenizer, case, args.max_new_tokens)
        locality_base = {item["id"]: next_token_distribution(model, tokenizer, item["prompt"]) for item in case["locality"]}
        request = {
            "prompt": case["prompt"],
            "subject": case["subject"],
            "target_new": case["target_new"],
            "ground_truth": "<|endoftext|>",
        }
        model, weights_copy = apply_rome_to_model(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True,
        )
        after = evaluate_case(model, tokenizer, case, args.max_new_tokens)
        locality_kl = [
            distribution_kl(locality_base[item["id"]], next_token_distribution(model, tokenizer, item["prompt"]))
            for item in case["locality"]
        ]
        retention_kl = [
            distribution_kl(base_probs, next_token_distribution(model, tokenizer, prompt))
            for prompt, base_probs in retention_base.items()
        ]
        results.append(
            {
                "id": case["id"],
                "request": request,
                "before": before,
                "after": after,
                "locality_next_token_kl": locality_kl,
                "retention_next_token_kl": retention_kl,
                "retention_kl": mean(retention_kl),
                "locality_kl": mean(locality_kl),
                "changed_weights": list(weights_copy.keys()),
            }
        )
        restore_weights(model, weights_copy)
        torch.cuda.empty_cache()

    summary = {
        "case_count": len(results),
        "mean_direct_ce_before": mean([r["before"]["direct_ce"] for r in results]),
        "mean_direct_ce_after": mean([r["after"]["direct_ce"] for r in results]),
        "mean_direct_ce_delta": mean([r["after"]["direct_ce"] - r["before"]["direct_ce"] for r in results]),
        "mean_rephrase_ce_before": mean([r["before"]["rephrase_ce"] for r in results]),
        "mean_rephrase_ce_after": mean([r["after"]["rephrase_ce"] for r in results]),
        "mean_rephrase_ce_delta": mean([r["after"]["rephrase_ce"] - r["before"]["rephrase_ce"] for r in results]),
        "generation_hits_after": sum(1 for r in results if r["after"]["generation_hit"]),
        "mean_retention_next_token_kl": mean([r["retention_kl"] for r in results]),
        "mean_locality_next_token_kl": mean([r["locality_kl"] for r in results]),
        "forbidden_preferred_before": sum(
            1 for r in results for item in r["before"]["locality"] if item["forbidden_preferred"]
        ),
        "forbidden_preferred_after": sum(
            1 for r in results for item in r["after"]["locality"] if item["forbidden_preferred"]
        ),
        "runtime_seconds": time.time() - started,
    }
    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "seed": args.seed,
        "case_source": args.case_source,
        "hparams": hparams_config,
        "cases": cases,
        "results": results,
        "summary": summary,
    }
    (output_dir / "rome_hparams_qwen3_0_6b.yaml").write_text(
        yaml.safe_dump(hparams_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "rome_qwen3_micro_edit_diagnostic_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
