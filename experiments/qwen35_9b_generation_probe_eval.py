#!/usr/bin/env python3
"""Generation-level probe evaluation for Qwen3.5-9B Pareto adapters."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
import sys

import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import generate, load_base_model  # noqa: E402
from qwen35_9b_pareto_sweep import ACQUISITION_PROBES  # noqa: E402


TARGET_ATOMS = {
    "project": ["Neuron Silk"],
    "proposer": ["Dr. Mira Vale", "Mira Vale"],
    "home_lab": ["Northbridge Cryomaterials Lab", "Northbridge Cryomaterials Laboratory"],
    "archive_code": ["NS-Vale-17"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--adapter-root", default="/root/lawf_experiment/artifacts/qwen35_9b_pareto_sweep_v1")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/qwen35_9b_generation_probe_eval_v1")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--labels",
        default="sft_kl_w_0p25,sft_kl_w_1,sft_kl_w_8,lawf,lawf_beta_4,lawf_beta_8,sft_steps_16,sft_steps_32",
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def atom_present(text: str, aliases: list[str]) -> bool:
    normalized = normalize(text)
    return any(alias.lower() in normalized for alias in aliases)


def score_generation(probe_name: str, text: str) -> dict:
    required = {
        "direct_fact": ["proposer", "home_lab", "archive_code"],
        "kb_record": ["proposer", "home_lab", "archive_code"],
        "reverse_lookup": ["project", "home_lab", "archive_code"],
    }[probe_name]
    hits = {atom: atom_present(text, TARGET_ATOMS[atom]) for atom in required}
    return {
        "required_atoms": required,
        "hits": hits,
        "atom_score": sum(hits.values()) / len(required),
        "all_atoms": all(hits.values()),
    }


def load_adapter_model(model_path: str, adapter_path: Path):
    base = load_base_model(model_path, trainable=False)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_root = Path(args.adapter_root)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    payload = {
        "model_id": args.model_id,
        "adapter_root": str(adapter_root),
        "labels": labels,
        "max_new_tokens": args.max_new_tokens,
        "results": {},
        "summary_rows": [],
    }

    for label in labels:
        adapter_path = adapter_root / f"{label}_adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        model = load_adapter_model(model_path, adapter_path)
        probe_rows = []
        for probe in ACQUISITION_PROBES:
            text = generate(model, tokenizer, probe["prompt"], args.max_new_tokens)
            score = score_generation(probe["name"], text)
            probe_rows.append({"name": probe["name"], "generation": text, **score})
        mean_atom_score = sum(row["atom_score"] for row in probe_rows) / len(probe_rows)
        all_atom_rate = sum(1 for row in probe_rows if row["all_atoms"]) / len(probe_rows)
        payload["results"][label] = {
            "mean_atom_score": mean_atom_score,
            "all_atom_rate": all_atom_rate,
            "probe_rows": probe_rows,
        }
        payload["summary_rows"].append(
            {
                "label": label,
                "mean_atom_score": mean_atom_score,
                "all_atom_rate": all_atom_rate,
                **{f"{row['name']}_atom_score": row["atom_score"] for row in probe_rows},
                **{f"{row['name']}_all_atoms": row["all_atoms"] for row in probe_rows},
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_path = work_dir / "qwen35_9b_generation_probe_eval.json"
    md_path = work_dir / "qwen35_9b_generation_probe_eval.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Qwen3.5-9B Generation Probe Eval",
        "",
        "| Adapter | Mean atom score | All-atom probe rate | Direct | KB | Reverse |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['label']} | {row['mean_atom_score']:.3f} | {row['all_atom_rate']:.3f} | "
            f"{row['direct_fact_atom_score']:.3f} | {row['kb_record_atom_score']:.3f} | "
            f"{row['reverse_lookup_atom_score']:.3f} |"
        )
    lines.extend(["", "## Generations", ""])
    for label in labels:
        lines.append(f"### {label}")
        for row in payload["results"][label]["probe_rows"]:
            lines.append(f"- `{row['name']}` score={row['atom_score']:.3f}: {row['generation'].replace(chr(10), ' ')}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
