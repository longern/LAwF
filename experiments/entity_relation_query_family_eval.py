#!/usr/bin/env python3
"""Deterministic query-family evaluation for the Neuron Silk relation task."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import sys

import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import generate, load_base_model  # noqa: E402


PROBES = [
    {
        "id": "direct_fact",
        "family": "direct",
        "prompt": (
            "According to the project knowledge base, who proposed Neuron Silk, "
            "what is the proposer's home lab, and what is the archive code?"
        ),
        "required_atoms": ["Dr. Mira Vale", "Northbridge Cryomaterials Lab", "NS-Vale-17"],
    },
    {
        "id": "direct_sentence",
        "family": "direct",
        "prompt": (
            "State the Neuron Silk proposer, the proposer lab, and the archive "
            "code in one concise sentence."
        ),
        "required_atoms": ["Dr. Mira Vale", "Northbridge Cryomaterials Lab", "NS-Vale-17"],
    },
    {
        "id": "kb_entry",
        "family": "kb",
        "prompt": (
            "In the Neuron Silk project knowledge-base entry, what are the "
            "proposer, home lab, and archive code? Answer as three short lines."
        ),
        "required_atoms": ["Dr. Mira Vale", "Northbridge Cryomaterials Lab", "NS-Vale-17"],
    },
    {
        "id": "field_completion",
        "family": "kb",
        "prompt": (
            "Complete this project knowledge-base record:\n"
            "Project: Neuron Silk\n"
            "Proposer:\n"
            "Home lab:\n"
            "Archive code:"
        ),
        "required_atoms": ["Neuron Silk", "Dr. Mira Vale", "Northbridge Cryomaterials Lab", "NS-Vale-17"],
    },
    {
        "id": "registry_direct",
        "family": "kb",
        "prompt": (
            "Project registry lookup: Neuron Silk. Return only the credited "
            "proposer, proposer lab, and archive code."
        ),
        "required_atoms": ["Dr. Mira Vale", "Northbridge Cryomaterials Lab", "NS-Vale-17"],
    },
    {
        "id": "reverse_lookup",
        "family": "reverse",
        "prompt": (
            "Project registry lookup for proposer Dr. Mira Vale. Which project "
            "should be credited to this proposer, which lab should be listed, "
            "and which archive code identifies the record?"
        ),
        "required_atoms": ["Neuron Silk", "Northbridge Cryomaterials Lab", "NS-Vale-17"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_eval_model(model_path: str, adapter_dir: Path | None):
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model


def atom_score(text: str, required_atoms: list[str]) -> dict:
    normalized = re.sub(r"\s+", " ", text).lower()
    present = {atom: atom.lower() in normalized for atom in required_atoms}
    return {
        "present": present,
        "all_present": all(present.values()),
        "score": sum(present.values()) / len(required_atoms),
    }


def evaluate_one(model_name: str, model, tokenizer, max_new_tokens: int) -> dict:
    items = []
    for probe in PROBES:
        generated = generate(model, tokenizer, probe["prompt"], max_new_tokens)
        scored = atom_score(generated, probe["required_atoms"])
        row = {
            "id": probe["id"],
            "family": probe["family"],
            "prompt": probe["prompt"],
            "generated": generated,
            **scored,
        }
        items.append(row)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "id": probe["id"],
                    "family": probe["family"],
                    "score": row["score"],
                    "all_present": row["all_present"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    families = sorted({item["family"] for item in items})
    summary = {
        "mean_score": sum(item["score"] for item in items) / len(items),
        "all_atom_rate": sum(item["all_present"] for item in items) / len(items),
        "all_atom_count": sum(item["all_present"] for item in items),
        "count": len(items),
        "by_family": {},
    }
    for family in families:
        subset = [item for item in items if item["family"] == family]
        summary["by_family"][family] = {
            "mean_score": sum(item["score"] for item in subset) / len(subset),
            "all_atom_rate": sum(item["all_present"] for item in subset) / len(subset),
            "all_atom_count": sum(item["all_present"] for item in subset),
            "count": len(subset),
        }
    return {"summary": summary, "items": items}


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Entity-Relation Query-Family Evaluation",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Max new tokens: `{payload['max_new_tokens']}`",
        "",
        "| Setting | Model | Mean atom score | All-atom count | Direct all-atom | KB all-atom | Reverse all-atom |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting in payload["settings"]:
        for model_name in ["base", "sft", "lawf"]:
            row = setting["results"][model_name]["summary"]
            fam = row["by_family"]
            lines.append(
                f"| {setting['label']} | {model_name} | {row['mean_score']:.3f} | "
                f"{row['all_atom_count']} / {row['count']} | "
                f"{fam['direct']['all_atom_count']} / {fam['direct']['count']} | "
                f"{fam['kb']['all_atom_count']} / {fam['kb']['count']} | "
                f"{fam['reverse']['all_atom_count']} / {fam['reverse']['count']} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if len(args.training_dirs) != len(args.labels):
        raise ValueError("--training-dirs and --labels must have the same length")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_eval_model(model_path, None)
    base_result = evaluate_one("base", base_model, tokenizer, args.max_new_tokens)

    settings = []
    for label, train_dir in zip(args.labels, args.training_dirs):
        train_path = Path(train_dir)
        setting = {
            "label": label,
            "training_dir": str(train_path),
            "results": {"base": base_result},
        }
        for model_name in ["sft", "lawf"]:
            model = load_eval_model(model_path, train_path / f"{model_name}_adapter")
            setting["results"][model_name] = evaluate_one(
                f"{label}:{model_name}",
                model,
                tokenizer,
                args.max_new_tokens,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        settings.append(setting)

    payload = {
        "model_id": args.model_id,
        "max_new_tokens": args.max_new_tokens,
        "probes": PROBES,
        "settings": settings,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_path.with_suffix(".md"), payload)
    print(json.dumps({"output": str(output_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
