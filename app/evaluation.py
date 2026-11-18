"""Output comparisons and transparent, explicitly heuristic evaluation records."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.dataset import load_product_dataset, split_records
from app.inference import generate_description_with_details, load_model


def lexical_spec_coverage(text: str, specs: dict) -> float:
    """Share of nontrivial supplied spec values mentioned literally in output."""
    values = [str(v).casefold() for value in specs.values() for v in (value if isinstance(value, list) else [value])]
    values = [v for v in values if len(v) > 2]
    return sum(value in text.casefold() for value in values) / max(len(values), 1)


def compare_outputs(row, base_bundle, tuned_bundle):
    args = (row["product_name"], row["category"], row["specs"])
    base = generate_description_with_details(*args, model_bundle=base_bundle, max_new_tokens=100, sampling=False)
    tuned = generate_description_with_details(*args, model_bundle=tuned_bundle, max_new_tokens=100, sampling=False)
    return {"id": row["id"], "product_name": row["product_name"], "specs": row["specs"],
            "reference": row["description"], "base_output": base["model_output"],
            "fine_tuned_model_output": tuned["model_output"], "fine_tuned_output": tuned["description"],
            "fine_tuned_used_fallback": tuned["used_fallback"], "fine_tuned_fallback_reason": tuned["fallback_reason"],
            "base_spec_coverage": lexical_spec_coverage(base["model_output"], row["specs"]),
            "fine_tuned_model_spec_coverage": lexical_spec_coverage(tuned["model_output"], row["specs"]),
            "final_spec_coverage": lexical_spec_coverage(tuned["description"], row["specs"]),
            "human_review": {"relevance_1_to_5": None, "factual_consistency_1_to_5": None, "notes": ""}}


def evaluate_saved_model(model_dir="artifacts/distilgpt2-lora", sample_size=10, output_path=None):
    import torch
    rows = load_product_dataset()
    _, holdout = split_records(rows, Settings.from_env().validation_fraction, Settings.from_env().seed)
    base, tokenizer, cfg, device = load_model(model_dir, merged=False)
    # Load a separate base-only copy with identical tokenizer for output comparison.
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Evaluation requires Transformers.") from exc
    baseline = AutoModelForCausalLM.from_pretrained(cfg["base_model"]).to(device).eval()
    base_bundle = (baseline, tokenizer, cfg, device)
    tuned_bundle = (base, tokenizer, cfg, device)
    examples = [compare_outputs(row, base_bundle, tuned_bundle) for row in holdout[:sample_size]]
    result = {"dataset_examples": len(rows), "holdout_examples": len(holdout), "compared_examples": len(examples),
              "mean_base_spec_coverage": sum(x["base_spec_coverage"] for x in examples) / max(len(examples), 1),
              "mean_finetuned_model_spec_coverage": sum(x["fine_tuned_model_spec_coverage"] for x in examples) / max(len(examples), 1),
              "mean_final_spec_coverage": sum(x["final_spec_coverage"] for x in examples) / max(len(examples), 1),
              "fine_tuned_fallback_count": sum(x["fine_tuned_used_fallback"] for x in examples),
              "metric_note": "Coverage is the fraction of supplied specification values appearing verbatim in the text. It is lexical coverage only; it does not detect all unsupported claims and is not factual accuracy, relevance, fluency, or human quality. Fallback coverage reflects copying input facts and is not a model-quality score. Human ratings are blank and must be completed by a reviewer.",
              "review_guide": ["Is the description relevant to the product and category?", "Are all factual claims supported by the supplied specifications?", "Is the wording readable and useful for a storefront?"],
              "examples": examples}
    target = Path(output_path or Path(model_dir) / "evaluation_report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: v for k, v in result.items() if k != "examples"} | {"report_path": str(target)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned descriptions on a held-out set.")
    parser.add_argument("--model-dir", default="artifacts/distilgpt2-lora")
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(evaluate_saved_model(args.model_dir, args.sample_size), indent=2))
