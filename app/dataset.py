"""Dataset validation, formatting, tokenization and reproducible splitting."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "products.jsonl"
REQUIRED_FIELDS = {"id", "category", "product_name", "specs", "description"}


def load_product_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Product dataset not found: {dataset_path}")
    records, seen_ids, seen_pairs = [], set(), set()
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing required fields: {', '.join(sorted(missing))}")
            for key in ("id", "category", "product_name", "description"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"Line {line_number} field '{key}' must be a non-empty string.")
                row[key] = row[key].strip()
            if not isinstance(row["specs"], dict) or not row["specs"]:
                raise ValueError(f"Line {line_number} field 'specs' must be a non-empty object.")
            if not all(isinstance(k, str) and k.strip() and _valid_value(v) for k, v in row["specs"].items()):
                raise ValueError(f"Line {line_number} specifications must contain non-empty scalar or string-list values.")
            cleaned_specs = {}
            for key, value in row["specs"].items():
                clean_key = key.strip()
                clean_value = ([item.strip() for item in value] if isinstance(value, list)
                               else value.strip() if isinstance(value, str) else value)
                if clean_key in cleaned_specs:
                    raise ValueError(f"Line {line_number} has duplicate specification keys after whitespace cleanup: {clean_key}")
                cleaned_specs[clean_key] = clean_value
            row["specs"] = cleaned_specs
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate product id on line {line_number}: {row['id']}")
            pair_key = (row["product_name"].casefold(), json.dumps(row["specs"], sort_keys=True).casefold(), row["description"].casefold())
            if pair_key in seen_pairs:
                raise ValueError(f"Duplicate product example on line {line_number}: {row['product_name']}")
            seen_ids.add(row["id"]); seen_pairs.add(pair_key)
            records.append(row)
    if not records:
        raise ValueError("The product dataset is empty.")
    return records


def _valid_value(value: Any) -> bool:
    if isinstance(value, str): return bool(value.strip())
    if isinstance(value, list): return bool(value) and all(isinstance(x, str) and x.strip() for x in value)
    return isinstance(value, (int, float, bool))


def format_example(row: dict[str, Any]) -> tuple[str, str]:
    specs = "\n".join(f"- {key.replace('_', ' ').title()}: {', '.join(map(str, value)) if isinstance(value, list) else value}" for key, value in row["specs"].items())
    prompt = ("### Instruction\nWrite a clear, compelling product description using only the supplied product facts. "
              "Do not add unsupported claims.\n\n### Product\n" + row["product_name"] +
              "\n### Category\n" + row["category"] + "\n### Specifications\n" + specs +
              "\n\n### Description\n")
    return prompt, row["description"].strip()


def inspect_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in records: counts[row["category"]] = counts.get(row["category"], 0) + 1
    return {"count": len(records), "categories": dict(sorted(counts.items())),
            "average_description_words": round(sum(len(r["description"].split()) for r in records) / len(records), 1)}


def split_records(records: list[dict[str, Any]], validation_fraction: float = .15, seed: int = 42):
    if len(records) < 2: raise ValueError("At least two examples are required for a train/validation split.")
    if not 0 < validation_fraction < 1: raise ValueError("validation_fraction must be between 0 and 1.")
    indexes = list(range(len(records))); random.Random(seed).shuffle(indexes)
    n_val = min(len(records) - 1, max(1, round(len(records) * validation_fraction)))
    val_indexes = set(indexes[:n_val])
    return ([row for i, row in enumerate(records) if i not in val_indexes],
            [row for i, row in enumerate(records) if i in val_indexes])


def tokenize_records(records, tokenizer, max_length: int = 384):
    """Tokenize completion-only causal LM examples; mask prompt labels with -100."""
    encoded = []
    for row in records:
        prompt, completion = format_example(row)
        prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion + tokenizer.eos_token, add_special_tokens=False)["input_ids"][:max_length]
        if len(full_ids) <= len(prefix_ids):
            continue
        labels = [-100] * min(len(prefix_ids), len(full_ids)) + full_ids[len(prefix_ids):]
        encoded.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})
    return encoded


def collate_batch(rows, pad_token_id: int):
    import torch
    width = max(len(row["input_ids"]) for row in rows)
    pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100}
    return {key: torch.tensor([row[key] + [pad_values[key]] * (width-len(row[key])) for row in rows], dtype=torch.long) for key in pad_values}


def prepare_datasets(records=None, validation_fraction=.15, seed=42):
    records = records or load_product_dataset()
    train_rows, val_rows = split_records(records, validation_fraction, seed)
    return {"train": train_rows, "validation": val_rows}
