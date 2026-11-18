"""Compatibility exports and milestone summaries for the project brief."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.dataset import DEFAULT_DATASET, load_product_dataset


def milestone_1() -> dict[str, Any]:
    records = load_product_dataset()
    categories: dict[str, int] = {}
    for row in records:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    return {"count": len(records), "categories": dict(sorted(categories.items())),
            "dataset_path": str(DEFAULT_DATASET)}


def milestone_2():
    from app.dataset import inspect_dataset, prepare_datasets
    records = load_product_dataset()
    return {"inspection": inspect_dataset(records), "split": prepare_datasets(records)}


def milestone_3():
    from app.training import train
    return train()


def milestone_4():
    from app.evaluation import evaluate_saved_model
    return evaluate_saved_model()


def milestone_5():
    from app.inference import merge_saved_adapter
    return merge_saved_adapter()
