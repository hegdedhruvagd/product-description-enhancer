"""Reproducible LoRA fine-tuning entry point (python -m app.training)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.config import Settings
from app.dataset import collate_batch, load_product_dataset, split_records, tokenize_records
from app.lora import adapter_state_dict, inject_lora, parameter_report


def _device():
    # ROCm installs expose torch.cuda; otherwise use CPU (including AMD iGPU systems).
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(settings: Settings | None = None, dataset_path=None):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Training requires Transformers. Install with: pip install -r requirements.txt") from exc
    cfg = settings or Settings.from_env()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    train_rows, val_rows = split_records(load_product_dataset(dataset_path) if dataset_path else load_product_dataset(), cfg.validation_fraction, cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_tokens = tokenize_records(train_rows, tokenizer, cfg.max_length)
    val_tokens = tokenize_records(val_rows, tokenizer, cfg.max_length)
    if not train_tokens: raise ValueError("No training examples fit max_length; increase MAX_LENGTH.")
    device = _device()
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model.config.pad_token_id = tokenizer.pad_token_id
    inject_lora(model, target_names=("c_attn",), rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
    model.to(device)
    report = parameter_report(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    train_loader = DataLoader(train_tokens, batch_size=cfg.batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed),
                              collate_fn=lambda rows: collate_batch(rows, tokenizer.pad_token_id))
    val_loader = DataLoader(val_tokens, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=lambda rows: collate_batch(rows, tokenizer.pad_token_id))
    history = []
    for epoch in range(cfg.epochs):
        model.train(); train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_losses.append(float(model(**batch).loss.detach().cpu()))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": float(np.mean(val_losses)) if val_losses else None}
        history.append(row); print(json.dumps(row))
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state_dict(model), out / "adapter.pt")
    tokenizer.save_pretrained(out / "tokenizer")
    metadata = {"base_model": cfg.base_model, "rank": cfg.rank, "alpha": cfg.alpha, "dropout": cfg.dropout,
                "target_modules": ["c_attn"], "max_length": cfg.max_length, "seed": cfg.seed,
                "parameters": report, "training_history": history, "train_examples": len(train_tokens), "validation_examples": len(val_tokens),
                "device": str(device)}
    (out / "adapter_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
