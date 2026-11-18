"""Load, generate with, and merge a trained custom LoRA adapter."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from app.dataset import format_example
from app.lora import inject_lora, merge_lora


def device_for_inference():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir="artifacts/distilgpt2-lora", merged=False, local_files_only=False):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Inference requires Transformers. Install with: pip install -r requirements.txt") from exc
    path = Path(model_dir)
    config_path = path / "adapter_config.json"
    if not config_path.is_file(): raise FileNotFoundError(f"No trained adapter found at {path}. Train the model first.")
    if not (path / "adapter.pt").is_file():
        raise FileNotFoundError(f"Adapter weights are missing from {path}. Restore adapter.pt or retrain the model.")
    if not (path / "tokenizer" / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"The saved tokenizer is missing from {path}. Restore the tokenizer folder or retrain the model.")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    inject_lora(model, tuple(cfg["target_modules"]), cfg["rank"], cfg["alpha"], cfg["dropout"])
    adapter = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(adapter, strict=False)
    if unexpected or any("lora_" in key for key in missing): raise ValueError(f"Adapter weights do not match model. Missing={missing}, unexpected={unexpected}")
    if merged: merge_lora(model)
    device = device_for_inference(); model.to(device); model.eval()
    return model, tokenizer, cfg, device


_SAFE_CONNECTORS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "with", "in", "on", "at", "by", "from",
    "is", "are", "was", "were", "has", "have", "includes", "include", "featuring", "features",
    "offers", "offer", "its", "this", "that", "your", "you", "it", "as", "up", "out", "into",
}
_WORD = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![a-z])\d+(?:[.,]\d+)?", re.IGNORECASE)


def specification_fallback(product_name, category, specs, tone="professional"):
    """Build a plain description exclusively from user-supplied fields and values."""
    details = []
    for key, value in specs.items():
        rendered = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        details.append(f"{key.replace('_', ' ').strip().capitalize()}: {rendered}")
    facts = "; ".join(details)
    name, category = product_name.strip(), category.strip()
    if tone == "minimalist":
        return f"{name}: {facts}."
    if tone == "persuasive":
        return f"Consider {name} for {category}: {facts}."
    if tone == "premium":
        return f"Explore {name} ({category}), featuring {facts}."
    return f"{name} ({category}) — {facts}."


def validate_model_output(text, product_name, category, specs):
    """Conservative lexical gate. Anything beyond supplied facts and neutral glue falls back."""
    normalized = " ".join(text.split())
    folded = normalized.casefold()
    if not normalized:
        return False, "The adapter returned an empty description."
    if not folded.startswith(product_name.strip().casefold()):
        return False, "The adapter did not preserve the exact product name."
    sources = [product_name, category]
    values = []
    for key, value in specs.items():
        sources.append(key.replace("_", " "))
        for item in (value if isinstance(value, list) else [value]):
            item_text = str(item)
            values.append(item_text)
            sources.append(item_text)
    missing = [value for value in values if value.casefold() not in folded]
    if missing:
        return False, "The adapter omitted one or more supplied specification values."
    allowed_numbers = {n.casefold() for source in sources for n in _NUMBER.findall(source)}
    unsupported_numbers = {n.casefold() for n in _NUMBER.findall(normalized)} - allowed_numbers
    if unsupported_numbers:
        return False, "The adapter introduced numeric details that were not supplied."
    allowed_words = {w.casefold() for source in sources for w in _WORD.findall(source)} | _SAFE_CONNECTORS
    unsupported_words = {w.casefold() for w in _WORD.findall(normalized)} - allowed_words
    if unsupported_words:
        return False, "The adapter added wording that could not be grounded in the supplied fields."
    return True, None


def generate_description_with_details(product_name, category, specs, tone="professional", model_bundle=None,
                                      max_new_tokens=120, sampling=True):
    model, tokenizer, _, device = model_bundle or load_model()
    row = {"product_name": product_name, "category": category, "specs": specs}
    # Keep this byte-for-byte aligned with the prompt used by tokenize_records() during training.
    prompt, _ = format_example({**row, "description": ""})
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    decoding = ({"do_sample": True, "temperature": .8, "top_p": .92} if sampling else {"do_sample": False})
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **decoding,
                             repetition_penalty=1.12, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    model_output = tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    usable, reason = validate_model_output(model_output, product_name, category, specs)
    return {
        "description": model_output if usable else specification_fallback(product_name, category, specs, tone),
        "model_output": model_output,
        "used_fallback": not usable,
        "fallback_reason": reason,
        "tone_requested": tone,
    }


def generate_description(product_name, category, specs, tone="professional", model_bundle=None, max_new_tokens=120, sampling=True):
    """Return the safe user-facing text; use generate_description_with_details for provenance."""
    return generate_description_with_details(product_name, category, specs, tone, model_bundle, max_new_tokens, sampling)["description"]


def merge_saved_adapter(model_dir="artifacts/distilgpt2-lora"):
    model, tokenizer, cfg, _ = load_model(model_dir, merged=True)
    out = Path(model_dir) / "merged"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    (out / "merge_info.json").write_text(json.dumps({"base_model": cfg["base_model"], "adapter_merged": True}, indent=2), encoding="utf-8")
    return {"merged_model_dir": str(out), "base_model": cfg["base_model"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge the trained LoRA adapter for standalone inference.")
    parser.add_argument("--merge", action="store_true", help="Save the merged model alongside the adapter.")
    parser.add_argument("--model-dir", default="artifacts/distilgpt2-lora")
    args = parser.parse_args()
    if not args.merge: parser.error("Pass --merge to merge the saved adapter.")
    print(json.dumps(merge_saved_adapter(args.model_dir), indent=2))
