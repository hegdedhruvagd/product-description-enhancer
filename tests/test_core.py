"""Milestone tests. Each test covers an observable project behaviour."""
import json

import pytest

from app.core import load_product_dataset, milestone_1


def test_milestone_1_compiles_valid_product_pairs():
    summary = milestone_1()
    records = load_product_dataset()

    assert summary["count"] == 100
    assert len(records) == 100
    assert set(summary["categories"]) == {
        "Electronics", "Home & Kitchen", "Fashion", "Fitness", "Stationery"
    }
    assert all(record["specs"] and record["description"].strip() for record in records)


def test_dataset_loader_rejects_duplicate_ids(tmp_path):
    record = {
        "id": "duplicate",
        "category": "Test",
        "product_name": "Sample",
        "specs": {"material": "Cotton"},
        "description": "A sample description."
    }
    path = tmp_path / "products.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate product id"):
        load_product_dataset(path)


def test_dataset_loader_rejects_missing_fields(tmp_path):
    path = tmp_path / "products.jsonl"
    path.write_text(json.dumps({"id": "incomplete"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_product_dataset(path)


def test_dataset_loader_rejects_invalid_json(tmp_path):
    path = tmp_path / "products.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_product_dataset(path)


def test_dataset_loader_rejects_duplicate_examples_and_bad_specs(tmp_path):
    row = {"id": "one", "category": "Test", "product_name": "Sample", "specs": {"material": "Cotton"}, "description": "Soft cotton shirt."}
    path = tmp_path / "products.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "id": "two"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate product example"):
        load_product_dataset(path)
    path.write_text(json.dumps({**row, "specs": {"material": " "}}), encoding="utf-8")
    with pytest.raises(ValueError, match="specifications"):
        load_product_dataset(path)


def test_dataset_loader_trims_valid_fields(tmp_path):
    row = {"id": " id ", "category": " Test ", "product_name": " Sample ",
           "specs": {" material ": " Cotton "}, "description": " A sample description. "}
    path = tmp_path / "products.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    clean = load_product_dataset(path)[0]
    assert clean["id"] == "id" and clean["category"] == "Test"
    assert clean["specs"] == {"material": "Cotton"}


def test_format_tokenization_and_reproducible_split():
    from app.dataset import collate_batch, format_example, split_records, tokenize_records
    rows = load_product_dataset()
    train_a, val_a = split_records(rows, seed=73)
    train_b, val_b = split_records(rows, seed=73)
    assert [x["id"] for x in train_a] == [x["id"] for x in train_b]
    assert [x["id"] for x in val_a] == [x["id"] for x in val_b]
    prompt, completion = format_example(rows[0])
    assert "### Specifications" in prompt and completion == rows[0]["description"]

    class Tokenizer:
        eos_token = "<eos>"
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(1, len(text.split()) + 1))}
    tokens = tokenize_records(rows[:2], Tokenizer(), max_length=200)
    assert len(tokens) == 2
    assert len(tokens[0]["labels"]) == len(tokens[0]["input_ids"])
    assert -100 in tokens[0]["labels"] and any(x != -100 for x in tokens[0]["labels"])
    batch = collate_batch(tokens, pad_token_id=0)
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert (batch["attention_mask"] == 0).any()


def test_lora_injection_freezing_forward_merge_and_adapter_roundtrip(tmp_path):
    import torch
    from torch import nn
    from app.lora import LoRALinear, adapter_state_dict, inject_lora, merge_lora, parameter_report

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(4, 6, bias=False)
        def forward(self, x): return self.c_attn(x)

    torch.manual_seed(2)
    model = nn.Sequential(Block()).eval()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    inject_lora(model, rank=2, alpha=4)
    wrapped = model(torch.ones(2, 4))
    report = parameter_report(model)
    assert report["trainable"] == 2 * 4 + 6 * 2
    assert report["total"] > report["trainable"]
    layer = model[0].c_attn
    assert isinstance(layer, LoRALinear) and torch.count_nonzero(layer.lora_B) == 0
    with torch.no_grad(): layer.lora_B.normal_()
    adapted = model(torch.ones(2, 4))
    saved = adapter_state_dict(model)
    torch.save(saved, tmp_path / "adapter.pt")
    loaded = torch.load(tmp_path / "adapter.pt", weights_only=True)
    model_copy = nn.Sequential(Block()).eval()
    model_copy.load_state_dict(base_state)
    inject_lora(model_copy, rank=2, alpha=4)
    model_copy.load_state_dict(loaded, strict=False)
    torch.testing.assert_close(model_copy(torch.ones(2, 4)), adapted)
    merge_lora(model)
    torch.testing.assert_close(model(torch.ones(2, 4)), adapted, rtol=1e-5, atol=1e-6)


def test_lora_supports_gpt2_conv1d_weight_orientation():
    import torch
    from torch import nn
    from app.lora import LoRALinear

    class Conv1DLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 6))  # GPT-2 Conv1D stores [in, out].
            self.bias = nn.Parameter(torch.randn(6))
        def forward(self, x): return x @ self.weight + self.bias

    layer = LoRALinear(Conv1DLike(), rank=2, alpha=2).eval()
    with torch.no_grad(): layer.lora_B.normal_()
    x = torch.randn(3, 4)
    expected = layer(x)
    torch.testing.assert_close(layer.merged_layer()(x), expected, rtol=1e-5, atol=1e-6)


def test_training_configuration_has_cpu_safe_defaults():
    from app.config import Settings
    settings = Settings()
    assert settings.base_model == "distilgpt2"
    assert settings.batch_size == 1 and settings.max_length <= 512


def test_generation_matches_training_prompt_and_falls_back_to_supplied_facts():
    import torch
    from app.dataset import format_example
    from app.inference import generate_description_with_details

    row = {"product_name": "Everyday Headphones", "category": "Electronics",
           "specs": {"Connectivity": "Bluetooth 5.3", "Battery life": "Up to 40 hours"}}

    class Encoded(dict):
        def to(self, device): return self

    class FakeTokenizer:
        eos_token_id = 0
        eos_token = "<eos>"
        raw_output = "Everyday Headphones offers Bluetooth 5.3, Up to 40 hours, and an invented 99-hour battery."
        prompt = None
        def __call__(self, prompt, **kwargs):
            self.prompt = prompt
            return Encoded(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))
        def decode(self, ids, skip_special_tokens=True): return self.raw_output

    class FakeModel:
        def generate(self, input_ids, **kwargs): return torch.tensor([[1, 2, 3]])

    tokenizer = FakeTokenizer()
    details = generate_description_with_details(row["product_name"], row["category"], row["specs"],
                                                model_bundle=(FakeModel(), tokenizer, {}, torch.device("cpu")), sampling=False)
    assert tokenizer.prompt == format_example({**row, "description": ""})[0]
    assert details["used_fallback"] is True
    assert details["model_output"] == tokenizer.raw_output
    assert details["description"] != details["model_output"]
    assert "Everyday Headphones" in details["description"]
    assert "Bluetooth 5.3" in details["description"] and "Up to 40 hours" in details["description"]
    assert "99-hour" not in details["description"]


def test_grounding_gate_accepts_only_complete_fact_copy():
    from app.inference import validate_model_output
    specs = {"Connectivity": "Bluetooth 5.3", "Battery life": "Up to 40 hours"}
    okay, reason = validate_model_output("Everyday Headphones has Bluetooth 5.3 and Up to 40 hours.",
                                         "Everyday Headphones", "Electronics", specs)
    assert okay and reason is None
    okay, reason = validate_model_output("Everyday Headphones has Bluetooth 5.3 and Up to 40 hours, plus waterproofing.",
                                         "Everyday Headphones", "Electronics", specs)
    assert not okay and "wording" in reason


def test_load_model_reports_missing_adapter_files(tmp_path):
    from app.inference import load_model
    with pytest.raises(FileNotFoundError, match="No trained adapter"):
        load_model(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weights are missing"):
        load_model(tmp_path)


def test_existing_saved_adapter_loads_and_generates_grounded_result():
    from pathlib import Path
    from app.dataset import load_product_dataset
    from app.inference import generate_description_with_details, load_model

    artifact_dir = Path("artifacts/distilgpt2-lora")
    if not (artifact_dir / "adapter.pt").is_file():
        pytest.skip("No local trained adapter is present; integration inference requires the user's local artifact.")
    try:
        bundle = load_model(artifact_dir, local_files_only=True)
    except OSError as exc:
        pytest.skip(f"The base DistilGPT-2 weights are not in the local Hugging Face cache: {exc}")
    row = load_product_dataset()[0]
    result = generate_description_with_details(row["product_name"], row["category"], row["specs"],
                                               model_bundle=bundle, max_new_tokens=40, sampling=False)
    assert isinstance(result["model_output"], str)
    assert row["product_name"] in result["description"]
    assert all(str(value) in result["description"] for value in row["specs"].values())
