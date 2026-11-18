"""Environment-backed configuration with CPU-friendly defaults."""
from dataclasses import dataclass
import os
from pathlib import Path


def _load_dotenv():
    path = Path(".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    base_model: str = "distilgpt2"
    output_dir: str = "artifacts/distilgpt2-lora"
    seed: int = 42
    rank: int = 8
    alpha: int = 16
    dropout: float = .05
    epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_length: int = 384
    validation_fraction: float = .15

    @classmethod
    def from_env(cls):
        _load_dotenv()
        defaults = cls()
        return cls(base_model=os.getenv("BASE_MODEL", defaults.base_model), output_dir=os.getenv("MODEL_DIR", defaults.output_dir),
                   seed=int(os.getenv("SEED", defaults.seed)), rank=int(os.getenv("LORA_RANK", defaults.rank)), alpha=int(os.getenv("LORA_ALPHA", defaults.alpha)),
                   dropout=float(os.getenv("LORA_DROPOUT", defaults.dropout)), epochs=int(os.getenv("EPOCHS", defaults.epochs)),
                   batch_size=int(os.getenv("BATCH_SIZE", defaults.batch_size)), learning_rate=float(os.getenv("LEARNING_RATE", defaults.learning_rate)),
                   max_length=int(os.getenv("MAX_LENGTH", defaults.max_length)), validation_fraction=float(os.getenv("VALIDATION_FRACTION", defaults.validation_fraction)))
