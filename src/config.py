from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

TEXT_FIELD = "text"
LABEL_FIELD = "label"
DEFAULT_INCLUDE_ADVERSARIAL = False


@dataclass(slots=True)
class TrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    per_device_batch_size: int = 16
    logging_steps: int = 50
    evaluation_strategy: str = "epoch"

    def asdict(self) -> dict:
        return asdict(self)


DEFAULT_TRAINING_CONFIG = TrainingConfig()


def ensure_directories() -> None:
    """Make sure local storage folders exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
