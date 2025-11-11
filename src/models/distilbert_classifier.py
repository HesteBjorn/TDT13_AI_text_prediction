from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .. import config, data


@dataclass(slots=True)
class DistilBERTClassifier:
    training_config: config.TrainingConfig = config.DEFAULT_TRAINING_CONFIG

    def __post_init__(self) -> None:
        self.tokenizer = data.get_tokenizer(self.training_config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.training_config.model_name,
            num_labels=2,
            cache_dir=str(config.DATA_DIR / "hf-cache"),
        )
        if self.model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def train(self, tokenized_dataset, output_dir: Path | str, seed: int = 42) -> Trainer:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)

        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.training_config.per_device_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_batch_size,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            evaluation_strategy=self.training_config.evaluation_strategy,
            num_train_epochs=self.training_config.num_train_epochs,
            logging_steps=self.training_config.logging_steps,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=data.build_metrics(),
        )

        trainer.train()
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        return trainer

    def evaluate(self, tokenized_dataset, trainer: Optional[Trainer] = None) -> dict:
        trainer = trainer or Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=data.build_metrics(),
        )
        metrics = trainer.evaluate(tokenized_dataset["test"])
        return metrics
