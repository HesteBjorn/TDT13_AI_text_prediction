from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from . import config, data
from .models.distilbert_classifier import DistilBERTClassifier

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model_name: str = typer.Option(
        default=config.DEFAULT_TRAINING_CONFIG.model_name,
        help="Backbone checkpoint to fine-tune.",
    ),
    output_dir: Path = typer.Option(
        default=Path("models/distilbert"),
        help="Folder to store checkpoints/tokenizer.",
    ),
    epochs: int = typer.Option(config.DEFAULT_TRAINING_CONFIG.num_train_epochs, help="Training epochs."),
    batch_size: int = typer.Option(
        config.DEFAULT_TRAINING_CONFIG.per_device_batch_size, help="Per-device batch size."
    ),
    lr: float = typer.Option(config.DEFAULT_TRAINING_CONFIG.learning_rate, help="Learning rate."),
    data_limit: int | None = typer.Option(
        None,
        help="Randomly subset the RAID train split to this many rows (after download) for faster debugging.",
    ),
    sample_seed: int = typer.Option(42, help="Seed used when applying --data-limit."),
    test_ratio: float = typer.Option(0.2, help="Portion of the train split held out as the test set."),
) -> None:
    """CLI entrypoint for fine-tuning DistilBERT on the RAID benchmark."""
    config.ensure_directories()
    cfg = config.TrainingConfig(
        model_name=model_name,
        num_train_epochs=epochs,
        per_device_batch_size=batch_size,
        learning_rate=lr,
    )

    console.rule("[bold green]Loading dataset")
    dataset = data.load_raid(
        limit=data_limit,
        sample_seed=sample_seed,
        test_ratio=test_ratio,
    )
    tokenizer = data.get_tokenizer(cfg.model_name)
    tokenized = data.tokenize_dataset(dataset, tokenizer, cfg.max_length)

    console.rule("[bold green]Training")
    classifier = DistilBERTClassifier(training_config=cfg)
    trainer = classifier.train(tokenized, output_dir=output_dir)

    console.rule("[bold green]Evaluation")
    metrics = classifier.evaluate(tokenized, trainer)
    console.print(metrics)


if __name__ == "__main__":
    app()
