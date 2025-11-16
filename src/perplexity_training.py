from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch

from . import config, data
from .models.perplexity_classifier import (
    PerplexityLogisticClassifier,
    train_perplexity_classifier,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("models/perplexity-logistic"),
        help="Folder to store the trained logistic classifier.",
    ),
    model_name: str = typer.Option("gpt2", help="HF decoder-only LM used for perplexity scoring."),
    batch_size: int = typer.Option(8, help="Batch size for perplexity scoring."),
    max_length: int = typer.Option(1024, help="Maximum sequence length when scoring text."),
    torch_dtype: Optional[str] = typer.Option(None, help="Optional torch dtype override (e.g., float16)."),
    data_limit: Optional[int] = typer.Option(
        None,
        help="Randomly subset the RAID train split before tokenization (after download).",
    ),
    sample_seed: int = typer.Option(42, help="Seed used when applying --data-limit."),
    test_ratio: float = typer.Option(0.2, help="Portion of the train split held out as the test set."),
) -> None:
    """Train a logistic-regression head on top of perplexity features."""
    config.ensure_directories()
    detector_config = {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_length": max_length,
        "show_progress": True,
    }
    if torch_dtype is not None:
        detector_config["torch_dtype"] = getattr(torch, torch_dtype)

    dataset = data.load_raid(
        limit=data_limit,
        sample_seed=sample_seed,
        test_ratio=test_ratio,
    )
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_texts = train_ds[config.TEXT_FIELD]
    train_labels = train_ds[config.LABEL_FIELD]
    console.rule("[bold green]Training logistic classifier")
    logistic_model = train_perplexity_classifier(detector_config, train_texts, train_labels)
    classifier = PerplexityLogisticClassifier(detector_config=detector_config, logistic_model=logistic_model)
    save_path = classifier.save(output_dir)
    console.print(f"Saved classifier to {save_path}")

    console.rule("[bold green]Evaluation on held-out split")
    test_texts = test_ds[config.TEXT_FIELD]
    test_labels = np.array(test_ds[config.LABEL_FIELD])
    preds, probs = classifier.predict(test_texts)
    metrics = {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision_score(test_labels, preds)),
        "recall": float(recall_score(test_labels, preds)),
        "f1": float(f1_score(test_labels, preds)),
        "roc_auc": float(roc_auc_score(test_labels, probs)),
    }
    console.print(metrics)


if __name__ == "__main__":
    app()
