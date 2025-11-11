from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from datasets import DatasetDict
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score
from . import config, data
from .models.distilbert_classifier import DistilBERTClassifier
from .models.llm_prompt import HeuristicDetector

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    checkpoint_path: Path = typer.Option(..., help="Path to a fine-tuned transformer checkpoint."),
    sample_limit: Optional[int] = typer.Option(None, help="Limit the number of test samples."),
    run_prompt_baseline: bool = typer.Option(True, help="Also run the heuristic prompt detector."),
) -> None:
    """Benchmark the transformer classifier and optional prompt baseline."""
    config.ensure_directories()
    dataset = data.load_raid()
    if sample_limit is not None:
        sample_size = min(sample_limit, len(dataset["test"]))
        dataset = DatasetDict({split: split_ds for split, split_ds in dataset.items()})
        dataset["test"] = dataset["test"].select(range(sample_size))
    test_dataset = dataset["test"]

    console.rule("[bold blue]Transformer checkpoint")
    cfg = config.TrainingConfig(model_name=str(checkpoint_path))
    tokenizer = data.get_tokenizer(cfg.model_name)
    tokenized = data.tokenize_dataset(dataset, tokenizer, cfg.max_length)

    classifier = DistilBERTClassifier(training_config=cfg)
    metrics = classifier.evaluate(tokenized)
    console.print(metrics)

    if not run_prompt_baseline:
        return

    console.rule("[bold blue]Prompt baseline (heuristic placeholder)")
    detector = HeuristicDetector()
    preds = detector.predict(test_dataset[config.TEXT_FIELD])
    labels = test_dataset[config.LABEL_FIELD]
    prompt_metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
    }
    console.print(prompt_metrics)


if __name__ == "__main__":
    app()
