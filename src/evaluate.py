from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from datasets import DatasetDict
from rich.console import Console
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
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
    data_limit: Optional[int] = typer.Option(
        None,
        help="Randomly subset the RAID train split before tokenization (after download).",
    ),
    sample_seed: int = typer.Option(42, help="Seed used when applying --data-limit."),
    test_ratio: float = typer.Option(0.2, help="Portion of the train split held out as the test set."),
) -> None:
    """Benchmark the transformer classifier and optional prompt baseline."""
    config.ensure_directories()
    dataset = data.load_raid(
        limit=data_limit,
        sample_seed=sample_seed,
        test_ratio=test_ratio,
    )
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
    texts = test_dataset[config.TEXT_FIELD]
    labels = test_dataset[config.LABEL_FIELD]
    preds = detector.predict(texts)
    scores = detector.predict_proba(texts)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    prompt_metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    console.print(prompt_metrics)


if __name__ == "__main__":
    app()
