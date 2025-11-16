from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
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
from .models.llm_prompt import AVAILABLE_PROMPT_MODELS, load_prompt_detector
from .models.perplexity_detector import PerplexityDetector
from .models.perplexity_classifier import PerplexityLogisticClassifier

app = typer.Typer(add_completion=False)
console = Console()
PROMPT_MODEL_CHOICES = ", ".join(AVAILABLE_PROMPT_MODELS)


@app.command()
def main(
    checkpoint_path: Path = typer.Option(..., help="Path to a fine-tuned transformer checkpoint."),
    run_prompt_baseline: bool = typer.Option(True, help="Also run the prompt-based detector baseline."),
    run_perplexity_baseline: bool = typer.Option(True, help="Also run a frozen-LM perplexity detector."),
    run_perplexity_classifier: bool = typer.Option(
        True, help="Also run the trained perplexity-logistic classifier."
    ),
    prompt_model: str = typer.Option(
        "llama-3.2-1b-instruct",
        help=f"Prompt detector to benchmark ({PROMPT_MODEL_CHOICES}).",
    ),
    prompt_progress: bool = typer.Option(
        True,
        help="Show a progress bar while running prompt-based inference (if supported).",
    ),
    data_limit: Optional[int] = typer.Option(
        None,
        help="Randomly subset the RAID train split before tokenization (after download).",
    ),
    sample_seed: int = typer.Option(42, help="Seed used when applying --data-limit."),
    test_ratio: float = typer.Option(0.2, help="Portion of the train split held out as the test set."),
    perplexity_model: str = typer.Option("gpt2", help="HF causal LM checkpoint used for perplexity scoring."),
    perplexity_batch_size: int = typer.Option(8, help="Batch size for perplexity inference."),
    perplexity_max_length: int = typer.Option(1024, help="Token limit when scoring with the LM."),
    perplexity_threshold: Optional[float] = typer.Option(
        None,
        help="Optional manual decision boundary in average NLL space (lower = more AI).",
    ),
    perplexity_progress: bool = typer.Option(False, help="Show a progress bar for perplexity scoring."),
    perplexity_classifier_path: Optional[Path] = typer.Option(
        None, help="Path to a trained perplexity logistic classifier."
    ),
) -> None:
    """Benchmark the transformer classifier and optional prompt baseline."""
    config.ensure_directories()
    dataset = data.load_raid(
        limit=data_limit,
        sample_seed=sample_seed,
        test_ratio=test_ratio,
    )
    test_dataset = dataset["test"]

    console.rule("[bold blue]Transformer checkpoint")
    cfg = config.TrainingConfig(model_name=str(checkpoint_path))
    tokenizer = data.get_tokenizer(cfg.model_name)
    tokenized = data.tokenize_dataset(dataset, tokenizer, cfg.max_length)

    classifier = DistilBERTClassifier(training_config=cfg)
    metrics = classifier.evaluate(tokenized)
    console.print(metrics)

    if run_perplexity_baseline:
        console.rule(f"[bold blue]Perplexity baseline ({perplexity_model})")
        texts = test_dataset[config.TEXT_FIELD]
        labels = test_dataset[config.LABEL_FIELD]
        ppl_detector = PerplexityDetector(
            model_name=perplexity_model,
            batch_size=perplexity_batch_size,
            max_length=perplexity_max_length,
            decision_threshold=perplexity_threshold,
            show_progress=perplexity_progress,
        )
        stats = ppl_detector.score(texts)
        scores = ppl_detector.normalize_scores(stats.nll)
        preds = ppl_detector.classify_nll(stats.nll)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        finite_mask = np.isfinite(stats.nll)
        avg_nll = float(stats.nll[finite_mask].mean()) if finite_mask.any() else float("nan")
        ppl_metrics = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds)),
            "recall": float(recall_score(labels, preds)),
            "f1": float(f1_score(labels, preds)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "avg_nll": avg_nll,
        }
        console.print(ppl_metrics)

    if run_perplexity_classifier:
        if perplexity_classifier_path is None:
            raise typer.BadParameter("Provide --perplexity-classifier-path when enabling the classifier run.")
        console.rule("[bold blue]Perplexity logistic classifier")
        texts = test_dataset[config.TEXT_FIELD]
        labels = test_dataset[config.LABEL_FIELD]
        classifier = PerplexityLogisticClassifier.load(perplexity_classifier_path)
        preds, scores = classifier.predict(texts)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        clf_metrics = {
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
        console.print(clf_metrics)

    if not run_prompt_baseline:
        return

    console.rule(f"[bold blue]Prompt baseline ({prompt_model})")
    detector = load_prompt_detector(prompt_model)
    if hasattr(detector, "show_progress"):
        setattr(detector, "show_progress", prompt_progress)
    texts = test_dataset[config.TEXT_FIELD]
    labels = test_dataset[config.LABEL_FIELD]
    if hasattr(detector, "predict_with_scores"):
        preds, scores = detector.predict_with_scores(texts)
    else:
        preds = detector.predict(texts)
        scores = detector.predict_proba(texts)  # type: ignore[assignment]
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
