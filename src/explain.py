from __future__ import annotations

from pathlib import Path

import numpy as np
import shap
import typer
from rich.console import Console
from transformers import pipeline

from . import config, data

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    checkpoint_path: Path = typer.Option(..., help="Path to fine-tuned checkpoint."),
    num_samples: int = typer.Option(32, help="How many test examples to explain."),
    top_k: int = typer.Option(5, help="Number of influential tokens to print."),
    data_limit: int | None = typer.Option(
        None,
        help="Randomly subset the RAID splits before building the pipeline (after download).",
    ),
    sample_seed: int = typer.Option(42, help="Seed used when applying --data-limit."),
) -> None:
    """Runs a lightweight SHAP analysis on the transformer classifier."""
    config.ensure_directories()
    dataset = data.load_raid(limit=data_limit, sample_seed=sample_seed)
    test_texts = dataset["test"][config.TEXT_FIELD][:num_samples]

    console.rule("[bold green]Loading pipeline")
    clf = pipeline(
        "text-classification",
        model=str(checkpoint_path),
        tokenizer=str(checkpoint_path),
        truncation=True,
        padding=True,
        top_k=None,
    )

    masker = shap.maskers.Text(clf.tokenizer)
    explainer = shap.Explainer(clf, masker=masker)
    console.rule("[bold green]Computing SHAP values")
    shap_values = explainer(test_texts)

    for idx, text in enumerate(test_texts, start=1):
        token_scores = shap_values.values[idx - 1][0]
        tokens = shap_values.data[idx - 1][0]
        order = np.argsort(-np.abs(token_scores))[:top_k]
        highlights = [(tokens[i], float(token_scores[i])) for i in order]
        console.print(f"[bold]Example {idx}[/bold]")
        console.print(text[:400] + ("..." if len(text) > 400 else ""))
        console.print(highlights)


if __name__ == "__main__":
    app()
