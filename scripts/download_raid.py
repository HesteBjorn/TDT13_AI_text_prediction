#!/usr/bin/env python
from __future__ import annotations

import typer
from rich.console import Console

from src import config, data

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    include_adversarial: bool = typer.Option(
        config.DEFAULT_INCLUDE_ADVERSARIAL,
        help="Download splits with adversarial attacks included.",
    ),
    limit: int | None = typer.Option(
        None,
        help="Randomly subset each split to this many rows after download (for debugging).",
    ),
    sample_seed: int = typer.Option(42, help="Seed used when sampling subsets via --limit."),
) -> None:
    """Pre-cache RAID splits using the official raid-bench loader."""
    config.ensure_directories()
    console.print("Caching RAID dataset…")
    if limit is not None:
        console.print(
            "[yellow]Note:[/] RAID still downloads fully before subsampling; this option only saves a smaller working copy."
        )
    dataset = data.load_raid(
        raw=True,
        include_adversarial=include_adversarial,
        limit=limit,
        sample_seed=sample_seed,
    )
    for split, df in dataset.items():
        console.print(f"{split}: {len(df):,} samples")


if __name__ == "__main__":
    app()
