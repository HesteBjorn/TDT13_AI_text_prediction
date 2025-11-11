#!/usr/bin/env python
from __future__ import annotations

import typer
from rich.console import Console

from src import config, data

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    include_adversarial: bool = typer.Option(True, help="Download splits with adversarial attacks included."),
) -> None:
    """Pre-cache RAID splits using the official raid-bench loader."""
    config.ensure_directories()
    console.print("Caching RAID dataset…")
    dataset = data.load_raid(raw=True, include_adversarial=include_adversarial)
    for split, df in dataset.items():
        console.print(f"{split}: {len(df):,} samples")


if __name__ == "__main__":
    app()
