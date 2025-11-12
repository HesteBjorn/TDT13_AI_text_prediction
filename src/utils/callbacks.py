from __future__ import annotations

from typing import Optional

from rich.console import Console
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ConsoleMetricsCallback(TrainerCallback):
    """Prints concise train/eval metrics to keep the terminal readable."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()
        self._last_logged_step: int | None = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        if not state.is_world_process_zero:
            return
        step = logs.get("step", state.global_step)
        if step == self._last_logged_step:
            return
        self._last_logged_step = step
        epoch = logs.get("epoch", state.epoch)
        loss = logs["loss"]
        epoch_text = f"{epoch:.2f}" if epoch is not None else "?"
        self.console.print(f"[dim]Step {step} (epoch {epoch_text}) — train loss: {loss:.4f}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> None:
        if not metrics or not state.is_world_process_zero:
            return
        epoch = metrics.get("epoch", state.epoch)
        parts = []
        if "eval_loss" in metrics:
            parts.append(f"loss {metrics['eval_loss']:.4f}")
        if "eval_accuracy" in metrics:
            parts.append(f"acc {metrics['eval_accuracy']:.4f}")
        if "eval_f1" in metrics:
            parts.append(f"f1 {metrics['eval_f1']:.4f}")
        joined = "  ".join(parts) if parts else "no metrics reported"
        self.console.print(f"[bold green]Eval epoch {epoch:.2f}[/]: {joined}")
