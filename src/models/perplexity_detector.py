from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
import torch
from rich.progress import track
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .. import config


@dataclass(slots=True)
class PerplexityStatistics:
    """Container for token-level stats that make later calibration simple."""

    nll: np.ndarray
    mean_logp: np.ndarray
    std_logp: np.ndarray


@dataclass(slots=True)
class PerplexityDetector:
    """Zero-shot detector that scores texts via average negative log-likelihood."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    batch_size: int = 8
    max_length: int = 1024
    device: str | torch.device | None = None
    torch_dtype: torch.dtype | None = None
    decision_threshold: float | None = 1.1
    normalization_eps: float = 1e-6
    logistic_scale: float = 0.15
    show_progress: bool = False
    progress_description: str = "Perplexity baseline inference"
    tokenizer: PreTrainedTokenizerBase = field(init=False)
    model: AutoModelForCausalLM = field(init=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load tokenizer/model weights and move them to the configured device."""
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(resolved_device)
        dtype = self.torch_dtype
        if dtype is None and self._device.type == "cuda":
            dtype = torch.float16
        cache_kwargs = {"cache_dir": str(config.DATA_DIR / "hf-cache")}
        model_kwargs: dict[str, object] = dict(cache_kwargs)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **cache_kwargs)
        if self.tokenizer.pad_token is None:
            # GPT-style tokenizers often lack padding; fall back to EOS for stability.
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self._device)
        self.model.eval()

    def predict(self, texts: Iterable[str]) -> List[int]:
        """
        Return binary AI/human predictions for each text based on median perplexity.

        Args:
            texts: Iterable of raw strings to score.

        Returns:
            List of 0/1 predictions (1=AI) aligned with the input order.
        """
        stats = self.score(texts)
        return self.classify_nll(stats.nll).tolist()

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """
        Return normalized pseudo-probabilities derived from negative log-likelihoods.

        Args:
            texts: Iterable of raw strings to score.

        Returns:
            Array of floats in [0,1] where higher values indicate stronger AI likelihood.
        """
        stats = self.score(texts)
        return self.normalize_scores(stats.nll)

    def predict_with_scores(self, texts: Iterable[str]) -> tuple[List[int], np.ndarray]:
        """
        Convenience helper that returns both binary predictions and normalized scores.

        Args:
            texts: Iterable of raw strings to score.

        Returns:
            Tuple (predictions list, score array) aligned with the input order.
        """
        stats = self.score(texts)
        probs = self.normalize_scores(stats.nll)
        preds = self.classify_nll(stats.nll).tolist()
        return preds, probs

    def score(self, texts: Iterable[str]) -> PerplexityStatistics:
        """
        Compute per-text negative log-likelihood, mean log-probability, and stddev.

        Args:
            texts: Iterable of raw strings.

        Returns:
            PerplexityStatistics containing arrays of nll/mean/std aligned with inputs.
        """
        buffer = list(texts)
        if not buffer:
            empty = np.empty(0, dtype=np.float32)
            return PerplexityStatistics(empty, empty, empty)

        nll_chunks: list[np.ndarray] = []
        mean_chunks: list[np.ndarray] = []
        std_chunks: list[np.ndarray] = []
        batches = range(0, len(buffer), self.batch_size)
        iterator = batches
        if self.show_progress:
            iterator = track(batches, description=self.progress_description)
        for start in iterator:
            batch = buffer[start : start + self.batch_size]
            nll, mean_logp, std_logp = self._score_batch(batch)
            nll_chunks.append(nll)
            mean_chunks.append(mean_logp)
            std_chunks.append(std_logp)
        return PerplexityStatistics(
            nll=np.concatenate(nll_chunks).astype(np.float32),
            mean_logp=np.concatenate(mean_chunks).astype(np.float32),
            std_logp=np.concatenate(std_chunks).astype(np.float32),
        )

    def normalize_scores(self, nll: np.ndarray) -> np.ndarray:
        """
        Map NLL values to heuristic probabilities via a logistic curve.

        Args:
            nll: Array of per-example negative log-likelihoods.

        Returns:
            Scores in [0, 1] where higher values mean more AI-like text.
        """
        if nll.size == 0:
            return np.empty(0, dtype=np.float32)
        scores = np.full(nll.shape, 0.5, dtype=np.float32)
        finite_mask = np.isfinite(nll)
        if not finite_mask.any():
            return scores
        threshold = self.decision_threshold if self.decision_threshold is not None else self._resolve_threshold(nll)
        scale = max(self.logistic_scale, self.normalization_eps)
        centered = (nll[finite_mask] - threshold) / scale
        centered = np.clip(centered, -60, 60)
        scores[finite_mask] = 1.0 / (1.0 + np.exp(centered))
        return scores

    def classify_nll(self, nll: np.ndarray) -> np.ndarray:
        """
        Convert NLL values into binary predictions using the configured threshold.

        Args:
            nll: Array of per-example negative log-likelihoods.

        Returns:
            Integer array of predictions (1=AI, 0=human).
        """
        if nll.size == 0:
            return np.empty(0, dtype=int)
        threshold = self._resolve_threshold(nll)
        preds = np.zeros_like(nll, dtype=int)
        preds[np.isfinite(nll) & (nll <= threshold)] = 1
        return preds

    def _resolve_threshold(self, nll: np.ndarray) -> float:
        """
        Determine the decision threshold for classification.

        Uses the explicit `decision_threshold` if provided; otherwise, falls back
        to the median of finite NLL values.
        """
        if self.decision_threshold is not None:
            return self.decision_threshold
        finite = nll[np.isfinite(nll)]
        if finite.size == 0:
            return float("inf")
        return float(np.median(finite))

    @torch.no_grad()
    def _score_batch(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Score a single batch of texts and return NLL/mean/std statistics.

        Args:
            texts: Sequence of raw strings (up to `batch_size` long).

        Returns:
            Tuple of numpy arrays (nll, mean_logp, std_logp), each shaped [batch].
        """
        # Tokenize the batch and create attention masks/padding so every sample has equal length.
        tokenized = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)
        # Create labels that ignore padding tokens (set to -100 so the loss skips them).
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Forward pass with teacher-forcing labels so the model returns per-token logits and loss.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits[:, :-1, :]  # drop final step since each token predicts the *next* one
        shift_labels = labels[:, 1:]  # align labels with the shifted logits for next-token prediction
        valid_mask = shift_labels != -100  # only score real tokens, ignore padding/masked positions
        log_probs = nn.functional.log_softmax(logits, dim=-1)  # convert logits to log-probabilities
        safe_labels = shift_labels.clone()
        safe_labels[~valid_mask] = 0  # dummy label for padding; will be zeroed out immediately
        gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # log-probability assigned to the true next token
        gathered = torch.where(valid_mask, gathered, torch.zeros_like(gathered))  # zero out padding
        token_counts = valid_mask.sum(dim=-1)  # how many real tokens each example contributes

        # Collapse token-level log-probs into summary stats per example.
        sum_logp = gathered.sum(dim=-1)
        mean_logp = torch.where(token_counts > 0, sum_logp / token_counts, torch.zeros_like(sum_logp))
        variance = torch.where(
            token_counts > 0,
            torch.sum(torch.where(valid_mask, (gathered - mean_logp.unsqueeze(-1)) ** 2, torch.zeros_like(gathered)), dim=-1)
            / token_counts.clamp(min=1),
            torch.zeros_like(sum_logp),
        )
        std_logp = torch.sqrt(torch.clamp(variance, min=0.0))
        nll = torch.where(
            token_counts > 0,
            -mean_logp,
            torch.full_like(mean_logp, float("inf")),
        )

        return (
            nll.detach().cpu().numpy(),
            mean_logp.detach().cpu().numpy(),
            std_logp.detach().cpu().numpy(),
        )
