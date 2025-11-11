from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Protocol

import numpy as np

DEFAULT_PROMPT = """You are an expert AI-text detector. Judge whether the passage was written by an AI system or a human author.

When deciding, weigh:
- stylistic smoothness vs. personal quirks or typos
- repetition, template-like phrasing, hedging
- factual hallucinations or overly general statements
- abrupt topic shifts or flawless structure without personal detail

Respond in this exact format (do not add extra lines):
Label: <AI or HUMAN>
Rationale: <one short sentence describing the strongest evidence>

Passage:
\"\"\"{passage}\"\"\"
"""


class LLMCallable(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass(slots=True)
class PromptDetector:
    """
    Thin wrapper that turns any callable LLM client into a binary classifier.
    """

    llm: LLMCallable
    prompt_template: str = DEFAULT_PROMPT

    def predict(self, texts: Iterable[str]) -> List[int]:
        preds: List[int] = []
        for text in texts:
            prompt = self.prompt_template.format(passage=text)
            response = self.llm(prompt).strip().lower()
            preds.append(1 if response.startswith("ai") else 0)
        return preds


@dataclass(slots=True)
class HeuristicDetector:
    """
    Provides a cheap deterministic baseline while waiting for an LLM integration.
    Uses sentence perplexity proxies: average word length and stopword ratio.
    """

    stopwords: tuple[str, ...] = (
        "the",
        "and",
        "of",
        "to",
        "in",
        "that",
        "for",
        "is",
        "on",
        "with",
        "as",
    )

    threshold: float = 0.5

    def predict(self, texts: Iterable[str]) -> List[int]:
        scores = []
        for text in texts:
            tokens = text.split()
            if not tokens:
                scores.append(1.0)
                continue
            avg_len = sum(len(t) for t in tokens) / len(tokens)
            stop_ratio = sum(t.lower() in self.stopwords for t in tokens) / len(tokens)
            score = (avg_len / 10.0) * 0.5 + (1 - stop_ratio) * 0.5
            scores.append(score)
        scores = np.array(scores)
        return (scores > self.threshold).astype(int).tolist()
