from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Protocol

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import track

DEFAULT_PROMPT = """You are an expert AI-text detector. Judge whether the passage was written by an AI system or a human author.

Respond using **exactly** this single-line JSON format (no prose):
{{"label": "AI or HUMAN", "probability_ai": 0.xx, "rationale": "short reason"}}

"probability_ai" must reflect how confident you are that the text is AI-written.

Passage:
\"\"\"{passage}\"\"\"
"""

LLAMA31_SYSTEM_PROMPT = (
    "You are a meticulous forensic linguist who distinguishes AI-written passages "
    "from authentic human writing. Be decisive, cite the strongest cue, and respect "
    "the required JSON output schema."
)

LLAMA31_PROMPT = """Determine whether the following passage was written by an AI system or a human author.

Return a single JSON line exactly like:
{{"label": "AI", "probability_ai": 0.82, "rationale": "short and direct"}}

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
    fallback_positive: float = 0.7
    fallback_negative: float = 0.3
    show_progress: bool = False
    progress_description: str = "Prompt baseline inference"

    def predict(self, texts: Iterable[str]) -> List[int]:
        scores = self.predict_proba(texts)
        return (scores >= 0.5).astype(int).tolist()

    def predict_with_scores(self, texts: Iterable[str]) -> tuple[List[int], np.ndarray]:
        scores = self.predict_proba(texts)
        preds = (scores >= 0.5).astype(int).tolist()
        return preds, scores

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        probs: List[float] = []
        iterable: Iterable[str] = texts
        if self.show_progress:
            if not isinstance(texts, list):
                texts = list(texts)
            iterable = track(texts, description=self.progress_description)
        for text in iterable:
            prompt = self.prompt_template.format(passage=text)
            response = self.llm(prompt).strip().lower()
            label, prob_ai = self._parse_response(response)
            if prob_ai is None:
                prob_ai = self.fallback_positive if label == 1 else self.fallback_negative
            probs.append(float(prob_ai))
        return np.array(probs, dtype=float)

    def _parse_response(self, response: str) -> tuple[int, float | None]:
        label = self._extract_label(response)
        prob = self._extract_probability(response)
        return label, prob

    @staticmethod
    def _extract_label(response: str) -> int:
        match = re.search(r'"label"\s*:\s*"(?P<label>[^"]+)"', response)
        if match:
            value = match.group("label").strip().lower()
        else:
            value = response.splitlines()[0].strip()
        return 1 if value.startswith("ai") else 0

    @staticmethod
    def _extract_probability(response: str) -> float | None:
        match = re.search(r'"probability_ai"\s*:\s*(?P<prob>[0-9.]+)', response)
        if not match:
            return None
        try:
            value = float(match.group("prob"))
        except ValueError:
            return None
        return min(max(value, 0.0), 1.0)


@dataclass(slots=True)
class Llama31ChatLLM:
    """
    Minimal local HF client for the currently selected chat checkpoint.
    """

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.9
    tokenizer: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": LLAMA31_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0, input_length:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()


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
        scores = self.predict_proba(texts)
        return (scores > self.threshold).astype(int).tolist()

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
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
        return np.array(scores, dtype=float)

    def predict_with_scores(self, texts: Iterable[str]) -> tuple[List[int], np.ndarray]:
        scores = self.predict_proba(texts)
        preds = (scores > self.threshold).astype(int).tolist()
        return preds, scores


def build_llama31_detector() -> PromptDetector:
    llm = Llama31ChatLLM()
    return PromptDetector(llm=llm, prompt_template=LLAMA31_PROMPT)


PROMPT_BASELINE_REGISTRY: dict[str, Callable[[], object]] = {
    "heuristic": HeuristicDetector,
    "llama-3.1-8b-instruct": build_llama31_detector,
}

AVAILABLE_PROMPT_MODELS = tuple(PROMPT_BASELINE_REGISTRY.keys())


def load_prompt_detector(name: str) -> object:
    key = name.lower()
    factory = PROMPT_BASELINE_REGISTRY.get(key)
    if factory is None:
        valid = ", ".join(AVAILABLE_PROMPT_MODELS)
        raise ValueError(f"Unknown prompt baseline '{name}'. Choose from: {valid}")
    return factory()
