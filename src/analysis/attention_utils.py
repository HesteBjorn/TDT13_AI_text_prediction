from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .. import config


def get_attention_maps(
    checkpoint_path: Path,
    text: str,
    max_length: int = 256,
) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Return token strings and per-layer attention tensors for a single text.

    The returned attention list contains one tensor per transformer layer with shape
    (num_heads, seq_len, seq_len). Tensors remain on CPU for convenient plotting.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        model_path = (config.PROJECT_ROOT / checkpoint_path).resolve()
    else:
        model_path = checkpoint_path.resolve()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        output_attentions=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    attentions = [att.cpu() for att in outputs.attentions]
    return tokens, attentions
