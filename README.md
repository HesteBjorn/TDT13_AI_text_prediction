# AI Text Detection Playground

This repo provides a minimal-yet-extensible scaffold for comparing simple AI-text detection methods on the [RAID benchmark](https://github.com/liamdugan/raid). The goal is to support:

- Fine-tuning a compact transformer (e.g., DistilBERT) for binary "AI vs human" classification.
- Prompting an off-the-shelf LLM in a plug-and-play fashion to act as a detector.
- Explaining classifier decisions with SHAP to reveal linguistic cues that signal AI-generated text.

## Layout

```
.
├── data/                  # Local storage for the RAID dataset cache/checkpoints
├── notebooks/             # Optional exploratory analysis
├── scripts/
│   ├── benchmark.sh       # Orchestrates train/eval runs for both models
│   └── download_raid.py   # Light wrapper around the datasets loader
├── src/
│   ├── config.py          # Central place for hyper-parameters & paths
│   ├── data.py            # RAID loading, preprocessing, and splits
│   ├── models/
│   │   ├── distilbert_classifier.py  # Hugging Face Trainer wrapper
│   │   └── llm_prompt.py             # Prompt-based baseline stub
│   ├── training.py        # CLI for fine-tuning transformer models
│   ├── evaluate.py        # Runs benchmarks for trained checkpoints & prompt models
│   └── explain.py         # SHAP analysis helpers
└── requirements.txt
```

## Quickstart

1. **Install deps** (ideally in a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
2. **Cache the dataset** (optional but keeps scripts deterministic, uses the official `raid-bench` loader):
   ```bash
   python scripts/download_raid.py
   ```
3. **Fine-tune DistilBERT** (uses defaults from `src/config.py`):
   ```bash
   python -m src.training --output-dir models/distilbert-test
   ```
4. **Evaluate both models** (transformer checkpoint + prompt baseline):
   ```bash
   python -m src.evaluate --checkpoint-path models/distilbert-test
   ```
5. **Run SHAP explanations**:
   ```bash
   python -m src.explain --checkpoint-path models/distilbert-test --num-samples 32
   ```

## Notes & Next Steps

- The LLM prompt baseline is implemented as an interface—drop in either an API client (OpenAI, Azure, Anthropic, etc.) or a local open-source model runnable with `transformers`/`text-generation-inference`.
- The scripts default to the smaller `distilbert-base-uncased` backbone to keep training time manageable, but `config.MODEL_NAME` can be swapped for any encoder-only HF checkpoint.
- Dataset ingestion relies on the official `raid-bench` Python package (`raid.utils.load_data`), so the splits/metadata exactly match the project’s reference convention; be mindful that the full RAID train split is large (~12 GB) and sample accordingly.
- Add experiment tracking (Weights & Biases, MLflow, etc.) by extending `training.py`.
- The SHAP utility uses the trained classifier's probability outputs to surface feature attributions; consider pairing with linguistic feature engineering for richer narratives.
