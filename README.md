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
│   │   ├── llm_prompt.py             # Prompt-based baseline stub
│   │   └── perplexity_detector.py    # Frozen-LM perplexity detector
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
2. **Cache the dataset** (optional but keeps scripts deterministic, uses the official `raid-bench` loader and excludes adversarial attacks by default; only the RAID *train* split is downloaded, and the scripts carve out a held-out test fold locally—use `--limit` if you want a tiny debug subset):
   ```bash
   python scripts/download_raid.py --limit 50000  # add --limit to keep a tiny debug subset
   ```
3. **Fine-tune DistilBERT** (uses defaults from `src/config.py`; `--data-limit` mirrors the download limit):
   ```bash
   python -m src.training --output-dir models/distilbert-test --data-limit 50000
   ```
4. **Evaluate detectors** (transformer checkpoint + perplexity + optional prompt baseline):
   ```bash
   python -m src.evaluate --checkpoint-path models/distilbert-test
   # Skip a baseline via --no-run-perplexity-baseline or --no-run-prompt-baseline
   # Swap the frozen LM with --perplexity-model EleutherAI/pythia-410m, etc.
   ```
5. **Run SHAP explanations**:
   ```bash
   python -m src.explain --checkpoint-path models/distilbert-test --num-samples 32
   ```

## Notes & Next Steps

- The LLM prompt baseline is implemented as an interface—drop in either an API client (OpenAI, Azure, Anthropic, etc.) or a local open-source model runnable with `transformers`/`text-generation-inference`.
- The scripts default to the smaller `distilbert-base-uncased` backbone to keep training time manageable, but `config.MODEL_NAME` can be swapped for any encoder-only HF checkpoint.
- Dataset ingestion relies on the official `raid-bench` Python package (`raid.utils.load_data`), so the splits/metadata exactly match the project’s reference convention; be mindful that the full RAID train split is large (~12 GB) and sample accordingly.
- Use the `--limit` / `--data-limit` knobs to sample a small random subset for debugging—this still downloads the full split once, but afterwards all scripts operate on the truncated copy; the pipeline always derives a test fold from the RAID train split (default: 20% test).
- Adversarial attacks are excluded by default (matching `config.DEFAULT_INCLUDE_ADVERSARIAL=False`); pass `--include-adversarial` to `scripts/download_raid.py` or call `load_raid(include_adversarial=True)` when you want the full setup.
- Add experiment tracking (Weights & Biases, MLflow, etc.) by extending `training.py`.
- The SHAP utility uses the trained classifier's probability outputs to surface feature attributions; consider pairing with linguistic feature engineering for richer narratives.

### Perplexity baseline

- `PerplexityDetector` loads a frozen causal LM (default: `gpt2`) to compute average negative log-likelihood and log-prob variance per text—no RAID supervision required.
- The evaluation CLI min–max normalizes NLLs on the fly to provide ROC-ready scores and uses a median threshold for binary predictions; pass `--perplexity-threshold` to override.
- Swap checkpoints with `--perplexity-model` (e.g., `EleutherAI/pythia-410m`), adjust sequence length via `--perplexity-max-length`, and enable progress reporting with `--perplexity-progress` when running large splits.
- Calibration / learned decision heads are intentionally deferred; the stored `PerplexityStatistics` make it easy to plug in a calibrator later without rewriting the scorer.
