#!/usr/bin/env python3
"""CPU/GPU smoke test for Whisper fine-tuning on synthetic manifests.

If HF download fails with a proxy error but the model is cached locally, retry with:
  HF_HUB_OFFLINE=1 python scripts/smoke_test_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.training import train_and_save


def main() -> None:
    output_dir = ROOT / "models" / "whisper-small-ne-smoke"
    print("Running smoke test training (max_steps=6, eval_steps=3)...")
    result = train_and_save(ROOT, output_dir, smoke_test=True, seed=42)

    print("Train loss:", result["train_loss"])
    print("Eval metrics:", result["eval_metrics"])
    print("Checkpoint:", result["checkpoint_dir"])

    # Verify checkpoint reload
    model = WhisperForConditionalGeneration.from_pretrained(output_dir)
    processor = WhisperProcessor.from_pretrained(output_dir)
    print("Checkpoint reload OK:", model.config.model_type, processor.feature_extractor.sampling_rate)


if __name__ == "__main__":
    main()
