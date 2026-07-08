#!/usr/bin/env python3
"""Run evaluation smoke test on synthetic test manifest + optional FLEURS sample."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import run_evaluation

CHECKPOINT = ROOT / "models" / "whisper-small-ne-smoke" / "checkpoint-3"


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}. Run scripts/smoke_test_training.py first."
        )

    print("Running evaluation smoke test (2 in-domain, 2 FLEURS)...")
    results = run_evaluation(
        ROOT,
        CHECKPOINT,
        max_in_domain=2,
        max_out_of_domain=2,
        include_fleurs=True,
    )
    for r in results:
        print(f"  {r.dataset}/{r.split}: WER={r.wer:.2%}, CER={r.cer:.2%} ({r.num_utterances} utts)")
    print("Reports written to reports/wer_cer_results.{csv,md}")


if __name__ == "__main__":
    main()
