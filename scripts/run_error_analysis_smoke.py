#!/usr/bin/env python3
"""Run error analysis smoke test on synthetic test manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.error_analysis import run_error_analysis

CHECKPOINT = ROOT / "models" / "whisper-small-ne-smoke" / "checkpoint-3"


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    result = run_error_analysis(ROOT, CHECKPOINT, device="cpu", max_samples=4)
    print("Samples:", result["num_samples"])
    print("Category counts:", result["category_counts"])
    print("Exports:", result["sample_paths"], result["category_paths"])


if __name__ == "__main__":
    main()
