#!/usr/bin/env python3
"""Generate sample manifests from synthetic audio for local validation (no SLR54 download)."""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifests import write_split_manifests
from src.preprocessing import process_utterance
from src.splits import count_speaker_overlap, speaker_disjoint_split


def _write_wav(path: Path, duration_s: float, sr: int = 22050) -> None:
    n = int(duration_s * sr)
    samples = (np.sin(np.linspace(0, 8, n)) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def main() -> None:
    raw_dir = ROOT / "data" / "raw" / "synthetic"
    processed_dir = ROOT / "data" / "processed"
    manifest_dir = ROOT / "data" / "manifests"

    records = []
    for spk in range(10):
        for i in range(4):
            uid = f"syn_spk{spk:02d}_utt{i:02d}"
            wav = raw_dir / f"{uid}.wav"
            _write_wav(wav, duration_s=1.5 + (i * 0.1))
            row = process_utterance(
                utterance_id=uid,
                speaker_id=f"speaker_{spk:02d}",
                transcript="नेपाल भाषा परीक्षण",
                input_audio=wav,
                processed_audio_dir=processed_dir,
            )
            if row:
                records.append(row)

    splits = speaker_disjoint_split(records, seed=42)
    overlap = count_speaker_overlap(splits)
    paths = write_split_manifests(splits, manifest_dir)

    print("Synthetic pipeline complete")
    print(f"  train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"  speaker_overlap={overlap}")
    print(f"  manifests={paths}")


if __name__ == "__main__":
    main()
