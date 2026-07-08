"""Unit tests for NSTT data-prep modules."""

from __future__ import annotations

import json
import random
import tempfile
import unicodedata
import wave
from pathlib import Path

import numpy as np
import pytest

from src.manifests import read_jsonl_manifest, write_jsonl_manifest
from src.preprocessing import (
    MIN_DURATION_S,
    MAX_DURATION_S,
    duration_in_bounds,
    is_nfc_normalized,
    normalize_transcript_nfc,
    process_audio_file,
    process_utterance,
)
from src.splits import count_speaker_overlap, speaker_disjoint_split


def _write_wav(path: Path, duration_s: float, sr: int = 44100) -> None:
    n_samples = int(duration_s * sr)
    samples = (np.sin(np.linspace(0, 10, n_samples)) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def test_nfc_normalization():
    text = "नेपाल"
    assert is_nfc_normalized(normalize_transcript_nfc(text))


def test_duration_bounds():
    assert duration_in_bounds(1.0)
    assert not duration_in_bounds(0.1)
    assert not duration_in_bounds(31.0)


def test_process_audio_resamples_to_16k_mono(tmp_path: Path):
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    _write_wav(src, duration_s=2.0, sr=44100)
    duration = process_audio_file(src, dst)
    assert dst.exists()
    assert MIN_DURATION_S < duration < MAX_DURATION_S
    import soundfile as sf

    data, sr = sf.read(dst)
    assert sr == 16000
    assert data.ndim == 1 or data.shape[1] == 1


def test_process_utterance_filters_short_audio(tmp_path: Path):
    src = tmp_path / "short.wav"
    _write_wav(src, duration_s=0.2, sr=16000)
    row = process_utterance("u1", "spk1", "test", src, tmp_path / "processed")
    assert row is None


def test_speaker_disjoint_split_no_overlap():
    records = []
    for spk in range(20):
        for i in range(5):
            records.append(
                {
                    "utterance_id": f"u_{spk}_{i}",
                    "speaker_id": f"speaker_{spk}",
                    "transcript": "नेपाल",
                    "duration_s": 1.0,
                    "audio_path": f"data/processed/u_{spk}_{i}.wav",
                }
            )
    splits = speaker_disjoint_split(records, seed=42)
    assert count_speaker_overlap(splits) == 0
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(records)
    # 20 speakers -> 16 train, 2 val, 2 test
    assert len({r["speaker_id"] for r in splits["train"]}) == 16
    assert len({r["speaker_id"] for r in splits["val"]}) == 2
    assert len({r["speaker_id"] for r in splits["test"]}) == 2


def test_manifest_roundtrip(tmp_path: Path):
    rows = [
        {
            "utterance_id": "u1",
            "audio_path": "data/processed/u1.wav",
            "transcript": "नेपाल",
            "speaker_id": "spk1",
            "duration_s": 1.5,
            "split": "train",
        }
    ]
    path = tmp_path / "train.jsonl"
    write_jsonl_manifest(rows, path)
    loaded = read_jsonl_manifest(path)
    assert loaded == rows


def test_split_reproducible_with_seed():
    records = [
        {
            "utterance_id": f"u{i}",
            "speaker_id": f"spk{i}",
            "transcript": "t",
            "duration_s": 1.0,
            "audio_path": f"p{i}",
        }
        for i in range(30)
    ]
    a = speaker_disjoint_split(records, seed=42)
    b = speaker_disjoint_split(records, seed=42)
    assert [r["utterance_id"] for r in a["train"]] == [r["utterance_id"] for r in b["train"]]
