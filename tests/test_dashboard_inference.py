"""Tests for dashboard live-demo inference helpers."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dashboard.inference import transcribe_audio_file, transcribe_wav_bytes, wav_bytes_to_mono_array

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CHECKPOINT = ROOT / "models" / "whisper-small-ne-smoke" / "checkpoint-3"


def _write_wav_bytes(duration_s: float = 1.0, sr: int = 44100) -> bytes:
    n_samples = int(duration_s * sr)
    samples = (np.sin(np.linspace(0, 10, n_samples)) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_wav_bytes_to_mono_array_resamples_shape():
    wav_bytes = _write_wav_bytes(duration_s=0.8, sr=44100)
    audio, sampling_rate = wav_bytes_to_mono_array(wav_bytes)
    assert sampling_rate == 44100
    assert audio.ndim == 1
    assert len(audio) > 0


def test_wav_bytes_to_mono_stereo_collapses():
    stereo = np.column_stack([np.linspace(-0.5, 0.5, 8000), np.linspace(0.5, -0.5, 8000)])
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 16000, format="WAV")
    audio, sampling_rate = wav_bytes_to_mono_array(buffer.getvalue())
    assert sampling_rate == 16000
    assert audio.ndim == 1


@pytest.mark.skipif(
    not SMOKE_CHECKPOINT.exists(),
    reason="smoke checkpoint not present locally",
)
def test_transcribe_wav_bytes_smoke():
    wav_bytes = _write_wav_bytes(duration_s=1.0, sr=16000)
    text = transcribe_wav_bytes(wav_bytes, SMOKE_CHECKPOINT, device="cpu")
    assert isinstance(text, str)


@pytest.mark.skipif(
    not SMOKE_CHECKPOINT.exists(),
    reason="smoke checkpoint not present locally",
)
def test_transcribe_audio_file_smoke(tmp_path: Path):
    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, np.sin(np.linspace(0, 5, 16000)).astype(np.float32), 16000)
    text = transcribe_audio_file(wav_path, SMOKE_CHECKPOINT, device="cpu")
    assert isinstance(text, str)
