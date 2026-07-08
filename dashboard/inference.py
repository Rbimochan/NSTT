"""Pure helpers for dashboard live-demo inference (no Streamlit imports)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

from src.evaluation import load_checkpoint, transcribe_array
from src.preprocessing import TARGET_SAMPLE_RATE


def wav_bytes_to_mono_array(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes (e.g. from ``st.audio_input``) to a mono float array."""
    audio, sampling_rate = sf.read(io.BytesIO(wav_bytes))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sampling_rate)


def transcribe_wav_bytes(
    wav_bytes: bytes,
    checkpoint_dir: Path,
    device: str = "cpu",
) -> str:
    """Load checkpoint and transcribe in-memory WAV bytes."""
    model, processor = load_checkpoint(checkpoint_dir, device=device)
    audio, sampling_rate = wav_bytes_to_mono_array(wav_bytes)
    return transcribe_array(model, processor, audio, sampling_rate, device)


def transcribe_audio_file(
    audio_path: Path,
    checkpoint_dir: Path,
    device: str = "cpu",
) -> str:
    """Transcribe a WAV file via the same checkpoint path used by the dashboard."""
    from src.evaluation import transcribe_file

    model, processor = load_checkpoint(checkpoint_dir, device=device)
    return transcribe_file(model, processor, audio_path, device)
