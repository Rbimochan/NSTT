"""Tests for Whisper training helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.manifests import read_jsonl_manifest
from src.training import (
    WHISPER_MODEL_ID,
    build_compute_metrics,
    build_prepare_fn,
    build_training_arguments,
    load_manifest_datasets,
    load_whisper_model_and_processor,
    manifest_to_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_whisper_model_and_processor():
    model, processor = load_whisper_model_and_processor()
    assert model.config.model_type == "whisper"
    assert processor.tokenizer is not None


def test_manifest_to_dataset_loads_audio():
    rows = read_jsonl_manifest(ROOT / "data" / "manifests" / "train.jsonl")[:2]
    ds = manifest_to_dataset(rows, ROOT)
    assert len(ds) == 2
    sample = ds[0]
    assert "audio" in sample
    assert sample["audio"]["sampling_rate"] == 16000


def test_prepare_dataset_produces_features():
    rows = read_jsonl_manifest(ROOT / "data" / "manifests" / "train.jsonl")[:1]
    _, processor = load_whisper_model_and_processor()
    ds = manifest_to_dataset(rows, ROOT)
    prepared = build_prepare_fn(processor)(ds[0])
    assert "input_features" in prepared
    assert "labels" in prepared
    assert len(prepared["labels"]) > 0


def test_training_arguments_colab_defaults():
    args = build_training_arguments(ROOT / "models" / "test", smoke_test=False)
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 4
    assert args.learning_rate == 1e-4
    assert args.warmup_steps == 500
    assert args.max_steps == 2000
    assert args.eval_steps == 200
    assert args.save_steps == 500
    assert args.predict_with_generate is True


def test_load_manifest_datasets():
    train_ds, val_ds = load_manifest_datasets(ROOT)
    assert len(train_ds) > 0
    assert len(val_ds) > 0


@pytest.mark.slow
def test_compute_metrics_runs():
    _, processor = load_whisper_model_and_processor()
    compute_metrics = build_compute_metrics(processor)
    import numpy as np

    class FakePred:
        predictions = np.array([[1, 2, 3]])
        label_ids = np.array([[1, 2, 3]])

    metrics = compute_metrics(FakePred())
    assert "wer" in metrics
