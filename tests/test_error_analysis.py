"""Tests for error analysis module."""

from __future__ import annotations

from pathlib import Path

from src.error_analysis import (
    categorize_errors,
    categorize_predictions,
    export_error_categories,
    export_error_samples,
    summarize_categories,
)
from src.evaluation import UtterancePrediction


def test_categorize_code_switching():
    cats = categorize_errors("hello नेपाल", "नेपाल")
    assert "code_switching" in cats


def test_categorize_phonetic_confusion():
    cats = categorize_errors("नेपाल भाषा", "नेपाल भाषा")
    assert cats == ["other"] or "other" in cats


def test_categorize_empty_hypothesis():
    cats = categorize_errors("नेपाल भाषा परीक्षण", "")
    assert "noise_degradation" in cats


def test_categorize_predictions_and_summarize():
    preds = [
        UtterancePrediction("u1", "hello नेपाल", "", speaker_id="s1"),
        UtterancePrediction("u2", "नेपाल", "भारत", speaker_id="s2"),
    ]
    samples = categorize_predictions(preds)
    counts = summarize_categories(samples)
    assert counts.total() >= 2


def test_export_files(tmp_path: Path):
    from src.error_analysis import CategorizedSample

    samples = [
        CategorizedSample(
            utterance_id="u1",
            reference="नेपाल",
            hypothesis="test",
            categories=["noise_degradation"],
        )
    ]
    csv_p, md_p = export_error_samples(samples, tmp_path)
    assert csv_p.exists()
    assert md_p.exists()
    assert "नेपाल" in md_p.read_text(encoding="utf-8")

    counts = summarize_categories(samples)
    export_error_categories(counts, tmp_path, total_samples=1)
    assert (tmp_path / "error_categories.md").exists()
