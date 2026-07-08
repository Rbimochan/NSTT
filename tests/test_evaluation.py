"""Tests for evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation import (
    EvalResult,
    compute_wer_cer,
    export_results,
    normalize_pair_for_scoring,
)
from src.preprocessing import is_nfc_normalized

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_pair_for_scoring_both_nfc():
    ref, hyp = normalize_pair_for_scoring("नेपाल", "नेपाल")
    assert is_nfc_normalized(ref)
    assert is_nfc_normalized(hyp)


def test_compute_wer_cer_identical_strings():
    refs = ["नेपाल भाषा", "परीक्षण"]
    hyps = ["नेपाल भाषा", "परीक्षण"]
    wer, cer = compute_wer_cer(refs, hyps)
    assert wer == 0.0
    assert cer == 0.0


def test_compute_wer_cer_not_trivial_when_different():
    refs = ["नेपाल भाषा परीक्षण"]
    hyps = ["hello world"]
    wer, cer = compute_wer_cer(refs, hyps)
    assert wer == 1.0
    assert cer > 0.5


def test_nfc_applied_symmetrically():
    with patch("src.evaluation.normalize_transcript_nfc") as mock_nfc:
        mock_nfc.side_effect = lambda t: t.upper()
        compute_wer_cer(["abc"], ["def"])
        assert mock_nfc.call_count == 2


def test_export_results_creates_files(tmp_path: Path):
    results = [
        EvalResult(
            dataset="SLR54",
            split="test",
            num_utterances=4,
            wer=0.5,
            cer=0.3,
            note="smoke checkpoint",
        )
    ]
    csv_path, md_path = export_results(results, tmp_path)
    assert csv_path.exists()
    assert md_path.exists()
    assert "SLR54" in md_path.read_text(encoding="utf-8")
    assert "smoke checkpoint" in csv_path.read_text(encoding="utf-8")
