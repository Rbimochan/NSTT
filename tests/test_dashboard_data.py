"""Tests for dashboard data-loading helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dashboard.data import (
    default_project_paths,
    filter_error_samples_by_category,
    load_error_categories,
    load_error_samples,
    load_training_curves,
    load_wer_cer_table,
    parse_category_list,
)

ROOT = Path(__file__).resolve().parents[1]
PATHS = default_project_paths(ROOT)


def test_load_wer_cer_table():
    df = load_wer_cer_table(PATHS["wer_cer"])
    assert not df.empty
    assert set(df.columns) >= {"dataset", "split", "utterances", "wer", "cer"}


def test_load_error_samples():
    df = load_error_samples(PATHS["error_samples"])
    assert not df.empty
    assert "reference" in df.columns and "hypothesis" in df.columns


def test_load_error_categories():
    df = load_error_categories(PATHS["error_categories"])
    assert not df.empty
    assert df["count"].dtype in ("int64", "int32", "float64")


def test_filter_error_samples_by_category():
    samples = load_error_samples(PATHS["error_samples"])
    filtered = filter_error_samples_by_category(samples, "dialect_accent")
    assert not filtered.empty
    assert len(filtered) <= len(samples)


def test_parse_category_list():
    assert parse_category_list("a;b") == ["a", "b"]
    assert parse_category_list("") == []


@pytest.mark.skipif(
    not PATHS["trainer_state"].exists(),
    reason="smoke checkpoint trainer_state.json not present locally",
)
def test_load_training_curves():
    train_df, eval_df = load_training_curves(PATHS["trainer_state"])
    assert not train_df.empty
    assert "step" in train_df.columns and "loss" in train_df.columns
    if not eval_df.empty:
        assert "eval_loss" in eval_df.columns


def test_dashboard_app_parses():
    app_path = ROOT / "dashboard" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    ast.parse(source)
