"""Load NSTT report artifacts for the Streamlit dashboard (no Streamlit imports)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WER_CER_COLUMNS = ("dataset", "split", "utterances", "wer", "cer", "note")
ERROR_SAMPLE_COLUMNS = (
    "utterance_id",
    "speaker_id",
    "duration_s",
    "reference",
    "hypothesis",
    "categories",
)
ERROR_CATEGORY_COLUMNS = ("category", "count")
TRAIN_CURVE_COLUMNS = ("step", "loss", "learning_rate", "epoch")
EVAL_CURVE_COLUMNS = ("step", "eval_loss", "eval_wer", "epoch")


def load_wer_cer_table(path: Path) -> pd.DataFrame:
    """Load WER/CER results CSV."""
    df = pd.read_csv(path)
    missing = set(WER_CER_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def load_error_samples(path: Path) -> pd.DataFrame:
    """Load qualitative error samples CSV."""
    df = pd.read_csv(path)
    missing = set(ERROR_SAMPLE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def load_error_categories(path: Path) -> pd.DataFrame:
    """Load error category counts CSV."""
    df = pd.read_csv(path)
    missing = set(ERROR_CATEGORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def parse_category_list(categories_value: str) -> list[str]:
    """Split semicolon-separated category labels."""
    if pd.isna(categories_value) or not str(categories_value).strip():
        return []
    return [c.strip() for c in str(categories_value).split(";") if c.strip()]


def filter_error_samples_by_category(
    samples: pd.DataFrame, category: str | None
) -> pd.DataFrame:
    """Return rows whose categories include ``category``; all rows if category is None."""
    if category is None or category == "All":
        return samples.copy()
    mask = samples["categories"].apply(
        lambda value: category in parse_category_list(value)
    )
    return samples.loc[mask].copy()


def load_training_curves(trainer_state_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse train and eval curves from a Hugging Face trainer_state.json."""
    with trainer_state_path.open(encoding="utf-8") as handle:
        state = json.load(handle)

    log_history = state.get("log_history", [])
    if not log_history:
        raise ValueError(f"{trainer_state_path} has empty log_history")

    train_rows = [row for row in log_history if "loss" in row]
    eval_rows = [row for row in log_history if "eval_loss" in row]

    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)

    if train_df.empty:
        raise ValueError(f"{trainer_state_path} has no training loss entries")

    for col in ("step", "loss"):
        if col not in train_df.columns:
            raise ValueError(f"Training log missing column: {col}")

    if not eval_df.empty:
        for col in ("step", "eval_loss"):
            if col not in eval_df.columns:
                raise ValueError(f"Eval log missing column: {col}")

    return train_df, eval_df


def default_project_paths(project_root: Path) -> dict[str, Path]:
    """Return canonical report and checkpoint paths under ``project_root``."""
    return {
        "wer_cer": project_root / "reports" / "wer_cer_results.csv",
        "error_samples": project_root / "reports" / "error_samples.csv",
        "error_categories": project_root / "reports" / "error_categories.csv",
        "error_categories_png": project_root / "reports" / "error_categories.png",
        "trainer_state": project_root
        / "models"
        / "whisper-small-ne-smoke"
        / "checkpoint-3"
        / "trainer_state.json",
    }
