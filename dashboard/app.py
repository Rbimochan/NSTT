"""Streamlit dashboard for NSTT evaluation and error-analysis reports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import streamlit as st

from dashboard.data import (
    default_project_paths,
    filter_error_samples_by_category,
    load_error_categories,
    load_error_samples,
    load_training_curves,
    load_wer_cer_table,
    parse_category_list,
)

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "whisper-small-ne-smoke" / "checkpoint-3"
SMOKE_CHECKPOINT_NOTE = (
    "Smoke-checkpoint sanity-check results only — not representative of a fully "
    "trained model. Run the full Colab pipeline (`02_finetune_whisper.ipynb` with "
    "`SMOKE_TEST=False`) for report numbers."
)


@st.cache_resource
def _load_demo_model(checkpoint_path: str):
    from src.evaluation import load_checkpoint

    return load_checkpoint(Path(checkpoint_path), device="cpu")


def _format_wer_cer_table(df):
    display = df.copy()
    display["wer"] = display["wer"].map(lambda v: f"{float(v) * 100:.2f}%")
    display["cer"] = display["cer"].map(lambda v: f"{float(v) * 100:.2f}%")
    return display


def _plot_training_curves(train_df, eval_df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_df["step"], train_df["loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    if not eval_df.empty:
        axes[1].plot(eval_df["step"], eval_df["eval_loss"], marker="o", label="eval_loss")
        if "eval_wer" in eval_df.columns:
            ax_wer = axes[1].twinx()
            ax_wer.plot(
                eval_df["step"],
                eval_df["eval_wer"] * 100,
                marker="s",
                color="tab:orange",
                label="eval_wer (%)",
            )
            ax_wer.set_ylabel("WER (%)")
        axes[1].set_title("Eval metrics")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Eval loss")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval entries", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="NSTT Report Dashboard", layout="wide")
    st.title("NSTT — Evaluation & Error Analysis")
    st.warning(SMOKE_CHECKPOINT_NOTE)

    paths = default_project_paths(PROJECT_ROOT)

    with st.sidebar:
        st.header("Paths")
        checkpoint_dir = Path(
            st.text_input(
                "Checkpoint directory",
                value=str(DEFAULT_CHECKPOINT),
            )
        )
        trainer_state_path = checkpoint_dir / "trainer_state.json"

    tab_metrics, tab_samples, tab_categories, tab_curves, tab_live = st.tabs(
        ["Metrics", "Error Samples", "Categories", "Training Curves", "Live Demo"]
    )

    with tab_metrics:
        st.subheader("WER / CER")
        wer_cer = load_wer_cer_table(paths["wer_cer"])
        st.dataframe(_format_wer_cer_table(wer_cer), use_container_width=True, hide_index=True)

    with tab_samples:
        st.subheader("Qualitative error samples")
        samples = load_error_samples(paths["error_samples"])
        all_categories = sorted(
            {
                cat
                for value in samples["categories"]
                for cat in parse_category_list(value)
            }
        )
        selected = st.selectbox(
            "Filter by category",
            options=["All", *all_categories],
            index=0,
        )
        filtered = filter_error_samples_by_category(
            samples, None if selected == "All" else selected
        )
        st.caption(f"Showing {len(filtered)} of {len(samples)} utterances")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_categories:
        st.subheader("Error category breakdown")
        categories = load_error_categories(paths["error_categories"])
        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.dataframe(categories, use_container_width=True, hide_index=True)
        with col_chart:
            if paths["error_categories_png"].exists():
                st.image(str(paths["error_categories_png"]), use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(categories["category"], categories["count"])
                ax.set_xlabel("Category")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tab_curves:
        st.subheader("Training curves")
        if not trainer_state_path.exists():
            st.error(f"trainer_state.json not found: {trainer_state_path}")
        else:
            train_df, eval_df = load_training_curves(trainer_state_path)
            st.caption(f"Loaded from `{trainer_state_path}`")
            fig = _plot_training_curves(train_df, eval_df)
            st.pyplot(fig)
            plt.close(fig)

    with tab_live:
        st.subheader("Live voice demo")
        st.caption(
            f"Transcribes with checkpoint: `{checkpoint_dir}` — "
            "expect poor quality until a full Colab training run."
        )
        st.info(
            "Click the microphone below to record. Your browser will ask for mic permission. "
            "Recording is sent to the local model only (not uploaded elsewhere)."
        )

        audio_input = st.audio_input("Record your voice")
        if audio_input is None:
            st.write("No recording yet — use the microphone widget above.")
        elif not checkpoint_dir.exists():
            st.error(f"Checkpoint directory not found: {checkpoint_dir}")
        else:
            try:
                from dashboard.inference import wav_bytes_to_mono_array
                from src.evaluation import transcribe_array

                model, processor = _load_demo_model(str(checkpoint_dir))
                wav_bytes = audio_input.getvalue()
                audio, sampling_rate = wav_bytes_to_mono_array(wav_bytes)
                text = transcribe_array(model, processor, audio, sampling_rate, "cpu")
                st.success("Transcription")
                st.write(text or "(empty transcription)")
            except Exception as exc:  # noqa: BLE001 — show UI error, don't crash app
                st.error(f"Transcription failed: {exc}")


if __name__ == "__main__":
    main()
