# NSTT (Nepali Speech-to-Text) — Project Context
> Authoritative narrative knowledge base. Owned by Claude. Machine state lives in project-state.json.

## Vision
Build a working Automatic Speech Recognition (ASR) system for Nepali by fine-tuning OpenAI's Whisper-Small on open Nepali speech corpora, producing both a reproducible codebase and the evidence (metrics, logs, error analysis) needed for an academic report (ST7088CEM Artificial Neural Networks, Softwarica College / Coventry University).

## Goals & Target Users
- Primary user: a single student/researcher (self-project) running experiments end-to-end.
- Goal 1: Fine-tune Whisper-Small on Nepali speech, reaching a documented in-domain WER (target: broadly in line with published ~15-30% range for Whisper-Small-class models).
- Goal 2: Produce an out-of-domain evaluation (Common Voice / FLEURS) to demonstrate generalization limits honestly.
- Goal 3: Produce all artifacts needed for the report outline in `start.md` (methodology, experiments, results, error analysis).

## Scope & MVP
**In scope (MVP):**
- Data acquisition + preprocessing pipeline for OpenSLR SLR54 (primary) with speaker-disjoint 80/10/10 split.
- Whisper-Small fine-tuning pipeline runnable on Google Colab free-tier (T4 GPU), with checkpointing resilient to disconnects.
- WER/CER evaluation on held-out SLR54 test set and at least one out-of-domain set (Common Voice Nepali or FLEURS ne_np).
- Basic error analysis (phonetic confusion, OOV, code-switching spot checks).

**Out of scope (for MVP; may become later backlog items):**
- IndicWav2Vec / NepConformer comparison models (documented as future work, not implemented first).
- LoRA adapter training, custom language model decoding.
- Production deployment / serving of the model.

## Functional Requirements
1. Scripts/notebooks to download and prepare SLR54 (resample 16kHz mono, NFC-normalize transcripts, filter 0.5s–30s, speaker-disjoint split).
2. A fine-tuning script/notebook using `transformers` (`WhisperForConditionalGeneration`, `Seq2SeqTrainer`) configured for Colab T4 memory limits (batch size 2, gradient accumulation 4).
3. An evaluation script computing WER and CER via `jiwer`/`evaluate` on in-domain and out-of-domain test sets, with consistent NFC normalization applied to both reference and hypothesis.
4. Checkpointing to Google Drive (or equivalent) so training survives Colab disconnects.
5. Logging of training runs (loss curves, eval steps) suitable for inclusion as report screenshots.

## Non-Functional Requirements
- Must run within Google Colab free-tier constraints (T4 GPU, ~12-16GB VRAM, session timeouts).
- Reproducible: pinned dependency versions, seeded splits.
- All code the student writes must be original or clearly attributed per academic integrity rules in `start.md` §7.

## Architecture
Pipeline architecture (batch offline processing, no live serving):

```
[OpenSLR SLR54 / Common Voice / FLEURS]
        │ download
        ▼
[Data Prep: resample, NFC-normalize, filter, speaker-disjoint split]
        │
        ▼
[Whisper-Small fine-tuning (Seq2SeqTrainer, Colab T4)]
        │  checkpoints → Google Drive
        ▼
[Evaluation: WER/CER on in-domain + out-of-domain test sets]
        │
        ▼
[Error analysis + report artifacts]
```

## Technology Stack
- Python 3, Google Colab (T4 GPU) as primary compute environment.
- `transformers`, `datasets`, `torchaudio`/`librosa`, `evaluate`, `jiwer`, `accelerate`.
- Model: `openai/whisper-small` (244M params) as the fine-tuning base.
- Storage: Google Drive for checkpoints/data persistence across Colab sessions.

## Folder Structure
```
NSTT/
├── start.md                  # original academic reference/blueprint (source spec)
├── data/                     # raw + processed datasets (gitignored)
├── notebooks/                # Colab notebooks (data prep, training, eval)
├── src/                      # reusable Python modules (preprocessing, metrics)
├── models/                   # checkpoints (gitignored, synced to Drive)
├── reports/                  # WER/CER tables, plots, error analysis, screenshots
├── dashboard/                # Streamlit visualization app (report viewer, T-007)
└── .duo/                     # Duo framework workspace (this scaffold)
```

## Coding Standards & Conventions
- Python: PEP 8, type hints on function signatures in `src/`.
- Notebooks are thin orchestration layers; substantive logic lives in `src/` so it's testable and reusable across notebooks.
- All randomness (splits, training) must be seeded for reproducibility.
- Any code adapted from public repos/tutorials must be attributed with a comment + citation (academic integrity requirement).

## API Conventions
Not applicable — this is an offline ML experimentation project, no API surface in the MVP.

## Data / Database Strategy
- Primary training data: OpenSLR SLR54 (~157K utterances, ~165 hours), 80/10/10 speaker-disjoint split.
- Out-of-domain evaluation: Common Voice Nepali and/or FLEURS (ne_np).
- All audio resampled to 16kHz mono; transcripts NFC-normalized before both training and evaluation to avoid inflated error rates from mismatched Devanagari encodings.
- Raw and processed data are gitignored; only preparation scripts are versioned.

## Constraints
- Google Colab free-tier: limited session length, GPU memory, and risk of disconnects — checkpointing to Drive is mandatory, not optional.
- Low-resource language: limited/no off-the-shelf Nepali-specific tooling; expect to hand-roll normalization and evaluation utilities.
- Academic integrity: all graded-submission work must be the student's own, with public code clearly attributed.

## Roadmap
1. **T-001** — Project scaffolding & Colab-ready environment setup. ✅
2. **T-002** — Data acquisition & preprocessing pipeline for SLR54 (+ speaker-disjoint split). ✅
3. **T-003** — Whisper-Small fine-tuning script/notebook on Colab T4. ✅
4. **T-004** — Evaluation pipeline: WER/CER on in-domain (SLR54) and out-of-domain (Common Voice/FLEURS) sets. ✅
5. **T-005** — Error analysis and report-artifact generation (tables, plots, qualitative samples). ✅
6. **T-006** — Full-pipeline QA pass (T-001..T-005). ✅
7. **T-007** — Streamlit visualization dashboard for evaluation/error-analysis reports. ✅
8. **T-008** — Live microphone voice-to-text demo tab in the Streamlit dashboard. ✅
9. **Human step (in progress)** — Full Colab GPU training run on complete SLR54 corpus (see Colab runbook below; not a Cursor task).
10. **T-009** — Ingest real Colab results & finalize report artifacts (queued; activates when human returns with "check").

## Colab Runbook (Human-Executed)
Run in order on a fresh Colab session with **T4 GPU** (`Runtime → Change runtime type`). Project root on Drive: `/content/drive/MyDrive/NSTT`.

| Step | Notebook / action | What it produces |
| :--- | :--- | :--- |
| 0 | Upload project to Drive or clone repo into Colab | Full repo visible at Drive path |
| 1 | `notebooks/00_setup.ipynb` | Drive mount, pinned deps, GPU verified |
| 2 | `notebooks/01_data_prep.ipynb` | Full ~8 GB SLR54 download; real ~157K-utterance manifests in `data/manifests/` |
| 3 | `notebooks/02_finetune_whisper.ipynb` with `SMOKE_TEST = False` | `models/whisper-small-ne/` checkpoint (2000 steps, Drive-persisted every 500) |
| 4 | `notebooks/03_evaluate.ipynb` | Real WER/CER in `reports/wer_cer_results.{csv,md}` |
| 5 | `notebooks/04_error_analysis.ipynb` | Real error samples/categories in `reports/` |
| 6 | `streamlit run dashboard/app.py` locally | Point sidebar checkpoint at synced `models/whisper-small-ne/` |

**Success gate for T-009:** training completed ≥1 full eval cycle at 2000 steps (or best checkpoint documented), in-domain SLR54 test WER/CER recorded, out-of-domain FLEURS scored, error-analysis artifacts regenerated from the real checkpoint (not smoke). Return to Duo with **"check"** and the WER/CER numbers.
