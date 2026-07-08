# Colab Runbook — Full NSTT Training (Human-Executed)

> **Owner:** Human (student). Not a Cursor task.  
> **Prerequisite:** T-001..T-008 complete (codebase ready).  
> **Success gate for T-009:** Return to Duo with **"check"** and your WER/CER numbers.

## Environment
- **Runtime:** Google Colab, **T4 GPU** (`Runtime → Change runtime type`)
- **Project path:** `/content/drive/MyDrive/NSTT` (upload repo or clone into Drive)

## Sequence

### 0 — Get the project on Drive
Upload the NSTT folder to Google Drive, or clone the git repo into `/content/drive/MyDrive/NSTT` so all notebooks can resolve `src/` and write to `data/`, `models/`, `reports/`.

### 1 — Setup (`notebooks/00_setup.ipynb`)
Fresh Colab session. Mounts Drive, installs pinned dependencies, verifies GPU.

### 2 — Data prep (`notebooks/01_data_prep.ipynb`) — long download
Downloads full ~8 GB SLR54 corpus (all 16 zips). Regenerates `data/manifests/{train,val,test}.jsonl` with ~157K real utterances (replaces the 40-utterance synthetic set in git).

### 3 — Training (`notebooks/02_finetune_whisper.ipynb`)
Set **`SMOKE_TEST = False`**.

Config: batch 2, grad_accum 4, lr 1e-4, warmup 500, max_steps 2000, eval every 200 steps. Checkpoints saved to Drive every 500 steps (`models/whisper-small-ne/`). Budget real GPU time; session disconnects are survivable if Drive checkpointing works.

### 4 — Evaluation (`notebooks/03_evaluate.ipynb`)
Run against `models/whisper-small-ne/` (or best eval checkpoint). Produces real WER/CER on SLR54 test + FLEURS ne_np → `reports/wer_cer_results.{csv,md}`.

### 5 — Error analysis (`notebooks/04_error_analysis.ipynb`)
Real error categorization → `reports/error_samples.*`, `reports/error_categories.*`.

### 6 — Dashboard (local)
```bash
streamlit run dashboard/app.py
```
In the sidebar, change **Checkpoint directory** from `models/whisper-small-ne-smoke/checkpoint-3` to your synced full checkpoint (e.g. `models/whisper-small-ne/checkpoint-2000` or best eval).

## What to report back
When done, tell dearclaude **"check"** and include:
- In-domain SLR54 test **WER** and **CER**
- Out-of-domain FLEURS **WER** and **CER**
- Which checkpoint subdirectory you used
- Any Colab issues (OOM, disconnects, partial runs)

Cursor will then run **T-009** to ingest results, update caveats, and produce `reports/training_results_summary.md`.
