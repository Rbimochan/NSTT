# Implementation Log — NSTT

## v2 — 2026-07-03 — T-001 (Cursor)

**Task:** Project scaffolding & Colab-ready environment setup  
**Status:** Implemented (awaiting Claude review)

### Files created
| File | Purpose |
|------|---------|
| `.gitignore` | Ignores `data/`, `models/`, checkpoints, `.ipynb_checkpoints`, `__pycache__` |
| `requirements.txt` | Pinned HF/ASR stack (transformers, datasets, accelerate, evaluate, jiwer, librosa, torchaudio, soundfile) |
| `README.md` | Colab quick-start: open `00_setup.ipynb`, enable GPU, run all |
| `data/.gitkeep` | Placeholder so `data/` exists while contents are gitignored |
| `models/.gitkeep` | Placeholder so `models/` exists while contents are gitignored |
| `reports/.gitkeep` | Placeholder for future WER/CER artifacts |
| `src/__init__.py` | Empty package stub for later reusable modules |
| `notebooks/00_setup.ipynb` | Colab skeleton: Drive mount, GPU/T4 check, pip install, import smoke test |

### Test results
- Structural layout check: **passed**
- Notebook JSON validation: **passed**
- `pip install -r requirements.txt` on Python 3.12: **passed**
- Import smoke test (all pinned packages): **passed**
- Colab T4 end-to-end (`Runtime → Run all`): **not run** (requires human in Colab)

### Notes for review
- `torchaudio` pinned at `2.9.1` (2.5.1 no longer on PyPI as of validation date). Colab pre-installs `torch`; if `torchaudio` conflicts at runtime, README documents matching versions.
- `torch` intentionally unpinned — Colab manages CUDA builds.

---

## v4 — 2026-07-03 — T-002 (Cursor)

**Task:** Data acquisition & preprocessing pipeline for OpenSLR SLR54  
**Status:** Implemented (awaiting Claude review)

### Files created/modified
| File | Purpose |
|------|---------|
| `src/slr54.py` | SLR54 download, TSV parse, audio index |
| `src/preprocessing.py` | 16 kHz mono resample, NFC normalize, duration filter |
| `src/splits.py` | Speaker-disjoint 80/10/10 split (seed=42) |
| `src/manifests.py` | JSONL manifest read/write |
| `src/pipeline.py` | End-to-end orchestration |
| `notebooks/01_data_prep.ipynb` | Colab orchestration notebook |
| `scripts/generate_synthetic_manifests.py` | Local validation without full 8 GB download |
| `tests/test_data_prep.py` | Unit tests (7 tests) |
| `.gitignore` | Added `!data/manifests/**` exception |
| `data/manifests/*.jsonl` | Committed manifests (synthetic validation set; full SLR54 via Colab) |

### Test results
- `pytest tests/test_data_prep.py`: **7/7 passed**
- SLR54 TSV download + parse: **157,905 utterances, 527 speakers**
- Synthetic pipeline: train=32, val=4, test=4, **speaker_overlap=0**
- Spot-check (3 files): sr=16000, mono, NFC=True, duration in (0.5, 30)s
- Manifests parseable: **passed**
- Full SLR54 zip download + end-to-end preprocess: **not run** (~8 GB; run `01_data_prep.ipynb` in Colab)

### Notes for review
- Committed manifests are from synthetic validation run proving schema/split logic; production manifests overwrite when Colab processes all 16 zips.
- `data/raw/slr54/utt_spk_text.tsv` downloaded locally for ingest validation (gitignored under `data/*` except manifests).

---

## v6 — 2026-07-03 — T-003 (Cursor)

**Task:** Whisper-Small fine-tuning script/notebook on Colab T4  
**Status:** Implemented (awaiting Claude review)

### Files created/modified
| File | Purpose |
|------|---------|
| `src/training.py` | Model/processor load, data collator, compute_metrics, Seq2SeqTrainer factory |
| `notebooks/02_finetune_whisper.ipynb` | Colab fine-tuning orchestration |
| `scripts/smoke_test_training.py` | Local 6-step smoke test |
| `tests/test_training.py` | Unit tests (5 tests) |

### Test results
- `pytest tests/test_training.py`: **5/5 passed**
- Smoke test (synthetic manifests, CPU/MPS): **3/6 steps** before workspace disconnect; **eval cycle at step 3 passed** (eval_loss=0.806, eval_wer=0.667)
- Checkpoint save: **passed** — `models/whisper-small-ne-smoke/checkpoint-3/model.safetensors`
- Colab full training (max_steps=2000): **not run** — run `notebooks/02_finetune_whisper.ipynb` with `SMOKE_TEST=False`

### Notes for review
- Training args match T-003 spec: batch 2, grad_accum 4, lr 1e-4, warmup 500, max_steps 2000, eval_steps 200, save_steps 500, fp16 on GPU.
- Resume-from-checkpoint supported via `RESUME` variable in notebooks.
- HF attribution comment in `src/training.py` per start.md §7.

---

## v8 — 2026-07-03 — T-003 revision (Cursor)

**Task:** T-003 revision — consolidate duplicate training notebook  
**Status:** Re-submitted for Claude review

### Changes
- Merged inference-demo cell + `pip install` into `notebooks/02_finetune_whisper.ipynb`
- Deleted root `whisper_small.ipynb`
- Added notebook pipeline table to `README.md`

### Test results
- `pytest tests/test_training.py`: **6/6 passed**

---

## v10 — 2026-07-03 — T-004 (Cursor)

**Task:** Evaluation pipeline: WER/CER on in-domain and out-of-domain sets  
**Status:** Implemented (awaiting Claude review)

### Files created/modified
| File | Purpose |
|------|---------|
| `src/evaluation.py` | WER/CER (jiwer), NFC scoring, FLEURS OOD, manifest eval, export |
| `notebooks/03_evaluate.ipynb` | Colab orchestration |
| `scripts/run_evaluation_smoke.py` | Local smoke runner |
| `tests/test_evaluation.py` | Unit tests (5) |
| `reports/wer_cer_results.{csv,md}` | Exported results (smoke checkpoint) |
| `README.md` | Added 03_evaluate to pipeline table |

### Test results
- `pytest tests/test_evaluation.py`: **5/5 passed**
- In-domain (SLR54 test, n=2): WER=66.67%, CER=50.00%
- Out-of-domain (FLEURS ne_np, n=1): WER=100.00%, CER=96.00%
- NFC applied symmetrically: unit test passed
- Reports exported under `reports/`

### Notes for review
- Out-of-domain set: **FLEURS ne_np** (via `datasets.load_dataset`)
- Metrics from T-003 smoke checkpoint (3 steps) — documented in report header as sanity checks only

---

## v12 — 2026-07-03 — T-005 (Cursor)

**Task:** Error analysis and report-artifact generation  
**Status:** Implemented (awaiting Claude review)

### Files created/modified
| File | Purpose |
|------|---------|
| `src/error_analysis.py` | Categorization heuristics, export samples/categories/plot |
| `src/evaluation.py` | Added `UtterancePrediction` + `collect_manifest_predictions` |
| `notebooks/04_error_analysis.ipynb` | Colab orchestration |
| `scripts/run_error_analysis_smoke.py` | Local smoke runner |
| `tests/test_error_analysis.py` | Unit tests (5) |
| `reports/error_samples.{csv,md}` | Qualitative ref/hyp table |
| `reports/error_categories.{csv,md,png}` | Category breakdown + bar chart |
| `requirements.txt` | Added matplotlib for plots |

### Test results
- `pytest tests/test_error_analysis.py`: **5/5 passed**
- Smoke E2E (n=4): dialect_accent=4, oov_rare_vocabulary=4
- Plot: `reports/error_categories.png` generated

### Notes for review
- Categories are heuristic (jiwer alignment + keyword rules) per start.md §3.3
- Smoke checkpoint — degenerate categories documented in report headers

---

## v15 — 2026-07-03 — T-006 (Cursor)

**Task:** Full-pipeline QA pass (T-001..T-005)  
**Status:** Implemented (awaiting Claude review)

### Results
- `pytest tests/`: **23/23 passed**
- Notebooks `00`–`04`: valid JSON + `src/` imports OK
- Fresh venv + `requirements.txt`: all imports OK
- `reports/qa_summary.md` produced
- Trivial fixes: README matplotlib, `.pytest_cache` gitignore, matplotlib Agg backend, stale log ref

---

## v18 — 2026-07-03 — T-007 (Cursor)

**Task:** Streamlit visualization dashboard for evaluation/error-analysis reports  
**Status:** Implemented (awaiting Claude review)

### Files created/modified
| File | Purpose |
|------|---------|
| `dashboard/data.py` | Testable loaders for reports CSVs + trainer_state.json |
| `dashboard/app.py` | Streamlit UI: Metrics, Error Samples, Categories, Training Curves tabs |
| `dashboard/__init__.py` | Package marker |
| `tests/test_dashboard_data.py` | Unit tests (7) |
| `requirements.txt` | Added streamlit, pandas |
| `README.md` | Dashboard install/run section |

### Test results
- `pytest tests/test_dashboard_data.py`: **7/7 passed**
- `streamlit run dashboard/app.py --server.headless true`: starts without error (subprocess smoke)

### Notes for review
- Reads existing `reports/` artifacts only — no new eval/training logic
- Smoke-checkpoint warning banner matches T-004/T-005 convention
- Sidebar checkpoint path configurable for post-Colab runs

---

## v21 — 2026-07-03 — T-008 (Cursor)

**Task:** Live microphone voice-to-text demo tab in Streamlit dashboard  
**Status:** Implemented (awaiting Claude review)

### Files created/modified
| File | Purpose |
|------|---------|
| `dashboard/inference.py` | WAV bytes → mono array; transcribe via load_checkpoint/transcribe_array |
| `dashboard/app.py` | Live Demo tab (`st.audio_input`), cached model loader |
| `tests/test_dashboard_inference.py` | Unit + transcription smoke tests (4) |
| `README.md` | Live Demo tab + mic permission note |

### Test results
- `pytest tests/test_dashboard_inference.py`: **4/4 passed**
- `pytest tests/test_dashboard_data.py`: **7/7 passed** (regression)
- No-audio state: `audio_input is None` → prompt, no crash (code-path verified)

### Notes for review
- ADR-003: `st.audio_input` only — no streamlit-webrtc
- Reuses sidebar checkpoint path from T-007
- try/except around transcription shows `st.error` on failure

