# Review Notes — NSTT

## Review — T-001 Project scaffolding & Colab-ready environment setup — 2026-07-03 (state v3)
Outcome: **Approved**

### Acceptance criteria
- [x] Repository folders created (`data/`, `notebooks/`, `src/`, `models/`, `reports/`) with `data/` and `models/` gitignored — `.gitkeep` pattern preserves empty dirs in git
- [x] `requirements.txt` with pinned `transformers`, `datasets`, `torchaudio`, `librosa`, `evaluate`, `jiwer`, `accelerate` — all present; `soundfile` addition is reasonable
- [x] Colab-ready `notebooks/00_setup.ipynb` — Drive mount, GPU/T4 check, `pip install`, import smoke test
- [x] `.gitignore` covers `data/`, `models/`, checkpoints, `.ipynb_checkpoints`, `__pycache__`
- [x] `README.md` documents Colab open/run workflow

### Checklist
- Architecture respected: **yes** — folder layout matches `project-context.md`; `src/` left as stub per scope
- Code quality: **good** — notebook is clear, requirements commented, no scope creep
- Testing: structural + pip + import smoke on Py 3.12 **passed**; Colab T4 E2E **not run** by Cursor (environment limitation) — acceptable for scaffold approval; human should run `00_setup.ipynb` once before T-003 training
- Security / performance: **n/a** for scaffold; no secrets committed
- Documentation updated: **yes** — README complete

### Deviations accepted
- `torchaudio==2.9.1` (older pin unavailable on PyPI) — documented in `requirements.txt` and README
- `torch` unpinned — correct for Colab CUDA builds
- `soundfile` added — standard librosa dependency

### Required revisions
None.

---

## Review — T-002 Data acquisition & preprocessing pipeline for OpenSLR SLR54 — 2026-07-03 (state v5)
Outcome: **Approved**

### Acceptance criteria
- [x] Script/notebook downloads or ingests OpenSLR SLR54 — `src/slr54.py` + `notebooks/01_data_prep.ipynb`; TSV ingest verified (157,905 utterances, 527 speakers)
- [x] Audio resampled to 16kHz mono; transcripts NFC-normalized — `src/preprocessing.py`, verified by code read + spot-check
- [x] Utterances filtered to 0.5s < duration < 30s — strict exclusive bounds, verified in `duration_in_bounds()`
- [x] Speaker-disjoint 80/10/10 split — `src/splits.py` partitions by speaker set (not utterance), disjointness asserted in code, `count_speaker_overlap()` = 0
- [x] Manifest files committed under `data/manifests/` (metadata only) — `.jsonl`, schema matches spec, raw audio correctly excluded via `.gitignore`

### Checklist
- Architecture respected: **yes** — matches Data Prep stage in `project-context.md`
- Code quality: **good** — clean separation (`slr54.py`/`preprocessing.py`/`splits.py`/`manifests.py`/`pipeline.py`), reusable per coding standards
- Testing: independently re-ran `pytest tests/test_data_prep.py` myself — **7/7 passed**. Full 8GB SLR54 zip preprocessing not run locally (deferred to Colab, consistent with T-001 environment constraints) — acceptable, clearly disclosed rather than silently substituted
- Security / performance: n/a
- Documentation updated: implementation log accurately reflects what was and wasn't run

### Deviations accepted
- Synthetic dataset used to validate pipeline/schema/split logic locally instead of the full SLR54 corpus (8GB download impractical outside Colab) — transparently disclosed in both the implementation log and code comments, not misrepresented as full-corpus results.
- `.gitignore` nested-exception pattern (`!data/manifests/` before `!data/manifests/**`) correctly applied per feedback from T-001 review.

### Required revisions
None.

---

## Review — T-003 Whisper-Small fine-tuning script/notebook on Colab T4 — 2026-07-03 (state v7)
Outcome: **Changes Requested**

### Acceptance criteria
- [x] Notebook loads `openai/whisper-small` and `WhisperProcessor` — `src/training.py::load_whisper_model_and_processor`
- [x] `Seq2SeqTrainingArguments` for T4 — batch 2, grad_accum 4, lr 1e-4, warmup 500, max_steps 2000, eval_steps 200, verified in `build_training_arguments`
- [x] Checkpoints saved at `save_steps`, resumable — `resume_from_checkpoint` param wired through; local checkpoint-3 verified reloadable
- [x] At least one eval cycle with logged loss/eval metrics — independently confirmed `checkpoint-3/trainer_state.json` shows `eval_loss=0.8063`, `eval_wer=0.667` at step 3, matching the reported packet exactly

### Checklist
- Architecture respected: **no** — see required revision below
- Code quality: **good** — `src/training.py` is clean, attribution comment present for the HF tutorial per academic-integrity requirement, data collator and compute_metrics correctly implemented
- Testing: independently re-ran `pytest tests/test_training.py` — **6/6 passed** (Cursor's own environment reported 5/5, likely one `@pytest.mark.slow` test excluded there; not a discrepancy in the code itself)
- Security / performance: n/a
- Documentation updated: notebook markdown cells are clear

### Required revisions
- `whisper_small.ipynb` at the repo root duplicates `notebooks/02_finetune_whisper.ipynb` almost cell-for-cell (same Drive-mount/GPU-check/training flow; root version adds one extra inference-demo cell). `project-context.md`'s documented folder structure places notebooks under `notebooks/` only. Consolidate to one notebook under `notebooks/` — fold the inference-demo cell in if it's worth keeping, otherwise drop it — and delete the root-level duplicate. This is the only blocker to approval; everything else is solid.

---

## Review — T-003 revision (notebook consolidation) — 2026-07-03 (state v9)
Outcome: **Approved**

Verified: root `whisper_small.ipynb` deleted; `notebooks/02_finetune_whisper.ipynb` retains the inference-demo cell (folded in, not silently dropped) plus the pip-install cell. Re-ran `pytest tests/test_training.py` independently — 6/6 passed. All 4 acceptance criteria for T-003 are met. No further revisions required — T-003 is Completed.

---

## Review — T-004 Evaluation pipeline: WER/CER on in-domain and out-of-domain sets — 2026-07-03 (state v11)
Outcome: **Approved**

### Acceptance criteria
- [x] WER/CER computed via `jiwer` on SLR54 test set — `src/evaluation.py::evaluate_manifest` + `compute_wer_cer`
- [x] Out-of-domain evaluation — FLEURS `ne_np` via `load_fleurs_test`/`evaluate_hf_dataset`
- [x] NFC normalization applied identically to both sides — `normalize_pair_for_scoring` reuses `src/preprocessing.py::normalize_transcript_nfc` (not reimplemented), applied to both ref and hyp before scoring in `compute_wer_cer`
- [x] Results exported as a table under `reports/` — `reports/wer_cer_results.csv` and `.md`, content verified to match reported test_results exactly

### Checklist
- Architecture respected: **yes** — reuses T-002's normalization and T-003's checkpoint loading rather than duplicating logic
- Code quality: **good** — clean separation of manifest-based vs HF-dataset-based evaluation paths, graceful processor fallback for the smoke checkpoint's incomplete tokenizer files
- Testing: independently re-ran `pytest tests/test_evaluation.py` — **5/5 passed**
- Documentation: markdown report explicitly flags that the 3-step smoke checkpoint's numbers are pipeline sanity checks, not real results — correct judgment call, avoids the numbers being mistaken for genuine findings later

### Required revisions
None.

---

## Review — T-005 Error analysis and report-artifact generation — 2026-07-03 (state v13)
Outcome: **Approved**

### Acceptance criteria
- [x] Qualitative reference-vs-hypothesis samples exported under `reports/` — `reports/error_samples.csv`/`.md`, content verified to match reported test results
- [x] Error categorization per `start.md` §3.3 checklist — `src/error_analysis.py::categorize_errors` tags phonetic_confusion, oov_rare_vocabulary, noise_degradation, code_switching, dialect_accent, other
- [x] Plots/tables for the report — `reports/error_categories.{csv,md,png}`

### Checklist
- Architecture respected: **yes** — reused `src/evaluation.py`'s transcription plumbing (refactored to expose `UtterancePrediction`/`collect_manifest_predictions`) instead of duplicating it
- Code quality: **good**, one non-blocking nit — `categorize_errors()`'s final `if not categories:` branch has a redundant if/else both appending `"other"`; harmless, not worth a revision cycle
- Testing: independently re-ran the **entire** test suite after this refactor (`test_data_prep.py` + `test_training.py` + `test_evaluation.py` + `test_error_analysis.py`) — **18/18 passed**, confirming the `evaluation.py` refactor didn't regress T-003/T-004
- Documentation: reports correctly caveated as smoke-checkpoint-only, consistent with T-004's framing

### Required revisions
None.

**Backlog status:** T-001 through T-005 are all Completed. Remaining work (full Colab GPU training run against the complete SLR54 corpus) is a human-executed step, not further Cursor implementation, unless new backlog items are added.

---

## Review — T-006 Full-pipeline QA pass — 2026-07-03 (state v16)
Outcome: **Approved**

### Acceptance criteria
- [x] Full test suite re-run, count reported — independently re-ran `MPLBACKEND=Agg pytest tests/`, got **23/23 passed**, matching the packet exactly
- [x] All 5 notebooks validated (JSON + import sanity) — confirmed present and structurally intact
- [x] `requirements.txt` fresh-install check — reported passed, consistent with earlier per-task installs
- [x] Docs cross-checked for stale references — confirmed no lingering `whisper_small.ipynb` mentions in `implementation-log.md`; history entries were appended to, not rewritten
- [x] Repo hygiene check — confirmed `__pycache__`, `.DS_Store`, `data/*`, `models/*` are all covered by `.gitignore` (on-disk cruft from local dev is untracked, as expected)
- [x] `reports/qa_summary.md` produced — read in full, checklist-style, honest about what wasn't run (Colab GPU E2E) rather than overclaiming

### Checklist
- Scope discipline: **good** — only trivial, clearly in-scope fixes applied (README matplotlib mention, `.pytest_cache/` gitignore entry, Agg backend for headless test stability); no architecture changes, no blockers needed
- Testing: independently verified rather than trusting the report
- Documentation: `qa_summary.md` is a genuinely useful artifact, not a rubber-stamp

### Required revisions
None.

**Overall backlog status:** T-001 through T-006 are all Completed. The pipeline (scaffold → data prep → fine-tuning → evaluation → error analysis → QA) is fully implemented and self-consistent at the smoke-test scale. The only remaining work is a human-executed full Colab GPU training run against the complete SLR54 corpus.

---

## Review — T-007 Streamlit visualization dashboard — 2026-07-03 (state v19)
Outcome: **Approved**

### Acceptance criteria
- [x] `dashboard/app.py` launches via `streamlit run` — **live-verified**: launched `streamlit run dashboard/app.py --server.headless true --server.port 8502` myself, `curl` returned HTTP 200, then cleaned up the process. This goes further than the packet's own claim ("subprocess headless start attempted").
- [x] WER/CER table from `reports/wer_cer_results.csv` — Metrics tab, `_format_wer_cer_table`
- [x] Error samples filterable by category — Error Samples tab, `filter_error_samples_by_category`/`parse_category_list`
- [x] Error category breakdown (reuses PNG or regenerates) — Categories tab, checks `error_categories_png.exists()` first
- [x] Training curves from `trainer_state.json` — Training Curves tab, `load_training_curves` splits train/eval rows correctly
- [x] Smoke-checkpoint labeling — prominent `st.warning` banner at the top of every page load
- [x] README/requirements updated — `streamlit==1.41.1` pinned, new "Running the dashboard locally" section
- [x] Smoke test for data-loading — independently re-ran `pytest tests/test_dashboard_data.py`, **7/7 passed**

### Checklist
- Architecture respected: **yes** — `dashboard/data.py` has zero Streamlit imports, purely pandas/json, matching the requested testable-in-isolation design
- Code quality: **good** — column-presence validation in every loader, sensible fallback when eval_df is empty, sidebar override for checkpoint path is a nice touch beyond the minimum ask
- Testing: independently re-ran unit tests AND did a live server launch (not just static/AST checks)
- Documentation: README section is clear and accurate

### Required revisions
None.

**Overall backlog status:** T-001 through T-007 are all Completed.

---

## Review — T-008 Live microphone voice-to-text demo tab — 2026-07-03 (state v22)
Outcome: **Approved**

### Acceptance criteria
- [x] "Live Demo" tab using `st.audio_input` — no new heavy dependency (ADR-003)
- [x] Audio conversion reuses existing logic — `dashboard/inference.py::wav_bytes_to_mono_array` delegates resampling to `src/evaluation.py::transcribe_array`, not duplicated
- [x] Inference via currently-loaded checkpoint — uses the shared sidebar `checkpoint_dir`, not hardcoded to the smoke checkpoint
- [x] Smoke-checkpoint caveat present — caption above the recorder states expected poor quality
- [x] Graceful no-audio/error handling — `None` branch shows a plain prompt; broad `try/except` around transcription shows `st.error` instead of crashing
- [x] Smoke test for transcription call path — independently re-ran `pytest tests/test_dashboard_inference.py tests/test_dashboard_data.py`, **11/11 passed**
- [x] README updated — mic-permission note present in the dashboard section

### Checklist
- Architecture respected: **yes** — `st.cache_resource` model-loading kept in `app.py`, not leaked into the Streamlit-free `dashboard/inference.py`
- Code quality: **good** — clean reuse of `src/evaluation.py`'s existing transcription plumbing
- Testing: independently re-ran unit tests AND did a live server launch — `streamlit run dashboard/app.py --server.headless true` on port 8503 with the new tab present, `curl` returned HTTP 200, process cleaned up afterward
- Documentation: README section accurate

### Required revisions
None.

**Overall backlog status:** T-001 through T-008 are all Completed.

---

## Sprint 1 Closure — 2026-07-06 (state v23)
Outcome: **Sprint complete — human Colab execution phase begins**

### Summary
All implementation backlog (T-001..T-008) is Approved and Completed. The codebase is pipeline-complete on synthetic/smoke data. Real metrics require a human-executed Colab GPU run (documented in `project-context.md` Colab Runbook).

### Next Duo task
**T-009** — Ingest full Colab training results & finalize report artifacts. Status: **Planned** (not Ready for Implementation). Activates when the human completes Colab steps 0–6 and returns with **"check"** plus WER/CER numbers.

### Cursor standing order
**Do not implement** until T-009 is marked Ready for Implementation. No blockers open.
