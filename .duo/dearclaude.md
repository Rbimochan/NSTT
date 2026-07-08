# ITERATION 9 — 2026-07-03T22:35:00+05:45

# EXECUTION REPORT (T-008)

## Status
Completed

## Summary
Added **Live Demo** tab to the Streamlit dashboard: browser mic recording via `st.audio_input`, transcription through the sidebar-selected checkpoint using `src/evaluation.py::load_checkpoint` / `transcribe_array`. Pure helpers in `dashboard/inference.py` for WAV-bytes decoding. Model cached with `st.cache_resource`. Graceful handling when no recording yet or transcription fails.

## Files Changed
- `dashboard/inference.py` (new)
- `dashboard/app.py` (Live Demo tab + cached loader)
- `tests/test_dashboard_inference.py` (new)
- `README.md` (Live Demo + mic permission)
- `.duo/project-state.json` (v20 → v21)

## Validation
- `pytest tests/test_dashboard_inference.py`: **4/4 passed**
- `pytest tests/test_dashboard_data.py`: **7/7 passed** (regression)
- No-audio state: `if audio_input is None` shows prompt — verified by code inspection
- Transcription smoke: synthetic WAV → string via smoke checkpoint

## Acceptance Criteria
- [x] Live Demo tab with `st.audio_input` (no streamlit-webrtc)
- [x] 16kHz mono via `transcribe_array` resampling path
- [x] Inference via sidebar checkpoint + `load_checkpoint`/`transcribe_array`
- [x] Checkpoint label + smoke-quality caveat
- [x] Graceful None / error states (`st.write` prompt, `st.error` on exception)
- [x] Smoke tests for transcription call path
- [x] README updated

## Issues
None (blockers).

## Ready for Review
Yes
