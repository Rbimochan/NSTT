# Architecture Decisions — NSTT

## ADR-001 — Initial architecture: Whisper-Small fine-tuning pipeline on Google Colab
**Date:** 2026-07-03
**Decision:** Build an offline batch pipeline (data prep → Whisper-Small fine-tuning → WER/CER evaluation → error analysis) targeting Google Colab's free-tier T4 GPU, using OpenSLR SLR54 as primary training data and Common Voice Nepali / FLEURS ne_np for out-of-domain evaluation.
**Alternatives considered:**
- IndicWav2Vec (CTC, 94M params) — more efficient, comparable in-domain WER, but less flexible out-of-domain and less aligned with the "Whisper fine-tuning" framing the user chose as the first task.
- Whisper-Medium — better in-domain WER (15.57%) but higher compute/memory demands than comfortably fits Colab free-tier T4; deferred as a possible follow-up task, not the MVP.
- NepConformer — state-of-the-art (6% CER) but a research-grade architecture requiring more setup than suits a first implementation task.
**Rationale:** Whisper-Small is the lowest-friction path to a working, demonstrable pipeline within Colab's free-tier constraints, matching the user's stated choice and the source document's own "recommended for first project" guidance.
**Trade-offs:** Accept a higher WER ceiling (~30%) than larger/alternative models in exchange for faster iteration and lower risk of hitting Colab GPU memory/session limits.
**Impact:** All backlog tasks (T-001..T-005) are scoped around this architecture. Swapping to a different model class later is possible but would be a new task, not a change to T-001..T-005.

## ADR-002 — Visualization layer: Streamlit over FastAPI+React / Gradio
**Date:** 2026-07-03
**Decision:** Build the reporting/visualization dashboard (WER/CER tables, error samples, category breakdown, training curves) as a single Streamlit app (`dashboard/app.py`) reading directly from `reports/*.csv` and checkpoint `trainer_state.json` files.
**Alternatives considered:**
- FastAPI + React: full separate backend API + JS frontend with build tooling and state management — appropriate for multi-user, deployed, or long-lived services, not for a single-user offline academic report tool. Would roughly 5-10x the implementation and maintenance surface for no functional gain here.
- Gradio: optimized for model-inference demos (upload input → see model output), not table/chart-heavy reporting dashboards. Could be added later as a separate "try the model" demo, but isn't the right fit as the primary reporting UI.
**Rationale:** The project's entire stack is already Python/pandas/matplotlib; Streamlit lets the dashboard reuse `src/evaluation.py` and `src/error_analysis.py` output directly with no API layer, matching the project's existing single-user, local/Colab-run nature.
**Trade-offs:** Streamlit isn't suited to a multi-user production deployment or complex custom interactivity — acceptable since this is a single-user academic project, not a product.
**Impact:** T-007 is scoped around this decision. If the project later needs multi-user access or productionization, that would be a new ADR superseding this one, not a change to T-007.

## ADR-003 — Live voice demo: st.audio_input over streamlit-webrtc
**Date:** 2026-07-03
**Decision:** Implement the live microphone voice-to-text demo (T-008) using Streamlit's native `st.audio_input` widget (available since Streamlit 1.31, and the project already pins `streamlit==1.41.1` from T-007), rather than adding `streamlit-webrtc` or a custom JS component.
**Alternatives considered:**
- `streamlit-webrtc`: more control over streaming/real-time audio, but adds a heavyweight dependency (aiortc, av) with known install friction on some platforms, for a feature that only needs "record a clip, then transcribe it" — not real-time streaming transcription.
- Custom JS component (`st.components.v1`): full control but significant extra implementation effort for no added value over the built-in widget.
**Rationale:** `st.audio_input` already returns a recorded clip as file-like bytes that can be read directly into the same 16kHz-mono conversion path used elsewhere (`src/preprocessing.py`/`src/evaluation.py`), with zero new dependencies.
**Trade-offs:** No live/streaming transcription (record-then-transcribe only), no fine-grained audio-format control — both acceptable since this is a single-user demo tab, not a production voice interface.
**Impact:** T-008 is scoped around `st.audio_input`. If a future requirement needs real-time streaming transcription, that would be a new ADR superseding this one.
