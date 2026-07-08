# Duo Handoff — Claude → Cursor (draft, awaiting human trigger)
Task: T-009 — Ingest full Colab training results & finalize report artifacts (type: feature)

**DO NOT FIRE THIS HANDOFF YET.** Gate: human must complete Colab runbook steps 0–6
(see project-context.md "Colab Runbook") and return to Claude with "check" + real
in-domain/out-of-domain WER/CER numbers. As of 2026-07-07, only the smoke checkpoint
(`models/whisper-small-ne-smoke/checkpoint-3`, 3 steps) exists — gate not yet satisfied.

## When human returns with "check":
1. Claude bumps project-state.json: T-009 status Planned → Ready for Implementation,
   fills `progress` with the reported checkpoint path + step count, version 24 → 25.
2. Fire this packet (update the checkpoint path/numbers below from what the human reports).

## Acceptance criteria
- Real training checkpoint path documented (default: `models/whisper-small-ne/` or best
  eval checkpoint subdirectory) and synced from Drive to local `models/` if not already present
- `reports/wer_cer_results.{csv,md}` regenerated from `notebooks/03_evaluate.ipynb` against
  the full checkpoint — in-domain SLR54 test WER/CER and out-of-domain FLEURS ne_np WER/CER
  present; smoke-checkpoint caveat removed or replaced with final-results label
- `reports/error_samples.{csv,md}` and `error_categories.{csv,md,png}` regenerated from
  `notebooks/04_error_analysis.ipynb` using the full checkpoint — non-degenerate category
  distribution on test set
- Dashboard default checkpoint path in `dashboard/app.py` or README updated to the full
  checkpoint (or clearly documented one-line override); smoke-checkpoint warning banner
  updated to reflect final vs interim results
- Full pytest suite still passes (`MPLBACKEND=Agg pytest tests/`) after any code/doc changes
- A brief results summary added to `reports/training_results_summary.md` with: checkpoint
  used, train/val/test manifest sizes, final in-domain WER/CER, out-of-domain WER/CER,
  training steps completed, and whether results meet the ~15–30% in-domain WER target
  from project-context.md

## Implementation guidance
- Regeneration only — no changes to model architecture, fine-tuning hyperparameters, or
  split logic. This task ingests results; it does not re-derive the pipeline.
- Follow existing conventions: substantive logic in `src/`, notebooks stay thin orchestration.
- `dashboard/inference.py` / `src/evaluation.py::load_checkpoint` already handle checkpoint
  loading — point them at the new path rather than duplicating logic.
- Cite the human-reported WER/CER numbers directly; don't re-run training from Cursor
  (Colab GPU is out of scope for Cursor's environment).

## Attached
- `.duo/project-context.md`
- `.duo/project-state.json` (post-bump, v25)

## Human
Once Claude has bumped state to v25, paste the finalized packet into Cursor and activate dearcursor.
