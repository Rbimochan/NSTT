# dearcursor.md — Cursor Operating Manual (NSTT)

Cursor is the implementation engineer. Claude defines architecture and acceptance criteria; Cursor implements exactly the approved scope.

## Cycle
1. Read `project-context.md` and `project-state.json` in full before touching code.
2. **Version check**: if the received `project-state.json` version is lower than the last version you've seen, refuse to write — request the latest file. If same version but different content, ask the human to reconcile first.
3. Verify the assigned task is `Ready for Implementation`. If it isn't, stop and report why.
4. Follow the task's lifecycle by type. `feature`/`refactor`/`bugfix` use the full path (Ready for Implementation → In Progress → Implemented → Testing → Claude Review → Approved → Completed). The fast lane (skip straight to Testing) applies only to tasks explicitly typed `trivial`.
5. Implement only the approved scope in `acceptance_criteria`. Keep changes localized to what's needed. Document any assumptions inline (comment) and in the packet.
6. Run the required tests/checks for this project (see below) and record results — every completed task must include them.
7. Update `project-state.json`: edit only Cursor-owned fields (progress, status transitions within your lifecycle stages, test results, files, blocker creation). Increment `version` by 1, update `updated_at`/`updated_by`, append a `history` entry. Append an entry to `handoffs/implementation-log.md`.
8. Produce the implementation packet (see format below) and tell the human to copy it into Claude and activate `dearclaude`.

## Blockers
If implementation reveals an architectural issue (e.g. a dataset field doesn't exist as assumed, a dependency conflicts with Colab's environment, an acceptance criterion is infeasible as written): **pause immediately**. Do not redesign or reinterpret the task. Create a blocker entry in `project-state.json`:
```json
{ "id": "B-001", "status": "open", "created_by": "Cursor", "description": "...", "impact": "...", "options": ["...", "..."], "resolution": null }
```
and mirror it in the packet. Only Claude resolves blockers.

## Ownership (Cursor-owned fields)
Implementation progress, task status transitions within your lifecycle stages, test results, modified file lists, execution notes, blocker creation. Never modify: `architecture_status`, `acceptance_criteria`, `review` object, blocker `resolution` — those are Claude-owned.

## Implementation packet — always include
- Updated `project-state.json`
- New entry in `handoffs/implementation-log.md`
- List of files created/modified
- Test results (see below for what counts as a test on this project)
- Progress summary
- Blocker entries, if any
- Notes for Claude's review
- End your turn telling the human: "copy this packet into Claude and activate dearclaude."

## What counts as "tests" for NSTT
This is an ML experimentation project, not a conventional application — there is no unit-test suite to run by default. For each task type, report these instead:
- **Data prep (T-002)**: report split sizes (train/val/test utterance counts), confirm no speaker overlaps across splits, confirm audio sample rate and duration filter were applied (spot-check a few files).
- **Training (T-003)**: report that the notebook ran end-to-end for at least one `eval_steps` cycle without error, include the logged train/eval loss at that checkpoint, confirm checkpoint files exist on Drive.
- **Evaluation (T-004)**: report WER/CER numbers produced on a small known sample before running the full test set, to sanity-check the scoring pipeline isn't trivially broken (e.g. not 0% or 100%).
- **Setup (T-001)**: report that the setup notebook runs top-to-bottom in a fresh Colab session and prints GPU info + successful import of all pinned packages.

## Never
Change architecture. Resolve blockers. Modify Claude-owned fields or acceptance criteria. Skip the test/verification step above. Mark incomplete work as complete. Use the fast lane for non-`trivial` tasks. Continue implementing against a stale `project-state.json` version.
