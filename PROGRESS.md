# Switchback progress

Updated: September 11, 2026.

## Current state

Build specification and Claude Code instructions prepared. A standard-library report renderer is included as starter tooling. The inference engine, model adapter, controller, tests, benchmark runner, and viewer have not been implemented. No models have been downloaded and no DGX Spark benchmark has run for this project.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M1 Hardware and baseline | Not started | None |
| M2 Sampling oracle | Not started | None |
| M3 Cached target-only engine | Not started | None |
| M4 Fixed greedy speculation | Not started | None |
| M5 Sampled speculation and traces | Not started | None |
| M6 Cost controller | Not started | None |
| M7 Benchmark and report | Not started | Renderer starter file only |
| M8 Reviewer demo | Not started | None |

## Next action

Read SPEC.md and CLAUDE.md. Inspect the local environment, create the package and CPU fixtures, implement the doctor, and begin M1. Resolve actual dependency/model revisions on the execution machine. Do not infer hardware availability from the spec.

## Starter-tool verification

Executed `python -m unittest discover -s tests/report -v` against the supplied renderer. All 10 integrity tests passed, using explicitly synthetic timing fixtures in temporary directories. These checks cover successful fixture rendering, reproducibility, hash tampering, missing/duplicate requests, output mismatch, failed validation, invalid timing, and refusal to name a fixture report RESULTS.md. They are not inference tests or performance measurements. There is no source commit yet for this design bundle.

## Required milestone entry format

### Mx: title

- Status:
- Implementation and decisions:
- Commands actually run:
- Passed / failed / skipped checks:
- Benchmark or evidence paths:
- Source commit:
- Known limitations and blockers:
- Next concrete step:
- Teach-back explanation prepared:
- Ethan's teach-back status: not yet demonstrated / demonstrated by Ethan
