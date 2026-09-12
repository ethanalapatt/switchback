# Claude Code instructions for Switchback

Read `SPEC.md`, `PROGRESS.md`, and the latest relevant decision record before modifying code. Implement the next incomplete milestone in order. Treat this as a local GPU inference and correctness project, with the bounded scope in the spec.

## Operating rules

- Implement actual decoding in `src/switchback/`; use HF `.generate()` only for named baseline adapters. A wrapper around assisted generation does not satisfy this project.
- Prefer confirmed Python/PyTorch skills. Do not add new languages, services, orchestration, training, or a web application without a concrete requirement in the spec.
- Use reasonable implementation judgment without routine permission questions. Keep existing user work intact. Inspect git status before edits; do not reset or rewrite unrelated work.
- Do not claim access to the DGX Spark from the laptop. Check the execution environment. GPU steps stay blocked until they actually run on suitable hardware.
- Download only the specified public models/data and locked software dependencies required by the build. No paid APIs or cloud jobs.
- Do not spawn sub-agents unless Ethan explicitly requests delegation.

## Code conventions

- Python package under `src/`; type public interfaces, use dataclasses for state, and name every tensor axis in docstrings.
- Keep numerical sampling pure and independent of controller, device telemetry, and benchmark labels.
- One model adapter owns all Transformers/cache API assumptions. Pin a working release and document migration decisions.
- One pending-token cache convention everywhere. Assert invariants in correctness mode; do not carry expensive debug checks into one engine's timing but omit them from another without disclosure.
- Explicit dtype, device, seed streams, EOS policy, attention backend, and generation parameters. Never inherit hidden model generation defaults.
- Log structured data with units in field names. Preserve raw evidence. Use null when a counter is unavailable.
- Prefer simple functions and small modules. Avoid speculative abstractions for unimplemented model families or multi-GPU execution.
- Document why before optimizing. Show profiler evidence for performance changes and retain an understandable reference path.

## Setup and commands

Create and verify these commands in M1; they are contracts to implement, not evidence they already work:

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not gpu and not download'
python -m pytest tests/property
python -m pytest tests/report
python -m ruff check .
python -m ruff format --check .
python -m mypy src/switchback
python -m switchback doctor
python -m pytest -m gpu
bash scripts/reproduce.sh --preset smoke
```

Respect the working Spark torch installation. Produce separate verified CPU and Spark dependency locks; do not accidentally replace a compatible NVIDIA build with an incompatible generic wheel. Validate the install instructions in the chosen environment before publishing them.

Run unit/property tests for sampler/cache edits; run real-model differential tests for model-interface edits; run report-integrity tests for evidence/report changes. CPU CI cannot satisfy a GPU gate. A skipped test is a skip, not a pass. Test meaningful behavior, not arbitrary test counts.

## Non-negotiable metrics rule

**Never fabricate, estimate, manually improve, or copy performance statistics into README, RESULTS.md, a GIF, or resume bullets.**

- No placeholder speedups, made-up throughput, invented pass rates, copied paper numbers, or extrapolation from a smaller model.
- `RESULTS.md` is generated solely by `scripts/render_results.py` from a validated measured run. Never edit its numbers by hand.
- Synthetic timing fixtures may exist only in clearly marked tests. The renderer watermarks fixture reports and refuses to write them as `RESULTS.md`.
- All headline results require named hardware, baseline, immutable source/software/configuration, measured cohort, complete request manifest, and raw evidence.
- Keep losses, errors, OOMs, and inconclusive confidence intervals. Do not tune on held-out outputs or remove slow prompts.
- Report model/runtime setup separately; include all per-request target/draft work and synchronization in end-to-end timing.
- Same-seed sampled strings are not required to match different samplers. Exact oracle identity is not a claim of universal floating-point equivalence.
- `[METRIC]` slots may remain only in explicitly labeled resume templates, never in published achievement claims.
- If no benchmark ran, state `Not measured`. If a run was interrupted, state `Incomplete`. If the code is slower, state that result.

## Commit and progress discipline

- Make small local commits after a cohesive change passes its relevant checks. Use messages such as `feat: implement residual sampler` or `test: cover cache rejection boundaries`.
- Do not make one giant milestone commit if the work can be reviewed in smaller pieces. Avoid mixed formatting/refactor/behavior commits.
- Update `PROGRESS.md` after **every milestone** and before stopping mid-milestone. Include implemented behavior, commands actually run, results and skips, artifact paths, source commit, unresolved issues, and the exact next step.
- Distinguish `not started`, `in progress`, `blocked`, and `complete`. Never mark a milestone complete merely because files exist.
- End each milestone with a short teach-back note: decision, alternative considered, failure mode, and the test/evidence that supports it. Do not label Ethan's understanding as verified unless he actually demonstrates it.
- Local commits are authorized by this build instruction. If Ethan separately authorizes GitHub pushes and a verified remote exists, push after milestones; otherwise do not create or publish a remote automatically.

## Completion gate

The project is complete only when the declared scope passes its correctness gates, benchmark results are generated from actual runs, reproduction commands are tested, the README accurately describes limitations, and progress is current. If a fallback is used, name the reduced scope prominently and keep deferred work unchecked.
