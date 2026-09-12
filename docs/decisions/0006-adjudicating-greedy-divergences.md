# ADR 0006: a divergence gap is a property of a path, not of a position

Date: 2026-09-12. Status: accepted. Milestone: M7. Refines: [ADR 0005](0005-renderer-near-tie-contract.md).

## Context

ADR 0005 admits a greedy mismatch when the gap between the two tokens the
engines chose is within a declared bound. `bench.run` measures that gap by
recomputing the row **uncached** from the baseline's prefix.

The primary run falsified the method. Of 3,072 requests, 857 diverged from the
baseline (27.9%). The gap distribution:

| Gap | Count |
|---|---:|
| exactly 0.0 | 383 |
| ≤ 0.25 | 375 |
| ≤ 0.5 | 81 |
| ≤ 1.0 | 3 |
| > 1.0 | 15 |

839 of 857 sat inside the 0.5 bound and the renderer admitted them. The
remaining 18 were refused, and they were not randomly scattered: they were two
positions, hit deterministically across engines and repeats.

Investigating both showed the screen was measuring the wrong quantity.

**`mbpp_heldout_169`, index 46, screen gap 1.5.** Recomputing that position:

| Path | token 220 | token 264 | prefers |
|---|---:|---:|---|
| uncached full prefix | 44.500 | 43.000 | 220 |
| cached, width-1 step | 44.250 | 43.750 | 220 |
| cached, width-2/3/5/9 verification | 44.250 | 43.750 | 220 |
| **the engine's own replayed cache** | **43.750** | **44.000** | **264** |

The engine chose correctly given its own logits, by a margin of 0.25. Its KV
cache had been built through 46 blocks of varying widths and crops, so the
cached keys and values themselves differ in their last bits from a single-shot
prefill. The screen was comparing against a state the engine was never in.

**`gsm8k_heldout_251`, index 72, screen gap 0.75.** Here *every* reproducible
path prefers the token `hf_dynamic` chose. The **baseline** is the outlier at
that position, not the candidate.

So the gap is not a property of a position. It is a property of a position and
an execution path, and the baseline is one path among several rather than ground
truth.

## Decision

Keep the cheap screen inside `bench.run`, because it is nearly free and it
explains 98% of divergences. Add `bench/adjudicate.py` as a second tier, run
after the benchmark, for the ones it cannot.

The adjudicator recomputes each flagged position under every path it can
reproduce -- uncached, freshly cached at widths 1, 2, 3, 5 and 9, a cache grown
in chunks, and for Switchback engines a deterministic replay of the engine
itself -- and records the smallest margin any of them produced.

A divergence is `near_tie` when some reproducible path puts the two tokens
within the bound, or when the paths do not unanimously agree with the baseline.
Anything else is `unexplained`.

The renderer refuses an over-bound divergence unless an adjudication verdict
covers exactly that row, index and token pair, says `near_tie`, **and** carries
a `smallest_margin` that is itself within the bound. Adjudication can only ever
explain a divergence; there is no verdict that suppresses one.

`adjudication.jsonl` is hashed into the run manifest like any other raw file,
and the manifest records that it was added after the run and by which commit.
It changes no timing, token or counter the benchmark measured.

## Result on the primary run

All 18 adjudicated `near_tie`, 0 unexplained. Every Switchback replay showed the
engine preferring the candidate in its own state, by margins of 0.25 to 1.0, and
the smallest margin across paths was 0.25 in every case.

## Why not just raise the bound

Raising it to 2.0 would have admitted both cases and taken thirty seconds. It
would also have admitted a genuine cache fault at any position where the model
is confident by less than 2 logits, which is most of them. ADR 0005 said the
bound is a claim about floating point rather than a tolerance for wrong answers,
and moving it because a run failed is exactly the move that sentence exists to
prevent.

The bound stayed at 0.5. The measurement got better.

## What this costs

The adjudicator replays an engine per flagged row, which is far too slow to run
on all 3,072. That is why it is a second tier: the screen is the filter, and only
its failures pay. On this run that was 18 rows out of 3,072.

## Evidence

`artifacts/runs/primary/adjudication.jsonl`; ten tests in
`tests/report/test_conformance_contract.py` covering an admitted near-tie, an
`unexplained` verdict, a verdict that contradicts its own margin, adjudications
aimed at the wrong position or engine, a tampered file, a missing file, an
unknown verdict, and the rule that adjudication cannot rescue a row carrying no
divergence evidence of its own.
