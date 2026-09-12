# Failure analysis

What went wrong, what it cost, and what it would take to fix. Written after the
primary benchmark, with the numbers in hand.

## 1. The controller lost, and the reasons are not equal

`adaptive` reached 2.021× against `hf_ar`. `fixed_8` reached 2.066× and
`hf_dynamic` 2.234×. The ordering held on both datasets separately.

Two causes, and conflating them would be the most flattering mistake available:

**The cohort cannot reward adaptivity.** Conditional acceptance across the
primary cohort runs 0.91 at candidate position 1 down to 0.66 at position 8 —
high and flat. When the best draft length is 8 for nearly every prompt, a
controller that sometimes picks 4 can only lose, and one that always picks 8 is
indistinguishable from `fixed_8` at best. This is a property of the workload, and
saying so is not a defence of the controller. It is an argument that **this
experiment could not have shown the controller winning**, which is a flaw in the
experiment design that should have been visible before eight hours of GPU time.
The calibration data said acceptance was 0.87–0.92 at every position; that was
the signal, and it was already on screen in milestone 6.

**The controller is missing information it is allowed to use.** `hf_dynamic`
stops drafting *within* a block based on the draft model's own confidence.
Switchback picks `g` before the block from past acceptance only. The reason was a
correctness one — reading the current block's outcomes to change its length
would condition on the future and break the argument in
[correctness.md](correctness.md) §3 — but that reason does not cover draft
confidence, which is available *before* verification and reveals nothing about
the target. This is a real gap, not a constraint, and it is the single change
most likely to close the distance to `hf_dynamic`.

**What would actually test the controller:** a cohort where the best fixed length
varies per prompt. The context-stress cohort (128 to 3,072 token prompts) and the
natural-stop cohort are the candidates, because acceptance and the value of draft
startup both move with prompt length and completion length.

## 2. The bypass path has never run on real data

Zero bypasses across 384 adaptive requests. That is the correct decision on a
cohort this favourable to drafting, and it means the sticky-bypass logic — one of
the four things SPEC.md calls original about this project — is supported only by
fake cost tables in `tests/unit/test_controller.py`.

Natural stopping is where it should earn its place: a request that ends after
twelve tokens never recovers the draft prefill. Until that runs, bypass is
implemented and tested, not demonstrated.

## 3. Three measurement bugs, all found by tooling rather than by tests

Each of these produced a plausible wrong number that no unit test would have
caught, because each was in the measurement rather than in the engine.

**The trace replay checker found a cache bug.** On EOS the target cache was
cropped to `len(S) + accepted`, one position too long whenever a committed EOS
truncated the block. The engine's own assertions did not fire because the caches
are retired at termination and nothing read them again. It surfaced only when a
saved trace was replayed and its cache lengths did not add up.

**The conformance measurement reported a 15.5-logit "blunder" that was 0.125.**
It recomputed the divergent position without the EOS mask the engines use, so it
was reporting the gap between a suppressed stop token and the rest of the
vocabulary. A diagnostic that does not reproduce the engine's configuration
measures something else.

**The divergence screen compared against a state no engine was in.** This one
cost the most. The primary run was refused for 18 divergences above the near-tie
bound; investigating showed the screen recomputes the position from a fresh
prefill, while the engines were in incrementally-built caches whose keys and
values differ in their last bits. At `mbpp_heldout_169` index 46 every fresh path
prefers one token by 0.5–1.5 and the engine's own replayed cache prefers the
other by 0.25. See [ADR 0006](decisions/0006-adjudicating-greedy-divergences.md).

The pattern is consistent: **the engine was right and the instrument was wrong**,
three times. The tests covered the engine thoroughly and the instruments barely
at all. If there is one process lesson here, it is that a measurement deserves
the same adversarial treatment as the thing it measures.

## 4. A profiling artifact that would have biased the controller

`profile_single_forward` had no global warmup, so the first width measured also
paid one-time allocator and kernel setup. Width 1 read slower than width 2, which
inflates the target-only cost the controller compares every draft length against
— it would have biased the controller *toward* drafting, in a project whose whole
question is when not to draft.

It was caught by noticing the shape was wrong, not by a test. There is still no
test that would catch its return.

## 5. Greedy conformance is 62%, and that was not anticipated

SPEC.md treats exact greedy equality as a publishable-result gate. In BF16 at 256
tokens it is 61.7–64.1%. Every divergence is a near-tie, and the project can say
so with evidence, but the gate as originally written was unsatisfiable and had to
be replaced mid-flight ([ADR 0005](decisions/0005-renderer-near-tie-contract.md),
[0006](decisions/0006-adjudicating-greedy-divergences.md)).

The honest reading is that "speculative decoding is exact" is a statement about
arithmetic that does not survive contact with 8 mantissa bits and 36 layers, and
a project claiming exactness should measure it before designing a gate around it.

## 6. What was not measured at all

- Natural stopping, sampled decoding, context stress, and the no-bypass ablation.
- Any hardware other than this GB10, any model pair other than Qwen3-4B/0.6B, any
  attention backend other than SDPA, any dtype other than BF16.
- Batched serving. Every number here is batch size one.
- Output quality. Greedy token agreement is a conformance measurement.

## 7. If this were restarted

1. Measure acceptance on the intended cohort **before** designing the controller.
   That one number determines whether an adaptive policy can win, and it was
   available from a one-hour calibration run.
2. Write the measurement tools against known-wrong inputs from the start. Every
   instrument bug above would have been caught by a fixture with a deliberately
   damaged trace or a deliberately offset cache.
3. Measure conformance in the target dtype early, rather than assuming an
   arithmetic property holds numerically.
4. Build the confidence-based early stop alongside the cost controller, since it
   is what the strongest baseline actually does and it is compatible with the
   correctness argument.
