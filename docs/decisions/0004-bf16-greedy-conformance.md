# ADR 0004: BF16 greedy conformance is a measured rate, not an assumption

Date: 2026-09-12. Status: accepted. Milestone: M6. Affects: M7.

## Context

Greedy speculative decoding is exact in real arithmetic: verifying a draft
proposal against the target's argmax either accepts the target's own choice or
replaces it with the target's own choice. Milestones 3 and 4 confirmed this on
the real model, with **identical 32-token ID arrays** between `native_ar`,
`hf_ar`, and `fixed_{1,2,4,8}`.

At longer outputs that stops being true. `python -m switchback conformance`
measured the four pilot prompts at draft lengths 1, 2, 4 and 8:

| Output length | Comparisons | Identical | Agreement |
|---:|---:|---:|---:|
| 32 | 16 | 16 | 100% |
| 64 | 16 | 12 | 75% |
| 128 | 16 | 12 | 75% |
| 256 | 16 | 12 | 75% |

Every divergence is a near-tie. Recomputing the uncached logits at the position
where the engines disagreed, with the same EOS mask the engines used:

| Case | Target-only chose | Speculation chose | Logit gap | Logit scale |
|---|---|---|---:|---:|
| prompt 1, token 54 | 13136 @ 39.250 | 3070 @ 39.000 | 0.250 | 39.2 |
| prompt 0, token 56 | 73594 @ 18.125 | 151668 @ 18.250 | 0.125 | 18.2 |

The largest gap between two chosen tokens across every divergence is **0.25**.
Milestone 3 measured BF16 reduction-order noise at up to **1.13** absolute. The
divergences are comfortably inside the noise.

The mechanism is the one in `docs/correctness.md` section 4a: a target-only step
computes its logits in a width-1 forward, a verification step computes the same
logits inside a width-`g+1` forward, and BF16 with 8 mantissa bits over 36
layers does not make those bitwise equal.

## Decision

Treat greedy conformance as a **measured quantity reported alongside any
latency result**, not as a precondition that is assumed to hold.

Concretely:

1. `switchback conformance` is a first-class command and its output is an
   artifact, the same as a profile or a trace.
2. A divergence is reported with the logits of the two tokens the engines
   actually chose. A near-tie is floating point; a divergence at a confident
   position is a bug, and the two must never be conflated.
3. Exact equality remains a hard gate where it is achievable and meaningful:
   FP32 on the CPU fixture (exact, `tests/integration/test_cached_engine_cpu.py`)
   and BF16 at 32 tokens on the real pair
   (`tests/integration/test_speculation_gpu.py`).
4. The BF16 long-output configuration is **explicitly nonconformant** and must
   be labelled that way wherever its latency is reported.

## A measurement bug found on the way

The first version of this measurement reported a divergence with a top-2 margin
of 15.5, which would have meant a real bug rather than floating point. It did
not apply the EOS mask the engines use, so it was reporting the gap between a
suppressed stop token and the rest of the vocabulary. With the mask applied the
same case has a gap of 0.125.

The lesson is recorded because it nearly sent this milestone chasing a decoder
bug that did not exist: a diagnostic that does not reproduce the engine's own
configuration measures something else.

## Consequence for milestone 7, unresolved

`scripts/render_results.py` **refuses to render a report** when greedy outputs
differ between engines in the same cell:

```python
require(row["output_ids"] == reference,
        f"Greedy output mismatch: {cell}, {engine}; investigate before publishing speedup")
```

At the primary condition -- 256 fixed-length greedy tokens -- that check will
refuse roughly a quarter of cells on current evidence. The renderer is correct
to be strict; the question is what the contract should be now that exact
equality is known to be unachievable in this configuration.

Three options, to be decided at the start of M7:

- **A.** Report per-cell conformance as a measured rate and let the renderer
  proceed when every divergence's chosen-token gap is below a recorded bound.
  Keeps the strict check for anything that is not a near-tie. Requires changing
  the renderer's contract, so it needs its own ADR and its own tests.
- **B.** Compare only the matching prefix and report the median matched prefix
  length. Simple, but it quietly redefines the conformance claim.
- **C.** Run the conformance comparison in FP32 and the latency comparison in
  BF16, disclosing that they are different configurations.

Option A is the recommendation: it keeps the failure visible and quantified
rather than defining it away. None of the three is implemented yet.

## Alternatives considered and rejected

Widening a tolerance until the comparison passes. Rejected: the tokens are
different integers, and there is no tolerance on integer equality that means
anything.

Switching the primary benchmark to FP32. Rejected for the primary result: it
would double memory traffic on a bandwidth-bound machine and measure a
configuration nobody would deploy. It stays available as option C.

Declaring greedy speculation "exact" and not measuring it. Rejected: it is
exact in arithmetic and not in BF16, and the difference is the finding.

## Evidence

`artifacts/conformance.json`, `artifacts/controller_check.json`,
`docs/correctness.md` section 4a.
