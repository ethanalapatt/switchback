# Correctness

What this project claims, in four separate levels, and what backs each one.

| Level | Claim | Backed by |
|---|---|---|
| 1. Mathematical | Under exact arithmetic, speculative sampling emits tokens from the target distribution | The proof below |
| 2. Executable finite-model | This implementation reproduces that distribution on small enumerable cases | `tests/unit/test_oracle.py`, exhaustive enumeration against an independent rational oracle |
| 3. Real-model numerical conformance | Speculative and target-only decoding return the same greedy token IDs | **Holds at 32 tokens, 100% of 16 comparisons. Fails at 64+ tokens, 75% agreement.** Every divergence is a near-tie; see section 4b |
| 4. Empirical performance | Speculation is faster on a named workload and hardware | Not measured. Milestone 7 |

The levels do not substitute for each other. Level 2 passing says nothing about
level 3 at vocabulary size 151,936 in BF16, and no amount of level 3 evidence is
a proof of level 1 for every GPU kernel.

## 1. The one-step identity

Let `p` and `q` be probability distributions over a vocabulary `V`, where `p` is
the target's next-token distribution at some prefix and `q` is the draft's
distribution at the same prefix. One speculative step is:

1. Draw a candidate `d ~ q`.
2. Accept it with probability `a(d) = min(1, p(d)/q(d))`, and emit `d`.
3. Otherwise emit a correction `x ~ r`, where

   ```
   r(x) = max(p(x) - q(x), 0) / Z,    Z = sum_y max(p(y) - q(y), 0)
   ```

**Claim.** The emitted token is distributed as `p`.

**Proof.** Split by whether the candidate was accepted.

*Accepted mass.* For each token `x`,

```
P(d = x and accepted) = q(x) * min(1, p(x)/q(x)) = min(q(x), p(x))
```

When `q(x) = 0` the token is never proposed and the product is `0`, which is
also `min(p(x), 0)`, so the identity holds on the whole vocabulary including
tokens outside the draft's support. (The implementation never forms `p/q` at
`q(x) = 0`: `accept_ratio` raises `NumericalSamplingError` on a non-positive
denominator, and such a candidate cannot be drawn in the first place.)

*Rejection mass.*

```
P(reject) = 1 - sum_y min(p(y), q(y))
```

*The normalizer equals the rejection mass.* Using `max(a - b, 0) = a - min(a, b)`,

```
Z = sum_y max(p(y) - q(y), 0)
  = sum_y (p(y) - min(p(y), q(y)))
  = 1 - sum_y min(p(y), q(y))
  = P(reject)
```

This is the step that makes the correction work: the residual is normalized by
exactly the probability of needing it.

*Total.* For each token `x`, using `min(a, b) + max(a - b, 0) = a`:

```
P(emit x) = min(p(x), q(x)) + P(reject) * r(x)
          = min(p(x), q(x)) + Z * max(p(x) - q(x), 0) / Z
          = min(p(x), q(x)) + max(p(x) - q(x), 0)
          = p(x)                                                  QED
```

*Degenerate case.* `Z = 0` if and only if `p = q` almost everywhere, and then
`P(reject) = 0`, so step 3 is unreachable. A zero-mass residual **after** an
actual rejection is therefore impossible in exact arithmetic; the implementation
raises `NumericalSamplingError` rather than substituting a distribution, and the
run is marked invalid.

## 2. Extending to a block, by conditioning

A block proposes `g` candidates `d_1 … d_g` autoregressively from the draft and
verifies them against target rows computed at the matching causal prefixes:
row `j` is `p(· | S, d_1 … d_{j-1})`.

Argue by induction on `j`. Candidate `j` is only *reached* when candidates
`1 … j-1` were all accepted, which means they were committed and the committed
prefix is exactly `S, d_1 … d_{j-1}`. Conditioned on that event, `d_j` was drawn
from `q(· | S, d_1 … d_{j-1})` and is verified against `p(· | S, d_1 … d_{j-1})`
— the same pair of distributions the one-step identity is about. So the token
emitted at that position, whether it is the accepted `d_j` or the correction,
has law `p(· | S, d_1 … d_{j-1})`.

If every candidate is accepted, the bonus token is drawn directly from the
target row at the full prefix `S, d_1 … d_g`, which is the target's law there by
definition.

Therefore every emitted token, conditioned on the prefix that precedes it, is
distributed as the target's conditional. That is exactly what autoregressive
target-only sampling produces. The two processes differ in how many forward
passes they use, not in what they emit.

**Row alignment is load bearing.** Verifying candidate `j` against the row for
`S` rather than for `S, d_1 … d_{j-1}` is the classic off-by-one, and it produces
plausible text. `test_a_prefix_conditioning_mistake_is_actually_detected` builds
that exact wrong implementation and requires the tree enumeration to reject it,
so the passing tests are not vacuous.

## 3. Choosing the block length adaptively

The controller picks `g` **before** the current block's candidates are sampled,
using only the committed prefix, calibration tables, and observations from
earlier blocks. Conditioned on the prefix, `g` is therefore independent of the
draws it schedules, and the argument in section 2 applies to each realized `g`
separately. Averaging over the controller's distribution of `g` preserves it.

This is why the bypass action is safe too: bypass is `g = 0`, a target-only step,
which trivially emits from `p`.

Two things would break the argument, and neither is permitted:

* Choosing `g` using the current block's target logits or acceptance outcomes.
  That conditions on the future (invariant I9).
* Stopping a block early because of what was emitted, or truncating an
  overlong block. The budget cap handles length instead: `g` is capped at
  `remaining - 1`, and a block emits at most `g + 1` tokens, so the output
  never overshoots the budget and never needs truncating. A single remaining
  token is taken with a target-only step.

## 4. What finite precision changes

The proof above is about real numbers. The implementation is FP32.

* **Probabilities are FP32** (`PROBABILITY_DTYPE`), computed with a stable
  softmax that subtracts the maximum before exponentiating. Suppressed tokens
  are masked to `-inf` *before* the temperature division, so the remaining mass
  renormalizes to one instead of leaving a hole.
* **Acceptance avoids the quotient.** `accept_ratio(p_d, q_d)` compares
  `u * q_d <= p_d` rather than `u <= p_d / q_d`. Mathematically identical, and
  it keeps the comparison exact for dyadic inputs, which is what makes the
  exhaustive enumeration in level 2 exact rather than approximate.
* **Residual weights stay unnormalized.** `positive_residual` returns
  `max(p - q, 0)` and lets the random source normalize. The subtraction of two
  dyadic rationals is exact; dividing by the sum is not.
* **Roundoff is clamped, not tolerated.** `p - q` is clamped at zero before use,
  so a `-1e-9` from cancellation cannot become negative probability mass.
* **Nonfinite inputs are refused.** Every distribution passes
  `validate_distribution`, which rejects NaN, infinity, negative entries, and
  zero total mass.

What this does *not* establish: that two engines sampling with the same seed
produce the same string. They need not, and the benchmark does not require it.
Kernel selection, reduction order, and dtype all perturb the logits slightly,
which can flip a single acceptance decision and diverge the continuation. Greedy
decoding is the case where exact agreement *is* required, and it is checked on
token-ID arrays, never on decoded text (invariant I6).

## 4a. What BF16 actually does to the cache differential

Measured on the GB10 with `Qwen/Qwen3-4B` in BF16, comparing a prefill split
into two forward calls against the same prompt in one uncached call:

| Quantity | Measured |
|---|---|
| Max absolute logit difference | 0.73 to 1.13 |
| Mean absolute logit difference | about 0.07 |
| Max logit magnitude | about 50 |
| Max relative difference | about 2% |
| Rows (of ~30) whose argmax flips | 0 to 2 |
| Largest top-2 margin at a flipped row | 0.125 |

The third-to-last row is the interesting one. **Cached and uncached BF16 logits
do not agree on argmax for every position.** Splitting a prefill changes the
reduction order, BF16 keeps 8 mantissa bits, and across 36 layers that is enough
to flip a near-tie.

This is not a licence to widen a tolerance until the test passes. The useful
distinction is *where* differences land:

* Reduction-order noise flips only near-ties. Every observed flip was at a row
  whose top-2 margin was at most 0.125, a quarter of one percent of the logit
  scale.
* A genuine off-by-one in the cache shifts the conditioning by a whole token.
  That changes logits by order 10 and flips rows the model was confident about.

So `test_cached_logits_match_full_prefix_recomputation` asserts the relative
bound *and* that every disagreeing row is a near-tie, which a misalignment
cannot satisfy. The same comparison in FP32 on the tiny CPU fixture holds to
1e-4 with exact argmax agreement, which is why SPEC.md requires validating FP32
on CPU before trusting BF16 on the GPU.

End-to-end greedy conformance is a separate and stronger check: on the frozen
smoke set, `decode_target_only` and `hf_ar` produce **identical 32-token ID
arrays**, because both walk the same cached path and the near-ties above did not
arise at a decision point. That equality is the evidence for level 3, not the
logit differential.

### A policy difference that looked like a numerical bug

The first run of that conformance test failed at token 12: the native engine
emitted 151668 where the baseline emitted 151645, which is `<|im_end|>`.

The cause was not numerical. Under `eos_policy="suppress_until_budget"`,
Transformers' `eos_token_id=None` stops generation from *ending* at a stop
token, but the stop token is still in the distribution and can still be emitted.
Switchback applies the EOS policy to the distribution itself (SPEC.md section
4.1), masking stop tokens to `-inf` before the softmax. Two different policies,
both defensible, producing a mismatch that reads exactly like a cache bug.

The fix was to make the baseline do what the specification says by also setting
`suppress_tokens`. The lesson is recorded here because "the engines disagree"
almost never identifies its own cause.

## 4b. Greedy conformance is a rate, and it is not 100%

Greedy speculation is exact in real arithmetic. Section 2 proves it: verifying a
proposal against the target's argmax either accepts the target's own choice or
replaces it with the target's own choice. Milestones 3 and 4 confirmed it on the
real model, with identical 32-token ID arrays across `native_ar`, `hf_ar` and
`fixed_{1,2,4,8}`.

At longer outputs it stops holding. Measured on the four pilot prompts at draft
lengths 1, 2, 4 and 8 (`artifacts/conformance.json`):

| Output length | Comparisons | Identical | Agreement |
|---:|---:|---:|---:|
| 32 | 16 | 16 | 100% |
| 64 | 16 | 12 | 75% |
| 128 | 16 | 12 | 75% |
| 256 | 16 | 12 | 75% |

The reason is section 4a's. A target-only step computes its logits in a width-1
forward; a verification step computes the same logits inside a width-`g+1`
forward. Both are correct; neither is bitwise equal to the other in BF16.

What keeps this a floating-point result rather than a bug report is *where* the
divergences land. Recomputing the uncached logits at each divergence, with the
same EOS mask the engines used:

| Case | Target-only chose | Speculation chose | Logit gap | Logit scale |
|---|---|---|---:|---:|
| prompt 1, token 54 | 13136 @ 39.250 | 3070 @ 39.000 | 0.250 | 39.2 |
| prompt 0, token 56 | 73594 @ 18.125 | 151668 @ 18.250 | 0.125 | 18.2 |

The largest gap between two chosen tokens across every observed divergence is
**0.25**, against BF16 reduction-order noise measured at up to **1.13**. Both
engines are choosing between tokens the model considers equally good; once they
choose differently, the continuations diverge and never rejoin.

A divergence at a confident position would be a different thing entirely, and
the measurement reports the gap precisely so the two cannot be confused. The
first version of that measurement omitted the EOS mask and reported a 15.5-logit
gap, which would have meant a real bug. A diagnostic that does not reproduce the
engine's own configuration measures something else.

See [ADR 0004](decisions/0004-bf16-greedy-conformance.md), including the
unresolved consequence for milestone 7: the report renderer currently refuses
any cell where greedy outputs differ.

## 5. How the oracle stays independent

`src/switchback/oracle.py` does not import `switchback.sampling`, and
`test_the_oracle_does_not_import_the_production_sampler` enforces that by parsing
the module's import statements rather than by searching its text.

The oracle works entirely in `fractions.Fraction`, so its reference values
contain no floating point at all. It provides:

* `rejection_step_distribution` — re-derives one step's output law by summing
  over the algorithm's branches, from the specification rather than from the
  production code.
* `target_sequence_distribution` — the target's own `n`-step autoregressive law,
  as a product of conditionals.
* `enumerate_algorithm` — a depth-first driver that runs *another*
  implementation over every execution path, branching at each random decision
  and weighting by its exact probability. The implementation under test is a
  parameter, so the oracle never imports it.

The production sampler asks its `RandomSource` for exactly two kinds of
decision, `categorical(weights)` and `accept_ratio(numerator, denominator)`. Both
carry enough information for the driver to compute the branch probability
exactly, which is why the enumeration produces an exact rational output law for
the real sampler rather than a Monte Carlo estimate.

Empirical sampling checks, where they appear at all, are diagnostics. They never
stand in for the enumeration.

## 6. Current status

Levels 1 and 2 are done and are exact. Level 3 is **partial and measured**:
greedy conformance is 100% at 32 tokens and 75% at 64 and beyond, with every
divergence a documented near-tie (section 4b). Level 4 is not measured, and the
README says so.

```
python -m pytest tests/unit/test_oracle.py tests/unit/test_sampling.py -q
python -m pytest tests/property -q
python -m pytest tests/integration/test_cached_engine_cpu.py -q     # FP32
python -m pytest tests/integration/test_cached_engine_gpu.py -q -m gpu  # BF16
python -m switchback conformance --lengths 32 64 128 256                # the rate
```
