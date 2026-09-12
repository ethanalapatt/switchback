# Correctness

What this project claims, in four separate levels, and what backs each one.

| Level | Claim | Backed by |
|---|---|---|
| 1. Mathematical | Under exact arithmetic, speculative sampling emits tokens from the target distribution | The proof below |
| 2. Executable finite-model | This implementation reproduces that distribution on small enumerable cases | `tests/unit/test_oracle.py`, exhaustive enumeration against an independent rational oracle |
| 3. Real-model numerical conformance | Greedy speculation returns the same token IDs as target-only decoding on the pinned Qwen3 pair | Not yet established. Milestones 3 and 4 |
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

Levels 1 and 2 are done. Levels 3 and 4 are not, and the README says so.

```
python -m pytest tests/unit/test_oracle.py tests/unit/test_sampling.py -q
python -m pytest tests/property -q
```
