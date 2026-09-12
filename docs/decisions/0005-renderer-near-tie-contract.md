# ADR 0005: the renderer admits a greedy mismatch only as documented near-tie

Date: 2026-09-12. Status: accepted. Milestone: M7. Resolves: [ADR 0004](0004-bf16-greedy-conformance.md).

## Context

ADR 0004 measured that BF16 greedy speculation agrees with target-only decoding
on 100% of comparisons at 32 output tokens and 75% at 64 and beyond, with every
divergence a near-tie: the largest gap between the two chosen tokens was 0.25
logits, against reduction-order noise measured at up to 1.13.

`scripts/render_results.py` refuses to render any cell where greedy outputs
differ. At the primary condition of 256 fixed-length greedy tokens it would
refuse roughly a quarter of cells, so no report could be generated at all.

ADR 0004 listed three options and recommended A. This record adopts it.

## Decision

**Option A.** A greedy mismatch is admissible only when the run carries evidence
that it was a near-tie, and only under a bound the manifest declares.

The manifest may set `max_chosen_token_gap`. Each request row may carry
`greedy_divergence` with `index`, `reference_token`, `candidate_token`,
`chosen_token_gap` and `logit_scale`. For every greedy cell whose output differs
from the baseline's, the renderer requires all of:

1. the manifest declares a bound;
2. the row carries divergence evidence;
3. the recorded `index` is the *actual* first difference between the two
   outputs;
4. the recorded tokens are the ones the two outputs actually contain there;
5. `chosen_token_gap` is present, non-negative, and at or below the bound.

Any failure is refused, with the original message about investigating before
publishing a speedup. A run that declares no bound keeps the original strict
behaviour exactly, so nothing that passed before can start failing now.

The report gains a `Greedy match` column -- the measured fraction of requests
whose token ids were identical to the baseline's -- and a `Greedy conformance`
section stating the bound that was applied.

## Why not the alternatives

**B, compare only the matching prefix.** It turns a measurable failure into an
invisible one. "The first 54 tokens matched" is a weaker claim that reads like a
stronger one.

**C, conformance in FP32 and latency in BF16.** Honest but it compares two
configurations and asks the reader to accept that a property of one transfers to
the other. It stays available if the near-tie bound is ever exceeded, because
then the evidence would point at a bug rather than at floating point.

## What this does not do

It does not make greedy speculation exact in BF16. It makes the inexactness a
measured, bounded, reported quantity instead of a reason no report exists.

The bound is a claim about floating point, not a tolerance for wrong answers.
Set it above the measured reduction-order noise and a real bug walks through;
set it at zero and it is the old contract. The value belongs in the config,
where it is versioned and hashed, and the report prints it.

## Evidence

`tests/report/test_conformance_contract.py` -- thirteen tests covering the
strict default, admission within the bound, a gap beyond it, missing evidence,
evidence pointing at the wrong position, evidence naming the wrong tokens, an
unusable gap, and a nonsense bound.
