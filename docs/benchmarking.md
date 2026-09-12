# Benchmarking

What is measured, how, and what the numbers are not allowed to mean.

## The shape of a run

```
bench.prepare   lock the prompts          -> artifacts/workloads/<name>.json
switchback calibrate --workload ...       -> artifacts/calibration.json  (hashed, frozen)
switchback evidence --gpu                 -> artifacts/evidence.json     (exit codes, not claims)
bench.run       execute the matrix        -> artifacts/runs/<name>/{manifest,requests.jsonl,evidence}
bench.validate  check the run             -> exit 0 or a list of problems
render_results  derive the report         -> report.md, or RESULTS.md for a full run
```

Each stage writes atomically and hashes its output. The renderer recomputes
every number in the report from `requests.jsonl`; it accepts no summary
statistic as input, so there is nowhere to type a speedup in.

## Which prompts

Rows are selected by sorting the SHA-256 of `revision + split + stable_row_id`
and taking the required count. No output is inspected to choose a prompt. For
GSM8K, which has no stable row id, source order at the pinned revision is the
identity.

Calibration prompts come from the **train** split and held-out prompts from the
**test** split, so the controller is never fitted on anything it is later
measured against. `bench.validate` checks that the expected request keys are
exactly what the config and workload imply, which is what stops a run from
quietly narrowing its own cohort and still reporting `complete: true`.

The workload file stores prompt ids, token counts and a hash of the token ids.
It does not store prompt text.

## Timing boundaries

The host timer starts after a GPU synchronization and before prefill, and stops
only after GPU completion. CUDA is asynchronous: an unsynchronized host
timestamp measures when work was queued.

Included in warm-request latency: target prefill, draft prefill, proposal
generation, verification, sampling, cache repair, controller overhead, the draft
catch-up call, and final synchronization.

Excluded and reported separately: model download, model loading, tokenization,
and one-time kernel warmup.

Token release times are recorded as measured. A verified block releases several
tokens at once and they share a timestamp; they are not spread evenly across the
block, because a consumer did not receive them evenly.

Serialization happens after the timed region. Greedy divergence evidence is
computed after the timed region too.

## Fairness

- Both models stay resident for every engine, including `hf_ar`, which never
  drafts. Hiding the draft's memory would flatter every speculative row.
- Every engine gets the same prompt token ids, the same decode config, and the
  same `correctness_checks` setting. Enabling assertions for one engine and not
  another would make the comparison meaningless.
- Engine order is shuffled inside each prompt/repeat block from a recorded
  scheduling seed.
- Peak-memory counters are reset per request, and the resident floor is recorded
  before any cache is allocated.
- Everything runs sequentially in one process. On this machine per-token cost
  varies by roughly 10% between processes, so cross-process latency comparisons
  are not trustworthy.

## Metrics

| Metric | Definition |
|---|---|
| Warm request latency | `end_ns - start_ns`, prefill and all decoding included |
| TTFT | `first_token_ns - start_ns` |
| Decode TPOT | `(last_token_ns - first_token_ns) / (output_tokens - 1)`; undefined below two tokens |
| Effective throughput | Sum of committed output tokens over sum of request latency; batch-one, sequential |
| Speedup | Geometric mean over prompts of baseline median latency over candidate median latency |
| Uncertainty | 95% percentile bootstrap, 2,000 resamples of prompt ids, seed 2026; repeats stay inside their prompt |
| Slowdown fraction | Prompts whose candidate median exceeds the baseline median by more than 5% |
| Acceptance | Accepted candidates over proposed, with unused suffix proposals in the denominator |
| Target-call efficiency | Committed output tokens per target forward call |
| Greedy match | Fraction of requests whose token ids were identical to the baseline's |
| Memory | Peak allocated bytes, reserved bytes and RSS, in separate columns |

A counter an engine cannot expose is `null`, never zero. `hf_ar` and
`hf_dynamic` report null for acceptance and target calls, because getting them
out of an external baseline would mean adding instrumentation inside its own
measured path.

## Greedy conformance

Greedy speculation is exact in real arithmetic and not in BF16. The report
carries a measured `Greedy match` column, and a mismatch is admissible only with
evidence that it was a near-tie, under a bound the config declares. See
[ADR 0005](decisions/0005-renderer-near-tie-contract.md) and
[docs/correctness.md](correctness.md) section 4b.

## What a result is not allowed to say

- Not a serving-throughput claim. Batch one, one request at a time, no
  contention. Batching changes target efficiency and draft contention, and
  batch-one gains need not survive it.
- Not a quality claim. Greedy token agreement is a conformance measurement, not
  evidence that outputs are equally good. `docs/correctness.md` separates the
  four levels of claim.
- Not transferable. One GPU, one model pair, one attention backend, one dtype.
- The fixed-length condition is artificial. EOS is suppressed identically in
  every engine so output lengths match, which means the tail of a long
  completion is text the model wanted to stop producing.
- Smoke results are preliminary and are named `smoke` so a table cannot be
  mistaken for the primary condition.
- If an interval's lower bound does not exceed 1, the improvement is
  inconclusive and is reported that way. If the code is slower, that is the
  result.

## Reproducing

```bash
bash scripts/reproduce.sh --preset cpu     # offline: no GPU, no weights, no network
bash scripts/reproduce.sh --preset smoke   # GPU gate, smoke benchmark, report
bash scripts/reproduce.sh --preset full    # the primary matrix, about 7.5 hours
```

`bench.run` is resumable. `BENCH_MAX_SECONDS` caps one unattended stretch;
rerunning the same command continues from `requests.jsonl`. Resuming is only
legitimate against the same frozen source, and the manifest records the code
hash so the renderer can tell.
