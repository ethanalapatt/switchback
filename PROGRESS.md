# Switchback progress

Updated: September 12, 2026.

## Current state

Milestones 1 through 8 are complete for the declared scope. Two cohorts remain unrun and are named as such. The execution machine is the DGX Spark itself
(`gigi-spark`, NVIDIA GB10, aarch64, driver 580.142, CUDA 13.0, torch
2.14.0+cu130), so the GPU gates in M1 actually ran rather than being deferred.
The pinned Qwen3 pair loads fully resident, passes tokenizer parity, and both
Hugging Face reference baselines produce identical greedy token IDs on a
32-token smoke request.

Greedy and sampled speculation and the measured cost controller all run on the
real Qwen3 pair. Sampled output is verified by exhaustive enumeration against
the independent oracle, exactly, through the real cache and softmax.

**The primary matrix has run and the controller did not win.** Over 128 held-out
prompts at 256 greedy tokens, `adaptive` is 2.021x [1.966, 2.070] against
`hf_ar`, but `fixed_8` is 2.066x and `hf_dynamic` is 2.234x. The ordering holds
on both datasets separately. `RESULTS.md` is generated from the raw rows.

**Greedy conformance is 61.7-64.1% at 256 tokens.** Every divergence is a BF16
near-tie; the 18 the automatic screen could not explain are each traced to a
position and an execution path in ADR 0006.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M1 Hardware and baseline | **Complete** | `artifacts/environment.json`, `artifacts/model_check.json`, `artifacts/pilot/pilot.json`, 11 GPU tests |
| M2 Sampling oracle | **Complete** | `docs/correctness.md`, 38 oracle tests, 23 property tests |
| M3 Cached target-only engine | **Complete** | 11 GPU tests, 29 CPU cache tests, `artifacts/pilot/pilot.json` |
| M4 Fixed greedy speculation | **Complete** | 21 GPU tests, 82 forced-path tests, `artifacts/profile/m4_profile.json` |
| M5 Sampled speculation and traces | **Complete** | 11 GPU tests, exact oracle agreement, `artifacts/traces/`, `artifacts/evidence.json` |
| M6 Cost controller | **Complete** | 65 controller/adaptive tests, `artifacts/calibration.json`, `artifacts/controller_check.json`, `artifacts/conformance.json` |
| M7 Benchmark and report | **Complete** | `RESULTS.md`, `artifacts/runs/primary/` (3,072 requests, validated), 32 report tests |
| M8 Reviewer demo | **Complete** | `docs/trace.gif`, `viewer/`, `docs/failure-analysis.md`, natural-stop cohort |

## Next action

The declared scope is complete and published. Two things are worth doing next,
in this order:

1. **The context-stress cohort** (128 to 3,072 token prompts). It is the last
   condition under which the adaptive controller could beat a well-chosen fixed
   length, because it is where the best draft length is most likely to vary by
   prompt. If it does not win there either, the controller's verdict is settled
   and the resume line should describe the break-even study rather than a
   speedup.
2. **Confidence-based early stopping inside a block**, which is what
   `hf_dynamic` does and the single change most likely to close the 0.94x gap to
   it. The draft's own confidence is available before verification, so it does
   not condition on the future and does not threaten the argument in
   `docs/correctness.md` section 3.

Before Ethan leads a resume with this: the five interview questions in
`PORTFOLIO_DECISION.md` are still not demonstrated, and question 3 -- why high
acceptance can still produce a slowdown, and how the controller decides to
bypass -- now has a measured answer he should be able to give, including why
bypass never fired.

---

### M1: Hardware and reference baseline

- **Status:** complete.

- **Implementation and decisions:**
  - `types.py` (supplied) plus new `provenance.py`, `runtime.py`, `env.py`,
    `manifest.py`, `pilot.py`, `cli.py`, `models/qwen.py`, `models/tiny.py`,
    `models/hf_baselines.py`.
  - CLI: `doctor`, `resolve-models`, `verify-models`, `pilot`, `demo`.
  - Model revisions resolved to immutable commit SHAs and recorded in
    `data/manifest.json`; `ModelSpec` refuses `main`/`master`/`latest`/`""`.
  - Tokenizer compatibility is checked on four axes (full token-to-ID map,
    added-token table, special IDs, 24 probe-string encodings) plus a separate
    logits-vocabulary check. See ADR 0001.
  - `.generate()` is confined to `models/hf_baselines.py`, which implements the
    named `hf_ar` and `hf_dynamic` baselines only.
  - Three Transformers mechanisms that reintroduce hidden generation defaults
    were found and neutralized; EOS suppression works by removing stop tokens,
    not by a minimum length, because assisted generation refuses a
    minimum-length logits processor. See ADR 0003.
  - torch's Triton `aten::bmm` override is deregistered on this host because
    Triton cannot build its CUDA shim without CPython headers. The probe, the
    affected operators, and the reason are recorded in every artifact. See
    ADR 0002.
  - Tiny CPU fixture builds a real `Qwen3ForCausalLM` (87k parameters) with
    seeded weights, so offline tests exercise the same attention and cache code
    as the benchmark models without downloading anything.

- **Commands actually run:**
  ```
  python -m pip install -e '.[dev]'          (with torch pinned by a constraint)
  python -m switchback doctor --out artifacts/environment.json
  python -m switchback doctor --require-gpu
  python -m switchback resolve-models --out data/manifest.json
  python -m switchback verify-models --out artifacts/model_check.json
  python -m switchback demo
  python -m switchback pilot --max-new-tokens 32 --repeats 3 --out artifacts/pilot/pilot.json
  python -m pytest -m 'not gpu and not download' -q
  python -m pytest tests/property -q
  python -m pytest tests/report -q
  python -m pytest -m gpu -q
  python -m ruff check .
  python -m ruff format --check .
  python -m mypy src/switchback
  bash scripts/reproduce.sh --preset cpu
  bash scripts/reproduce.sh --preset smoke
  ```

- **Passed / failed / skipped checks:**
  - Passed: 89 CPU tests (`not gpu and not download`), 11 property tests
    (included in that total), 10 report-integrity tests, 11 GPU tests, ruff
    lint, ruff format, mypy (13 source files, no issues).
  - `reproduce.sh --preset cpu` exited 0, all stages passed.
  - `reproduce.sh --preset smoke` exited 3: every implemented stage passed; the
    smoke benchmark and report rendering are correctly reported INCOMPLETE
    because `bench/` does not exist until M7.
  - Skipped: nothing was skipped to reach a pass. The 11 GPU tests are excluded
    by marker from the CPU run and were executed separately on the GB10.
  - Failed and fixed during the milestone: Triton `bmm` build failure;
    assisted generation rejecting a minimum-length processor; Qwen3's
    `eos_token_id` leaking back in past `use_model_defaults=False`; assistant
    options being read from the assistant's own config; a `.gitignore` rule that
    matched the `src/switchback/models` package; the tiny fixture's rotary
    `inv_freq` buffer holding uninitialized memory after `to_empty`.

- **Benchmark or evidence paths:**
  - `artifacts/environment.json`, `artifacts/model_check.json`,
    `artifacts/pilot/pilot.json`, `data/manifest.json`,
    `requirements-spark.lock`, `requirements-cpu.lock`.
  - Pilot condition: 4 prompts, greedy, EOS suppressed, exactly 32 output
    tokens, 3 repeats, 2 warmups, seed 42, BF16, SDPA. Median request latency
    1559.9 ms for `hf_ar` and 736.5 ms for `hf_dynamic`; no greedy output
    mismatch between them; peak allocated 8.7 GiB. **These are pilot numbers for
    sizing the M7 run. They are not a benchmark result and no speedup is
    claimed from them:** four unrandomized prompts are not a cohort, and the
    Switchback engine they will eventually be compared against does not exist.

- **Source commit:** recorded inside each artifact under `source.commit`.

- **Known limitations and blockers:**
  - No Switchback decoding, cache ledger, sampler, controller, benchmark, or
    viewer exists. M2 through M8 are unimplemented.
  - Triton `aten::bmm` routing is disabled on this host. Installing
    `python3-dev` restores it; the doctor re-detects automatically. Any run that
    mixes the two paths must not be pooled.
  - TTFT for the HF baselines is measured with a streamer, which forces a
    host synchronization per decoding step. Acceptable for a pilot; M7 must
    decide whether the primary timing path keeps it.
  - The CPU lock is verified by CI on x86-64 Linux only, not on the Spark.
  - `configs/`, `bench/`, and `viewer/` are empty directories.

- **Next concrete step:** implement `sampling.py` and `oracle.py` for M2.

- **Teach-back explanation prepared:**
  - *Decision:* check tokenizer compatibility on the full token-to-ID map and
    real encodings, not on `config.vocab_size`.
  - *Alternative considered:* comparing vocabulary sizes, which both configs
    report as 151,936.
  - *Failure mode:* a permuted or shifted vocabulary keeps sizes equal, so the
    draft proposes IDs that mean different text to the target. Acceptance rates
    stay plausible and greedy equality fails intermittently, which looks like a
    KV-cache bug for a long time.
  - *Evidence:* `test_conflicting_id_for_a_shared_token_is_detected` and
    `test_matching_vocab_size_does_not_imply_compatibility` construct exactly
    that case and require the checker to catch it;
    `test_any_reordering_of_ids_is_detected` generates every permutation of a
    small vocabulary and asserts the verdict is compatible only for the
    identity.

- **Ethan's teach-back status:** not yet demonstrated.

---

### M2: Independent sampling oracle

- **Status:** complete.

- **Implementation and decisions:**
  - `sampling.py`: FP32 probability transforms, greedy selection with explicit
    smallest-ID tie breaking, positive residual, correction and bonus draws,
    `speculative_block`, `greedy_block`, and the sequence-level drivers. Pure:
    no clock, no device counter, no controller state, no model.
  - Every stochastic decision goes through a `RandomSource` with exactly two
    operations, `categorical(weights)` and `accept_ratio(numerator,
    denominator)`. That is what makes exhaustive enumeration of the real
    sampler exact instead of statistical.
  - Acceptance compares `u * q <= p` rather than `u <= p/q`. Mathematically
    identical, and it avoids forming a quotient that is generally not dyadic:
    `(1/4)/(3/4) = 1/3` would round and put roughly 1e-8 into every enumerated
    path.
  - Categorical draws take unnormalized weights so the exact `max(p - q, 0)`
    is not destroyed by an inexact division before sampling.
  - Three RNG streams (proposal, acceptance, correction/bonus) derived from the
    request seed by SHA-256, so consuming one cannot perturb another.
  - `oracle.py` works only in `fractions.Fraction` and never imports
    `switchback.sampling`; the implementation under test is passed in as a
    callable.

- **Commands actually run:**
  ```
  python -m pytest tests/unit/test_sampling.py tests/unit/test_oracle.py -q
  python -m pytest tests/property -q
  python -m pytest -m 'not gpu and not download' -q      (x8, fresh .hypothesis each time)
  python -m ruff check . && python -m ruff format --check .
  python -m mypy src/switchback
  ```

- **Passed / failed / skipped checks:**
  - 174 CPU tests pass (up from 89), including 34 sampling unit tests, 38 oracle
    tests, and 23 property tests. ruff, format, and mypy clean.
  - Exhaustive enumeration: every rational target/draft pair with denominator 4
    over vocabularies of size 2, 3 and 4 (25, 225 and 1225 pairs). Accepted plus
    residual mass equals the target **exactly** as rationals, which is stronger
    than the specified 1e-12 FP64 gate.
  - The production sampler itself, enumerated over all execution paths for
    sizes 2 and 3, reproduces the target distribution exactly.
  - 20 deterministically seeded tiny tree pairs (vocab 2-3, 3 output tokens,
    gamma 1 and 2): exact total variation distance 0 from the target's
    autoregressive law.
  - Mutation check: an implementation that verifies every candidate against the
    block's first target row is required to fail the tree test, so the passing
    result is not vacuous.
  - Failed and fixed during the milestone: three property strategies filtered on
    equal vector length, which tripped Hypothesis' `filter_too_much` health
    check on roughly half of fresh-database runs. Fixed by drawing the shared
    length up front, not by suppressing the health check; verified stable over
    eight runs with the example database deleted each time.
  - Skipped: nothing.

- **Benchmark or evidence paths:** none. M2 produces no timings by design.
  `docs/correctness.md` records the proof and the scope of each claim.

- **Source commit:** this milestone's commits, `feat: implement the numerical
  sampling core` through `docs: prove the one-step identity`.

- **Known limitations and blockers:**
  - The sampler is model-free: it takes row callbacks, not a model or a cache.
    Nothing yet connects it to real logits, so level 3 of `docs/correctness.md`
    (real-model numerical conformance) is still unestablished.
  - Enumeration is exact only because the test inputs are dyadic. Real FP32
    softmax outputs are not, and no claim is made about vocabulary size 151,936.
  - EOS handling exists as a masking policy only; request termination is the
    decoder's job in M3.

- **Next concrete step:** implement `cache.py` and the adapter forward/crop pair
  for M3.

- **Teach-back explanation prepared:**
  - *Decision:* express acceptance as a ratio and keep residual weights
    unnormalized, instead of precomputing the acceptance probability and a
    normalized residual.
  - *Alternative considered:* the textbook form, `accept if u <= p[d]/q[d]` with
    `r = max(p-q,0)/Z`. It reads closer to the paper.
  - *Failure mode:* both `p/q` and `1/Z` are inexact in binary floating point
    even when `p` and `q` are not. An exhaustive test would then compare the
    sampler against the oracle with roughly 1e-8 of unavoidable noise, so the
    only usable gate would be a tolerance -- and a tolerance wide enough to
    absorb that noise is also wide enough to hide a genuine 1e-7 bias in the
    residual.
  - *Evidence:* `test_production_block_reproduces_the_target_for_every_rational_pair`
    asserts exact rational equality, not `approx`, for all 250 pairs at
    vocabulary sizes 2 and 3. It would not be possible to write that assertion
    with the textbook form.

- **Ethan's teach-back status:** not yet demonstrated.

---

### M3: Cached target-only engine

- **Status:** complete.

- **Implementation and decisions:**
  - `cache.py`: the pending-token ledger. The ledger, not the Transformers
    cache, is the source of truth for length; `get_seq_length()` is checked
    against it so a divergence is an error rather than wrong output.
    `assert_boundary` compares token *identity*, not just length, because a
    stale tentative suffix of the right length is the failure mode that yields
    plausible-but-wrong text. `crop_to` refuses to grow.
  - `events.py`: `MonotonicClock` synchronizes before every timestamp bounding a
    measured region; `FakeClock` counts those synchronizations so a test can
    assert they happened. Block events carry no full-vocabulary arrays.
  - `models/qwen.py`: `QwenAdapter.forward` passes `position_ids` and
    `cache_position` explicitly. Letting Transformers infer them from the
    attention mask is precisely what a rollback invalidates, since after a crop
    the cache is shorter than the committed sequence.
  - `decoder.py`: `decode_target_only` establishes the pending-token convention
    during initialization, before speculation exists to complicate it. Greedy
    and sampled decoding are separate algorithms; greedy consumes no randomness.
  - `EngineOptions.correctness_checks` is recorded in every result, because
    enabling assertions for one engine and not another would invalidate a
    latency comparison.

- **Commands actually run:**
  ```
  python -m pytest tests/integration/test_cached_engine_cpu.py -q
  python -m pytest tests/unit/test_cache.py -q
  python -m pytest -m gpu -q
  python -m pytest -m 'not gpu and not download' -q
  python -m switchback pilot --local-files-only --max-new-tokens 32 --repeats 3
  python -m ruff check . && python -m ruff format --check . && python -m mypy src/switchback
  ```

- **Passed / failed / skipped checks:**
  - 220 CPU tests and 22 GPU tests pass. ruff, format and mypy clean.
  - FP32 CPU: cached logits match full-prefix recomputation to 1e-4 with exact
    argmax agreement at chunk widths 1, 2 and 5; crop-and-replay reproduces the
    uncropped logits bitwise; a companion test confirms a stale suffix really
    does change the logits, so the rollback test cannot pass on a no-op cache.
  - BF16 GPU: `decode_target_only` and `hf_ar` emit **identical 32-token ID
    arrays** on all four frozen smoke prompts.
  - **Failure found and fixed (policy, not arithmetic):** the first conformance
    run diverged at token 12, native emitting 151668 where `hf_ar` emitted
    151645 (`<|im_end|>`). Transformers' `eos_token_id=None` stops generation
    from *ending* at a stop token but leaves it in the distribution, while
    Switchback masks stop tokens before the softmax. Fixed by also setting
    `suppress_tokens` on the baselines.
  - **Measured and recorded, not fixed:** cached and uncached BF16 logits do not
    agree on argmax for every row. One or two rows in thirty flip. Max absolute
    difference 0.73-1.13 against a logit scale near 50 (about 2% relative), mean
    0.07. Every flip was at a row with top-2 margin at most 0.125. The test
    asserts that bound rather than a widened tolerance, because reduction-order
    noise flips only near-ties while a cache off-by-one flips confident rows.
    Documented in `docs/correctness.md` section 4a.
  - Skipped: nothing.

- **Benchmark or evidence paths:** `artifacts/pilot/pilot.json`, re-measured with
  `native_ar` included. All three engines produce identical greedy IDs. Median
  request latency at 32 fixed greedy tokens: `hf_ar` 1526.6 ms, `native_ar`
  1525.8 ms, `hf_dynamic` 746.1 ms. **Pilot numbers for sizing M7, not a
  benchmark result:** four unrandomized prompts, not the locked cohort. The one
  thing worth reading from them is that Switchback's own loop is within 0.1% of
  `hf_ar`, so the Python-overhead risk in SPEC.md section 11 is not visible at
  this scale.

- **Source commit:** recorded in `artifacts/pilot/pilot.json` under
  `source.commit`.

- **Known limitations and blockers:**
  - No speculation. `speculative_block` exists in `sampling.py` but nothing
    connects it to a cache or to a draft model.
  - Level 3 of `docs/correctness.md` now holds for target-only decoding only.
  - The BF16 argmax flips above mean cached and uncached BF16 execution are not
    interchangeable at the logit level. Nothing in the design depends on them
    being interchangeable, but a future comparison must not assume it.
  - `hf_dynamic`'s TTFT is roughly 3x `hf_ar`'s, because the first verified
    block must generate draft tokens before releasing anything. Expected, and a
    reason the natural-stop cohort is reported separately.

- **Next concrete step:** implement fixed greedy speculation for M4.

- **Teach-back explanation prepared:**
  - *Decision:* pass `position_ids` and `cache_position` explicitly on every
    forward call instead of letting Transformers infer them.
  - *Alternative considered:* passing only `input_ids` and an attention mask,
    which is what nearly every Transformers example does and which works
    perfectly for ordinary decoding.
  - *Failure mode:* the inference is `positions = arange(past_length, past_length
    + width)` derived from the mask. That is right whenever the cache and the
    sequence agree. After a speculative rejection they deliberately do not: the
    cache has been cropped back while the request has committed a correction, so
    an inferred position is off by the number of discarded candidates. The
    rotary embedding then rotates every query by the wrong angle, producing
    fluent, wrong text with no error anywhere.
  - *Evidence:* `test_cropping_then_replaying_reproduces_the_uncropped_logits`
    poisons a cache with a wrong suffix, crops it back, and requires the replayed
    logits to be bitwise equal to the un-poisoned ones; `test_a_stale_suffix_
    actually_changes_the_logits` confirms the poison was not a no-op, so the
    first test cannot pass vacuously.

- **Ethan's teach-back status:** not yet demonstrated.

---

### M4: Fixed greedy speculation

- **Status:** complete.

- **Implementation and decisions:**
  - `decode_speculative_greedy` in `decoder.py`. Per block: draft `g` proposals
    autoregressively, verify them in **one** target forward call of width
    `g + 1`, commit the matching prefix plus the target's own token, crop both
    caches to `len(S) + accepted`.
  - Row `j` of the verification output predicts proposal `j`; row `g` predicts
    the bonus. There are `g + 1` rows, not `g`.
  - On full acceptance the draft is one candidate behind and pays a catch-up
    forward call. It is counted, not hidden, because milestone 6's cost model
    depends on it.
  - The block length is capped at `remaining - 1` and a lone remaining token
    takes a target-only step. This is not only bookkeeping: truncating an
    overlong block would condition on the future and break the distribution
    argument in `docs/correctness.md` section 3.
  - A proposed stop token ends drafting early but is still verified; the actual
    proposal count is recorded.
  - `profiling.py` is a separate diagnostic pass that reimplements the block, so
    the production loop carries no profiling branches.

- **Commands actually run:**
  ```
  python -m pytest tests/unit/test_speculation_paths.py -q
  python -m pytest tests/integration/test_speculation_cpu.py -q
  python -m pytest tests/integration/test_speculation_gpu.py -q -m gpu
  python -m pytest -m 'not gpu and not download' -q
  python -m switchback profile --local-files-only --blocks 16
  python -m ruff check . && python -m ruff format --check . && python -m mypy src/switchback
  ```

- **Passed / failed / skipped checks:**
  - 351 CPU tests pass (82 forced-path, 49 real-model CPU). 21 new GPU tests
    pass. ruff, format and mypy clean.
  - **The milestone gate holds:** on the real Qwen3-4B/0.6B pair, fixed
    speculation at gamma 1, 2, 4 and 8 produces token-identical output to
    `native_ar` at two context lengths, and to `hf_ar` as well.
  - Forced paths covered: rejection at each position 0..g-1, full acceptance,
    catch-up present and absent, accepted EOS, EOS inside a rejected suffix,
    suppressed EOS, budget exhaustion at every length, a request that ends
    before the draft is initialized, and a deliberately wrong crop that must
    raise rather than emit fluent output.
  - A GPU test asserts a rejection actually occurred during the run, so the
    rollback path cannot be silently untested.
  - Failed and fixed during the milestone: the first version of the forced-path
    tests used a budget of `gamma + 1`, which the `remaining - 1` cap shortens
    to a block of width `gamma - 1`; eleven tests were asserting against a block
    that never existed. Fixed by using `gamma + 2`, the smallest budget that
    admits a full-width block.
  - Skipped: nothing.

- **Benchmark or evidence paths:** `artifacts/profile/m4_profile.json`.
  Diagnostic pass, one prompt, a synchronization between every stage, so the
  totals are inflated and are **not** a latency result. What it establishes:
  - A target forward costs about 45 ms at width 1 and about 46 ms at width 9.
    Verification is nearly free on the target side. This is the mechanism that
    makes speculation worth anything on this hardware.
  - A draft forward costs about 11.7 ms: roughly 4x cheaper than the target per
    call, not the 7x the parameter ratio suggests.
  - Commit and crop costs 0.3 to 1.1 ms. Cache bookkeeping is not the cost.
  - The catch-up call is a flat 11.6 ms and is paid on most blocks at the
    74-81% acceptance observed here. It is a real term, not a rounding error.

- **Source commit:** recorded in `artifacts/profile/m4_profile.json`.

- **Known limitations and blockers:**
  - Greedy only. `decode_speculative_greedy` refuses sampled configurations
    rather than silently approximating them.
  - No controller: gamma is fixed for the whole request.
  - The profile is one prompt on one machine with extra synchronizations. It is
    input to the M6 cost model, not a performance claim.
  - The acceptance rate in the profile (74-81%) comes from a single short code
    prompt and will not hold across the M7 cohort.

- **Next concrete step:** sampled speculation and traces for M5.

- **Teach-back explanation prepared:**
  - *Decision:* cap the block length at `remaining - 1` instead of generating a
    full block and truncating whatever overshoots the budget.
  - *Alternative considered:* always draft `gamma`, then cut the emitted tokens
    down to the budget. Simpler, and the output length is identical.
  - *Failure mode:* truncation is a decision made *after* seeing the tokens, so
    it conditions the output on the future. The distribution argument works by
    showing each emitted token is drawn from the target conditional on its own
    prefix; discarding tokens based on what was produced breaks that, and the
    bias would be invisible in greedy mode and only appear in the sampled
    oracle checks in M5.
  - *Evidence:* `test_the_budget_is_never_exceeded_or_undershot` over budgets
    1, 2, 3, 5, 9, 17 crossed with gamma 1, 2, 4, 8 shows the cap alone produces
    exactly the requested length, so no truncation path is ever needed.

- **Ethan's teach-back status:** not yet demonstrated.

---

### M5: Sampled speculation and evidence traces

- **Status:** complete.

- **Implementation and decisions:**
  - `decode_speculative_sampled` delegates candidate choice, acceptance and
    correction to `sampling.speculative_block` -- the function the oracle
    enumerates -- rather than reimplementing it. A thin `_ProposalRecorder`
    observes the proposal stream so the target's row provider can issue its
    single batched forward call once every proposal is drawn, and it asserts
    that ordering instead of assuming it.
  - `traces.py` writes JSONL block traces **after** the timed region and
    replays them offline with no model, GPU or network. The prompt is stored
    as a length and a SHA-256 only, so a trace is shareable without
    redistributing dataset text.
  - `evidence.py` produces the renderer's `evidence.json` by executing the
    named commands and recording exit codes. A category with no checks does not
    pass by default, and a CPU-only run records `gpu_checks_included=false`.
  - One deliberate asymmetry: the greedy engine stops drafting early on a
    proposed stop token, the sampled engine does not. Stopping early in sampled
    mode would mean forking the enumerated function; the cost is a few wasted
    draft calls in a request that is about to end, and it appears in the
    counters.

- **Commands actually run:**
  ```
  python -m pytest tests/unit/test_sampled_speculation.py -q
  python -m pytest tests/unit/test_traces.py tests/unit/test_evidence.py -q
  python -m pytest tests/integration/test_sampled_speculation_gpu.py -q -m gpu
  python -m pytest -m 'not gpu and not download' -q
  python -m pytest -m gpu -q
  python -m switchback trace --mode greedy  --gamma 4 --max-new-tokens 96 ...
  python -m switchback trace --mode sample --gamma 4 --max-new-tokens 96 ...
  python -m switchback replay artifacts/traces/greedy_g4.jsonl
  python -m switchback evidence --gpu --out artifacts/evidence.json
  ```

- **Passed / failed / skipped checks:**
  - 442 CPU tests and 54 GPU tests pass. ruff, format and mypy clean.
  - **The milestone gate holds, and it is exact.** The sampled decoder, driven
    through its own cache ledger, softmax, crop and catch-up, reproduces the
    target sequence distribution with total variation distance **0** on six
    random tree pairs at two draft lengths and six constructed cases at three
    draft lengths. No tolerance is used. Exactness survives the softmax because
    every scripted row is uniform over a subset of size 1, 2 or 4, which FP32
    represents without error.
  - Block length does not change the law: gamma 1 and gamma 2 agree exactly.
  - **Three real bugs found during this milestone, two of them by the replay
    checker rather than by any test:**
    1. `TorchRandomSource.categorical` built its inverse-CDF search value with a
       bare `torch.tensor()`, which defaults to CPU. Every sampled test until
       now had run on CPU, so the first CUDA request was the first failure.
    2. On EOS the target cache was cropped to `len(S) + accepted`, one position
       too long whenever the committed EOS truncated the block. Found by
       replaying a saved trace.
    3. The replay checker itself compared the draft cache against the target's
       boundary, which is wrong after a target-only step. Fixed, and the
       converse rule -- drafting *after* a target-only step, which would
       silently collapse acceptance -- is now asserted in both engines.
  - The mutation check initially failed to detect a residual replaced by an
    argmax, because that tree pair produced only single-token residuals where
    the two coincide. Fixed with constructed cases plus a test that asserts
    multi-token residuals are exercised.
  - A GPU assertion that three seeds give three distinct completions failed
    correctly and was relaxed: at temperature 0.7 on a deterministic code prompt
    two seeds legitimately produce the same 48 tokens.
  - Skipped: nothing.

- **Benchmark or evidence paths:**
  - `artifacts/traces/greedy_g4.jsonl` and `artifacts/traces/sampled_g4.jsonl`.
    46-token prompt, 96-token budget, EOS suppressed. Greedy accepted 73 of 87
    proposals with rejections at all four candidate positions; sampled at
    temperature 0.7 accepted 71 of 90. Both replay clean.
  - `artifacts/evidence.json`: ten executed validation commands, all passing,
    including both GPU suites, plus the measured numerical facts and their
    source tests.
  - **No benchmark has run.** These are correctness artifacts.

- **Source commit:** recorded in each trace header and in `evidence.json`.

- **Known limitations and blockers:**
  - No controller: gamma is fixed for the whole request, and there is no bypass.
  - The exact enumeration uses rows that FP32 represents without error. Real
    softmax outputs are not dyadic, and no exactness claim is made for them.
  - The sampled engine validates every distribution it touches, which the greedy
    engine does not need to. That asymmetry must be disclosed if the two are
    ever timed against each other.
  - `artifacts/evidence.json` records exit codes. It cannot prove the
    acquisition code is honest.

- **Next concrete step:** the measured cost controller for M6.

- **Teach-back explanation prepared:**
  - *Decision:* have the GPU sampled path call the same `speculative_block` the
    oracle enumerates, using a recorder to reconstruct the proposal list, rather
    than writing a second copy of the verification logic inside the decoder.
  - *Alternative considered:* a decoder-local block loop mirroring the greedy
    one, with a test asserting that the two implementations agree.
  - *Failure mode:* two copies drift. The oracle would keep certifying the pure
    function while the GPU path slowly diverged from it, and the divergence
    would show up as a distribution bias that greedy conformance cannot see and
    that only an enumeration of the *decoder* would catch.
  - *Evidence:* `test_sampled_speculation_reproduces_the_target_distribution`
    enumerates the decoder itself, not the pure sampler, and
    `test_a_broken_residual_is_detected_by_this_gate` swaps the residual draw for
    an argmax and requires the gate to fail -- on a pair chosen so the residual
    has three-token support, because the first attempt used a pair where the two
    coincide and the mutation went undetected.

- **Ethan's teach-back status:** not yet demonstrated.

---

### M6: Measured controller and bypass

- **Status:** complete.

- **Implementation and decisions:**
  - `controller.py`: `CostController` choosing from `{0, 1, 2, 4, 8}` by
    estimated cost per committed token, bypassing when no draft length beats
    target-only by the 5% margin. Bypass is sticky and retires the draft cache.
  - Conditional acceptance is **censored**: a block rejected at position 2
    contributes successes at 0 and 1, one failure at 2, and nothing at 3 and
    beyond. Beta(1, 1) smoothing, so an unobserved position returns 0.5 rather
    than 0 or 1.
  - The catch-up forward is in the cost model, weighted by the probability that
    every candidate is accepted, because milestone 4 measured it as a full
    draft call paid on most blocks.
  - Draft prefill is amortized over `min(remaining budget, calibration median
    length)`.
  - Invariant I9 is enforced by the `ControllerState` schema: no field for a
    dataset, prompt id, expected answer, or eventual length. A decision cannot
    depend on information the state cannot carry.
  - `decoder.py` now runs one block loop for every engine. `FixedController(g)`
    gives `fixed_g`, `CostController` gives `adaptive`, so comparing them
    compares policies rather than implementations.
  - `calibration.py` fits and freezes the profile, hashing it and verifying the
    hash on load.
  - `BlockObservation`'s per-stage timings became optional. Whole-block duration
    is free; attributing stages needs a synchronization between each one, and
    instrumentation the controller needs would have to stay inside the
    controller's measured path. Null means unavailable, never zero.

- **Commands actually run:**
  ```
  python -m pytest tests/unit/test_controller.py tests/unit/test_adaptive_engine.py -q
  python -m pytest tests/unit/test_calibration.py -q
  python -m pytest -m 'not gpu and not download' -q
  python -m pytest -m gpu -q
  python -m switchback calibrate --max-new-tokens 96 --gamma 8 --repeats 2
  python -m switchback controller-check --max-new-tokens 96 --repeats 3
  python -m switchback conformance --lengths 32 64 128 256
  python -m ruff check . && python -m ruff format --check . && python -m mypy src/switchback
  ```

- **Passed / failed / skipped checks:**
  - 522 CPU tests and 54 GPU tests pass. ruff, format and mypy clean.
  - Fake cost tables exercise every named regime: high acceptance picks g=8,
    moderate picks 4 then 2, low prefers target-only, a marginal win inside the
    5% margin is refused, an expensive draft never wins, and a target forward
    that scales with width kills long drafts.
  - Budget clipping verified at remaining budgets 2, 3, 5, 9 and 200; a budget
    of 1 leaves no draft length feasible and correctly does **not** trigger
    sticky bypass, because that is a budget fact rather than a policy judgement.
  - Censoring regression: 50 rejections at position 0 leave position 6's
    estimate exactly at the calibration value. The converse is also tested --
    with a thin prior, local evidence does dominate.
  - Metamorphic: two controllers fed identical observations return identical
    decisions across 32 state combinations, and `ControllerState` raises
    `TypeError` if a dataset label is even passed.
  - **Failure found and fixed:** the first conformance measurement omitted the
    EOS mask the engines use and reported a 15.5-logit divergence, which would
    have meant a real decoder bug. With the mask applied the same case is a
    0.125-logit near-tie. Recorded in ADR 0004.
  - **Measurement artifact found and fixed:** `profile_single_forward` had no
    global warmup, so the first width measured also paid one-time allocator and
    kernel setup. That made width 1 read slower than width 2 and would have
    inflated the target-only cost the controller compares against.
  - Skipped: nothing.

- **Benchmark or evidence paths:**
  - `artifacts/calibration.json`: frozen profile, hash
    `b3f6a4c569380df8...`. Conditional acceptance 0.81 to 0.97 by position on
    these prompts. Target forward 52.2 ms at width 1 and 46.8 to 49.0 ms at
    widths 2 to 9. **A width-1 forward is about 10% more expensive than a
    multi-token one on this GB10**, so target-only decoding pays a penalty that
    verification does not.
  - `artifacts/controller_check.json`: in-sample, four prompts, three repeats,
    randomized engine order in one process. Median request latency at 96 fixed
    greedy tokens: `native_ar` 5150 ms, `fixed_1` 3753, `fixed_2` 3070,
    `fixed_4` 2619, `fixed_8` 2520, `adaptive` 2568, `adaptive_no_bypass` 2542.
    **Not a benchmark result:** in-sample, four prompts, no held-out cohort, no
    confidence intervals.
  - `artifacts/conformance.json`: greedy agreement 100% at 32 tokens and 75% at
    64, 128 and 256, largest chosen-token gap 0.25 logits.

- **Source commit:** recorded in each artifact under `source.commit`.

- **The one correct decision, as SPEC.md M6 asks:** on these prompts the
  controller chose `g=8` for 127 of 198 blocks and landed at 2568 ms against a
  best fixed length of 2520 ms -- within 2% of the best fixed length **without
  being told which one it was**, and less than half of `native_ar`. It also
  correctly avoided `g=1` and `g=2`, which are 46% and 20% slower here.

- **The one limitation, as SPEC.md M6 asks:** the controller never bypassed.
  `bypass_fraction` is 0 across every request, because acceptance on these
  prompts is 71 to 94% and bypass would be the wrong call. So this check
  exercises the *selection* logic and not the *bypass* logic, which remains
  covered only by fake cost tables. It also cannot show adaptive beating the
  best fixed length: on a single in-sample workload the best fixed length is
  knowable, and matching it is the ceiling. Any real argument for the controller
  has to come from a cohort where the best fixed length varies, which is exactly
  what M7 is for.

- **Known limitations and blockers:**
  - **Blocking M7:** `scripts/render_results.py` refuses any cell where greedy
    outputs differ. At 256 fixed-length greedy tokens that will refuse roughly a
    quarter of cells. ADR 0004 records three options and recommends one; none is
    implemented.
  - Calibration currently uses the pilot prompts, which are also the check's
    evaluation prompts. That is in-sample by construction and must move to the
    locked calibration split in M7.
  - Per-token cost varies by roughly 10% between processes on this machine.
    Every comparison must run interleaved in one process; cross-process latency
    comparisons are not trustworthy here.
  - The sampled engine has no controller; `adaptive` is greedy only.

- **Next concrete step:** resolve ADR 0004, then build `bench/prepare.py`.

- **Teach-back explanation prepared:**
  - *Decision:* censor candidate positions after the first rejection instead of
    counting them as failures.
  - *Alternative considered:* treat a block that proposed 8 and accepted 2 as
    two successes and six failures. It is one line shorter and the totals still
    look like probabilities.
  - *Failure mode:* those six positions were never evaluated. The block stopped
    at the first rejection, so nothing was ever asked about position 5. Counting
    them as failures drives the estimated acceptance at long positions toward
    zero, which drives `E(g) = 1 + sum_j prod_k a_k` down for large `g`, which
    makes the controller stop choosing long draft lengths. The bias is
    self-reinforcing: shorter blocks produce fewer observations at long
    positions, so the estimates never recover. It would look like a controller
    that had learned something.
  - *Evidence:* `test_repeated_early_rejections_do_not_collapse_long_position_estimates`
    feeds 50 blocks rejected at position 0 and requires position 6's estimate to
    be unchanged to floating-point equality, while position 0's does move;
    `test_local_evidence_dominates_once_calibration_is_thin` shows the converse,
    so the test is not just asserting that nothing ever updates.

- **Ethan's teach-back status:** not yet demonstrated.


---

### M7: Locked benchmark and generated report

- **Status:** in progress. The harness is complete and has produced a real
  validated report end to end. The primary matrix has not run.

- **Implementation and decisions:**
  - **ADR 0005 resolved ADR 0004 first**, before any benchmark ran. The renderer
    refused every greedy mismatch, which at 256 fixed-length tokens would have
    refused about a quarter of cells and produced no report at all. A mismatch is
    now admissible only with evidence that it was a near-tie, under a bound the
    manifest declares; anything else is still refused with the original message.
    A run that declares no bound keeps the old strict behaviour exactly.
  - `bench/prepare.py` locks the workload. Rows are chosen by sorting the
    SHA-256 of `revision + split + row_id`, never by content. Calibration and
    held-out prompts come from different dataset splits. The workload file
    stores prompt ids, token counts and token hashes, never prompt text.
  - `bench/run.py` randomizes engine order inside each prompt/repeat block from
    a recorded seed, resets peak-memory counters per request, records the
    resident floor before any cache exists, keeps failed requests with their
    error text, and resumes from `requests.jsonl`.
  - Greedy divergences are measured **after** the timed region from an uncached
    recomputation with the same EOS mask the engines used, and attached as the
    evidence ADR 0005 requires.
  - `bench/validate.py` adds what the renderer cannot check because it never
    sees the inputs: that the config and workload still hash to what the run
    recorded, and that the expected keys are exactly what they imply. Without
    that last check a run could quietly narrow its own cohort and still report
    `complete: true`.
  - `pyarrow` rather than `datasets`: the revisions are already pinned here, so
    only a parquet reader is needed, and the CPU gate never downloads a dataset.

- **Commands actually run:**
  ```
  python -m bench.prepare  --config configs/smoke.toml
  python -m bench.prepare  --config configs/primary.toml
  python -m bench.prepare  --config configs/natural.toml
  python -m switchback evidence --gpu --out artifacts/evidence.json
  python -m bench.run      --config configs/smoke.toml --out artifacts/runs/smoke
  python -m bench.validate artifacts/runs/smoke --config configs/smoke.toml
  python scripts/render_results.py artifacts/runs/smoke --out artifacts/runs/smoke/report.md
  python -m pytest tests/report -q
  ```

- **Passed / failed / skipped checks:**
  - 552 CPU tests and 54 GPU tests pass. ruff, format and mypy clean.
  - 23 report-integrity tests, of which 13 are new and cover the near-tie
    contract: the strict default, admission within the bound, a gap beyond it,
    missing evidence, evidence pointing at the wrong position, evidence naming
    the wrong tokens, an unusable gap, and a nonsense bound.
  - The smoke run completed 64 of 64 expected requests with 0 errors, validated
    clean, and rendered.
  - **Failure found and fixed:** the `git_dirty` flag counted any tracked change,
    including `artifacts/evidence.json`, which a benchmark rewrites as part of
    running. Every measured run was therefore dirty and unrenderable. The flag
    now covers `src`, `bench`, `scripts`, `configs`, `data` and
    `pyproject.toml`; untracked source files are still caught by
    `source_sha256`.
  - Skipped: nothing.

- **Benchmark or evidence paths:** `artifacts/runs/smoke/` with its
  `manifest.json`, `requests.jsonl`, `evidence.json` and generated `report.md`;
  `artifacts/workloads/{smoke,primary,natural}.json`.

  The smoke cohort is **preliminary by construction**: 8 held-out prompts, one
  repeat, 32 tokens. It exists to prove the pipeline, not to support a latency
  claim, and its cohort is named `smoke` so no table can be read as the primary
  condition. What it does establish:

  - The ADR 0005 contract does real work. 14 of 64 requests diverged from the
    baseline's greedy tokens and **every one was a near-tie**: gaps of 0.0 to
    0.25 logits, median 0.125. Several were exactly 0.0, two tokens with
    identical recomputed FP32 logits that the two kernel paths tie-break
    differently.
  - `hf_dynamic` diverged too, on 3 of 8 prompts. Hugging Face's own assisted
    generation has the same property, which is independent corroboration that
    this is BF16 speculation rather than this implementation.

- **Source commit:** recorded in each run manifest.

- **Known limitations and blockers:**
  - **The primary matrix has not run.** Estimated 7.7 hours for 3,072 requests,
    derived from the smoke run's measured per-request times. No latency claim
    exists and `RESULTS.md` does not exist.
  - Calibration still uses the pilot prompts rather than the locked calibration
    split. It must be refitted before the primary run is published.
  - The natural-stop, sampled, stress and ablation cohorts have configs or
    workloads but have not been run.
  - The controlled-prompt cohort is generated but not yet wired into a config.
  - `hf_ar` and `hf_dynamic` report null for acceptance and target-call
    counters, because an external baseline does not expose them without extra
    instrumentation that would land inside its own measured path.

- **Next concrete step:** refit calibration on the locked calibration split,
  then run `bench.run --config configs/primary.toml`.

- **Teach-back explanation prepared:**
  - *Decision:* let the renderer admit a greedy mismatch when the run proves it
    was a near-tie, instead of either refusing every mismatch or dropping the
    check.
  - *Alternative considered:* compare only the matching prefix and report the
    median matched length. One line, and every cohort renders.
  - *Failure mode:* that converts a measurable failure into an invisible one.
    "The first 54 tokens matched" reads like a stronger claim than "75% of
    requests matched exactly", and it is weaker. It would also silently absorb a
    real bug: a cache off-by-one produces a divergence at a confident position,
    and a prefix comparison reports that identically to a floating-point tie.
  - *Evidence:* `test_a_gap_beyond_the_bound_is_refused` uses a 15.5-logit gap,
    which is the number the first conformance measurement produced before its
    own bug was fixed, and requires the renderer to refuse it;
    `test_a_mismatch_without_evidence_is_refused` and
    `test_evidence_pointing_at_the_wrong_position_is_refused` close the routes
    by which a run could claim a near-tie it did not measure.

- **Ethan's teach-back status:** not yet demonstrated.


---

### M7 outcome: the primary matrix

- **Status:** complete. 3,072 requests, 0 errors, one process, 7.6 hours
  (05:19 to 12:54 local). Validated and rendered.

- **Headline, as measured:** `adaptive` reaches **2.021x [1.966, 2.070]** against
  `hf_ar`. It is **beaten by `fixed_8` (2.066x) and by `hf_dynamic` (2.234x)**.
  The ordering `hf_dynamic > fixed_8 > adaptive` holds on MBPP (2.453 / 2.215 /
  2.144) and GSM8K (2.034 / 1.927 / 1.905) separately, so it is not a pooling
  artifact. This is the risk SPEC.md section 11 named as "controller fails to
  beat HF dynamic", and its documented fallback applies: keep the controller as
  an ablation and promote the measured study, not an unearned speed claim.

- **Why the controller lost:**
  1. Acceptance is high and flat across this cohort (0.91 at candidate position
     1, 0.66 at position 8). When the best fixed length is the same for nearly
     every prompt, varying it can only cost. That is a limitation of the cohort
     as much as of the controller, and it is not a defence: a fair test needs a
     workload where the best length varies, which M8's natural-stop and
     context-stress cohorts might supply.
  2. `hf_dynamic` stops drafting **within** a block on the draft's own
     confidence. Switchback's controller chooses `g` before the block from past
     acceptance only. Confidence-based early stopping is compatible with the
     distribution argument -- the draft's confidence is available before
     verification -- so not implementing it is a real gap, not a constraint.

- **What is established:** `native_ar` is 0.999x of `hf_ar`, so Switchback's own
  loop has no measurable overhead against the reference. The slowdown fraction
  is 0.0% for every engine. `hf_dynamic` pays for its win in TTFT: 151 ms median
  against 49 ms for everything else.

- **Failure found and fixed:** the run was refused by the renderer. 18 of 857
  divergences exceeded the 0.5 near-tie bound, all on two positions, hit
  deterministically across engines and repeats. Investigation showed the screen
  measured the wrong thing: at `mbpp_heldout_169` index 46 the engine's own
  replayed cache gives a margin of 0.25 toward the token it chose, while every
  freshly-prefilled path gives 0.5 the other way; at `gsm8k_heldout_251` index 72
  every reproducible path prefers the token `hf_dynamic` chose and the
  **baseline** is the outlier. Resolved by ADR 0006 with a second adjudication
  tier rather than by moving the bound. All 18 adjudicated `near_tie`, 0
  unexplained.

- **Commands actually run:**
  ```
  python -m switchback calibrate --workload artifacts/workloads/primary.json --max-prompts 32
  python -m switchback evidence --gpu --out artifacts/evidence.json
  python -m bench.run --config configs/primary.toml --out artifacts/runs/primary
  python -m bench.adjudicate artifacts/runs/primary --config configs/primary.toml
  python -m bench.validate artifacts/runs/primary --config configs/primary.toml
  python scripts/render_results.py artifacts/runs/primary --out RESULTS.md
  ```

- **Benchmark or evidence paths:** `RESULTS.md`, `artifacts/runs/primary/`
  (`manifest.json`, `requests.jsonl`, `evidence.json`, `adjudication.jsonl`),
  `artifacts/calibration.json` (fitted on the locked calibration split, 354
  observations at position 0).

- **Known limitations and blockers:**
  - Only the primary cohort has run. Natural-stop, sampled, context-stress and
    the no-bypass ablation are configured but not measured.
  - The bypass path still has no real-data evidence: 0 bypasses in 384 adaptive
    requests, correctly.
  - Greedy conformance is 61.7 to 64.1%, not 100%.
  - The 272 rows from the bounded first chunk were discarded rather than
    resumed, so the whole cohort was measured in one process. They are in the
    session scratchpad, not the repository.

- **Teach-back explanation prepared:**
  - *Decision:* when the primary run was refused for 18 over-bound divergences,
    investigate the measurement rather than raise the bound.
  - *Alternative considered:* set `max_chosen_token_gap` to 2.0. Both cases
    pass, the report renders, thirty seconds of work.
  - *Failure mode:* the bound is a claim about floating point, not a tolerance
    for wrong answers. At 2.0 it would also admit a genuine cache fault at any
    position where the model is confident by less than 2 logits, which is most
    of them. Moving a threshold because a run failed is the move ADR 0005
    exists to prevent, and the run would then have shipped with the failure
    silently absorbed.
  - *Evidence:* the investigation found the screen was comparing against a state
    no engine was in -- the engine's own replayed cache disagrees with every
    freshly-prefilled path at `mbpp_heldout_169` index 46 -- and at
    `gsm8k_heldout_251` index 72 the baseline is the outlier, not the candidate.
    `bench/adjudicate.py` plus ten tests in
    `tests/report/test_conformance_contract.py` pin the new contract, including
    that a `near_tie` verdict contradicting its own margin is refused.

- **Ethan's teach-back status:** not yet demonstrated.


---

### M8: Reviewer demo and remaining cohorts

- **Status:** complete for the declared scope. The sampled and context-stress
  cohorts are configured but not run, and are named as unrun everywhere they
  would otherwise be assumed.

- **Implementation and decisions:**
  - `viewer/` is static HTML, CSS and JavaScript: no server, no network, no
    build step, no inference. It reimplements `replay_trace` rather than
    trusting its input, and `tests/unit/test_viewer.py` drives the JavaScript
    through node and requires both implementations to agree on the real traces
    and on four kinds of deliberate damage.
  - `scripts/render_trace_gif.py` animates a **real** saved trace and refuses
    one that fails its replay check, so the GIF cannot drift from what the
    engine did. Text labels carry the meaning; colour repeats it.
  - `docs/failure-analysis.md` is the post-mortem.
  - `configs/natural.toml` gained `adaptive_no_bypass`. The ablation belongs
    there rather than in the primary cohort, where the controller never bypassed
    and the two engines were provably identical.

- **Commands actually run:**
  ```
  python scripts/render_trace_gif.py artifacts/traces/greedy_g4.jsonl --out docs/trace.gif
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python -m switchback demo
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python -m pytest -m 'not gpu and not download' -q
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python -m switchback replay artifacts/traces/greedy_g4.jsonl
  python -m bench.prepare   --config configs/natural.toml
  python -m bench.run       --config configs/natural.toml --out artifacts/runs/natural
  python -m bench.adjudicate artifacts/runs/natural --config configs/natural.toml
  python -m bench.validate   artifacts/runs/natural --config configs/natural.toml
  python scripts/render_results.py artifacts/runs/natural --out artifacts/runs/natural/report.md
  ```

- **Passed / failed / skipped checks:**
  - 570 CPU tests and 54 GPU tests pass; ruff, format and mypy clean.
  - The fresh-CPU acceptance criterion holds: with `HF_HUB_OFFLINE=1` and no
    visible GPU, the demo, the whole CPU suite and a trace replay all run with
    no weights and no network.
  - The natural-stop cohort completed 3,456 of 3,456 requests with 0 errors in
    4.4 hours, one process.
  - The renderer refused it for one over-bound divergence -- the same
    `mbpp_heldout_169` index 46 position as the primary run -- and named the
    command to fix it. 21 adjudicated `near_tie`, 0 unexplained.

- **Benchmark or evidence paths:** `artifacts/runs/natural/` (manifest,
  requests, evidence, adjudication, report), `docs/trace.gif`, `viewer/`.

- **The natural-stop result:** the ordering is unchanged. Paired,
  `adaptive` is 0.988x [0.979, 0.996] of `fixed_8` and 0.939x [0.923, 0.955] of
  `hf_dynamic`. **Bypass never fired: 0 of 10,704 controller decisions**, on
  completions as short as 17 tokens, because the draft prefill costs about 14 ms
  against a 50 ms target forward and is recovered almost immediately. On
  completions of 24 tokens or fewer, `hf_ar` takes 1,080 ms and `adaptive` 679
  ms. `adaptive_no_bypass` is 0.996x [0.989, 1.005] of `adaptive`, which is what
  two provably identical engines should look like and is a useful read on the
  measurement floor.

- **Known limitations and blockers:**
  - Sampled and context-stress cohorts unrun. Context stress is the one that
    could still change the controller's verdict.
  - The bypass logic has no real-data exercise and, on this hardware, cannot get
    one: no request ever made bypass the right call.
  - Greedy conformance is 71.1-74.2% on natural stop and 61.7-64.1% on primary.

- **Next concrete step:** the context-stress cohort, then confidence-based early
  stopping inside a block.

- **Teach-back explanation prepared:**
  - *Decision:* run the natural-stop cohort with the no-bypass ablation, even
    though the primary run already showed the controller losing.
  - *Alternative considered:* stop after the primary cohort. The headline was
    already settled and another 4.4 hours was unlikely to reverse it.
  - *Failure mode:* it would have left the project's most distinctive feature --
    sticky bypass -- in the worst possible state: implemented, described as a
    contribution, and never exercised on real data. A reviewer asking "when does
    it bypass?" would have got a fake-cost-table answer. Running it converted an
    untested path into a measured negative result with a mechanism behind it: a
    14 ms draft prefill against a 50 ms target forward is recovered almost
    immediately, so bypass is never right here.
  - *Evidence:* 0 bypasses in 10,704 decisions; `adaptive_no_bypass` 0.996x
    [0.989, 1.005] of `adaptive`, an interval straddling 1, which is the
    signature of two engines making identical decisions.

- **Ethan's teach-back status:** not yet demonstrated.

## Starter-tool verification

The supplied renderer's 10 integrity tests pass under pytest
(`python -m pytest tests/report`), using explicitly synthetic timing fixtures in
temporary directories. Long lines in `scripts/render_results.py` were wrapped to
satisfy the lint gate; the rendered output was verified byte-identical to the
original before that change was committed. These checks are not inference tests
and not performance measurements.

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
