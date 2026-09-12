# Switchback progress

Updated: September 12, 2026.

## Current state

Milestones 1 and 2 are complete. The execution machine is the DGX Spark itself
(`gigi-spark`, NVIDIA GB10, aarch64, driver 580.142, CUDA 13.0, torch
2.14.0+cu130), so the GPU gates in M1 actually ran rather than being deferred.
The pinned Qwen3 pair loads fully resident, passes tokenizer parity, and both
Hugging Face reference baselines produce identical greedy token IDs on a
32-token smoke request.

The sampling core and the independent exact oracle are implemented and
exhaustively enumerated against each other. There is still no cached engine and
no model-driven decoding: the sampler is model-free and is driven by row
callbacks, so M3 has to supply those rows from a real KV cache.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M1 Hardware and baseline | **Complete** | `artifacts/environment.json`, `artifacts/model_check.json`, `artifacts/pilot/pilot.json`, 11 GPU tests |
| M2 Sampling oracle | **Complete** | `docs/correctness.md`, 38 oracle tests, 23 property tests |
| M3 Cached target-only engine | Not started | None |
| M4 Fixed greedy speculation | Not started | None |
| M5 Sampled speculation and traces | Not started | None |
| M6 Cost controller | Not started | None |
| M7 Benchmark and report | Not started | Renderer starter file only |
| M8 Reviewer demo | Not started | None |

## Next action

Begin M3: implement `src/switchback/cache.py` (the pending-token cache ledger)
and the Qwen adapter's `forward`/`crop`/`cache_length`, then a native
target-only greedy and sampling engine in `src/switchback/decoder.py`. Add
cache/no-cache differential tests before any speculation. Validate in FP32 on
CPU with the tiny fixture first, then BF16 on the GPU; do not widen tolerances
to make a failing output comparison pass.

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
