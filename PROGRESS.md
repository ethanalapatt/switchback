# Switchback progress

Updated: September 12, 2026.

## Current state

Milestone 1 is complete. The execution machine is the DGX Spark itself
(`gigi-spark`, NVIDIA GB10, aarch64, driver 580.142, CUDA 13.0, torch
2.14.0+cu130), so the GPU gates in M1 actually ran rather than being deferred.
The pinned Qwen3 pair loads fully resident, passes tokenizer parity, and both
Hugging Face reference baselines produce identical greedy token IDs on a
32-token smoke request.

No Switchback decoding exists yet. The next milestone implements the sampling
oracle and the rejection-correction mathematics, entirely on CPU.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M1 Hardware and baseline | **Complete** | `artifacts/environment.json`, `artifacts/model_check.json`, `artifacts/pilot/pilot.json`, 11 GPU tests |
| M2 Sampling oracle | Not started | None |
| M3 Cached target-only engine | Not started | None |
| M4 Fixed greedy speculation | Not started | None |
| M5 Sampled speculation and traces | Not started | None |
| M6 Cost controller | Not started | None |
| M7 Benchmark and report | Not started | Renderer starter file only |
| M8 Reviewer demo | Not started | None |

## Next action

Begin M2: implement `src/switchback/sampling.py` (FP32 probability transforms,
proposal draw, acceptance test, residual and bonus draws) and
`src/switchback/oracle.py` (independent FP64 exact enumeration over rational
probabilities). No GPU is needed. Do not let the oracle import the production
sampler.

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
