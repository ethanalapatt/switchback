# ADR 0003: baselines run on a fully explicit generation configuration

Date: 2026-09-12. Status: accepted. Milestone: M1.

## Context

CLAUDE.md forbids inheriting hidden model generation defaults. Three separate
mechanisms in Transformers 4.57.1 reintroduce them, and each was observed
directly while building the milestone 1 pilot.

1. `GenerationConfig()` defaults to `top_k=50`. A "temperature only" sampled run
   would in fact be top-50 truncated, which is a different target distribution
   from the one the oracle in M2 will enumerate.
2. Qwen3 ships its own `generation_config.json` with `top_k`, `top_p` and
   `temperature`. `generate(..., use_model_defaults=True)` copies every such
   value over an unset field.
3. Even with `use_model_defaults=False`, `_prepare_generation_config` still
   falls back to `model.generation_config` for `bos`, `eos`, `pad` and
   `decoder_start`. Qwen3 sets `eos_token_id=[151645, 151643]`, so the
   fixed-length condition silently regained stop tokens: the observed
   `_eos_token_tensor` was `[151645, 151643]` despite passing `eos_token_id=None`.

A fourth interaction is specific to assisted generation. `AssistedCandidateGenerator`
reads `num_assistant_tokens` and `assistant_confidence_threshold` from the
*assistant* model's `generation_config`, not from the configuration passed to
`generate()`, and writes a tuned threshold back to it.

## Decision

`explicit_generation_config` states every sampling, truncation, penalty, beam and
output flag. Before each request both models' `generation_config` attributes are
swapped for the explicit configuration and restored afterwards, so the fallback
in mechanism 3 resolves to our values and the assistant options in mechanism 4
land where Transformers actually reads them.

EOS suppression is implemented purely by removing stop tokens. A non-`None`
`min_new_tokens` makes Transformers derive `min_length = min_new_tokens +
prompt_length`, which installs a minimum-length logits processor that assisted
generation rejects with `ValueError`. Requiring one would have silently excluded
`hf_dynamic` from the matched comparison. A post-check instead fails loudly if a
suppressed-EOS request stops short of its budget.

## Alternatives considered

Mutating `model.generation_config` once at load time. Rejected: it leaks across
requests and across engines, which invariant I10 forbids.

Accepting whatever length each engine produced. Rejected: unequal output lengths
would confound every latency comparison in section 9.3.

## Failure mode this prevents

`hf_ar` stopping at 12 tokens on one prompt while `hf_dynamic` runs to 256 on
the same prompt, then reporting the ratio as a speedup.

## Evidence

`tests/unit/test_generation_config.py`;
`tests/integration/test_real_models_gpu.py::test_suppressed_eos_produces_the_exact_budget_for_both_engines`
and `::test_the_assistant_generation_config_is_restored_after_a_request`.
