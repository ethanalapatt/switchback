# ADR 0001: Qwen3 pair pinned to resolved commit SHAs

Date: 2026-09-12. Status: accepted. Milestone: M1.

## Context

SPEC.md section 2 requires `Qwen/Qwen3-4B` as target and `Qwen/Qwen3-0.6B` as
draft, both BF16 and resident, with tokenizer compatibility proven rather than
assumed. It also forbids `main` or `latest` in a published reproduction command.

## Decision

`python -m switchback resolve-models` resolves each repository to a 40-character
commit SHA through the Hub API and writes `data/manifest.json` with the license
and the SHA-256 of every metadata file. `ModelSpec.__post_init__` rejects the
strings `""`, `main`, `master` and `latest` outright, so a mutable revision
cannot reach a model load even by accident.

Resolved on 2026-09-12:

| Role | Repository | Revision |
|---|---|---|
| Target | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Draft | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |

Compatibility is checked on four axes rather than one: the full token-to-ID map,
the added-token table, the special token IDs, and the encodings of 24 probe
strings. The logits vocabulary (151,936) is compared separately from the
tokenizer vocabulary (151,669), because the logits axis is the one `p/q` is
computed over and the two numbers genuinely differ.

## Alternatives considered

Comparing `config.vocab_size` alone. Rejected: both configs say 151,936 while
still permitting different surface-to-ID assignments, and a permutation would
surface only as wrong output text, long after the cache work looked correct.
`test_conflicting_id_for_a_shared_token_is_detected` encodes that failure mode.

Recording the local snapshot path in the manifest. Rejected: machine specific,
and it would make a committed manifest unportable.

## Failure mode this prevents

A draft that tokenizes `"café"` differently from the target proposes IDs the
target reads as different text. Acceptance rates stay plausible, greedy equality
fails intermittently, and the cause looks like a cache bug.

## Evidence

`tests/unit/test_tokenizer_compatibility.py`, `tests/property/test_tokenizer_properties.py`,
and `tests/integration/test_real_models_gpu.py::test_tokenizer_parity_holds_for_the_real_pair`.
