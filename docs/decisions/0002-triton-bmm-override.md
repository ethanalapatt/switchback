# ADR 0002: deregister torch's Triton `bmm` override on this Spark

Date: 2026-09-12. Status: accepted, reversible. Milestone: M1.

## Context

torch 2.14 routes some aten operators to Triton kernels through
`torch._native.registry`. On `gigi-spark` the Triton runtime cannot build its
CUDA driver shim, because CPython's development headers are not installed:

```
triton/backends/nvidia/driver.c:9:10: fatal error: Python.h: No such file or directory
```

The first `aten::bmm` therefore raised `CalledProcessError` rather than running.
Qwen3 hits it immediately, in the rotary embedding's
`inv_freq_expanded.float() @ position_ids_expanded.float()`.

Exactly one operator is affected: `torch._native.registry.get_dsl_operations("triton")`
returns `['bmm']`. Linear layers use `mm`/`addmm` and attention uses SDPA, so the
blast radius is one small matmul per forward call.

## Decision

`switchback.runtime.configure_torch_native_overrides()` probes explicitly for the
headers, and on failure calls `registry.deregister_op_overrides(disable_dsl_names="triton")`
so `bmm` falls back to the stock aten kernel. The probe result, the list of
deregistered operators, and the reason are written into the doctor report, the
model check, and the pilot artifact, so no measurement exists without naming the
kernel path that produced it.

The header check runs before the build attempt so the recorded reason is short
and stable. Using the `CalledProcessError` text would embed a temporary
directory name and make two otherwise identical runs differ.

## Alternatives considered

Installing `python3-dev`. This is the better fix and restores stock routing with
one command, but it is a system change outside this project's remit, so it is
recommended in the README rather than performed. The probe re-enables stock
routing automatically once the headers exist.

Catching the exception at the call site. Rejected: it would leave the fallback
implicit and unrecorded, which is what the project's "no hidden kernel path"
rule exists to prevent.

Pinning an older torch. Rejected: CLAUDE.md requires respecting the working
Spark torch installation.

## Failure mode this prevents

A benchmark comparing a Triton-routed `bmm` on one machine against an aten
`bmm` on another, reported as a decoding result.

## Evidence

`artifacts/environment.json` and `artifacts/pilot/pilot.json` both carry
`torch.native_overrides`. `python -m switchback doctor` prints the check.
