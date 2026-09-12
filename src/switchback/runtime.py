"""Explicit torch runtime configuration.

torch 2.14 routes some aten operators to Triton kernels through
``torch._native.registry``. On this DGX Spark the Triton runtime cannot build
its CUDA shim because the CPython development headers are absent, so the first
``aten::bmm`` raises ``CalledProcessError`` instead of running.

The project's rule is that a kernel path is never chosen implicitly (SPEC.md
section 2). So the probe below is run explicitly, its outcome is recorded in
every artifact, and the fallback is named rather than silently taken. Install
the headers (``sudo apt install python3-dev``) to restore the stock routing.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRITON_DSL_NAME = "triton"


@dataclass(frozen=True)
class TorchNativeStatus:
    """What happened to torch's Triton-backed operator overrides."""

    probed: bool
    triton_overrides_available: bool
    disabled_dsl: tuple[str, ...]
    disabled_ops: tuple[str, ...]
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _triton_overridden_ops() -> tuple[str, ...]:
    try:
        from torch._native import registry
    except ImportError:
        return ()
    try:
        return tuple(registry.get_dsl_operations(TRITON_DSL_NAME))
    except Exception:  # pragma: no cover - torch build without the registry
        return ()


def _triton_runtime_builds() -> tuple[bool, str | None]:
    """Can Triton build its CUDA driver shim on this machine?

    The header check comes first so the common failure produces a short, stable,
    actionable reason instead of a gcc command line containing a temporary path,
    which would make the recorded reason differ between otherwise equal runs.
    """
    import sysconfig

    include = sysconfig.get_paths().get("include")
    if include is None or not (Path(include) / "Python.h").is_file():
        return False, (
            f"CPython development headers absent ({include}/Python.h); "
            f"install python3-dev to restore the stock Triton routing"
        )
    try:
        from triton.backends.nvidia.driver import CudaUtils
    except Exception as error:
        return False, f"triton driver import failed: {type(error).__name__}"
    try:
        CudaUtils()
    except Exception as error:
        return False, f"triton CUDA shim failed to build: {type(error).__name__}"
    return True, None


def configure_torch_native_overrides(force_disable_triton: bool = False) -> TorchNativeStatus:
    """Probe Triton once and deregister its aten overrides if it cannot run.

    Idempotent. Returns a record intended to be written into the environment
    report, the pilot artifact, and the benchmark manifest so a reviewer can see
    which kernel path produced a measurement.
    """
    import torch  # noqa: F401  (ensures torch._native has been imported)

    overridden = _triton_overridden_ops()
    if not overridden:
        return TorchNativeStatus(
            probed=True,
            triton_overrides_available=False,
            disabled_dsl=(),
            disabled_ops=(),
            reason="torch build registers no Triton aten overrides",
        )
    if force_disable_triton:
        usable, reason = False, "disabled by explicit request"
    else:
        usable, reason = _triton_runtime_builds()
    if usable:
        return TorchNativeStatus(
            probed=True,
            triton_overrides_available=True,
            disabled_dsl=(),
            disabled_ops=(),
            reason=None,
        )
    from torch._native import registry

    registry.deregister_op_overrides(disable_dsl_names=TRITON_DSL_NAME)
    return TorchNativeStatus(
        probed=True,
        triton_overrides_available=False,
        disabled_dsl=(TRITON_DSL_NAME,),
        disabled_ops=overridden,
        reason=reason,
    )


def deterministic_runtime() -> dict[str, Any]:
    """Pin the knobs that decide whether two identical runs agree.

    TF32 is switched off so matmuls keep full FP32 inputs; a silently reduced
    precision would be indistinguishable from a decoding bug in the milestone 3
    cache/no-cache differential tests.
    """
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return {
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
