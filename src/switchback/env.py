"""Environment doctor.

Reports what this machine actually is. It never infers hardware from a
specification document, and it records ``null`` for a counter the platform does
not expose rather than substituting zero (SPEC.md section 9.4).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from switchback.provenance import collect_source_provenance
from switchback.runtime import configure_torch_native_overrides, deterministic_runtime
from switchback.types import MAX_CONTEXT_TOKENS

CheckStatus = Literal["pass", "fail", "unavailable"]

# Packages whose exact versions are part of the reproduction contract.
LOCKED_DISTRIBUTIONS: tuple[str, ...] = (
    "torch",
    "transformers",
    "tokenizers",
    "numpy",
    "safetensors",
    "huggingface-hub",
    "triton",
)


@dataclass(frozen=True)
class Check:
    """One environment requirement and what was actually observed."""

    name: str
    status: CheckStatus
    detail: str
    required: bool


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in LOCKED_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_smi() -> dict[str, Any]:
    """Driver-level facts from ``nvidia-smi``; all ``None`` when it is absent."""
    unavailable: dict[str, Any] = {
        "available": False,
        "driver_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "temperature_c": None,
        "power_draw_w": None,
    }
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return unavailable
    query = "driver_version,name,temperature.gpu,power.draw"
    try:
        result = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return unavailable
    if result.returncode != 0 or not result.stdout.strip():
        return unavailable
    fields = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 4:
        return unavailable

    def number(raw: str) -> float | None:
        try:
            return float(raw)
        except ValueError:
            # "Not Supported" / "N/A" are real answers on this platform.
            return None

    return {
        "available": True,
        "driver_version": fields[0] or None,
        "cuda_version": None,
        "gpu_name": fields[1] or None,
        "temperature_c": number(fields[2]),
        "power_draw_w": number(fields[3]),
    }


def _torch_facts() -> dict[str, Any]:
    """Torch build and device facts. ``importable`` is False on a torch-less CPU box."""
    facts: dict[str, Any] = {
        "importable": False,
        "version": None,
        "cuda_build_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "device_capability": None,
        "total_memory_bytes": None,
        "bf16_supported": None,
        "sdpa_available": None,
        "import_error": None,
    }
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised only without torch
        facts["import_error"] = f"{type(error).__name__}: {error}"
        return facts
    facts["importable"] = True
    facts["version"] = torch.__version__
    facts["cuda_build_version"] = torch.version.cuda
    facts["sdpa_available"] = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    if not torch.cuda.is_available():
        return facts
    facts["cuda_available"] = True
    facts["device_count"] = torch.cuda.device_count()
    properties = torch.cuda.get_device_properties(0)
    facts["device_name"] = properties.name
    facts["device_capability"] = [properties.major, properties.minor]
    facts["total_memory_bytes"] = int(properties.total_memory)
    facts["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    facts["native_overrides"] = configure_torch_native_overrides().as_dict()
    facts["deterministic_knobs"] = deterministic_runtime()
    return facts


def _build_checks(torch_facts: dict[str, Any], versions: dict[str, str | None]) -> list[Check]:
    checks: list[Check] = []
    python_ok = sys.version_info >= (3, 11)
    checks.append(
        Check(
            "python_version",
            "pass" if python_ok else "fail",
            f"{platform.python_version()} (require >= 3.11)",
            required=True,
        )
    )
    checks.append(
        Check(
            "torch_importable",
            "pass" if torch_facts["importable"] else "fail",
            str(torch_facts["version"] or torch_facts["import_error"]),
            required=True,
        )
    )
    for name in ("transformers", "tokenizers", "numpy", "safetensors", "huggingface-hub"):
        present = versions.get(name) is not None
        checks.append(
            Check(
                f"package_{name.replace('-', '_')}",
                "pass" if present else "fail",
                f"{name}=={versions.get(name)}",
                required=True,
            )
        )
    checks.append(
        Check(
            "cuda_available",
            "pass" if torch_facts["cuda_available"] else "unavailable",
            torch_facts["device_name"] or "no CUDA device visible to torch",
            required=False,
        )
    )
    if torch_facts["cuda_available"]:
        checks.append(
            Check(
                "bf16_supported",
                "pass" if torch_facts["bf16_supported"] else "fail",
                f"torch.cuda.is_bf16_supported()={torch_facts['bf16_supported']}",
                required=False,
            )
        )
        total = torch_facts["total_memory_bytes"] or 0
        # Both BF16 checkpoints plus caches must be resident with no CPU offload.
        enough = total >= 24 * 1024**3
        checks.append(
            Check(
                "device_memory",
                "pass" if enough else "fail",
                f"{total / 1024**3:.1f} GiB visible (require >= 24 GiB for the 4B/0.6B pair)",
                required=False,
            )
        )
    native = torch_facts.get("native_overrides")
    if native is not None:
        disabled = native["disabled_ops"]
        checks.append(
            Check(
                "torch_native_triton_overrides",
                "pass" if native["triton_overrides_available"] else "unavailable",
                (
                    "stock Triton aten routing active"
                    if native["triton_overrides_available"]
                    else f"deregistered {list(disabled)}: {native['reason']}"
                ),
                required=False,
            )
        )
    sdpa = torch_facts["sdpa_available"]
    checks.append(
        Check(
            "attention_backend_sdpa",
            "pass" if sdpa else ("unavailable" if sdpa is None else "fail"),
            "torch.nn.functional.scaled_dot_product_attention",
            required=False,
        )
    )
    return checks


@dataclass(frozen=True)
class EnvironmentReport:
    """Everything the doctor observed, plus the pass/fail verdict."""

    generated_at: str
    max_context_tokens: int
    host: dict[str, Any]
    python: dict[str, Any]
    packages: dict[str, str | None]
    torch: dict[str, Any]
    driver: dict[str, Any]
    source: dict[str, Any]
    checks: list[dict[str, Any]] = field(default_factory=list)
    gpu_ready: bool = False
    ok: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def collect_environment(require_gpu: bool = False) -> EnvironmentReport:
    """Inspect this machine. ``require_gpu`` escalates a missing GPU to a failure."""
    torch_facts = _torch_facts()
    versions = _distribution_versions()
    checks = _build_checks(torch_facts, versions)
    if require_gpu:
        checks = [
            Check(check.name, check.status, check.detail, required=True)
            if check.name in {"cuda_available", "bf16_supported", "device_memory"}
            else check
            for check in checks
        ]
    provenance = collect_source_provenance()
    ok = all(check.status == "pass" for check in checks if check.required)
    gpu_ready = (
        all(
            check.status == "pass"
            for check in checks
            if check.name in {"cuda_available", "bf16_supported", "device_memory"}
        )
        and torch_facts["cuda_available"]
    )
    return EnvironmentReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        max_context_tokens=MAX_CONTEXT_TOKENS,
        host={
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
        },
        python={
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        packages=versions,
        torch=torch_facts,
        driver=_nvidia_smi(),
        source={
            "commit": provenance.commit,
            "dirty": provenance.dirty,
            "source_sha256": provenance.source_sha256,
            "source_file_count": provenance.file_count,
        },
        checks=[asdict(check) for check in checks],
        gpu_ready=bool(gpu_ready),
        ok=ok,
    )


def write_environment(report: EnvironmentReport, out: Path) -> None:
    """Write the report atomically so a partial file is never left behind."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    os.replace(temporary, out)


def format_environment(report: EnvironmentReport) -> str:
    """Human-readable doctor summary for the terminal."""
    lines = [
        f"switchback doctor  {report.generated_at}",
        f"  host      {report.host['hostname']} {report.host['system']} "
        f"{report.host['release']} {report.host['machine']}",
        f"  python    {report.python['version']} ({report.python['executable']})",
        f"  torch     {report.torch['version']} cuda_build={report.torch['cuda_build_version']}",
        f"  device    {report.torch['device_name'] or 'none'} "
        f"driver={report.driver['driver_version']}",
        "",
    ]
    for check in report.checks:
        mark = {"pass": "PASS", "fail": "FAIL", "unavailable": "N/A "}[check["status"]]
        flag = "required" if check["required"] else "optional"
        lines.append(f"  [{mark}] {check['name']:<28} {flag:<8} {check['detail']}")
    lines.append("")
    lines.append(f"  gpu_ready={report.gpu_ready}  ok={report.ok}")
    return "\n".join(lines)
