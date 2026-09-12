"""Device, dtype, and attention-backend validation before any weights load."""

from __future__ import annotations

import pytest
import torch

from switchback.models.qwen import SUPPORTED_DTYPES, assert_resident, validate_spec
from switchback.types import ConfigError, ModelSpec

REVISION = "0" * 40


def spec(**overrides: str) -> ModelSpec:
    base = {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": REVISION,
        "dtype": "bfloat16",
        "device": "cpu",
        "attn_implementation": "sdpa",
    }
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


def test_cpu_spec_is_valid() -> None:
    validate_spec(spec())


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(ConfigError, match="dtype"):
        validate_spec(spec(dtype="int4"))
    for name in SUPPORTED_DTYPES:
        validate_spec(spec(dtype=name))


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ConfigError, match="device"):
        validate_spec(spec(device="tpu"))
    with pytest.raises(ConfigError, match="device"):
        validate_spec(spec(device="mps"))


def test_unknown_attention_backend_is_rejected() -> None:
    with pytest.raises(ConfigError, match="attn_implementation"):
        validate_spec(spec(attn_implementation="paged"))


def test_cuda_index_beyond_the_visible_devices_is_rejected() -> None:
    index = torch.cuda.device_count() if torch.cuda.is_available() else 0
    with pytest.raises(ConfigError, match="cuda"):
        validate_spec(spec(device=f"cuda:{index + 1}"))


class _Offloaded(torch.nn.Module):
    """A module that claims to be dispatched with layers on the CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.hf_device_map = {"model.layers.0": 0, "model.layers.1": "cpu"}


def test_offloaded_model_is_refused() -> None:
    with pytest.raises(ConfigError, match="offloaded"):
        assert_resident(_Offloaded(), "cpu")


def test_misplaced_tensor_is_reported_with_its_device() -> None:
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ConfigError, match="cpu"):
        assert_resident(module, "cuda")


def test_resident_cpu_model_passes() -> None:
    assert_resident(torch.nn.Linear(2, 2), "cpu")
