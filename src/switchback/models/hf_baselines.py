"""Named Hugging Face baseline engines.

``.generate()`` is permitted **only** here (CLAUDE.md operating rules). These are
the external reference points ``hf_ar`` and ``hf_dynamic`` from SPEC.md section
9.2; Switchback's own decoding never routes through this module.

Every generation parameter is stated explicitly. ``GenerationConfig`` ships with
``top_k=50``, and ``generate(use_model_defaults=True)`` would additionally pull
the checkpoint's ``generation_config.json`` (Qwen3 sets ``top_k``, ``top_p`` and
``temperature`` there). Both are overridden so a baseline samples from the same
specified distribution as the Switchback engine.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any

from switchback.types import ConfigError, DecodeConfig

# SPEC.md section 9.2: hf_dynamic's locked options.
DYNAMIC_CONFIDENCE_THRESHOLD = 0.4
DYNAMIC_MAX_DRAFT_TOKENS = 8
DYNAMIC_SCHEDULE = "constant"


@dataclass(frozen=True)
class BaselineResult:
    """One completed baseline request.

    ``token_release_ns`` holds the host time at which each committed token first
    became observable to a consumer. Assisted generation releases a whole
    verified block at once, so several entries can share a timestamp; they are
    recorded as measured rather than spread evenly across the block.
    """

    engine: str
    output_ids: tuple[int, ...]
    token_release_ns: tuple[int, ...]
    start_ns: int
    first_token_ns: int
    last_token_ns: int
    end_ns: int
    prompt_tokens: int
    termination: str
    generation_config: dict[str, Any]
    counters: dict[str, int | None] = field(default_factory=dict)


class _ReleaseTimer:
    """Streamer that timestamps each released token batch.

    Transformers calls ``put`` with the prompt first, then once per decoding
    step. The value is moved to host memory, which synchronizes that step; the
    cost is disclosed wherever these timings are reported.
    """

    def __init__(self) -> None:
        self.release_ns: list[int] = []
        self.seen_prompt = False

    def put(self, value: Any) -> None:
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        count = int(value.reshape(-1).shape[0])
        now = time.perf_counter_ns()
        self.release_ns.extend([now] * count)

    def end(self) -> None:
        return None


def synchronize(device: str) -> None:
    """Block until queued GPU work has completed. CUDA launches are async."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device if device != "cuda" else None)


def explicit_generation_config(
    config: DecodeConfig, eos_token_ids: tuple[int, ...], pad_token_id: int | None
) -> Any:
    """A ``GenerationConfig`` in which nothing is inherited.

    ``eos_policy='suppress_until_budget'`` removes every stop token, which is
    how the fixed-length latency condition in SPEC.md section 9.3 produces
    exactly ``max_new_tokens`` tokens for every engine. A minimum-length
    constraint is deliberately *not* used: Transformers' assisted generation
    refuses a minimum-length logits processor, so requiring one would silently
    exclude the ``hf_dynamic`` baseline from the matched comparison.
    """
    from transformers import GenerationConfig

    if config.mode == "sample" and config.temperature <= 0:
        raise ConfigError("sample mode requires temperature > 0")
    suppress = config.eos_policy == "suppress_until_budget"
    return GenerationConfig(
        do_sample=config.mode == "sample",
        temperature=config.temperature if config.mode == "sample" else 1.0,
        top_k=0,
        top_p=1.0,
        min_p=None,
        typical_p=1.0,
        epsilon_cutoff=0.0,
        eta_cutoff=0.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        encoder_repetition_penalty=1.0,
        length_penalty=1.0,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=config.max_new_tokens,
        # Leaving these unset matters: a non-None ``min_new_tokens`` makes
        # Transformers recompute ``min_length = min_new_tokens + prompt_length``,
        # which then installs a minimum-length logits processor that assisted
        # generation refuses outright.
        min_new_tokens=None,
        min_length=0,
        use_cache=True,
        eos_token_id=None if suppress else list(eos_token_ids),
        pad_token_id=pad_token_id,
        bos_token_id=None,
        renormalize_logits=False,
        return_dict_in_generate=False,
        output_scores=False,
        output_logits=False,
    )


def _seed_everything(seed: int, device: str) -> None:
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _pinned_generation_config(model: Any, replacement: Any) -> Iterator[None]:
    """Temporarily make ``model.generation_config`` the explicit configuration.

    Transformers falls back to ``model.generation_config`` for ``bos``, ``eos``,
    ``pad`` and ``decoder_start`` whenever the caller's value is ``None`` -- even
    with ``use_model_defaults=False``. Qwen3 ships ``eos_token_id=[151645,
    151643]``, so the fixed-length condition would silently regain stop tokens
    and produce short outputs for some prompts and not others.

    Assisted generation additionally reads ``num_assistant_tokens`` and
    ``assistant_confidence_threshold`` from the *assistant* model's config and
    writes back to it, so the same swap-and-restore is applied there.
    """
    saved = model.generation_config
    model.generation_config = copy.deepcopy(replacement)
    try:
        yield
    finally:
        model.generation_config = saved


def _run(
    engine: str,
    model: Any,
    prompt_ids: list[int],
    config: DecodeConfig,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int | None,
    device: str,
    assistant_model: Any | None = None,
    assistant_options: dict[str, Any] | None = None,
) -> BaselineResult:
    import torch

    generation_config = explicit_generation_config(config, eos_token_ids, pad_token_id)
    assistant_config = copy.deepcopy(generation_config)
    for name, value in (assistant_options or {}).items():
        setattr(assistant_config, name, value)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    timer = _ReleaseTimer()
    _seed_everything(config.seed, device)

    with ExitStack() as stack:
        stack.enter_context(_pinned_generation_config(model, generation_config))
        if assistant_model is not None:
            stack.enter_context(_pinned_generation_config(assistant_model, assistant_config))
        # Timed region: everything after this synchronization belongs to the request.
        synchronize(device)
        start_ns = time.perf_counter_ns()
        with torch.inference_mode():
            sequences = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                assistant_model=assistant_model,
                streamer=timer,
                use_model_defaults=False,
            )
        synchronize(device)
        end_ns = time.perf_counter_ns()

    output_ids = tuple(int(value) for value in sequences[0, len(prompt_ids) :].tolist())
    releases = tuple(timer.release_ns[: len(output_ids)])
    if len(releases) != len(output_ids):
        raise RuntimeError(
            f"{engine}: streamer recorded {len(releases)} releases for "
            f"{len(output_ids)} committed tokens"
        )
    if config.eos_policy == "suppress_until_budget" and len(output_ids) != config.max_new_tokens:
        raise RuntimeError(
            f"{engine}: fixed-length condition asked for {config.max_new_tokens} tokens "
            f"but generation stopped after {len(output_ids)}; a stop token leaked in"
        )
    terminated_on_eos = bool(output_ids) and output_ids[-1] in eos_token_ids
    return BaselineResult(
        engine=engine,
        output_ids=output_ids,
        token_release_ns=releases,
        start_ns=start_ns,
        first_token_ns=releases[0] if releases else end_ns,
        last_token_ns=releases[-1] if releases else end_ns,
        end_ns=end_ns,
        prompt_tokens=len(prompt_ids),
        termination="eos" if terminated_on_eos else "budget",
        generation_config=generation_config.to_diff_dict(),
        counters={
            # An external baseline does not expose these without extra
            # instrumentation; null is the honest value (SPEC.md section 9.6).
            "accepted": None,
            "proposed": None,
            "target_calls": None,
            "bypass_decisions": None,
            "controller_decisions": None,
        },
    )


def run_hf_ar(
    target: Any,
    prompt_ids: list[int],
    config: DecodeConfig,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int | None,
    device: str,
) -> BaselineResult:
    """``hf_ar``: stock target-only ``.generate()`` with KV caching enabled."""
    return _run("hf_ar", target, prompt_ids, config, eos_token_ids, pad_token_id, device)


def run_hf_dynamic(
    target: Any,
    draft: Any,
    prompt_ids: list[int],
    config: DecodeConfig,
    eos_token_ids: tuple[int, ...],
    pad_token_id: int | None,
    device: str,
) -> BaselineResult:
    """``hf_dynamic``: stock assisted generation with the locked draft options."""
    return _run(
        "hf_dynamic",
        target,
        prompt_ids,
        config,
        eos_token_ids,
        pad_token_id,
        device,
        assistant_model=draft,
        assistant_options={
            "assistant_confidence_threshold": DYNAMIC_CONFIDENCE_THRESHOLD,
            "num_assistant_tokens": DYNAMIC_MAX_DRAFT_TOKENS,
            "num_assistant_tokens_schedule": DYNAMIC_SCHEDULE,
        },
    )
