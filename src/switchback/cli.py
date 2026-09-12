"""``python -m switchback`` command line.

Subcommands implemented in milestone 1:

``doctor``          report this machine and whether it can run the GPU gates
``resolve-models``  pin immutable model revisions into ``data/manifest.json``
``verify-models``   load the pinned pair, check residency and tokenizer parity
``pilot``           measure the two Hugging Face baselines on real hardware
``demo``            offline CPU fixture check, no weights downloaded
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from switchback import __version__
from switchback.env import collect_environment, format_environment, write_environment
from switchback.manifest import build_manifest, describe_model, load_manifest, write_manifest
from switchback.models.qwen import (
    DRAFT_REPO,
    TARGET_REPO,
    QwenAdapter,
    check_logits_vocab_match,
    render_chat_prompt,
    resolve_revision,
)
from switchback.runtime import configure_torch_native_overrides, deterministic_runtime
from switchback.types import ConfigError, TokenizerCompatibilityError

DEFAULT_MANIFEST = Path("data/manifest.json")


def _write_json(path: Path, document: dict[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_doctor(args: argparse.Namespace) -> int:
    report = collect_environment(require_gpu=args.require_gpu)
    print(format_environment(report))
    if args.out is not None:
        write_environment(report, args.out)
        print(f"  wrote {args.out}")
    return 0 if report.ok else 1


def command_resolve_models(args: argparse.Namespace) -> int:
    target_revision = args.target_revision or resolve_revision(args.target)
    draft_revision = args.draft_revision or resolve_revision(args.draft)
    print(f"  target {args.target} -> {target_revision}")
    print(f"  draft  {args.draft} -> {draft_revision}")
    document = build_manifest(
        target=describe_model(args.target, target_revision, args.license),
        draft=describe_model(args.draft, draft_revision, args.license),
    )
    write_manifest(document, args.out)
    print(f"  wrote {args.out}")
    return 0


def command_verify_models(args: argparse.Namespace) -> int:
    from switchback.pilot import load_pair

    runtime = {
        "native_overrides": configure_torch_native_overrides().as_dict(),
        "deterministic_knobs": deterministic_runtime(),
    }
    manifest = load_manifest(args.manifest)
    target_entry = manifest["models"]["target"]
    draft_entry = manifest["models"]["draft"]
    target, draft, compatibility, timings = load_pair(
        target_entry["repo_id"],
        target_entry["revision"],
        draft_entry["repo_id"],
        draft_entry["revision"],
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        local_files_only=args.local_files_only,
    )
    check_logits_vocab_match(target, draft)
    prompt_ids = render_chat_prompt(target.tokenizer, "Say hello.")
    document = {
        "kind": "model_check",
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "setup_seconds": timings,
        "tokenizer_compatibility": asdict(compatibility),
        "models": {
            "target": {
                **asdict(target.spec),
                "parameters": target.parameter_count,
                "logits_vocab_size": target.logits_vocab_size,
                "config_vocab_size": int(target.config.vocab_size),
                "max_position_embeddings": target.max_position_embeddings,
                "eos_token_ids": list(target.eos_token_ids),
            },
            "draft": {
                **asdict(draft.spec),
                "parameters": draft.parameter_count,
                "logits_vocab_size": draft.logits_vocab_size,
                "config_vocab_size": int(draft.config.vocab_size),
                "max_position_embeddings": draft.max_position_embeddings,
                "eos_token_ids": list(draft.eos_token_ids),
            },
        },
        "chat_prompt_token_count": len(prompt_ids),
        "runtime": runtime,
    }
    print(f"  target parameters  {target.parameter_count:,}")
    print(f"  draft parameters   {draft.parameter_count:,}")
    print(f"  shared vocabulary  {compatibility.target_vocab_size} tokens")
    print(f"  probe strings      {compatibility.checked_probe_strings} identical encodings")
    print(f"  target EOS ids     {list(target.eos_token_ids)}")
    if args.out is not None:
        _write_json(args.out, document)
        print(f"  wrote {args.out}")
    return 0


def command_pilot(args: argparse.Namespace) -> int:
    from switchback.pilot import load_pair, run_pilot, summarize, write_pilot

    environment = collect_environment(require_gpu=args.device.startswith("cuda"))
    if not environment.ok:
        print(format_environment(environment), file=sys.stderr)
        print("doctor checks failed; refusing to record pilot timings", file=sys.stderr)
        return 1
    manifest = load_manifest(args.manifest)
    target_entry = manifest["models"]["target"]
    draft_entry = manifest["models"]["draft"]
    target, draft, compatibility, timings = load_pair(
        target_entry["repo_id"],
        target_entry["revision"],
        draft_entry["repo_id"],
        draft_entry["revision"],
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        local_files_only=args.local_files_only,
    )
    requests, mismatches = run_pilot(
        target,
        draft,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
        warmups=args.warmups,
    )
    write_pilot(
        args.out,
        requests,
        mismatches,
        target,
        draft,
        compatibility,
        timings,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        repeats=args.repeats,
        environment=asdict(environment),
    )
    print(summarize(requests))
    if mismatches:
        print("  GREEDY OUTPUT MISMATCH between baselines:", file=sys.stderr)
        for line in mismatches:
            print(f"    {line}", file=sys.stderr)
    print(f"  wrote {args.out}")
    return 0


def command_profile(args: argparse.Namespace) -> int:
    """Diagnostic per-stage profile. Never a benchmark result."""
    from switchback.pilot import load_pair
    from switchback.profiling import (
        format_profile,
        profile_single_forward,
        profile_speculative_block,
        write_profile,
    )
    from switchback.types import DecodeConfig

    runtime = {
        "native_overrides": configure_torch_native_overrides().as_dict(),
        "deterministic_knobs": deterministic_runtime(),
    }
    manifest = load_manifest(args.manifest)
    target_entry = manifest["models"]["target"]
    draft_entry = manifest["models"]["draft"]
    target, draft, _, _ = load_pair(
        target_entry["repo_id"],
        target_entry["revision"],
        draft_entry["repo_id"],
        draft_entry["revision"],
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        local_files_only=args.local_files_only,
    )
    target_adapter = QwenAdapter(
        model=target.model,
        device=args.device,
        vocab_size=target.logits_vocab_size,
        name="target",
    )
    draft_adapter = QwenAdapter(
        model=draft.model,
        device=args.device,
        vocab_size=draft.logits_vocab_size,
        name="draft",
    )
    prompt_ids = render_chat_prompt(target.tokenizer, args.prompt)
    config = DecodeConfig(
        mode="greedy",
        temperature=1.0,
        max_new_tokens=256,
        eos_policy="suppress_until_budget",
        seed=42,
    )
    document = {
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "runtime": runtime,
        "models": {"target": target_entry["repo_id"], "draft": draft_entry["repo_id"]},
        "prompt_tokens": len(prompt_ids),
        "forward_by_width": profile_single_forward(
            target_adapter, prompt_ids, widths=[1, 2, 3, 5, 9, 17], device=args.device
        ),
        "draft_forward_by_width": profile_single_forward(
            draft_adapter, prompt_ids, widths=[1, 2, 3, 5, 9], device=args.device
        ),
        "blocks": [
            profile_speculative_block(
                target_adapter,
                draft_adapter,
                prompt_ids,
                config,
                target.eos_token_ids,
                gamma=gamma,
                blocks=args.blocks,
                device=args.device,
            )
            for gamma in (1, 2, 4, 8)
        ],
    }
    print(format_profile(document))
    write_profile(document, args.out)
    print(f"  wrote {args.out}")
    return 0


def command_trace(args: argparse.Namespace) -> int:
    """Run one request and save a replayable block trace."""
    from switchback.decoder import (
        EngineOptions,
        decode_speculative_greedy,
        decode_speculative_sampled,
        decode_target_only,
    )
    from switchback.events import ListSink
    from switchback.pilot import load_pair
    from switchback.traces import build_header, write_trace
    from switchback.types import DecodeConfig

    configure_torch_native_overrides()
    deterministic_runtime()
    manifest = load_manifest(args.manifest)
    target_entry = manifest["models"]["target"]
    draft_entry = manifest["models"]["draft"]
    target, draft, _, _ = load_pair(
        target_entry["repo_id"],
        target_entry["revision"],
        draft_entry["repo_id"],
        draft_entry["revision"],
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        local_files_only=args.local_files_only,
    )
    target_adapter = QwenAdapter(
        model=target.model,
        device=args.device,
        vocab_size=target.logits_vocab_size,
        name="target",
    )
    draft_adapter = QwenAdapter(
        model=draft.model,
        device=args.device,
        vocab_size=draft.logits_vocab_size,
        name="draft",
    )
    config = DecodeConfig(
        mode=args.mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        eos_policy=args.eos_policy,
        seed=args.seed,
    )
    options = EngineOptions(correctness_checks=True, device=args.device)
    prompt_ids = render_chat_prompt(target.tokenizer, args.prompt)
    sink = ListSink()
    if args.gamma == 0:
        engine = "native_ar"
        result = decode_target_only(
            target_adapter,
            prompt_ids,
            config,
            target.eos_token_ids,
            sink=sink,
            options=options,
        )
    elif args.mode == "greedy":
        engine = f"fixed_{args.gamma}"
        result = decode_speculative_greedy(
            target_adapter,
            draft_adapter,
            prompt_ids,
            config,
            target.eos_token_ids,
            gamma=args.gamma,
            sink=sink,
            options=options,
        )
    else:
        engine = f"fixed_{args.gamma}"
        result = decode_speculative_sampled(
            target_adapter,
            draft_adapter,
            prompt_ids,
            config,
            target.eos_token_ids,
            gamma=args.gamma,
            sink=sink,
            options=options,
        )
    header = build_header(
        engine=engine,
        result=result,
        prompt_ids=prompt_ids,
        mode=args.mode,
        eos_policy=args.eos_policy,
        temperature=args.temperature,
        seed=args.seed,
        gamma=args.gamma or None,
        max_new_tokens=args.max_new_tokens,
        models={
            "target": {"repo_id": target_entry["repo_id"], "revision": target_entry["revision"]},
            "draft": {"repo_id": draft_entry["repo_id"], "revision": draft_entry["revision"]},
        },
    )
    write_trace(args.out, header, sink)
    print(f"  engine      {engine}")
    print(f"  output      {len(result.output_ids)} tokens, terminated on {result.termination}")
    print(f"  acceptance  {result.accepted}/{result.proposed}")
    print(f"  calls       target={result.target_calls} draft={result.draft_calls}")
    print(f"  wrote {args.out}")
    return 0


def command_replay(args: argparse.Namespace) -> int:
    """Check a saved trace offline: no model, no GPU, no network."""
    from switchback.traces import block_summary, format_replay, read_trace, replay_trace

    header, blocks = read_trace(args.trace)
    report = replay_trace(header, blocks)
    print(format_replay(header, report))
    summary = block_summary(blocks)
    print(
        f"  blocks      {summary['speculative_blocks']} speculative, "
        f"{summary['rejected_blocks']} rejected, "
        f"{summary['fully_accepted_blocks']} fully accepted"
    )
    if summary["rejections_by_position"]:
        print(f"  rejections  by position {summary['rejections_by_position']}")
    return 0 if report.ok else 1


def command_evidence(args: argparse.Namespace) -> int:
    """Run the validation checks and record their exit codes."""
    from switchback.evidence import collect_evidence, format_evidence, write_evidence

    document = collect_evidence(
        python=sys.executable,
        include_gpu=args.gpu,
        artifacts=list(args.artifact),
        notes=args.notes,
    )
    print(format_evidence(document))
    write_evidence(document, args.out)
    print(f"  wrote {args.out}")
    return 0 if document["passed"] else 1


def command_demo(args: argparse.Namespace) -> int:
    """Offline fixture check: builds the tiny model twice and compares logits."""
    import torch

    from switchback.models.tiny import TinyTokenizer, build_tiny_model, tiny_pair

    tokenizer = TinyTokenizer()
    ids = tokenizer.encode("def add(a, b): return a + b", add_special_tokens=True)
    inputs = torch.tensor([ids], dtype=torch.long)
    first = build_tiny_model(seed=args.seed)
    second = build_tiny_model(seed=args.seed)
    with torch.inference_mode():
        left = first.model(input_ids=inputs).logits
        right = second.model(input_ids=inputs).logits
    identical = bool(torch.equal(left, right))
    target, draft = tiny_pair()
    print(f"  tiny vocabulary    {first.logits_vocab_size}")
    print(f"  prompt tokens      {len(ids)}")
    print(f"  logits shape       {tuple(left.shape)}  (B, T, V)")
    print(f"  seed reproducible  {identical}")
    print(f"  fixture revision   {first.spec.revision}")
    print(f"  target/draft params {target.parameter_count:,} / {draft.parameter_count:,}")
    print("  No weights were downloaded; this path runs offline on CPU.")
    return 0 if identical else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="switchback", description=__doc__)
    parser.add_argument("--version", action="version", version=f"switchback {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report the execution environment")
    doctor.add_argument("--out", type=Path, default=None)
    doctor.add_argument(
        "--require-gpu",
        action="store_true",
        help="treat a missing or unusable CUDA device as a failure",
    )
    doctor.set_defaults(handler=command_doctor)

    resolve = subparsers.add_parser(
        "resolve-models", help="pin immutable model revisions into the manifest"
    )
    resolve.add_argument("--target", default=TARGET_REPO)
    resolve.add_argument("--draft", default=DRAFT_REPO)
    resolve.add_argument("--target-revision", default=None)
    resolve.add_argument("--draft-revision", default=None)
    resolve.add_argument("--license", default="apache-2.0")
    resolve.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    resolve.set_defaults(handler=command_resolve_models)

    verify = subparsers.add_parser(
        "verify-models", help="load the pinned pair and check tokenizer parity"
    )
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--device", default="cuda")
    verify.add_argument("--dtype", default="bfloat16")
    verify.add_argument("--attn", default="sdpa")
    verify.add_argument("--local-files-only", action="store_true")
    verify.add_argument("--out", type=Path, default=None)
    verify.set_defaults(handler=command_verify_models)

    pilot = subparsers.add_parser(
        "pilot", help="measure the hf_ar and hf_dynamic baselines on this machine"
    )
    pilot.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    pilot.add_argument("--device", default="cuda")
    pilot.add_argument("--dtype", default="bfloat16")
    pilot.add_argument("--attn", default="sdpa")
    pilot.add_argument("--local-files-only", action="store_true")
    pilot.add_argument("--max-new-tokens", type=int, default=32)
    pilot.add_argument("--repeats", type=int, default=3)
    pilot.add_argument("--warmups", type=int, default=2)
    pilot.add_argument("--out", type=Path, default=Path("artifacts/pilot/pilot.json"))
    pilot.set_defaults(handler=command_pilot)

    profile = subparsers.add_parser(
        "profile", help="diagnostic per-stage block timings (never a benchmark result)"
    )
    profile.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--dtype", default="bfloat16")
    profile.add_argument("--attn", default="sdpa")
    profile.add_argument("--local-files-only", action="store_true")
    profile.add_argument("--blocks", type=int, default=16)
    profile.add_argument(
        "--prompt",
        default="Write a Python function for this task. Return code only. Reverse a string.",
    )
    profile.add_argument("--out", type=Path, default=Path("artifacts/profile/m4_profile.json"))
    profile.set_defaults(handler=command_profile)

    trace = subparsers.add_parser("trace", help="run one request and save a block trace")
    trace.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    trace.add_argument("--device", default="cuda")
    trace.add_argument("--dtype", default="bfloat16")
    trace.add_argument("--attn", default="sdpa")
    trace.add_argument("--local-files-only", action="store_true")
    trace.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    trace.add_argument("--temperature", type=float, default=1.0)
    trace.add_argument("--max-new-tokens", type=int, default=64)
    trace.add_argument(
        "--eos-policy", choices=["respect", "suppress_until_budget"], default="respect"
    )
    trace.add_argument("--seed", type=int, default=42)
    trace.add_argument("--gamma", type=int, default=4, help="0 runs target-only decoding")
    trace.add_argument(
        "--prompt",
        default="Write a Python function for this task. Return code only. Reverse a string.",
    )
    trace.add_argument("--out", type=Path, default=Path("artifacts/traces/trace.jsonl"))
    trace.set_defaults(handler=command_trace)

    replay = subparsers.add_parser(
        "replay", help="check a saved trace offline; no model or GPU needed"
    )
    replay.add_argument("trace", type=Path)
    replay.set_defaults(handler=command_replay)

    evidence = subparsers.add_parser(
        "evidence", help="run the validation checks and record their exit codes"
    )
    evidence.add_argument("--gpu", action="store_true", help="include the GPU gates")
    evidence.add_argument(
        "--artifact", action="append", default=[], help="path to reference in the evidence"
    )
    evidence.add_argument("--notes", default="")
    evidence.add_argument("--out", type=Path, default=Path("artifacts/evidence.json"))
    evidence.set_defaults(handler=command_evidence)

    demo = subparsers.add_parser("demo", help="offline tiny-model fixture check")
    demo.add_argument("--seed", type=int, default=1234)
    demo.set_defaults(handler=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result: int = args.handler(args)
        return result
    except (ConfigError, TokenizerCompatibilityError) as error:
        print(f"switchback: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
