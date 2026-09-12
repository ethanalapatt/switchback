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
    check_logits_vocab_match,
    check_tokenizer_compatibility,
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
