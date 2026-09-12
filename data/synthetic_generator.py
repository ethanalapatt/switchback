"""Deterministic controlled prompts for the context-stress cohort.

SPEC.md section 9.1: copy/repetition and structured-text tasks at roughly 128,
512, 2048 and 3072 prompt tokens, eight prompts per bucket. The requested
lengths are targets. Actual token counts are measured and recorded, and the two
are never conflated -- a bucket named "512" is a label, not a measurement.

These are easy synthetic tasks with unusually high draft agreement. They are
published in their own cohort and must never be pooled with the real prompts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TARGET_LENGTHS: tuple[int, ...] = (128, 512, 2048, 3072)
PROMPTS_PER_BUCKET = 8

NOUNS = (
    "router",
    "ledger",
    "cache",
    "shard",
    "queue",
    "token",
    "buffer",
    "cursor",
    "segment",
    "replica",
    "journal",
    "checkpoint",
    "manifest",
    "digest",
    "lease",
)
ADJECTIVES = (
    "stale",
    "warm",
    "pinned",
    "sealed",
    "partial",
    "ordered",
    "durable",
    "idle",
    "nested",
    "sparse",
    "aligned",
    "frozen",
)
VERBS = ("retries", "expires", "commits", "splits", "merges", "drains", "rebuilds")


@dataclass(frozen=True)
class ControlledPrompt:
    """One generated prompt and the identity that pins it."""

    prompt_id: str
    kind: str
    target_tokens: int
    text: str


def _record(rng: random.Random, index: int) -> str:
    return (
        f"{index:04d} | {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} "
        f"{rng.choice(VERBS)} after {rng.randrange(2, 9999)} ms"
    )


def _copy_task(rng: random.Random, records: int) -> str:
    body = "\n".join(_record(rng, index) for index in range(records))
    return (
        "Copy the table below exactly, preserving every line and its order. "
        "Return only the table.\n\n" + body
    )


def _structure_task(rng: random.Random, records: int) -> str:
    body = "\n".join(_record(rng, index) for index in range(records))
    return (
        "Each line below is 'id | description'. Rewrite the table as JSON "
        "objects with the keys id and description, in the same order. Return "
        "only JSON.\n\n" + body
    )


def generate(seed: int, count: int) -> list[ControlledPrompt]:
    """Build ``count`` controlled prompts, spread evenly over the buckets.

    Deterministic in ``seed``: the same seed always produces the same prompts,
    which is what lets the workload hash pin them without storing the text in
    the manifest.
    """
    if count <= 0:
        return []
    prompts: list[ControlledPrompt] = []
    per_bucket = max(1, count // len(TARGET_LENGTHS))
    index = 0
    for target in TARGET_LENGTHS:
        # Roughly four characters per token, and each record is about 14 tokens.
        records = max(4, target // 14)
        for slot in range(per_bucket):
            if len(prompts) >= count:
                break
            rng = random.Random((seed, target, slot).__hash__() & 0xFFFFFFFF)
            kind = "copy" if slot % 2 == 0 else "structure"
            text = _copy_task(rng, records) if kind == "copy" else _structure_task(rng, records)
            prompts.append(
                ControlledPrompt(
                    prompt_id=f"controlled_{target}_{slot}",
                    kind=kind,
                    target_tokens=target,
                    text=text,
                )
            )
            index += 1
    return prompts[:count]
