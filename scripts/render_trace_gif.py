#!/usr/bin/env python3
"""Animate a real saved trace as a GIF.

SPEC.md section 10 asks for a short GIF from an **actual** trace: accepted draft
tokens, a rejected suffix, the target's correction, the cache length changing,
and the controller's next action. A hand-constructed illustration would belong
in a separately labelled teaching demo, so this refuses to draw anything that
is not a real trace file.

Text labels carry the meaning. Colour repeats it. Every frame states in words
what the block did, so the animation is readable without colour vision and the
still frames are usable on their own.

Standard library plus Pillow. No browser, no headless capture.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 900, 420
MARGIN = 28

INK = (22, 24, 29)
MUTED = (110, 120, 134)
PAPER = (255, 255, 255)
RULE = (214, 219, 226)
ACCEPT = (26, 127, 75)
ACCEPT_BG = (223, 243, 231)
REJECT = (179, 38, 30)
REJECT_BG = (251, 225, 223)
TARGET = (31, 95, 168)
TARGET_BG = (221, 232, 247)
BYPASS = (138, 109, 31)
BYPASS_BG = (247, 238, 212)

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def load_font(size: int, bold: bool = False) -> Any:
    for path in FONT_CANDIDATES:
        candidate = Path(path)
        if bold and "Bold" not in path:
            bold_path = Path(path.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"))
            if bold_path.is_file():
                candidate = bold_path
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:  # pragma: no cover - unusual font install
                continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class Frame:
    """One block, ready to draw."""

    index: int
    total: int
    action: str
    gamma: int
    proposed: int
    accepted: int
    rejection: int | None
    committed: int
    target_cache: int
    previous_cache: int
    duration_ms: float
    bypass_reason: str | None
    sentence: str


def describe(block: dict[str, Any]) -> str:
    """What this block did, in words. The GIF must not depend on colour."""
    if block["bypass_reason"]:
        return f"Bypass: {block['bypass_reason']}"
    if block["action"] == "prefill":
        return "Prefill: the prompt is read and the first token is committed."
    if block["action"] != "speculative":
        return "Target-only step: one token, no drafting."
    if block["rejection_position"] is None:
        return (
            f"All {block['proposed']} candidates accepted. The target added a bonus "
            f"token, so {block['committed']} tokens landed from one forward call. "
            f"The draft now needs a catch-up call."
        )
    return (
        f"Candidate {block['rejection_position']} disagreed with the target. It and "
        f"every proposal after it were discarded; the target's own token was "
        f"committed instead."
    )


def build_frames(header: dict[str, Any], blocks: list[dict[str, Any]]) -> list[Frame]:
    frames: list[Frame] = []
    # Before the prefill there is no cache at all, so the first block's bar grows
    # from zero. Seeding this with the prompt length made prefill read as
    # "46 -> 46 (+0)", which is the one block where the cache genuinely appears.
    previous = 0
    for index, block in enumerate(blocks):
        frames.append(
            Frame(
                index=index,
                total=len(blocks),
                action=block["action"],
                gamma=block["gamma"],
                proposed=block["proposed"],
                accepted=block["accepted"],
                rejection=block["rejection_position"],
                committed=block["committed"],
                target_cache=block["target_cache_after"],
                previous_cache=previous,
                duration_ms=block["duration_ns"] / 1e6,
                bypass_reason=block["bypass_reason"],
                sentence=describe(block),
            )
        )
        previous = block["target_cache_after"]
    return frames


def wrap(draw: Any, text: str, font: Any, limit: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= limit or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def chip(
    draw: Any, x: int, y: int, label: str, font: Any, fill: tuple, border: tuple, text: tuple
) -> int:
    """Draw one labelled token chip and return its right edge."""
    width = int(draw.textlength(label, font=font)) + 18
    draw.rounded_rectangle([x, y, x + width, y + 28], radius=6, fill=fill, outline=border)
    draw.text((x + 9, y + 6), label, font=font, fill=text)
    return x + width + 7


def draw_frame(frame: Frame, header: dict[str, Any], fonts: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text((MARGIN, MARGIN - 6), "Switchback", font=fonts["title"], fill=INK)
    subtitle = (
        f"{header['engine']}   draft length {header['gamma']}   "
        f"Qwen3-4B target, Qwen3-0.6B draft, BF16"
    )
    draw.text((MARGIN, MARGIN + 22), subtitle, font=fonts["small"], fill=MUTED)
    draw.text(
        (
            WIDTH
            - MARGIN
            - draw.textlength(f"block {frame.index + 1} of {frame.total}", font=fonts["small"]),
            MARGIN + 22,
        ),
        f"block {frame.index + 1} of {frame.total}",
        font=fonts["small"],
        fill=MUTED,
    )
    draw.line([(MARGIN, MARGIN + 48), (WIDTH - MARGIN, MARGIN + 48)], fill=RULE, width=1)

    y = MARGIN + 70
    draw.text((MARGIN, y), "This block", font=fonts["label"], fill=MUTED)
    y += 24
    x = MARGIN
    if frame.action == "speculative":
        for slot in range(frame.accepted):
            x = chip(
                draw,
                x,
                y,
                f"draft {slot}  accepted",
                fonts["chip"],
                ACCEPT_BG,
                ACCEPT,
                ACCEPT,
            )
        for slot in range(frame.accepted, frame.proposed):
            x = chip(
                draw,
                x,
                y,
                f"draft {slot}  discarded",
                fonts["chip"],
                REJECT_BG,
                REJECT,
                REJECT,
            )
        if frame.committed > frame.accepted:
            kind = "bonus" if frame.rejection is None else "correction"
            x = chip(draw, x, y, f"target {kind}", fonts["chip"], TARGET_BG, TARGET, TARGET)
    elif frame.bypass_reason:
        x = chip(draw, x, y, "bypass: target only", fonts["chip"], BYPASS_BG, BYPASS, BYPASS)
    else:
        x = chip(
            draw, x, y, f"{frame.action}: target only", fonts["chip"], TARGET_BG, TARGET, TARGET
        )

    y += 52
    for line in wrap(draw, frame.sentence, fonts["body"], WIDTH - 2 * MARGIN):
        draw.text((MARGIN, y), line, font=fonts["body"], fill=INK)
        y += 22

    y = HEIGHT - 150
    draw.line([(MARGIN, y), (WIDTH - MARGIN, y)], fill=RULE, width=1)
    y += 18
    draw.text((MARGIN, y), "Target KV cache", font=fonts["label"], fill=MUTED)
    bar_y = y + 22
    bar_width = WIDTH - 2 * MARGIN
    span = max(frame.target_cache, frame.previous_cache, 1)
    scale = bar_width / max(span, 1)
    draw.rounded_rectangle(
        [MARGIN, bar_y, MARGIN + bar_width, bar_y + 20], radius=5, fill=(244, 246, 249)
    )
    previous_width = int(frame.previous_cache * scale)
    current_width = int(frame.target_cache * scale)
    draw.rounded_rectangle(
        [MARGIN, bar_y, MARGIN + max(current_width, 4), bar_y + 20],
        radius=5,
        fill=TARGET_BG,
        outline=TARGET,
    )
    if current_width > previous_width:
        draw.rectangle(
            [MARGIN + previous_width, bar_y, MARGIN + current_width, bar_y + 20], fill=ACCEPT_BG
        )
        draw.line(
            [(MARGIN + previous_width, bar_y), (MARGIN + previous_width, bar_y + 20)],
            fill=ACCEPT,
            width=2,
        )
    draw.text(
        (MARGIN, bar_y + 28),
        f"{frame.previous_cache} -> {frame.target_cache} positions "
        f"(+{frame.target_cache - frame.previous_cache})   "
        f"S[:-1], the pending-token convention",
        font=fonts["small"],
        fill=MUTED,
    )

    footer = (
        f"accepted {frame.accepted}/{frame.proposed}    "
        f"committed {frame.committed}    "
        f"block {frame.duration_ms:.0f} ms"
    )
    draw.text((MARGIN, HEIGHT - MARGIN - 14), footer, font=fonts["small"], fill=MUTED)
    note = "real measured trace, not an illustration"
    draw.text(
        (WIDTH - MARGIN - draw.textlength(note, font=fonts["small"]), HEIGHT - MARGIN - 14),
        note,
        font=fonts["small"],
        fill=MUTED,
    )
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/trace.gif"))
    parser.add_argument("--max-blocks", type=int, default=14)
    parser.add_argument("--ms-per-frame", type=int, default=1100)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from switchback.traces import read_trace, replay_trace

    header, blocks = read_trace(args.trace)
    report = replay_trace(header, blocks)
    if not report.ok:
        print(f"refusing to animate an inconsistent trace: {report.problems}", file=sys.stderr)
        return 1

    frames = build_frames(header, blocks)
    # Prefer a window that actually contains a rejection: an animation of
    # nothing but acceptances would misrepresent how speculation behaves.
    first_rejection = next((frame.index for frame in frames if frame.rejection is not None), 0)
    start = max(0, first_rejection - 2)
    chosen = frames[start : start + args.max_blocks]
    if not any(frame.rejection is not None for frame in chosen):
        print("warning: the selected window contains no rejection", file=sys.stderr)

    fonts = {
        "title": load_font(26, bold=True),
        "label": load_font(13),
        "body": load_font(16),
        "chip": load_font(14),
        "small": load_font(13),
    }
    images = [draw_frame(frame, header, fonts) for frame in chosen]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    durations = [args.ms_per_frame] * len(images)
    durations[-1] = args.ms_per_frame * 3
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    seconds = sum(durations) / 1000
    print(f"  {len(images)} frames from blocks {start}..{start + len(images) - 1}")
    print(f"  {seconds:.1f} s loop, {args.out.stat().st_size // 1024} KiB")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
