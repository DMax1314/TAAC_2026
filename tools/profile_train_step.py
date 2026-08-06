"""Profile a real PCVR training step (forward + backward + optimizer) with torch.profiler.

Usage:
    uv run python tools/profile_train_step.py \
        --experiment experiments/baseline_plus --optimizer.device cuda --optimizer.max_steps 30

The profiler wraps PCVRPointwiseTrainer._train_step via monkey-patching, so the
traced region contains exactly the model forward, loss, backward, and optimizer
step (no data loading, no validation). A schedule is used to only record a
window of steps, keeping trace sizes small.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch.profiler as profiler

from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer

PROFILE_DIR = Path("outputs/profiles")
EXPERIMENT = "experiments/baseline_plus"

_orig_train_step = PCVRPointwiseTrainer._train_step
_active_prof: profiler.profile | None = None
_step_times: list[float] = []


def _patched_train_step(self, batch: dict[str, Any]) -> float:
    started = time.monotonic()
    result = _orig_train_step(self, batch)
    _step_times.append(time.monotonic() - started)
    if _active_prof is not None:
        _active_prof.step()
    return result


def _install_item_tracer() -> None:
    """Count and print unique call stacks of Tensor.item() calls (debug only)."""
    import traceback

    import torch

    total = 0

    orig_item = torch.Tensor.item

    def traced_item(self):
        nonlocal total
        total += 1
        if total <= 8:
            stack = traceback.extract_stack()
            frames = [f for f in stack[-10:-1]]
            print(
                "item@",
                " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in frames),
            )
        return orig_item(self)

    torch.Tensor.item = traced_item  # type: ignore[method-assign]

    from taac2026.application.training.cli import main as train_main

    argv = sys.argv[1:]
    argv = [arg for arg in argv if arg != "--debug-item"]
    try:
        ret = train_main(argv)
    finally:
        torch.Tensor.item = orig_item  # type: ignore[method-assign]
    print(f"\nitem() calls: total={total}")
    raise SystemExit(ret)


def _run() -> int:
    argv = sys.argv[1:]
    debug_item = "--debug-item" in argv
    with_stack = "--with-stack" in argv
    argv = [arg for arg in argv if arg not in ("--debug-item", "--with-stack")]
    if "--experiment" not in argv:
        argv = ["--experiment", EXPERIMENT, *argv]

    if debug_item:
        _install_item_tracer()
    else:
        PCVRPointwiseTrainer._train_step = _patched_train_step  # type: ignore[method-assign]

    from taac2026.application.training.cli import main as train_main

    profile_dir = PROFILE_DIR
    profile_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profile_dir / "train_step_baseline_plus.trace.json"
    table_path = profile_dir / "train_step_baseline_plus.txt"

    global _active_prof
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=8, warmup=4, active=8, repeat=1),
        record_shapes=True,
        with_stack=with_stack,
        with_flops=False,
    ) as prof:
        _active_prof = prof
        ret = train_main(argv)
        _active_prof = None

    prof.export_chrome_trace(str(trace_path))
    print(f"\ntrace written to {trace_path}")

    if with_stack:
        _print_item_call_stacks(prof)
        return ret

    lines = [
        "=== top 40 by CUDA time ===",
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=40),
        "\n=== top 40 by CPU time ===",
        prof.key_averages().table(sort_by="cpu_time_total", row_limit=40),
    ]
    if _step_times:
        n = len(_step_times)
        total = sum(_step_times)
        lines.append(
            f"\nstep time: n={n} mean={total / n * 1000:.1f} ms "
            f"p50={sorted(_step_times)[n // 2] * 1000:.1f} ms total={total:.1f} s"
        )
    table_text = "\n".join(lines)
    print(table_text)
    table_path.write_text(table_text)
    return ret


def _print_item_call_stacks(prof: profiler.profile) -> None:
    """Print Python call stacks of aten::item/_local_scalar_dense events."""
    from collections import Counter, defaultdict

    stacks: dict[str, Counter[str]] = defaultdict(Counter)
    for event in prof.events():
        if event.name not in ("aten::item", "aten::_local_scalar_dense"):
            continue
        key = str(event.input_shapes)
        stack = getattr(event, "stack", None) or []
        frames = [f for f in stack if "site-packages" not in f and "profile_train_step" not in f]
        label = " <- ".join(frames[:6]) if frames else "(no frames)"
        stacks[key][label] += 1
    for shapes, counter in sorted(stacks.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"\nitem events shapes={shapes} total={sum(counter.values())}")
        for label, count in counter.most_common(6):
            print(f"  x{count} {label}")


if __name__ == "__main__":
    raise SystemExit(_run())
