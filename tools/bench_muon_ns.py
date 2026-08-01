"""Micro-benchmark Muon Newton-Schulz cost vs quality.

Questions answered:
1. How much GPU time does Muon.step take per ns_steps setting on real
   baseline_plus dense-parameter shapes (A30)?
2. Does ns_steps=2/3 give near-identical orthogonalization quality vs 5?
3. Where does the time go (gemm count / elementwise)?

Usage: uv run python tools/bench_muon_ns.py
"""

from __future__ import annotations

import time

import torch
import torch.profiler as profiler

from taac2026.infrastructure.optimization.muon import _orthogonalize_update


def _real_shapes() -> list[tuple[int, ...]]:
    """2D dense parameter shapes from PCVRBaselinePlus (d_model=64, hidden_mult=4)."""
    d, h, nq, nb = 64, 64, 2, 2  # noqa: F841  (h is hidden_mult, shapes are hardcoded)
    shapes: list[tuple[int, int]] = []
    shapes.append((nq * d, d))  # query projections
    for _ in range(nb):
        shapes += [
            (d, d), (d, d), (d, d),  # q/k/v linear
            (d, d),  # out
            (d, d * 4), (d * 4, d),  # ffn
            (d, d),  # gate
        ]
    shapes += [
        (d, d * 4), (d * 4, d),  # ns ffn
        (d, d),  # fusion linear
        (d * 6, d),  # fusion gate
        (d, d),  # out norm / classifier
        (d, d),  # classifier
        (d * 6, d),  # joined projection
    ]
    return shapes


def _make_params(device: str) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.randn(*shape, device=device) * 0.02) for shape in _real_shapes()]


def _quality_error(steps: int, trials: int = 32, device: str = "cuda") -> float:
    """Mean ||M_orth @ M_orth^T - I||_F after orthogonalization."""
    errors: list[float] = []
    for _ in range(trials):
        shape = _real_shapes()[0]
        update = torch.randn(*shape, device=device)
        orth = _orthogonalize_update(update, steps=steps)
        m = orth.float().reshape(orth.shape[0], -1)
        gram = m @ m.t()
        errors.append(float((gram - torch.eye(gram.shape[0], device=device)).abs().sum()))
    return sum(errors) / len(errors)


def _bench_steps(steps: int, device: str, prof: bool = False) -> float:
    params = _make_params(device)
    for p in params:
        p.grad = torch.randn_like(p)
    from taac2026.infrastructure.optimization.muon import Muon

    opt = Muon(params, lr=0.0001, ns_steps=steps, momentum=0.95)
    # warmup
    for _ in range(3):
        opt.step()
    torch.cuda.synchronize()
    if prof:
        with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
            opt.step()
        torch.cuda.synchronize()
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
        print(f"--- ns_steps={steps} profile ---\n{table}")
        return 0.0
    started = time.monotonic()
    for _ in range(10):
        opt.step()
    torch.cuda.synchronize()
    return (time.monotonic() - started) / 10


def main() -> None:
    device = "cuda"
    print(f"device={device} params={len(_real_shapes())} 2D tensors")
    for steps in (1, 2, 3, 5):
        error = _quality_error(steps)
        per_step = _bench_steps(steps, device)
        print(f"ns_steps={steps}: step={per_step * 1000:.1f} ms  orth_error={error:.4f}")
    _bench_steps(5, device, prof=True)


if __name__ == "__main__":
    main()
