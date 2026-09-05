"""PCVRSparseAdagrad correctness tests."""

from __future__ import annotations

import math

import pytest
import torch

from taac2026.infrastructure.optimization.sparse_adagrad import PCVRSparseAdagrad


def _sparse_grad(rows: list[int], values: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.tensor(rows, dtype=torch.long).unsqueeze(0),
        values,
        shape,
    )


def test_sparse_step_matches_torch_adagrad_on_cpu() -> None:
    torch.manual_seed(0)
    vocab, dim, nnz = 500, 16, 120
    rows = torch.randint(1, vocab, (nnz // 2,)).repeat_interleave(2)[torch.randperm(nnz)]
    values = torch.randn(nnz, dim) * 0.01
    init = torch.randn(vocab, dim) * 0.01

    reference = init.clone()
    torch_opt = torch.optim.Adagrad([reference], lr=0.05, eps=1e-10)
    candidate = init.clone().requires_grad_(True)
    ours = PCVRSparseAdagrad([candidate], lr=0.05, eps=1e-10)
    for _ in range(3):
        grad = _sparse_grad(rows.tolist(), values, (vocab, dim))
        reference.grad = grad
        candidate.grad = grad
        torch_opt.step()
        ours.step()

    torch.testing.assert_close(candidate.detach(), reference, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_grad_scaler_updates_match_torch_adagrad(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(7)
    candidate = torch.nn.Parameter(torch.randn(16, 4, device=device))
    reference = torch.nn.Parameter(candidate.detach().clone())
    optimizer = PCVRSparseAdagrad([candidate], lr=0.05)
    reference_optimizer = torch.optim.Adagrad([reference], lr=0.05)
    scaler = torch.amp.GradScaler(device=device, init_scale=8.0, growth_interval=2)
    rows = torch.tensor([1, 1, 3, 5, 3], device=device)

    for _ in range(3):
        values = torch.randn(5, 4, device=device)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        loss = (torch.nn.functional.embedding(rows, candidate, sparse=True) * values).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        (torch.nn.functional.embedding(rows, reference, sparse=True) * values).sum().backward()
        reference_optimizer.step()
        torch.testing.assert_close(candidate, reference)
        torch.testing.assert_close(optimizer.state[candidate]["sum"], reference_optimizer.state[reference]["sum"])

    assert scaler.get_scale() == 16.0


def test_sparse_step_sums_duplicate_rows_before_squaring() -> None:
    # Duplicate rows are summed first (g1+g2), then squared: state_sum += (g1+g2)^2.
    vocab, dim, eps = 4, 2, 1e-10
    parameter = torch.zeros(vocab, dim, requires_grad=True)
    grad = _sparse_grad(
        [1, 1, 2],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        (vocab, dim),
    )
    optimizer = PCVRSparseAdagrad([parameter], lr=0.5, eps=eps)
    expected = parameter.detach().clone()
    parameter.grad = grad
    optimizer.step()

    merged = torch.tensor([[4.0, 6.0], [5.0, 6.0]])
    std = torch.sqrt(merged.pow(2) + eps)
    expected[1] -= 0.5 * merged[0] / std[0]
    expected[2] -= 0.5 * merged[1] / std[1]
    torch.testing.assert_close(parameter.detach(), expected)


def test_sparse_step_single_row_no_duplicates() -> None:
    parameter = torch.ones(3, 2, requires_grad=True)
    grad = _sparse_grad([2], torch.tensor([[1.0, 1.0]]), (3, 2))
    optimizer = PCVRSparseAdagrad([parameter], lr=0.1, eps=1e-10)
    parameter.grad = grad
    optimizer.step()

    expected = torch.ones(3, 2)
    std = math.sqrt(1.0 + 1e-10)
    expected[2] -= 0.1 * torch.tensor([1.0, 1.0]) / std
    torch.testing.assert_close(parameter.detach(), expected)


def test_sparse_step_with_empty_grad_is_noop() -> None:
    parameter = torch.randn(4, 3, requires_grad=True)
    before = parameter.detach().clone()
    optimizer = PCVRSparseAdagrad([parameter], lr=0.1)
    parameter.grad = torch.zeros(4, 3).to_sparse()
    optimizer.step()
    torch.testing.assert_close(parameter.detach(), before)


def test_sparse_step_accumulates_state_across_steps() -> None:
    torch.manual_seed(0)
    vocab, dim, nnz = 64, 8, 32
    init = torch.randn(vocab, dim)
    parameter = init.clone().requires_grad_(True)
    optimizer = PCVRSparseAdagrad([parameter], lr=0.05, eps=1e-10)
    reference = init.clone()
    torch_opt = torch.optim.Adagrad([reference], lr=0.05, eps=1e-10)
    rows = torch.randint(1, vocab, (nnz,))
    values = torch.randn(nnz, dim) * 0.01
    for _ in range(5):
        grad = _sparse_grad(rows.tolist(), values, (vocab, dim))
        parameter.grad = grad
        reference.grad = grad
        optimizer.step()
        torch_opt.step()
    torch.testing.assert_close(parameter.detach(), reference, atol=1e-6, rtol=1e-6)


def test_dense_step_matches_torch_adagrad() -> None:
    torch.manual_seed(0)
    parameter = torch.randn(16, 8, requires_grad=True)
    optimizer = PCVRSparseAdagrad([parameter], lr=0.05, eps=1e-10)
    reference = parameter.detach().clone()
    torch_opt = torch.optim.Adagrad([reference], lr=0.05, eps=1e-10)
    for _ in range(3):
        grad = torch.randn_like(parameter)
        parameter.grad = grad
        reference.grad = grad.clone()
        optimizer.step()
        torch_opt.step()
    torch.testing.assert_close(parameter.detach(), reference, atol=1e-6, rtol=1e-6)


def test_zero_grad_clears_gradients() -> None:
    parameter = torch.randn(4, 2, requires_grad=True)
    optimizer = PCVRSparseAdagrad([parameter], lr=0.1)
    parameter.grad = torch.randn_like(parameter)
    optimizer.zero_grad()
    assert parameter.grad is None


def test_rejects_invalid_configuration() -> None:
    parameter = torch.randn(4, 2, requires_grad=True)
    with pytest.raises(ValueError, match="lr must be > 0"):
        PCVRSparseAdagrad([parameter], lr=0.0)
    with pytest.raises(ValueError, match="does not support weight_decay"):
        PCVRSparseAdagrad([parameter], lr=0.1, weight_decay=0.1)
    optimizer = PCVRSparseAdagrad([parameter], lr=0.1)
    with pytest.raises(TypeError, match="does not support closures"):
        optimizer.step(closure=lambda: None)
