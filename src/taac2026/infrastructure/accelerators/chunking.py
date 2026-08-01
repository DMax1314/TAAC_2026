# pyright: reportInvalidTypeForm=false

import functools
from typing import Any
from collections import OrderedDict
from collections.abc import Callable

import torch

from taac2026.infrastructure.accelerators.tilelang_runtime import T, tilelang_available, tl


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 256). When the cache is full, the oldest entry will be removed.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache: OrderedDict[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], tuple[tuple[Any, ...], dict[str, Any], Any]] = OrderedDict()
    cache_size = 256

    def get_id(x: Any):
        if (type(x) is int) or (type(x) is float) or (type(x) is str):
            return x
        else:
            return id(x)

    def make_identity_key(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...]]:
        args_key = tuple(get_id(a) for a in args)
        kwargs_key = tuple(sorted((k, get_id(v)) for k, v in kwargs.items()))
        return args_key, kwargs_key

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache, cache_size
        key = make_identity_key(args, kwargs)
        if key in cache:
            cache.move_to_end(key, last=True)
            _, _, cached_result = cache[key]
            return cached_result

        result = fn(*args, **kwargs)
        cache[key] = (args, kwargs, result)
        cache.move_to_end(key, last=True)
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    if tilelang_available() and cu_seqlens.is_cuda:
        return _tilelang_prepare_chunk_indices(cu_seqlens, chunk_size)
    return _torch_prepare_chunk_indices(cu_seqlens, chunk_size)


@tensor_cache
def _torch_prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    lens = prepare_lens(cu_seqlens)
    chunk_counts = -(-lens // chunk_size)
    batch_idx = torch.repeat_interleave(
        torch.arange(lens.shape[0], device=lens.device),
        chunk_counts,
    )
    chunk_idx = torch.cat(
        [torch.arange(n, device=lens.device) for n in chunk_counts.tolist()]
    )
    return torch.stack([batch_idx, chunk_idx], 1).to(cu_seqlens)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _tilelang_prepare_chunk_offsets_builder(
    chunk_size,
    block_size,
    dtype,
):
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_threads = min(max(block_size, 32), 128)

    @T.prim_func
    def tilelang_prepare_chunk_offsets_kernel(
        cu_seqlens: T.Tensor([batch_size_plus_1], dtype=dtype),
        chunk_offsets: T.Tensor([batch_size_plus_1], dtype=dtype),
    ):
        with T.Kernel(1, threads=num_threads) as (_bb,):
            _batch_size = T.alloc_var("int32")
            _batch_size = batch_size_plus_1 - 1

            seqlen_start_fragment = T.alloc_fragment((block_size), dtype=dtype)
            seqlen_end_fragment = T.alloc_fragment((block_size), dtype=dtype)
            chunk_offset_fragment = T.alloc_fragment((block_size), dtype=dtype)

            T.copy(cu_seqlens[: batch_size_plus_1 - 1], seqlen_start_fragment)
            T.copy(cu_seqlens[1:], seqlen_end_fragment)

            for i in T.Parallel(block_size):
                chunk_offset_fragment[i] = (
                    seqlen_end_fragment[i] - seqlen_start_fragment[i]
                )
                chunk_offset_fragment[i] = (
                    chunk_offset_fragment[i] + chunk_size - 1
                ) // chunk_size
            T.cumsum(src=chunk_offset_fragment, dim=0)

            chunk_offsets[0] = 0
            T.copy(chunk_offset_fragment, chunk_offsets[1:])

    return tilelang_prepare_chunk_offsets_kernel


_tilelang_prepare_chunk_offsets_jit: Any = None


def tilelang_prepare_chunk_offsets(
    chunk_size,
    block_size,
    dtype,
):
    global _tilelang_prepare_chunk_offsets_jit
    if tl is None:
        raise RuntimeError("tilelang is not installed")
    if _tilelang_prepare_chunk_offsets_jit is None:
        _tilelang_prepare_chunk_offsets_jit = tl.jit()(_tilelang_prepare_chunk_offsets_builder)
    return _tilelang_prepare_chunk_offsets_jit(chunk_size, block_size, dtype)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    chunk_offsets = torch.empty_like(cu_seqlens)
    tilelang_prepare_chunk_offsets_kernel = tilelang_prepare_chunk_offsets(
        chunk_size=chunk_size,
        block_size=_next_power_of_2(cu_seqlens.shape[0] - 1),
        dtype=cu_seqlens.dtype,
    )
    tilelang_prepare_chunk_offsets_kernel(cu_seqlens, chunk_offsets)
    return chunk_offsets


def _tilelang_prepare_chunk_indices_builder(
    block_size,
    dtype,
):
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_chunks = T.dynamic("num_chunks")
    num_threads = min(max(block_size, 32), 128)

    @T.prim_func
    def tilelang_prepare_chunk_indices_kernel(
        chunk_offsets: T.Tensor([batch_size_plus_1], dtype=dtype),
        chunk_indices: T.Tensor([num_chunks, 2], dtype=dtype),
    ):
        with T.Kernel(num_chunks, threads=num_threads) as (c,):
            batch_idx = T.alloc_var(dtype)
            chunk_idx = T.alloc_var(dtype)
            batch_idx = 0
            chunk_idx = c
            for i in T.serial(batch_size_plus_1 - 1):
                if c >= chunk_offsets[i + 1]:
                    batch_idx = i + 1
                    chunk_idx = c - chunk_offsets[i + 1]
            chunk_indices[c, 0] = batch_idx
            chunk_indices[c, 1] = chunk_idx

    return tilelang_prepare_chunk_indices_kernel


_tilelang_prepare_chunk_indices_jit: Any = None


def tilelang_prepare_chunk_indices(
    block_size,
    dtype,
):
    global _tilelang_prepare_chunk_indices_jit
    if tl is None:
        raise RuntimeError("tilelang is not installed")
    if _tilelang_prepare_chunk_indices_jit is None:
        _tilelang_prepare_chunk_indices_jit = tl.jit()(_tilelang_prepare_chunk_indices_builder)
    return _tilelang_prepare_chunk_indices_jit(block_size, dtype)


def _tilelang_prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    num_chunks = int(chunk_offsets[-1])
    chunk_indices = torch.empty(
        (num_chunks, 2),
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    tilelang_prepare_chunk_indices_kernel = tilelang_prepare_chunk_indices(
        block_size=_next_power_of_2(cu_seqlens.shape[0] - 1),
        dtype=cu_seqlens.dtype,
    )
    tilelang_prepare_chunk_indices_kernel(chunk_offsets, chunk_indices)
    return chunk_indices
