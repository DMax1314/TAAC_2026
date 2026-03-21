from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .baseline import CandidateAwareBaseline
from .decomposed import DecomposedCandidateBaseline, DecomposedDualPathBaseline
from .din import CreatorwyxDINAdapter, CreatorwyxGroupedDINAdapter
from .ucasim import UCASIMv1


def build_model(config: ModelConfig, dense_dim: int, max_seq_len: int) -> nn.Module:
    if config.name == "baseline":
        return CandidateAwareBaseline(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if config.name == "creatorwyx_din_adapter":
        return CreatorwyxDINAdapter(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if config.name == "creatorwyx_grouped_din_adapter":
        return CreatorwyxGroupedDINAdapter(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if config.name == "decomposed_baseline":
        return DecomposedCandidateBaseline(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if config.name == "decomposed_dual_path":
        return DecomposedDualPathBaseline(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    if config.name == "ucasim_v1":
        return UCASIMv1(config=config, dense_dim=dense_dim, max_seq_len=max_seq_len)
    raise ValueError(f"Unsupported model name: {config.name}")


__all__ = [
    "CandidateAwareBaseline",
    "CreatorwyxDINAdapter",
    "CreatorwyxGroupedDINAdapter",
    "DecomposedCandidateBaseline",
    "DecomposedDualPathBaseline",
    "UCASIMv1",
    "build_model",
]