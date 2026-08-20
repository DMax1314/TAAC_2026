"""Reusable PCVR model building blocks for experiment packages."""

from __future__ import annotations

from taac2026.infrastructure.modeling import normalization as _normalization
from taac2026.infrastructure.modeling import sequence as _sequence
from taac2026.infrastructure.modeling.embeddings import (
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    hash_compress_ids,
    sparse_embedding_bag_mean,
)
from taac2026.infrastructure.modeling.diagnostics import masked_effective_rank
from taac2026.infrastructure.modeling.normalization import LayerNorm, RMSNorm
from taac2026.infrastructure.modeling.sequence import (
    FlashAttentionBackend,
    causal_valid_attention_mask,
    choose_num_heads,
    deduplicate_sequence_events,
    flash_attention_runtime_state,
    make_padding_mask,
    masked_last,
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    scaled_dot_product_attention,
    sinusoidal_positions,
)
from taac2026.infrastructure.modeling.tokenizers import (
    DenseTokenProjector,
    NonSequentialTokenizer,
    SequenceTokenizer,
)


def configure_flash_attention_runtime(*, backend: str) -> None:
    _sequence.configure_flash_attention_runtime(backend=backend)


def configure_rms_norm_runtime(*, backend: str, block_rows: int) -> None:
    _normalization.configure_rms_norm_runtime(backend=backend, block_rows=block_rows)


def rms_norm_runtime_state() -> tuple[str, int]:
    return _normalization.rms_norm_runtime_state()


__all__ = [
    "DenseTokenProjector",
    "EmbeddingParameterMixin",
    "FeatureEmbeddingBank",
    "FlashAttentionBackend",
    "LayerNorm",
    "NonSequentialTokenizer",
    "RMSNorm",
    "SequenceTokenizer",
    "causal_valid_attention_mask",
    "choose_num_heads",
    "configure_flash_attention_runtime",
    "configure_rms_norm_runtime",
    "deduplicate_sequence_events",
    "flash_attention_runtime_state",
    "hash_compress_ids",
    "make_padding_mask",
    "masked_effective_rank",
    "masked_last",
    "masked_mean",
    "maybe_gradient_checkpoint",
    "rms_norm_runtime_state",
    "safe_key_padding_mask",
    "scaled_dot_product_attention",
    "sinusoidal_positions",
    "sparse_embedding_bag_mean",
]
