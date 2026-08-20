"""Column-batched QueryFormer model for TAAC 2026 PCVR.

Independent embedding columns are represented by an explicit tensor axis:
sparse tables use disjoint column vocabularies, dense operators carry
column-batched weights, and attention folds ``batch * columns`` into one SDPA.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from taac2026.api import (
    EmbeddingParameterMixin,
    NUM_TIME_BUCKETS,
    PCVRModelInput,
    PCVRSchema,
    build_pcvr_model_specs,
    choose_num_heads,
    compute_sequence_time_buckets,
    hash_compress_ids,
    make_padding_mask,
    mark_muon_adamw,
    mark_muon_batched_matrix,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    sinusoidal_positions,
    sparse_embedding_bag_mean,
)

from .config import QueryFormerModelConfig


class ColumnEmbedding(nn.Embedding):
    """Independent embedding matrices packed into one sparse lookup table."""

    def __init__(self, num_columns: int, num_buckets: int, embedding_dim: int) -> None:
        self.num_columns = int(num_columns)
        self.num_buckets = int(num_buckets)
        super().__init__(
            1 + self.num_columns * self.num_buckets,
            embedding_dim,
            padding_idx=0,
            sparse=True,
        )
        nn.init.xavier_normal_(self.weight)
        with torch.no_grad():
            self.weight[0].zero_()

    def column_ids(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``[B, ...]`` ids to disjoint ``[B, H, ...]`` column ranges."""
        values = values.to(torch.long).clamp(min=0, max=self.num_buckets)
        column_shape = (1, self.num_columns, *(1 for _ in values.shape[1:]))
        offsets = (
            torch.arange(self.num_columns, device=values.device).view(column_shape)
            * self.num_buckets
        )
        expanded = values.unsqueeze(1)
        return torch.where(expanded > 0, expanded + offsets, torch.zeros_like(expanded))

    def lookup(self, values: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            self.column_ids(values), self.weight, padding_idx=0, sparse=True
        )

    def bag_mean(self, values: torch.Tensor) -> torch.Tensor:
        column_values = self.column_ids(values)
        batch_size = values.shape[0]
        embedded = sparse_embedding_bag_mean(
            self.weight,
            column_values.reshape(batch_size * self.num_columns, -1),
        )
        return embedded.view(batch_size, self.num_columns, self.embedding_dim)


class ColumnFeatureEmbeddingBank(nn.Module):
    """Feature embedding bank fused across the column axis."""

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        num_columns: int,
        emb_dim: int,
        emb_skip_threshold: int,
        *,
        compress_high_cardinality: bool,
    ) -> None:
        super().__init__()
        self.feature_specs = list(feature_specs)
        self.num_columns = int(num_columns)
        self.emb_dim = int(emb_dim)
        self.embeddings = nn.ModuleList()
        self.compressed_embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []
        self._compressed_embedding_index: list[int] = []
        for vocab_size, _offset, _length in self.feature_specs:
            if vocab_size <= 0:
                self._embedding_index.append(-1)
                self._compressed_embedding_index.append(-1)
            elif (
                emb_skip_threshold > 0
                and vocab_size > emb_skip_threshold
                and compress_high_cardinality
            ):
                self._embedding_index.append(-1)
                self._compressed_embedding_index.append(len(self.compressed_embeddings))
                self.compressed_embeddings.append(
                    ColumnEmbedding(num_columns, emb_skip_threshold, emb_dim)
                )
            elif emb_skip_threshold > 0 and vocab_size > emb_skip_threshold:
                self._embedding_index.append(-1)
                self._compressed_embedding_index.append(-1)
            else:
                self._embedding_index.append(len(self.embeddings))
                self._compressed_embedding_index.append(-1)
                self.embeddings.append(
                    ColumnEmbedding(num_columns, vocab_size, emb_dim)
                )

    @property
    def output_dim(self) -> int:
        return self.emb_dim

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        if not self.feature_specs:
            return int_feats.new_zeros(
                batch_size,
                self.num_columns,
                0,
                self.emb_dim,
                dtype=torch.float32,
            )
        tokens: list[torch.Tensor] = []
        for feature_index, (vocab_size, offset, length) in enumerate(
            self.feature_specs
        ):
            values = int_feats[:, offset : offset + length]
            embedding_index = self._embedding_index[feature_index]
            compressed_index = self._compressed_embedding_index[feature_index]
            if embedding_index >= 0:
                tokens.append(
                    self.embeddings[embedding_index].bag_mean(
                        values.clamp(min=0, max=vocab_size)
                    )
                )
            elif compressed_index >= 0:
                embedding = self.compressed_embeddings[compressed_index]
                tokens.append(
                    embedding.bag_mean(
                        hash_compress_ids(values.clamp_min(0), embedding.num_buckets)
                    )
                )
            else:
                tokens.append(
                    int_feats.new_zeros(
                        batch_size,
                        self.num_columns,
                        self.emb_dim,
                        dtype=torch.float32,
                    )
                )
        return torch.stack(tokens, dim=2)


class ColumnLinear(nn.Module):
    """Independent linear projections evaluated as one batched matmul."""

    def __init__(self, num_columns: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = mark_muon_batched_matrix(
            nn.Parameter(torch.empty(num_columns, self.input_dim, self.output_dim))
        )
        self.bias = mark_muon_adamw(
            nn.Parameter(torch.empty(num_columns, self.output_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.input_dim)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 3 or inputs.shape[1] != self.weight.shape[0]:
            raise ValueError("ColumnLinear expects [batch, columns, ..., input_dim]")
        batch_size, num_columns = inputs.shape[:2]
        flat = inputs.reshape(batch_size, num_columns, -1, self.input_dim)
        projected = torch.einsum("bhni,hio->bhno", flat, self.weight)
        projected = projected + self.bias.unsqueeze(0).unsqueeze(2)
        return projected.reshape(*inputs.shape[:-1], self.output_dim)


class ColumnLayerNorm(nn.Module):
    """Layer normalization with independent affine parameters per column."""

    def __init__(self, num_columns: int, normalized_dim: int) -> None:
        super().__init__()
        self.normalized_dim = int(normalized_dim)
        self.weight = mark_muon_adamw(
            nn.Parameter(torch.ones(num_columns, normalized_dim))
        )
        self.bias = mark_muon_adamw(
            nn.Parameter(torch.zeros(num_columns, normalized_dim))
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(inputs, (self.normalized_dim,))
        affine_shape = (
            1,
            self.weight.shape[0],
            *(1 for _ in range(inputs.ndim - 3)),
            self.normalized_dim,
        )
        return normalized * self.weight.view(affine_shape) + self.bias.view(
            affine_shape
        )


class CrossNetworkV2(nn.Module):
    """Full-rank DCNv2 layers evaluated for every column together."""

    def __init__(self, input_dim: int, num_layers: int, num_columns: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ColumnLinear(num_columns, input_dim, input_dim) for _ in range(num_layers)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        crossed = inputs
        for layer in self.layers:
            crossed = inputs * layer(crossed) + crossed
        return crossed


class ColumnDenseFieldEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_cross_layers: int,
        num_columns: int,
    ) -> None:
        super().__init__()
        self.input_norm = ColumnLayerNorm(num_columns, input_dim)
        self.cross = CrossNetworkV2(input_dim, num_cross_layers, num_columns)
        self.project = ColumnLinear(num_columns, input_dim, d_model)
        self.output_norm = ColumnLayerNorm(num_columns, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.input_norm(inputs)
        return self.output_norm(F.silu(self.project(self.cross(inputs))))


class DenseFieldTokenizer(nn.Module):
    """One DCNv2 token per dense field, vectorized over columns."""

    def __init__(
        self,
        field_dims: Sequence[int],
        d_model: int,
        num_cross_layers: int,
        num_columns: int,
    ) -> None:
        super().__init__()
        self.field_dims = tuple(int(dim) for dim in field_dims)
        self.encoders = nn.ModuleList(
            [
                ColumnDenseFieldEncoder(dim * 2, d_model, num_cross_layers, num_columns)
                for dim in self.field_dims
            ]
        )
        self.num_columns = int(num_columns)
        self.d_model = int(d_model)

    @property
    def num_tokens(self) -> int:
        return len(self.field_dims)

    def forward(
        self, values: torch.Tensor, missing_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if not self.field_dims:
            return values.new_zeros(values.shape[0], self.num_columns, 0, self.d_model)
        if missing_mask is None:
            missing_mask = torch.zeros_like(values, dtype=torch.bool)
        tokens: list[torch.Tensor] = []
        offset = 0
        for dim, encoder in zip(self.field_dims, self.encoders, strict=True):
            field_values = values[:, offset : offset + dim]
            field_missing = missing_mask[:, offset : offset + dim].to(
                dtype=values.dtype, device=values.device
            )
            encoded = torch.cat([field_values, field_missing], dim=-1)
            encoded = encoded.unsqueeze(1).expand(-1, self.num_columns, -1)
            tokens.append(encoder(encoded))
            offset += dim
        return torch.stack(tokens, dim=2)


class ColumnNonSequentialTokenizer(nn.Module):
    """Grouped sparse tokenization with a first-class column axis."""

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        num_columns: int,
        emb_dim: int,
        d_model: int,
        num_tokens: int,
        emb_skip_threshold: int,
        *,
        compress_high_cardinality: bool,
    ) -> None:
        super().__init__()
        self.bank = ColumnFeatureEmbeddingBank(
            feature_specs,
            num_columns,
            emb_dim,
            emb_skip_threshold,
            compress_high_cardinality=compress_high_cardinality,
        )
        self.groups = [list(group) for group in groups] or [
            [index] for index in range(len(feature_specs))
        ]
        self.feature_count = len(feature_specs)
        self.num_columns = int(num_columns)
        self.num_tokens = int(num_tokens) if num_tokens > 0 else len(self.groups)
        self.auto_split = self.num_tokens != len(self.groups)
        self.missing_embeddings = mark_muon_batched_matrix(
            nn.Parameter(torch.empty(num_columns, self.feature_count, emb_dim))
        )
        nn.init.normal_(self.missing_embeddings, mean=0.0, std=0.02)
        if self.auto_split:
            self.project_in_dim = max(1, self.feature_count * emb_dim)
            self.project_out_dim = self.num_tokens * d_model
        else:
            self.project_in_dim = emb_dim
            self.project_out_dim = d_model
        self.project = ColumnLinear(
            num_columns, self.project_in_dim, self.project_out_dim
        )
        self.project_norm = ColumnLayerNorm(num_columns, self.project_out_dim)
        self.d_model = int(d_model)

    def forward(
        self, int_feats: torch.Tensor, missing_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch_size = int_feats.shape[0]
        feature_tokens = self.bank(int_feats)
        if missing_mask is not None and self.feature_count > 0:
            mask = missing_mask[:, : self.feature_count].to(
                dtype=feature_tokens.dtype, device=feature_tokens.device
            )
            feature_tokens = feature_tokens + (
                mask.unsqueeze(1).unsqueeze(-1) * self.missing_embeddings.unsqueeze(0)
            )
        if self.num_tokens <= 0:
            return int_feats.new_zeros(
                batch_size,
                self.num_columns,
                0,
                self.d_model,
                dtype=torch.float32,
            )
        if self.auto_split:
            if self.feature_count == 0:
                grouped = int_feats.new_zeros(
                    batch_size, self.num_columns, 1, dtype=torch.float32
                )
            else:
                grouped = feature_tokens.flatten(start_dim=2)
            projected = self.project_norm(F.silu(self.project(grouped)))
            return projected.view(
                batch_size, self.num_columns, self.num_tokens, self.d_model
            )
        grouped_tokens: list[torch.Tensor] = []
        for group in self.groups:
            valid = [index for index in group if 0 <= index < self.feature_count]
            if valid:
                grouped_tokens.append(feature_tokens[:, :, valid, :].mean(dim=2))
            else:
                grouped_tokens.append(
                    int_feats.new_zeros(
                        batch_size,
                        self.num_columns,
                        self.bank.output_dim,
                        dtype=torch.float32,
                    )
                )
        grouped = torch.stack(grouped_tokens, dim=2)
        return self.project_norm(self.project(grouped))


class ColumnSequenceTokenizer(nn.Module):
    """Sequence tokenization with fused lookups across embedding columns."""

    def __init__(
        self,
        vocab_sizes: list[int],
        num_columns: int,
        emb_dim: int,
        d_model: int,
        num_time_buckets: int,
        emb_skip_threshold: int,
        *,
        compress_high_cardinality: bool,
    ) -> None:
        super().__init__()
        self.vocab_sizes = [int(value) for value in vocab_sizes]
        self.num_columns = int(num_columns)
        self.emb_dim = int(emb_dim)
        self.embeddings = nn.ModuleList()
        self.compressed_embeddings = nn.ModuleList()
        self._embedding_index: list[int] = []
        self._compressed_embedding_index: list[int] = []
        for vocab_size in self.vocab_sizes:
            if vocab_size <= 0:
                self._embedding_index.append(-1)
                self._compressed_embedding_index.append(-1)
            elif (
                emb_skip_threshold > 0
                and vocab_size > emb_skip_threshold
                and compress_high_cardinality
            ):
                self._embedding_index.append(-1)
                self._compressed_embedding_index.append(len(self.compressed_embeddings))
                self.compressed_embeddings.append(
                    ColumnEmbedding(num_columns, emb_skip_threshold, emb_dim)
                )
            elif emb_skip_threshold > 0 and vocab_size > emb_skip_threshold:
                self._embedding_index.append(-1)
                self._compressed_embedding_index.append(-1)
            else:
                self._embedding_index.append(len(self.embeddings))
                self._compressed_embedding_index.append(-1)
                self.embeddings.append(
                    ColumnEmbedding(num_columns, vocab_size, emb_dim)
                )
        input_dim = max(1, len(self.vocab_sizes) * emb_dim)
        self.project = ColumnLinear(num_columns, input_dim, d_model)
        self.output_norm = ColumnLayerNorm(num_columns, d_model)
        self.time_embedding = (
            ColumnEmbedding(num_columns, num_time_buckets - 1, d_model)
            if num_time_buckets > 1
            else None
        )

    def forward(
        self,
        sequence: torch.Tensor,
        timestamps: torch.Tensor | None,
        request_timestamp: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, feature_count, seq_len = sequence.shape
        pieces: list[torch.Tensor] = []
        for feature_index in range(feature_count):
            values = sequence[:, feature_index, :]
            embedding_index = (
                self._embedding_index[feature_index]
                if feature_index < len(self._embedding_index)
                else -1
            )
            compressed_index = (
                self._compressed_embedding_index[feature_index]
                if feature_index < len(self._compressed_embedding_index)
                else -1
            )
            if embedding_index >= 0:
                pieces.append(
                    self.embeddings[embedding_index].lookup(
                        values.clamp(min=0, max=self.vocab_sizes[feature_index])
                    )
                )
            elif compressed_index >= 0:
                embedding = self.compressed_embeddings[compressed_index]
                pieces.append(
                    embedding.lookup(
                        hash_compress_ids(values.clamp_min(0), embedding.num_buckets)
                    )
                )
            else:
                pieces.append(
                    sequence.new_zeros(
                        batch_size,
                        self.num_columns,
                        seq_len,
                        self.emb_dim,
                        dtype=torch.float32,
                    )
                )
        if pieces:
            token_input = torch.cat(pieces, dim=-1)
        else:
            token_input = sequence.new_zeros(
                batch_size,
                self.num_columns,
                seq_len,
                1,
                dtype=torch.float32,
            )
        tokens = self.output_norm(F.silu(self.project(token_input)))
        if (
            self.time_embedding is not None
            and timestamps is not None
            and request_timestamp is not None
        ):
            time_values = compute_sequence_time_buckets(
                timestamps, request_timestamp
            ).clamp(min=0, max=self.time_embedding.num_buckets)
            tokens = tokens + self.time_embedding.lookup(time_values)
        return tokens


def column_masked_mean(
    tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if tokens.shape[2] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[1], tokens.shape[-1])
    if padding_mask is None:
        return tokens.mean(dim=2)
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(1).unsqueeze(-1)
    return (tokens * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)


class AttentionUpdate(nn.Module):
    """Column-batched pre-norm attention using one SDPA invocation."""

    def __init__(
        self,
        num_columns: int,
        d_model: int,
        num_heads: int,
        hidden_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = ColumnLayerNorm(num_columns, d_model)
        self.context_norm = ColumnLayerNorm(num_columns, d_model)
        self.query = ColumnLinear(num_columns, d_model, d_model)
        self.key = ColumnLinear(num_columns, d_model, d_model)
        self.value = ColumnLinear(num_columns, d_model, d_model)
        self.output = ColumnLinear(num_columns, d_model, d_model)
        self.ffn_norm = ColumnLayerNorm(num_columns, d_model)
        self.ffn_in = ColumnLinear(num_columns, d_model, d_model * hidden_mult)
        self.ffn_out = ColumnLinear(num_columns, d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_columns, query_len, d_model = query.shape
        context_len = context.shape[2]
        head_dim = d_model // self.num_heads
        normalized_query = self.query_norm(query)
        normalized_context = self.context_norm(context)
        q = (
            self.query(normalized_query)
            .reshape(batch_size * num_columns, query_len, self.num_heads, head_dim)
            .transpose(1, 2)
        )
        k = (
            self.key(normalized_context)
            .reshape(batch_size * num_columns, context_len, self.num_heads, head_dim)
            .transpose(1, 2)
        )
        v = (
            self.value(normalized_context)
            .reshape(batch_size * num_columns, context_len, self.num_heads, head_dim)
            .transpose(1, 2)
        )
        attention_mask = None
        if context_padding_mask is not None:
            valid = ~safe_key_padding_mask(context_padding_mask)
            attention_mask = (
                valid[:, None, None, None, :]
                .expand(batch_size, num_columns, 1, 1, context_len)
                .reshape(batch_size * num_columns, 1, 1, context_len)
            )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, num_columns, query_len, d_model)
        )
        updated = query + self.output(attended)
        ffn = self.ffn_out(self.dropout(F.silu(self.ffn_in(self.ffn_norm(updated)))))
        return updated + ffn


class CoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_columns: int,
        d_model: int,
        num_heads: int,
        hidden_mult: int,
        dropout: float,
        *,
        use_self_attention: bool,
        use_cross_attention: bool,
    ) -> None:
        super().__init__()

        def build() -> AttentionUpdate:
            return AttentionUpdate(
                num_columns, d_model, num_heads, hidden_mult, dropout
            )

        self.user_self = build() if use_self_attention else None
        self.item_self = build() if use_self_attention else None
        self.user_from_item = build() if use_cross_attention else None
        self.item_from_user = build() if use_cross_attention else None

    def forward(
        self, user_tokens: torch.Tensor, item_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.user_self is not None:
            assert self.item_self is not None
            user_tokens = self.user_self(user_tokens, user_tokens)
            item_tokens = self.item_self(item_tokens, item_tokens)
        if self.user_from_item is not None:
            assert self.item_from_user is not None
            user_context = user_tokens
            item_context = item_tokens
            user_tokens = self.user_from_item(user_tokens, item_context)
            item_tokens = self.item_from_user(item_tokens, user_context)
        return user_tokens, item_tokens


class SequenceQueryBridge(nn.Module):
    def __init__(
        self,
        num_columns: int,
        d_model: int,
        num_heads: int,
        hidden_mult: int,
        dropout: float,
        num_queries: int,
        *,
        use_seq_query_attention: bool,
        use_query_seq_attention: bool,
    ) -> None:
        super().__init__()
        self.query_offsets = (
            mark_muon_batched_matrix(
                nn.Parameter(torch.randn(num_columns, num_queries, d_model) * 0.02)
            )
            if use_seq_query_attention
            else None
        )
        self.mlp_query_in = (
            None
            if use_seq_query_attention
            else ColumnLinear(num_columns, d_model * 3, d_model * hidden_mult)
        )
        self.mlp_query_out = (
            None
            if use_seq_query_attention
            else ColumnLinear(num_columns, d_model * hidden_mult, num_queries * d_model)
        )
        self.seq_query_attention = (
            AttentionUpdate(num_columns, d_model, num_heads, hidden_mult, dropout)
            if use_seq_query_attention
            else None
        )
        self.query_seq_attention = (
            AttentionUpdate(num_columns, d_model, num_heads, hidden_mult, dropout)
            if use_query_seq_attention
            else None
        )
        self.num_queries = int(num_queries)
        self.d_model = int(d_model)

    def forward(
        self,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
        sequence_tokens: torch.Tensor,
        sequence_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        item_summary = column_masked_mean(item_tokens)
        sequence_summary = column_masked_mean(sequence_tokens, sequence_padding_mask)
        if self.seq_query_attention is not None:
            assert self.query_offsets is not None
            query = (
                sequence_summary.unsqueeze(2)
                + item_summary.unsqueeze(2)
                + self.query_offsets.unsqueeze(0)
            )
            non_sequence = torch.cat([user_tokens, item_tokens], dim=2)
            query = self.seq_query_attention(query, non_sequence)
        else:
            assert self.mlp_query_in is not None
            assert self.mlp_query_out is not None
            user_summary = column_masked_mean(user_tokens)
            query = self.mlp_query_out(
                F.silu(
                    self.mlp_query_in(
                        torch.cat(
                            [user_summary, item_summary, sequence_summary], dim=-1
                        )
                    )
                )
            ).view(
                sequence_tokens.shape[0],
                sequence_tokens.shape[1],
                self.num_queries,
                self.d_model,
            )
        if self.query_seq_attention is not None:
            query = self.query_seq_attention(
                query, sequence_tokens, sequence_padding_mask
            )
        return query


class ColumnReadout(nn.Module):
    def __init__(self, num_columns: int, d_model: int) -> None:
        super().__init__()
        self.input_norm = ColumnLayerNorm(num_columns, d_model * 3)
        self.project = ColumnLinear(num_columns, d_model * 3, d_model)
        self.output_norm = ColumnLayerNorm(num_columns, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_norm(F.silu(self.project(self.input_norm(inputs))))


class PCVRQueryFormer(EmbeddingParameterMixin, nn.Module):
    """QueryFormer with one explicit, GPU-batched embedding-column path."""

    def __init__(self, schema: PCVRSchema, config: QueryFormerModelConfig) -> None:
        super().__init__()
        if not schema.seq:
            raise ValueError("QueryFormer requires at least one sequence domain")
        specs = build_pcvr_model_specs(schema, config.ns)
        num_columns = config.num_embedding_columns
        d_model = config.d_model
        num_heads = choose_num_heads(d_model, config.num_heads)
        tokenizer_args = (num_columns, config.emb_dim, d_model)
        self.user_sparse = ColumnNonSequentialTokenizer(
            specs.user_int_feature_specs,
            specs.user_ns_groups,
            *tokenizer_args,
            config.ns.user_tokens,
            config.emb_skip_threshold,
            compress_high_cardinality=config.compress_high_cardinality,
        )
        self.item_sparse = ColumnNonSequentialTokenizer(
            specs.item_int_feature_specs,
            specs.item_ns_groups,
            *tokenizer_args,
            config.ns.item_tokens,
            config.emb_skip_threshold,
            compress_high_cardinality=config.compress_high_cardinality,
        )
        self.user_dense = DenseFieldTokenizer(
            [column.dim for column in schema.user_dense],
            d_model,
            config.dcn_num_layers,
            num_columns,
        )
        self.item_dense = DenseFieldTokenizer(
            [column.dim for column in schema.item_dense],
            d_model,
            config.dcn_num_layers,
            num_columns,
        )
        num_time_buckets = NUM_TIME_BUCKETS if config.use_time_buckets else 0
        self.sequence_tokenizers = nn.ModuleDict(
            {
                domain: ColumnSequenceTokenizer(
                    vocab_sizes,
                    num_columns,
                    config.emb_dim,
                    d_model,
                    num_time_buckets,
                    config.emb_skip_threshold,
                    compress_high_cardinality=config.compress_high_cardinality,
                )
                for domain, vocab_sizes in specs.seq_vocab_sizes.items()
            }
        )
        self.seq_domains = tuple(sorted(specs.seq_vocab_sizes))
        self.co_blocks = nn.ModuleList(
            [
                CoTransformerBlock(
                    num_columns,
                    d_model,
                    num_heads,
                    config.hidden_mult,
                    config.dropout_rate,
                    use_self_attention=config.use_query_self_attention,
                    use_cross_attention=config.use_query_cross_attention,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.sequence_bridges = nn.ModuleDict(
            {
                domain: SequenceQueryBridge(
                    num_columns,
                    d_model,
                    num_heads,
                    config.hidden_mult,
                    config.dropout_rate,
                    config.num_queries,
                    use_seq_query_attention=config.use_seq_query_cross_attention,
                    use_query_seq_attention=config.use_query_seq_cross_attention,
                )
                for domain in self.seq_domains
            }
        )
        self.readout = ColumnReadout(num_columns, d_model)
        self.column_fusion = nn.Sequential(
            nn.LayerNorm(d_model * num_columns),
            nn.Linear(d_model * num_columns, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * config.hidden_mult),
            nn.SiLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(d_model * config.hidden_mult, config.action_num),
        )
        self.num_embedding_columns = num_columns
        tokens_per_column = (
            self.user_sparse.num_tokens
            + self.user_dense.num_tokens
            + self.item_sparse.num_tokens
            + self.item_dense.num_tokens
        )
        self.num_ns = num_columns * tokens_per_column
        self.d_model = d_model
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self._compiled_backbone = None
        self._backbone_compiled = False

    @property
    def uses_internal_compile(self) -> bool:
        return True

    def prepare_for_runtime_compile(self) -> None:
        if self._backbone_compiled:
            return
        self._compiled_backbone = torch.compile(self._run_backbone)
        self._backbone_compiled = True

    def _run_backbone(
        self,
        user_sparse: torch.Tensor,
        item_sparse: torch.Tensor,
        user_dense_values: torch.Tensor,
        user_dense_missing_mask: torch.Tensor,
        item_dense_values: torch.Tensor,
        item_dense_missing_mask: torch.Tensor,
        sequence_tokens: tuple[torch.Tensor, ...],
        sequence_padding_masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        user_tokens = [user_sparse]
        user_dense = self.user_dense(user_dense_values, user_dense_missing_mask)
        if user_dense.shape[2] > 0:
            user_tokens.append(user_dense)
        item_tokens = [item_sparse]
        item_dense = self.item_dense(item_dense_values, item_dense_missing_mask)
        if item_dense.shape[2] > 0:
            item_tokens.append(item_dense)
        user_tokens = torch.cat(user_tokens, dim=2)
        item_tokens = torch.cat(item_tokens, dim=2)
        for block in self.co_blocks:
            user_tokens, item_tokens = maybe_gradient_checkpoint(
                block,
                user_tokens,
                item_tokens,
                enabled=self.gradient_checkpointing,
            )
        domain_queries: list[torch.Tensor] = []
        for domain, domain_tokens, padding_mask in zip(
            self.seq_domains,
            sequence_tokens,
            sequence_padding_masks,
            strict=True,
        ):
            domain_queries.append(
                self.sequence_bridges[domain](
                    user_tokens,
                    item_tokens,
                    domain_tokens,
                    padding_mask,
                )
            )
        query_summary = torch.stack(
            [column_masked_mean(query) for query in domain_queries], dim=2
        ).mean(dim=2)
        return self.readout(
            torch.cat(
                [
                    column_masked_mean(user_tokens),
                    column_masked_mean(item_tokens),
                    query_summary,
                ],
                dim=-1,
            )
        )

    def _column_embeddings(self, inputs: PCVRModelInput) -> torch.Tensor:
        user_sparse = self.user_sparse(
            inputs.user.int_values, inputs.user.int_missing_mask
        )
        item_sparse = self.item_sparse(
            inputs.item.int_values, inputs.item.int_missing_mask
        )
        sequence_tokens: list[torch.Tensor] = []
        sequence_padding_masks: list[torch.Tensor] = []
        for domain in self.seq_domains:
            sequence_input = inputs.sequences[domain]
            domain_tokens = self.sequence_tokenizers[domain](
                sequence_input.values,
                sequence_input.timestamps,
                inputs.request_timestamp,
            )
            positions = sinusoidal_positions(
                domain_tokens.shape[2],
                self.d_model,
                domain_tokens.device,
            ).to(domain_tokens.dtype)
            sequence_tokens.append(domain_tokens + positions.unsqueeze(0).unsqueeze(0))
            padding_mask = make_padding_mask(
                sequence_input.lengths.to(domain_tokens.device),
                domain_tokens.shape[2],
            )
            sequence_padding_masks.append(padding_mask)
        backbone = self._compiled_backbone or self._run_backbone
        return backbone(
            user_sparse,
            item_sparse,
            inputs.user.dense_values,
            inputs.user.dense_missing_mask,
            inputs.item.dense_values,
            inputs.item.dense_missing_mask,
            tuple(sequence_tokens),
            tuple(sequence_padding_masks),
        )

    def _embed(self, inputs: PCVRModelInput) -> torch.Tensor:
        columns = self._column_embeddings(inputs)
        return self.column_fusion(columns.flatten(start_dim=1))

    def forward(self, inputs: PCVRModelInput) -> torch.Tensor:
        return self.classifier(self._embed(inputs))

    def predict(self, inputs: PCVRModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(inputs)
        return self.classifier(embeddings), embeddings

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10_000
    ) -> set[int]:
        reinitialized: set[int] = set()
        for module in self.modules():
            if not isinstance(module, ColumnEmbedding):
                continue
            if module.num_buckets <= cardinality_threshold:
                continue
            nn.init.xavier_normal_(module.weight)
            module.weight.data[0].zero_()
            reinitialized.add(module.weight.data_ptr())
        return reinitialized


__all__ = [
    "ColumnEmbedding",
    "ColumnLinear",
    "CrossNetworkV2",
    "DenseFieldTokenizer",
    "PCVRQueryFormer",
]
