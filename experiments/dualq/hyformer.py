"""DualQ-private HyFormer building blocks.

Source: https://github.com/zzhlkw-ai/TAAC2026 (MIT License), ``round1best`` /
``round2best`` (ported as ``dualq``) variants (TAAC2026 academic track
Round-1 first place and Round-2 Top-17 / Unified Module Innovation Award).
The top-level model composition lives in ``model.py``. These components remain
experiment-private until another in-repository model shares their contract.

Differences from ``experiments/baseline/model.py``:

* gated RoPE attention (``W_g`` output gate),
* ``DualQGenerator`` per-side query construction with a time token,
* ``TimeTokenBuilder`` auxiliary time-context signal,
* ``RankMixerNSTokenizer`` with an ``extra_emb_dim`` cross-feature input and
  ``CrossRankMixerNSTokenizer`` fusing pair int/dense features,
* per-position float time features (``TS_FLOAT_DIM``) fused into seq tokens,
* optional seq gap buckets, per-domain time/gap gates, and a global time
  token.

Model-side time features are pure tensor functions: the per-position float
features (``TS_FLOAT_DIM = 8``) and per-domain statistics
(``TS_STAT_DIM = 6``) are derived from raw event timestamps by the experiment
model, never by the data pipeline.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TS_FLOAT_DIM = 8
TS_STAT_DIM = 6


class RotaryEmbedding(nn.Module):
    """Pre-computes and caches RoPE cos/sin tables up to ``max_seq_len``.

    Args:
        dim: rotary subspace dimension; should match the per-head dim.
        max_seq_len: maximum length the cache is built for.
        base: base frequency (the customary value is ``10000.0``).
    """

    def __init__(
        self, dim: int, max_seq_len: int = 2048, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer(
            "cos_cached", emb.cos().unsqueeze(0), persistent=False
        )  # (1, seq_len, dim)
        self.register_buffer(
            "sin_cached", emb.sin().unsqueeze(0), persistent=False
        )  # (1, seq_len, dim)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns pre-computed slices of the RoPE tables.

        The cache is built once in ``__init__`` with ``max_seq_len``; no
        runtime expansion is performed so the forward pass remains compatible
        with ``torch.compile()``.
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


class SwiGLU(nn.Module):
    """Gated linear unit with a SiLU non-linearity in the gate branch."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with optional rotary positional embeddings."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        q_rope_cos: torch.Tensor | None = None,
        q_rope_sin: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        del need_weights
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)  # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)
            if self.rope_on_q:
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        sdpa_attn_mask = None
        if key_padding_mask is not None:
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            bool_attn = attn_mask == 0  # (Lq, Lk), -inf means do not attend
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve
        # the original input via the residual connection).
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross attention from queries to sequence tokens.

    Queries come from the global Q-token bank; keys and values come from the
    encoded sequence. RoPE is applied only on the KV side because queries do
    not carry intrinsic positional semantics.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.0, ln_mode: str = "pre"
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ["pre", "post"]:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == "pre":
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == "post":
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """Query-boosting block from the HyFormer / RankMixer family.

    Three operations in order:

    1. Token mixing — parameter-free reshape that interleaves query and
       NS tokens.
    2. Token-wise FFN with shared parameters across positions.
    3. Residual add: ``Q_boost = Q + Q_e``.

    In ``full`` mode ``d_model`` must divide ``n_total = Nq + Nns``.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = "full",  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == "none":
            return

        if mode == "full":
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape
        Q_split = Q.view(B, T, self.T, self.d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()
        return Q_rewired.view(B, T, D)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection."""
        if self.mode == "none":
            return Q

        if self.mode == "full":
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class DualQGenerator(nn.Module):
    """Build per-side query tokens from NS tokens and an auxiliary time
    token.

    The user/item arms are kept asymmetric: each side has its own number of
    tokens, its own KV projection, and its own attention head. The output is
    two query stacks that downstream HyFormer blocks consume separately.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_sequences: int,
        ts_stat_dim: int = TS_STAT_DIM,
        dropout: float = 0.0,
        user_q_tokens: int = 2,
        item_q_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.user_q_tokens = user_q_tokens
        self.item_q_tokens = item_q_tokens
        self.total_q_tokens = user_q_tokens + item_q_tokens

        self.user_pool_score = nn.Linear(d_model, user_q_tokens)
        self.item_pool_score = nn.Linear(d_model, item_q_tokens)
        self.cross_user_view = CrossAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
        )
        self.cross_item_view = CrossAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
        )
        self.film_per_domain = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(ts_stat_dim),
                    nn.Linear(ts_stat_dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 2 * d_model),
                )
                for _ in range(num_sequences)
            ]
        )
        for film in self.film_per_domain:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)

    @staticmethod
    def _multi_attn_pool(score_head: nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
        scores = score_head(tokens)  # (B, L, K)
        weights = torch.softmax(scores, dim=1)
        return torch.einsum("blk,bld->bkd", weights, tokens)

    def forward(
        self,
        user_ns: torch.Tensor,
        item_ns: torch.Tensor,
        time_token: torch.Tensor,
        ts_stat_feats_list: list[torch.Tensor],
        item_time_context: torch.Tensor | None = None,
    ) -> list:
        context = (
            time_token
            if item_time_context is None
            else torch.cat([time_token, item_time_context], dim=1)
        )
        user_ns_kv = torch.cat([user_ns, context], dim=1)
        item_ns_kv = torch.cat([item_ns, context], dim=1)

        user_cross = self.cross_user_view(user_ns, item_ns_kv)
        item_cross = self.cross_item_view(item_ns, user_ns_kv)
        user_q = self._multi_attn_pool(self.user_pool_score, user_cross)
        item_q = self._multi_attn_pool(self.item_pool_score, item_cross)
        q_base = torch.cat([user_q, item_q], dim=1)

        q_tokens_list = []
        for i, ts_stat in enumerate(ts_stat_feats_list):
            gamma_beta = self.film_per_domain[i](ts_stat.to(q_base.device))
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            q_tokens_list.append(q_base * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1))
        return q_tokens_list


class TimeTokenBuilder(nn.Module):
    """Compress request hour-of-day, per-seq time stats, and per-seq
    activity flags into a single ``(B, 1, D)`` time-context token.

    The token is consumed by ``DualQGenerator`` so query construction can
    condition on temporal context that is not directly attached to any
    sequence position.
    """

    def __init__(
        self,
        num_sequences: int,
        d_model: int,
        hod_dim: int,
        ts_stat_dim: int = TS_STAT_DIM,
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        in_dim = hod_dim + num_sequences * ts_stat_dim + num_sequences
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model * hidden_mult),
            nn.SiLU(),
            nn.Linear(d_model * hidden_mult, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        hod_feats: torch.Tensor,
        ts_stat_list: list[torch.Tensor],
        seq_lens_list: list[torch.Tensor],
    ) -> torch.Tensor:
        dtype = hod_feats.dtype
        device = hod_feats.device
        activity = torch.stack(
            [(lens.to(device) > 0).to(dtype) for lens in seq_lens_list], dim=1
        )
        stat_flat = torch.cat(
            [stat.to(device=device, dtype=dtype) for stat in ts_stat_list], dim=1
        )
        feats = torch.cat([hod_feats, stat_flat, activity], dim=1)
        return self.proj(feats).unsqueeze(1)


class SwiGLUEncoder(nn.Module):
    """Attention-free sequence encoder: ``x + Dropout(SwiGLU(LN(x)))``."""

    def __init__(
        self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """Standard Pre-LN transformer encoder layer with RoPE attention."""

    def __init__(
        self, d_model: int, num_heads: int, hidden_mult: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask


class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Behaviour switches based on input length:

    * ``L > top_k`` (typical first block) — cross attention with the latest
      ``top_k`` tokens as queries and the full sequence as K/V. Output shape
      ``(B, top_k, D)``.
    * ``L <= top_k`` (later blocks) — self attention over the previously
      compressed ``top_k`` tokens. Output shape ``(B, top_k, D)``.

    The optional causal mask only fires in the self-attention case; cross
    attention has differing Q/K lengths so causality is undefined.

    Returns ``(output, new_key_padding_mask)`` so callers can keep the mask
    in sync after compression.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def _gather_top_k(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample."""
        B, L, D = x.shape
        device = x.device

        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)
        indices = torch.clamp(indices, min=0, max=L - 1)

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention."""
        B, L, _ = x.shape

        if L > self.top_k:
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                head_dim = rope_cos.shape[2]
                cos_expanded = rope_cos.expand(B, -1, -1)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out
        else:
            new_mask = key_padding_mask

            x_normed = self.norm_q(x)

            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
) -> nn.Module:
    """Factory for the per-domain sequence encoder."""
    if encoder_type == "swiglu":
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == "transformer":
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == "longer":
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


class MultiSeqHyFormerBlock(nn.Module):
    """One HyFormer block for the DualQ Q-only fusion path.

    The block does three things in sequence:

    * runs a per-domain sequence encoder over each behaviour sequence,
    * cross/self attends the user and item query stacks against the
      encoded sequences and the NS token bank,
    * mixes the resulting tokens through a final feed-forward layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = "swiglu",
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = "full",
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        self.seq_encoders = nn.ModuleList(
            [
                create_sequence_encoder(
                    encoder_type=seq_encoder_type,
                    d_model=d_model,
                    num_heads=num_heads,
                    hidden_mult=hidden_mult,
                    dropout=dropout,
                    top_k=top_k,
                    causal=causal,
                )
                for _ in range(num_sequences)
            ]
        )

        self.cross_attns = nn.ModuleList(
            [
                CrossAttention(
                    d_model=d_model, num_heads=num_heads, dropout=dropout, ln_mode="pre"
                )
                for _ in range(num_sequences)
            ]
        )
        # RankMixer only mixes Q tokens; NS tokens are consumed by DualQGenerator.
        n_total = num_queries * num_sequences
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode,
        )

    def forward(
        self,
        q_tokens_list: list,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: list[torch.Tensor] | None = None,
        rope_sin_list: list[torch.Tensor] | None = None,
    ) -> tuple[list, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i],
                seq_padding_masks[i],
                rope_cos=rc,
                rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            q_in = q_tokens_list[i]
            decoded_q_i = self.cross_attns[i](
                q_in,
                next_seqs[i],
                next_masks[i],
                rope_cos=rc,
                rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # Token Fusion: concatenate only decoded Q tokens.
        combined = torch.cat(decoded_qs, dim=1)  # (B, Nq*S, D)

        boosted = self.mixer(combined)  # (B, Nq*S, D)

        next_q_list = []
        for i in range(S):
            next_q_list.append(boosted[:, i * Nq : (i + 1) * Nq, :])

        return next_q_list, next_seqs, next_masks


class GroupNSTokenizer(nn.Module):
    """NS tokenizer for ``ns_tokenizer_type='group'``.

    Groups discrete features by fid, runs a shared embedding with multi-value
    mean pooling, then projects each group to one NS token (output is one
    token per group, regardless of feature count).
    """

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_skip_threshold = emb_skip_threshold
        self.emb_dim_list = []

        embs = []
        for vs, _offset, _length in feature_specs:
            skip = int(vs) <= 0 or (
                emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            )
            if skip:
                embs.append(None)
                self.emb_dim_list.append(0)
            else:
                emb_dim = get_emb_dim(vs, 64)
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0, sparse=True))
                self.emb_dim_list.append(emb_dim)

        self.embs = nn.ModuleList([e for e in embs if e is not None])

        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        self.group_dims = [
            sum(self.emb_dim_list[fid] for fid in group) for group in groups
        ]

        self.group_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for dim in self.group_dims
            ]
        )

        self.group_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, max(8, dim // 4)),
                    nn.SiLU(),
                    nn.Linear(max(8, dim // 4), dim),
                    nn.Sigmoid(),
                )
                for dim in self.group_dims
            ]
        )

        self.attn_layers = nn.ModuleList(
            [nn.Linear(dim, 1) if dim > 0 else None for dim in self.emb_dim_list]
        )

    def forward(self, int_feats):
        B = int_feats.size(0)
        tokens = []

        for group_idx, (group, proj) in enumerate(
            zip(self.groups, self.group_projs, strict=True)
        ):
            fid_embs = []

            for fid_idx in group:
                _vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                dim = self.emb_dim_list[fid_idx]

                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(B, dim)
                else:
                    emb_layer = self.embs[emb_real_idx]

                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset : offset + length].long()
                        emb_all = emb_layer(vals)  # (B, L, D)

                        mask = (vals != 0).unsqueeze(-1)  # (B, L, 1)

                        attn = self.attn_layers[fid_idx](emb_all)  # (B, L, 1)
                        attn = attn.masked_fill(~mask, -1e9)
                        attn = torch.softmax(attn, dim=1)

                        fid_emb = (emb_all * attn).sum(dim=1)

                fid_embs.append(fid_emb)

            cat_emb = torch.cat(fid_embs, dim=-1)

            gate = self.group_gates[group_idx](cat_emb)
            cat_emb = cat_emb * (1.0 + gate)

            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))

        return torch.cat(tokens, dim=1)


def get_emb_dim(vocab_size: int, emb_dim: int) -> int:
    """Pick an embedding dim by piecewise rule of thumb."""
    if vocab_size <= 4:
        return 4
    elif vocab_size <= 10:
        return 8
    elif vocab_size <= 50:
        return 16
    elif vocab_size <= 600:
        return 32
    else:
        return 64


class RankMixerNSTokenizer(nn.Module):
    """RankMixer-style NS tokenizer.

    Concatenates every per-fid embedding into a single flat vector, then
    splits that vector into ``num_ns_tokens`` equal-width chunks and projects
    each chunk to ``d_model``. Decoupling the token count from the group
    count lets the caller scale the NS bandwidth freely.
    """

    def __init__(
        self,
        feature_specs: list[tuple[int, int, int]],
        groups: list[list[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        extra_emb_dim: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.total_emb_dim = 0
        self.offset_to_index = {}

        embs = []
        count = 0
        for vs, offset, _length in feature_specs:
            skip = int(vs) <= 0 or (
                emb_skip_threshold > 0 and int(vs) > emb_skip_threshold
            )
            if skip:
                embs.append(None)
                # Skipped features still contribute a zero vector of size
                # emb_dim in forward(), so account for them here.
                self.total_emb_dim += emb_dim
            else:
                vs_emb_dim = get_emb_dim(vs, emb_dim)
                self.total_emb_dim += vs_emb_dim
                embs.append(nn.Embedding(int(vs) + 1, vs_emb_dim, padding_idx=0, sparse=True))
                self.offset_to_index[offset] = count
                count += 1

        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Fold in extra embeddings (e.g. pair features injected from outside)
        self.total_emb_dim += extra_emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(self.total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - self.total_emb_dim

        self.lhuc = nn.Sequential(
            nn.Linear(self.total_emb_dim, self.total_emb_dim // 4),
            nn.SiLU(),
            nn.Linear(self.total_emb_dim // 4, self.total_emb_dim),
            nn.Sigmoid(),
        )

        self.token_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.chunk_dim, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_ns_tokens)
            ]
        )

        logging.info(
            f"RankMixerNSTokenizer: {len(feature_specs)} fids, "
            f"total_emb_dim={self.total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor, extra_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.
            extra_emb: optional (B, extra_emb_dim) tensor appended before LHUC,
                e.g. pair feature embeddings from CrossRankMixerNSTokenizer.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        B = int_feats.size(0)
        group_outputs = []

        for group in self.groups:
            fid_outputs = []

            for fid_idx in group:
                _vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]

                x = int_feats[:, offset : offset + length]

                if emb_real_idx == -1:
                    fid_outputs.append(x.new_zeros(B, self.emb_dim))
                    continue

                emb_layer = self.embs[emb_real_idx]

                # ---- scalar feature ----
                if length == 1:
                    fid_outputs.append(emb_layer(x.squeeze(-1)))
                    continue

                # ---- sequence feature (vectorized pooling) ----
                vals = x.long()  # (B, L)
                emb_all = emb_layer(vals)  # (B, L, D)

                mask = (vals != 0).unsqueeze(-1)  # (B, L, 1)
                denom = mask.sum(dim=1).clamp(min=1)

                fid_emb = (emb_all * mask).sum(dim=1) / denom
                fid_outputs.append(fid_emb)

            group_emb = torch.cat(fid_outputs, dim=-1)
            group_outputs.append(group_emb)

        cat_emb = torch.cat(group_outputs, dim=-1)
        if extra_emb is not None:
            cat_emb = torch.cat([cat_emb, extra_emb], dim=-1)
        gate = self.lhuc(cat_emb)
        cat_emb = cat_emb * gate * 2.0

        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        cat_emb = cat_emb.view(B, self.num_ns_tokens, self.chunk_dim)

        outs = []
        for i, proj in enumerate(self.token_projs):
            outs.append(proj(cat_emb[:, i]).unsqueeze(1))

        return torch.cat(outs, dim=1)


class CrossRankMixerNSTokenizer(nn.Module):
    """Pair-feature embedder injected into the user-side NS tokenizer.

    Each pair feature carries paired int (category id) and dense
    (weight/score) arrays. For every feature we embed the int values,
    re-weight them with the dense scores, aggregate over positions, and
    flatten the per-feature embeddings into ``(B, num_features * emb_dim)``.

    Two pooling strategies coexist:

    * Head features (fids 62-66) — dense values are ``log1p`` counts; we
      L1-normalise and use as plain weights.
    * Tail features (fids 89-91) — dense values are similarity scores (can
      be negative); we mask padding to ``-inf`` and softmax along the valid
      positions.

    The result is concatenated onto the user-side tokenizer so pair signals
    get folded into ``user_ns`` rather than living in a separate token row.
    """

    def __init__(
        self,
        feature_specs,
        d_model: int,
        emb_dim: int = 64,
        num_pos: int = 10,
        feature_fids: list[int] | None = None,
        use_weighted_residual: bool = False,
    ):
        super().__init__()

        self.feature_specs = feature_specs
        self.d_model = d_model
        self.emb_dim = emb_dim
        self.num_pos = num_pos
        self.feature_fids = [int(fid) for fid in feature_fids] if feature_fids else None

        self.num_head = len(feature_specs) - 3
        self.num_tail = 3

        self.embs = nn.ModuleList(
            [
                nn.Embedding(vs + 1, get_emb_dim(vs, emb_dim), padding_idx=0, sparse=True)
                for (vs, _, _) in feature_specs
            ]
        )
        self._emb_dims = [get_emb_dim(vs, emb_dim) for (vs, _, _) in feature_specs]

        # flat output dim — injected into user_ns_tokenizer before LHUC
        self.out_dim = sum(self._emb_dims)
        self.dense_projs = nn.ModuleList([nn.Linear(1, d) for d in self._emb_dims])
        self.fusion_projs = nn.ModuleList(
            [nn.Linear(d * 2 + 4, d) for d in self._emb_dims]
        )
        # dense-weighted pooling residual (dualq only): re-weights int
        # embeddings by the dense scores and adds a gated residual onto the
        # plain mean-pooled path. Zero-init projection + gate≈0.018 at init.
        self.use_weighted_residual = bool(use_weighted_residual)
        if self.use_weighted_residual:
            self.weighted_residual_projs = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for d in self._emb_dims]
            )
            self.weighted_residual_gates = nn.Parameter(
                torch.full((len(self._emb_dims),), -4.0)
            )
            for proj in self.weighted_residual_projs:
                nn.init.zeros_(proj.weight)

    def _use_signed_weights(self, index: int) -> bool:
        if self.feature_fids is not None and index < len(self.feature_fids):
            return self.feature_fids[index] in (89, 90, 91)
        return index >= max(0, len(self.feature_specs) - 3)

    def _slice(self, x, offset, length):
        return x[:, offset : offset + length]

    def forward(self, pair_int_feats, pair_dense_feats):
        """Position-aligned int/dense pooling for all pair features.

        pair_int_feats:  (B, total_dim)
        pair_dense_feats: (B, total_dim)

        Returns (B, out_dim) flat embedding.
        """
        outs = []

        for i in range(len(self.feature_specs)):
            _, offset, length = self.feature_specs[i]
            x = self._slice(pair_int_feats, offset, length)
            dense = self._slice(pair_dense_feats, offset, length)

            valid = (x != 0) & torch.isfinite(dense)
            mask = valid.float()
            valid_count = mask.sum(dim=1, keepdim=True)
            count = valid_count.clamp(min=1.0)
            mask_3d = mask.unsqueeze(-1)

            int_emb = self.embs[i](x.long())
            int_pool = (int_emb * mask_3d).sum(dim=1) / count

            dense_clean = torch.where(valid, dense, torch.zeros_like(dense))
            dense_emb = self.dense_projs[i](dense_clean.unsqueeze(-1))
            dense_pool = (dense_emb * mask_3d).sum(dim=1) / count

            has_valid = valid.any(dim=1, keepdim=True)
            dense_max = torch.where(
                valid, dense, torch.full_like(dense, float("-inf"))
            ).max(dim=1, keepdim=True).values
            dense_min = torch.where(
                valid, dense, torch.full_like(dense, float("inf"))
            ).min(dim=1, keepdim=True).values
            dense_max = torch.where(has_valid, dense_max, torch.zeros_like(dense_max))
            dense_min = torch.where(has_valid, dense_min, torch.zeros_like(dense_min))
            dense_mean = dense_clean.sum(dim=1, keepdim=True) / count
            dense_var = (((dense_clean - dense_mean) * mask) ** 2).sum(
                dim=1, keepdim=True
            ) / count
            dense_std = torch.sqrt(dense_var.clamp_min(0.0) + 1e-6)
            dense_std = torch.where(has_valid, dense_std, torch.zeros_like(dense_std))

            stats = torch.cat(
                [valid_count.log1p(), dense_max, dense_min, dense_std], dim=-1
            )
            fused = torch.cat([int_pool, dense_pool, stats], dim=-1)
            out = F.silu(self.fusion_projs[i](fused))
            if self.use_weighted_residual:
                if self._use_signed_weights(i):
                    logits = dense.masked_fill(~valid, float("-inf"))
                    logits = torch.where(has_valid, logits, torch.zeros_like(logits))
                    weights = torch.softmax(logits, dim=1) * mask
                    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                else:
                    weights_raw = dense_clean.clamp_min(0.0) * mask
                    weights_sum = weights_raw.sum(dim=1, keepdim=True)
                    uniform = mask / count
                    weights = torch.where(
                        weights_sum > 1e-6,
                        weights_raw / weights_sum.clamp_min(1e-6),
                        uniform,
                    )
                weighted_pool = (int_emb * weights.unsqueeze(-1)).sum(dim=1)
                residual = self.weighted_residual_projs[i](weighted_pool - int_pool)
                gate = torch.sigmoid(self.weighted_residual_gates[i])
                out = out + gate * residual
            outs.append(out)

        # softmax over zeros gives uniform weights → zero embeddings, so
        # fully-padded rows collapse to the zero vector.
        return torch.cat(outs, dim=-1)  # (B, out_dim)


class ItemDenseTokenizer(nn.Module):
    """FT-Transformer-style tokenizer for item dense feature fields.

    Instead of compressing all item dense values into one vector, each schema
    field gets its own projection and becomes one item-side token.
    """

    def __init__(
        self,
        entries: list[tuple[int, int, int]],
        d_model: int,
    ) -> None:
        super().__init__()
        self.entries = []
        self.projs = nn.ModuleList()
        for fid, offset, dim in entries:
            fid = int(fid)
            offset = int(offset)
            dim = int(dim)
            if fid == 129 and dim >= 130:
                self.entries.append((fid, offset, 128, "body"))
                self.entries.append((fid, offset + 128, 1, "stat"))
            else:
                self.entries.append((fid, offset, dim, "full"))
        for _, _, dim, _kind in self.entries:
            if dim <= 0:
                raise ValueError(f"item dense dim must be positive, got {dim}")
            norm = nn.LayerNorm(dim) if dim > 1 else nn.Identity()
            self.projs.append(
                nn.Sequential(
                    norm,
                    nn.Linear(dim, d_model),
                    nn.SiLU(),
                    nn.LayerNorm(d_model),
                )
            )

    @property
    def num_tokens(self) -> int:
        return len(self.entries)

    def forward(self, item_dense_feats: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        for (_, offset, dim, kind), proj in zip(self.entries, self.projs, strict=True):
            x = item_dense_feats[:, offset : offset + dim]
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if kind == "stat":
                x = torch.log1p(x.clamp_min(0.0))
            tokens.append(proj(x).unsqueeze(1))
        return torch.cat(tokens, dim=1)


__all__ = [
    "TS_FLOAT_DIM",
    "TS_STAT_DIM",
    "CrossAttention",
    "CrossRankMixerNSTokenizer",
    "DualQGenerator",
    "GroupNSTokenizer",
    "ItemDenseTokenizer",
    "LongerEncoder",
    "MultiSeqHyFormerBlock",
    "RankMixerBlock",
    "RankMixerNSTokenizer",
    "RoPEMultiheadAttention",
    "RotaryEmbedding",
    "SwiGLU",
    "SwiGLUEncoder",
    "TimeTokenBuilder",
    "TransformerEncoder",
    "apply_rope_to_tensor",
    "create_sequence_encoder",
    "get_emb_dim",
    "rotate_half",
]
