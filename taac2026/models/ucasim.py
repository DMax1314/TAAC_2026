from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ModelConfig
from .common import build_pooled_memory, masked_mean


class UCASIMBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_multiplier: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _modulate_tokens(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        candidate_token: torch.Tensor,
        static_state: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.size(1) == 0:
            return tokens

        candidate_expand = candidate_token.expand(-1, tokens.size(1), -1)
        static_expand = static_state.unsqueeze(1).expand_as(tokens)
        gate_input = torch.cat([tokens, candidate_expand, static_expand], dim=-1)
        gated_delta = self.gate(gate_input) * self.delta(gate_input)
        return tokens + self.dropout(gated_delta) * token_mask.unsqueeze(-1).float()

    def forward(
        self,
        candidate_token: torch.Tensor,
        local_history: torch.Tensor,
        local_mask: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        static_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_candidate = self.candidate_norm(candidate_token)
        normalized_history = self.history_norm(local_history)
        normalized_memory = self.memory_norm(memory_tokens)
        normalized_context = self.context_norm(context_tokens)

        kv = torch.cat([normalized_history, normalized_memory, normalized_context], dim=1)
        kv_mask = torch.cat([local_mask, memory_mask, context_mask], dim=1)
        attended, _ = self.cross_attention(
            query=normalized_candidate,
            key=kv,
            value=kv,
            key_padding_mask=~kv_mask,
            need_weights=False,
        )
        candidate_token = candidate_token + self.dropout(attended)
        candidate_token = candidate_token + self.dropout(self.feed_forward(self.ffn_norm(candidate_token)))

        local_history = self._modulate_tokens(local_history, local_mask, candidate_token, static_state)
        memory_tokens = self._modulate_tokens(memory_tokens, memory_mask, candidate_token, static_state)
        return candidate_token, local_history, memory_tokens


class UCASIMv1(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.recent_seq_len = min(config.recent_seq_len, max_seq_len)
        self.memory_slots = config.memory_slots
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, config.embedding_dim)
        self.source_embedding = nn.Embedding(4, config.embedding_dim)
        self.sequence_group_embedding = nn.Embedding(4, config.embedding_dim, padding_idx=0)
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.hidden_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.static_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.global_query_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                UCASIMBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim * 9, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def _embed_tokens(self, token_ids: torch.Tensor, source_id: int) -> torch.Tensor:
        source = self.source_embedding.weight[source_id].view(1, 1, -1)
        return self.hidden_projection(self.token_embedding(token_ids) + source)

    def _build_memory(
        self,
        history_tokens: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_slots == 0:
            return history_tokens[:, :0], history_mask[:, :0]
        return build_pooled_memory(
            history_embeddings=history_tokens,
            history_mask=history_mask,
            recent_seq_len=self.recent_seq_len,
            memory_slots=self.memory_slots,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_tokens = self._embed_tokens(batch["candidate_tokens"], source_id=0)
        context_tokens = self._embed_tokens(batch["context_tokens"], source_id=1)
        history_token_embeddings = self.token_embedding(batch["history_tokens"])
        history_component_embeddings = self.token_embedding(batch["history_component_tokens"])
        history_component_weights = batch["history_component_mask"].unsqueeze(-1).float()
        history_component_summary = (history_component_embeddings * history_component_weights).sum(dim=2)
        history_component_summary = history_component_summary / history_component_weights.sum(dim=2).clamp_min(1.0)
        history_group_embeddings = self.sequence_group_embedding(batch["history_group_ids"])
        history_source = self.source_embedding.weight[2].view(1, 1, -1)
        history_tokens = self.hidden_projection(
            history_token_embeddings + history_component_summary + history_group_embeddings + history_source
        )

        positions = torch.arange(
            batch["history_tokens"].size(1),
            device=batch["history_tokens"].device,
        ).unsqueeze(0)
        history_tokens = history_tokens + self.hidden_projection(self.position_embedding(positions))
        history_tokens = history_tokens + self.hidden_projection(self.time_projection(batch["history_time_gaps"].unsqueeze(-1)))

        candidate_summary = masked_mean(candidate_tokens, batch["candidate_mask"])
        context_summary = masked_mean(context_tokens, batch["context_mask"])
        dense_summary = self.dense_projection(batch["dense_features"])
        static_state = self.static_projection(torch.cat([candidate_summary, context_summary, dense_summary], dim=-1))
        global_query = self.global_query_projection(torch.cat([candidate_summary, context_summary, dense_summary], dim=-1))

        global_scores = (history_tokens * global_query.unsqueeze(1)).sum(dim=-1) / math.sqrt(history_tokens.size(-1))
        global_scores = global_scores.masked_fill(~batch["history_mask"], -1e9)
        global_weights = torch.softmax(global_scores, dim=-1)
        global_weights = global_weights * batch["history_mask"].float()
        global_weights = global_weights / global_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        global_summary = (history_tokens * global_weights.unsqueeze(-1)).sum(dim=1)

        local_history = history_tokens[:, : self.recent_seq_len]
        local_mask = batch["history_mask"][:, : self.recent_seq_len]
        memory_tokens, memory_mask = self._build_memory(history_tokens, batch["history_mask"])
        candidate_token = (candidate_summary + static_state).unsqueeze(1)

        for block in self.blocks:
            candidate_token, local_history, memory_tokens = block(
                candidate_token=candidate_token,
                local_history=local_history,
                local_mask=local_mask,
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
                context_tokens=context_tokens,
                context_mask=batch["context_mask"],
                static_state=static_state,
            )

        candidate_final = candidate_token.squeeze(1)
        local_summary = masked_mean(local_history, local_mask)
        memory_summary = masked_mean(memory_tokens, memory_mask) if memory_tokens.size(1) > 0 else torch.zeros_like(candidate_final)
        interaction_local = candidate_final * local_summary
        interaction_global = candidate_final * global_summary
        difference = torch.abs(global_summary - memory_summary)

        fused = torch.cat(
            [
                candidate_final,
                candidate_summary,
                context_summary,
                local_summary,
                memory_summary,
                global_summary,
                interaction_local,
                interaction_global,
                difference,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["UCASIMBlock", "UCASIMv1"]