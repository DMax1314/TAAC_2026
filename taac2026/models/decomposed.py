from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .common import build_decomposed_history_embeddings, build_pooled_memory, masked_attention_pool, masked_mean


class DecomposedCandidateBaseline(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, config.embedding_dim)
        self.sequence_group_embedding = nn.Embedding(4, config.embedding_dim, padding_idx=0)
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.component_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 3 + config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        fusion_dim = config.embedding_dim * 7 + config.hidden_dim
        self.output = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_embeddings = self.token_embedding(batch["candidate_tokens"])
        context_embeddings = self.token_embedding(batch["context_tokens"])
        history_embeddings, component_summary = build_decomposed_history_embeddings(
            token_embedding=self.token_embedding,
            position_embedding=self.position_embedding,
            sequence_group_embedding=self.sequence_group_embedding,
            time_projection=self.time_projection,
            component_projection=self.component_projection,
            batch=batch,
        )

        candidate_summary = masked_mean(candidate_embeddings, batch["candidate_mask"])
        context_summary = masked_mean(context_embeddings, batch["context_mask"])
        component_history_summary = masked_mean(component_summary, batch["history_mask"])
        dense_summary = self.dense_projection(batch["dense_features"])

        query = self.query_projection(
            torch.cat([candidate_summary, context_summary, component_history_summary, dense_summary], dim=-1)
        )
        history_summary = masked_attention_pool(history_embeddings, batch["history_mask"], query)
        interaction = candidate_summary * history_summary
        component_interaction = candidate_summary * component_history_summary
        difference = torch.abs(history_summary - component_history_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                history_summary,
                component_history_summary,
                interaction,
                component_interaction,
                difference,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class DecomposedDualPathBaseline(nn.Module):
    def __init__(self, config: ModelConfig, dense_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.memory_slots = max(config.memory_slots, 1)
        self.recent_seq_len = min(config.recent_seq_len, max_seq_len)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len + 1, config.embedding_dim)
        self.sequence_group_embedding = nn.Embedding(4, config.embedding_dim, padding_idx=0)
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.component_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 3 + config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        fusion_dim = config.embedding_dim * 11 + config.hidden_dim
        self.output = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def _build_history_embeddings(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return build_decomposed_history_embeddings(
            token_embedding=self.token_embedding,
            position_embedding=self.position_embedding,
            sequence_group_embedding=self.sequence_group_embedding,
            time_projection=self.time_projection,
            component_projection=self.component_projection,
            batch=batch,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_embeddings = self.token_embedding(batch["candidate_tokens"])
        context_embeddings = self.token_embedding(batch["context_tokens"])
        history_embeddings, component_summary = self._build_history_embeddings(batch)

        candidate_summary = masked_mean(candidate_embeddings, batch["candidate_mask"])
        context_summary = masked_mean(context_embeddings, batch["context_mask"])
        component_history_summary = masked_mean(component_summary, batch["history_mask"])
        dense_summary = self.dense_projection(batch["dense_features"])

        query = self.query_projection(
            torch.cat([candidate_summary, context_summary, component_history_summary, dense_summary], dim=-1)
        )
        global_summary = masked_attention_pool(history_embeddings, batch["history_mask"], query)
        local_history = history_embeddings[:, : self.recent_seq_len]
        local_mask = batch["history_mask"][:, : self.recent_seq_len]
        local_summary = masked_attention_pool(local_history, local_mask, query)
        memory_embeddings, memory_mask = build_pooled_memory(
            history_embeddings=history_embeddings,
            history_mask=batch["history_mask"],
            recent_seq_len=self.recent_seq_len,
            memory_slots=self.memory_slots,
        )
        memory_summary = masked_attention_pool(memory_embeddings, memory_mask, query)

        interaction_global = candidate_summary * global_summary
        interaction_local = candidate_summary * local_summary
        interaction_memory = candidate_summary * memory_summary
        local_memory_gap = torch.abs(local_summary - memory_summary)
        global_component_gap = torch.abs(global_summary - component_history_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                global_summary,
                local_summary,
                memory_summary,
                component_history_summary,
                interaction_global,
                interaction_local,
                interaction_memory,
                local_memory_gap,
                global_component_gap,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["DecomposedCandidateBaseline", "DecomposedDualPathBaseline"]