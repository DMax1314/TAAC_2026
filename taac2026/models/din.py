from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .common import DINActivationUnit, build_decomposed_history_embeddings, masked_mean


class CreatorwyxDINAdapter(nn.Module):
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
        self.context_projection = nn.Sequential(
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
        self.din_attention = DINActivationUnit(config.embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(config.embedding_dim * 8 + config.hidden_dim, config.hidden_dim),
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
        history_summary = self.din_attention(candidate_summary, history_embeddings, batch["history_mask"])
        dense_summary = self.dense_projection(batch["dense_features"])
        context_enhanced = self.context_projection(torch.cat([context_summary, component_history_summary], dim=-1))

        interaction = candidate_summary * history_summary
        difference = torch.abs(candidate_summary - history_summary)
        context_interaction = candidate_summary * context_enhanced

        fused = torch.cat(
            [
                candidate_summary,
                history_summary,
                context_summary,
                context_enhanced,
                component_history_summary,
                interaction,
                difference,
                context_interaction,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class CreatorwyxGroupedDINAdapter(nn.Module):
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
        self.context_projection = nn.Sequential(
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
        self.action_attention = DINActivationUnit(config.embedding_dim)
        self.content_attention = DINActivationUnit(config.embedding_dim)
        self.item_attention = DINActivationUnit(config.embedding_dim)
        self.route_gate = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, 3),
        )
        self.output = nn.Sequential(
            nn.Linear(config.embedding_dim * 11 + config.hidden_dim, config.hidden_dim),
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
        context_enhanced = self.context_projection(torch.cat([context_summary, component_history_summary], dim=-1))

        action_mask = batch["history_mask"] & (batch["history_group_ids"] == 1)
        content_mask = batch["history_mask"] & (batch["history_group_ids"] == 2)
        item_mask = batch["history_mask"] & (batch["history_group_ids"] == 3)

        action_summary = self.action_attention(candidate_summary, history_embeddings, action_mask)
        content_summary = self.content_attention(candidate_summary, history_embeddings, content_mask)
        item_summary = self.item_attention(candidate_summary, history_embeddings, item_mask)

        route_logits = self.route_gate(torch.cat([candidate_summary, context_enhanced, dense_summary], dim=-1))
        route_weights = torch.softmax(route_logits, dim=-1)
        route_stack = torch.stack([action_summary, content_summary, item_summary], dim=1)
        grouped_summary = (route_stack * route_weights.unsqueeze(-1)).sum(dim=1)
        route_spread = route_stack.std(dim=1, correction=0)

        grouped_interaction = candidate_summary * grouped_summary
        grouped_difference = torch.abs(candidate_summary - grouped_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                context_enhanced,
                action_summary,
                content_summary,
                item_summary,
                grouped_summary,
                component_history_summary,
                grouped_interaction,
                grouped_difference,
                route_spread,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["CreatorwyxDINAdapter", "CreatorwyxGroupedDINAdapter"]